"""Beta native-ComfyUI H3 video VAE decoder.

Keeps the native H3 VAE decode path and routes decoder attention through
ComfyUI optimized_attention. The tiled path also preserves global H3 rotary
positions across spatial tiles and blends against already-composited neighbours.
"""
from __future__ import annotations

import types

import torch

from comfy.ldm.modules.attention import optimized_attention


def _h3_optimized_attention(self, query, key, value):
    out = optimized_attention(
        query, key, value, self.heads, skip_reshape=True
    )
    return torch.nan_to_num(out)


def _h3_create_token_ids(patch_dims, device, dtype, full_dims, offset):
    """Build RoPE coordinates in the full latent grid, not tile-local space."""
    coords_list = []
    for dim_size, full_size, off in zip(patch_dims, full_dims, offset):
        coords = torch.arange(
            0.5, dim_size, dtype=dtype, device=device
        ) + off
        coords = coords / full_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(
        torch.meshgrid(*coords_list, indexing="ij"), dim=-1
    )
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


def _h3_position_aware_decoder_forward(
    self, x, full_dims=None, offsets=None
):
    """Native ViT3D decoder forward with global coordinates for each tile."""
    import comfy.ops

    batch, channels, latent_t, latent_h, latent_w = x.shape
    hidden = self.x_embedder(x.flatten(2).transpose(1, 2))
    num_patches = hidden.shape[1]
    num_suffix = 1 + self.num_register_tokens

    register_tokens = comfy.ops.cast_to_input(
        self.register_tokens, hidden
    ).expand(batch, -1, -1)
    hidden = torch.cat(
        [hidden, register_tokens, torch.zeros_like(hidden[:, 0:1, :])],
        dim=1,
    )

    patch_dims = (latent_t, latent_h, latent_w)
    if full_dims is None:
        full_dims = patch_dims
    if offsets is None:
        offsets = (0, 0, 0)
    if isinstance(offsets, tuple):
        offsets = [offsets]

    if batch % len(offsets):
        raise ValueError(
            "Tile offset count must divide the decoder batch size."
        )
    per_offset_batch = batch // len(offsets)
    img_ids = torch.cat(
        [
            _h3_create_token_ids(
                patch_dims, x.device, x.dtype, full_dims, offset
            ).expand(per_offset_batch, -1, -1)
            for offset in offsets
        ],
        dim=0,
    )
    suffix_ids = torch.zeros(
        (batch, num_suffix, self.pos_embed.n_dim),
        device=x.device,
        dtype=img_ids.dtype,
    )
    img_ids = torch.cat([img_ids, suffix_ids], dim=1)
    rotary_pos_emb = self.pos_embed(img_ids)

    for block in self.transformer_blocks:
        hidden = block(hidden, rotary_pos_emb)

    output = self.proj_out(self.norm_out(hidden))[:, :num_patches, :]
    output = output.view(
        batch,
        latent_t,
        latent_h,
        latent_w,
        self.out_channels,
        self.patch_size_t,
        self.patch_size,
        self.patch_size,
    )
    output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return output.reshape(
        batch,
        self.out_channels,
        latent_t * self.patch_size_t,
        latent_h * self.patch_size,
        latent_w * self.patch_size,
    )


def _h3_position_aware_decode_pixels(
    self, latents, full_dims=None, offsets=None
):
    decoded_latents = self.post_quant_conv(latents)
    return self.decoder(
        decoded_latents, full_dims=full_dims, offsets=offsets
    )


def _h3_decode_tile_row(self, z_row, x_idx, x_len, full_dims, zi):
    """Retain ComfyUI's memory-aware tile batching while passing global offsets."""
    import comfy.model_management

    free = comfy.model_management.get_free_memory(z_row.device)
    batch = int(max(1, min(4, free // (128 * 2**20 * z_row.shape[0]))))
    slices = [
        z_row[..., j_pos // self.vae_ratio:(j_pos + j_len) // self.vae_ratio]
        for j_pos, j_len in zip(x_idx, x_len)
    ]
    offsets = [
        (0, zi, j_pos // self.vae_ratio) for j_pos in x_idx
    ]
    for start in range(0, len(slices), batch):
        group = slices[start:start + batch]
        group_offsets = offsets[start:start + batch]
        decoded = self._decode_pixels(
            torch.cat(group), full_dims=full_dims, offsets=group_offsets
        )
        yield from decoded.chunk(len(group))


def _h3_tiled_decode(self, z, frames=slice(None)):
    """H3 spatial tiling with global RoPE coordinates and continuous blending.

    The row strip and left tail are captured after the preceding blend. This
    prevents overlap intersections from blending against stale raw tile pixels.
    """
    ratio = int(self.vae_ratio)
    height = int(z.shape[-2]) * ratio
    width = int(z.shape[-1]) * ratio
    full_dims = (int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1]))

    y_idx, y_len, y_overlap = self.split_tiles(height)
    x_idx, x_len, x_overlap = self.split_tiles(width)

    canvas = None
    strip = None
    out_y = 0

    for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
        zi = i_pos // ratio
        zl = i_len // ratio
        new_strip = None
        left_tail = None
        out_x = 0
        tile_height = 0
        tiles = self._decode_tile_row(
            z[..., zi:zi + zl, :], x_idx, x_len, full_dims, zi
        )

        for j in range(len(x_idx)):
            tile = next(tiles)[:, :, frames]

            if i > 0:
                previous_row = strip[..., :, x_idx[j]:x_idx[j] + x_len[j]]
                tile = self.blend(
                    previous_row, tile, y_overlap[i - 1], dim=-2
                )
            if j > 0:
                tile = self.blend(
                    left_tail, tile, x_overlap[j - 1], dim=-1
                )

            left_tail = (
                tile[..., :, -x_overlap[j]:].clone()
                if j < len(x_idx) - 1
                else None
            )
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]

            if canvas is None:
                canvas = torch.empty(
                    *tile.shape[:-2],
                    height,
                    width,
                    dtype=tile.dtype,
                    device=tile.device,
                )

            if i < len(y_idx) - 1:
                if new_strip is None:
                    new_strip = torch.empty(
                        *tile.shape[:-2],
                        y_overlap[i],
                        width,
                        dtype=tile.dtype,
                        device=tile.device,
                    )
                new_strip[
                    ..., :, out_x:out_x + tile.shape[-1]
                ] = tile[..., -y_overlap[i]:, :]
                tile = tile[..., :-y_overlap[i], :]

            canvas[
                ..., out_y:out_y + tile.shape[-2],
                out_x:out_x + tile.shape[-1]
            ].copy_(tile)
            tile_height = tile.shape[-2]
            out_x += tile.shape[-1]
            del tile

        strip = new_strip
        out_y += tile_height

    if canvas is None:
        raise RuntimeError("H3 tiled decoder produced no spatial tiles.")
    return canvas


class H3VideoVAEDecodeBeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "video_latent": ("LATENT",),
                "tiling": ("BOOLEAN", {"default": True}),
                "tile_size": (
                    "INT",
                    {"default": 256, "min": 128, "max": 1024, "step": 16},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 64, "min": 16, "max": 512, "step": 16},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "diagnostic")
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE Decode (BETA)"

    @torch.inference_mode()
    def decode(
        self, vae, video_latent, tiling=True, tile_size=256, tile_overlap=64
    ):
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise TypeError("Expected LATENT dictionary containing 'samples'.")

        samples = video_latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise TypeError("LATENT['samples'] must be a torch.Tensor.")
        if samples.ndim != 5:
            raise ValueError(
                "Expected video latent [B,C,T,H,W], "
                f"received {tuple(samples.shape)}."
            )
        if samples.shape[1] != 24:
            raise ValueError(
                "Expected 24-channel H3 video latents, "
                f"received {samples.shape[1]} channels."
            )
        if not torch.isfinite(samples).all():
            raise ValueError("Video latent contains NaN or Inf.")

        first_stage = getattr(vae, "first_stage_model", None)
        decoder = getattr(first_stage, "decoder", None)
        blocks = getattr(decoder, "transformer_blocks", None)
        if blocks is None or len(blocks) == 0:
            raise TypeError(
                "The supplied VAE does not expose "
                "decoder.transformer_blocks from the native H3 VAE."
            )
        if not all(
            hasattr(decoder, name)
            for name in (
                "x_embedder", "register_tokens", "pos_embed",
                "num_register_tokens", "patch_size", "patch_size_t",
                "out_channels", "norm_out", "proj_out",
            )
        ):
            raise TypeError(
                "The loaded H3 decoder does not expose the expected "
                "ComfyUI ViT3D decoder interface."
            )
        if not callable(getattr(first_stage, "_decode_pixels", None)):
            raise TypeError("The loaded H3 VAE is missing _decode_pixels().")
        if not callable(getattr(first_stage, "split_tiles", None)):
            raise TypeError("The loaded H3 VAE is missing split_tiles().")
        if not callable(getattr(first_stage, "blend", None)):
            raise TypeError("The loaded H3 VAE is missing blend().")

        ratio = int(getattr(first_stage, "vae_ratio", 16))
        if tile_size % ratio:
            raise ValueError(
                f"tile_size ({tile_size}) must be divisible by H3 vae_ratio ({ratio})."
            )
        if tile_overlap % ratio:
            raise ValueError(
                f"tile_overlap ({tile_overlap}) must be divisible by H3 vae_ratio ({ratio})."
            )
        if tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size.")

        settings = {}
        for name, value in (
            ("tiling", bool(tiling)),
            ("tile_size", int(tile_size)),
            ("tile_overlap_min", int(tile_overlap)),
        ):
            if not hasattr(first_stage, name):
                raise TypeError(
                    f"Loaded H3 VAE is missing expected setting {name!r}."
                )
            settings[name] = getattr(first_stage, name)

        def install_method(obj, name, function):
            had_override = name in obj.__dict__
            previous_override = obj.__dict__.get(name)
            setattr(obj, name, types.MethodType(function, obj))
            return (obj, name, had_override, previous_override)

        patched_methods = []
        patched_attention = []
        try:
            for block in blocks:
                attention = getattr(block, "attn", None)
                if attention is None:
                    continue
                original = getattr(attention, "attention", None)
                if not callable(original):
                    continue

                had_override = "attention" in attention.__dict__
                previous_override = attention.__dict__.get("attention")
                attention.attention = types.MethodType(
                    _h3_optimized_attention, attention
                )
                patched_attention.append(
                    (attention, had_override, previous_override)
                )

            if not patched_attention:
                raise RuntimeError(
                    "No compatible H3 decoder attention methods found."
                )

            patched_methods.append(
                install_method(
                    decoder, "forward", _h3_position_aware_decoder_forward
                )
            )
            patched_methods.append(
                install_method(
                    first_stage, "_decode_pixels",
                    _h3_position_aware_decode_pixels,
                )
            )
            patched_methods.append(
                install_method(first_stage, "tiled_decode", _h3_tiled_decode)
            )
            patched_methods.append(
                install_method(first_stage, "_decode_tile_row", _h3_decode_tile_row)
            )

            first_stage.tiling = bool(tiling)
            first_stage.tile_size = int(tile_size)
            first_stage.tile_overlap_min = int(tile_overlap)

            mode = (
                f"spatial tiling ON (tile={tile_size}px, overlap={tile_overlap}px)"
                if tiling
                else "spatial tiling OFF"
            )
            print(
                "[H3 Video VAE BETA] Decoding "
                f"{tuple(samples.shape)} with ComfyUI optimized_attention; "
                f"patched {len(patched_attention)} decoder blocks; "
                "global tile RoPE positions and composited overlap blending; "
                f"{mode}.",
                flush=True,
            )

            images = vae.decode(samples)

        finally:
            for name, value in settings.items():
                setattr(first_stage, name, value)

            for obj, name, had_override, previous_override in reversed(
                patched_methods
            ):
                if had_override:
                    setattr(obj, name, previous_override)
                else:
                    delattr(obj, name)

            for attention, had_override, previous_override in reversed(
                patched_attention
            ):
                if had_override:
                    attention.attention = previous_override
                else:
                    delattr(attention, "attention")

        if not isinstance(images, torch.Tensor) or images.ndim != 5:
            shape = (
                tuple(images.shape)
                if isinstance(images, torch.Tensor)
                else type(images).__name__
            )
            raise RuntimeError(
                "Expected decoded H3 video [B,T,H,W,C], received "
                f"{shape}."
            )
        if images.shape[-1] != 3:
            raise RuntimeError(
                f"Expected RGB output, received {images.shape[-1]} channels."
            )
        if not torch.isfinite(images).all():
            raise RuntimeError("Decoded video contains NaN or Inf.")

        batch, frames, height, width, channels = images.shape
        image_batch = images.reshape(
            batch * frames, height, width, channels
        ).contiguous()

        diagnostic = (
            "BETA decode completed with ComfyUI optimized_attention; "
            f"patched {len(patched_attention)} H3 attention blocks; "
            "global tile RoPE coordinates and composited overlap blending; "
            f"spatial tiling={'on' if tiling else 'off'}, "
            f"tile={tile_size}px, overlap={tile_overlap}px; "
            f"output={tuple(image_batch.shape)}. "
            "INT8 checkpoint weights were not modified."
        )
        return (image_batch, diagnostic)
