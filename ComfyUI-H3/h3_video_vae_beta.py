"""Beta native-ComfyUI H3 video VAE decoder.

Uses the loaded ComfyUI VAE and the H3 VAE's own spatial tiled decoder.
During decoding, H3 decoder attention is temporarily routed through
ComfyUI optimized_attention instead of the incompatible INT8 Sage kernel.
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

        # These are the actual controls used by MiniMaxH3VideoVAE._adaptive_decode
        # and tiled_decode. tile_size/overlap are in decoded pixel-space units.
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
                raise TypeError(f"Loaded H3 VAE is missing expected setting {name!r}.")
            settings[name] = getattr(first_stage, name)

        patched = []
        images = None
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
                patched.append((attention, had_override, previous_override))

            if not patched:
                raise RuntimeError("No compatible H3 decoder attention methods found.")

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
                f"{tuple(samples.shape)} with regular ComfyUI attention; "
                f"patched {len(patched)} decoder blocks; {mode}. "
                "Quantized decoder weights remain unchanged.",
                flush=True,
            )

            # H3's own decode() calls _adaptive_decode(), which dispatches to
            # tiled_decode() when first_stage.tiling is enabled. ComfyUI's VAE
            # wrapper still manages model loading, offloading and output device.
            images = vae.decode(samples)

        finally:
            for name, value in settings.items():
                setattr(first_stage, name, value)
            for attention, had_override, previous_override in reversed(patched):
                if had_override:
                    attention.attention = previous_override
                else:
                    delattr(attention, "attention")

        # ComfyUI's IMAGE type is [N,H,W,C]. H3 video decode returns
        # [B,T,H,W,C]; flatten batch and time so Save Image receives 3D frames.
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
            f"patched {len(patched)} H3 attention blocks; "
            f"spatial tiling={'on' if tiling else 'off'}, "
            f"tile={tile_size}px, overlap={tile_overlap}px; "
            f"output={tuple(image_batch.shape)}. "
            "INT8 checkpoint weights were not modified."
        )
        return (image_batch, diagnostic)
