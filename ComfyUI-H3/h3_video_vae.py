"""LightVAE-Lite H3 video decoder nodes for ComfyUI.

Uses the repository's checkpoint-aware standalone LightVAE decoder and the
same CPU-resident, one-transformer-block-at-a-time CUDA execution used by
minimax_h3_lightvae_decoder.py. It does not route LightVAE checkpoints through
ComfyUI's unrelated native VAE loader.
"""
from __future__ import annotations

import gc
import sys
import types
from pathlib import Path

import torch


# The repository keeps the standalone LightVAE implementation beside the
# ComfyUI-H3 custom-node directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIGHTVAE_ROOT = REPOSITORY_ROOT / "lightvae_h3_standalone"
MODEL_DIR = REPOSITORY_ROOT / "models" / "vae"
DECODER_FILENAME = "lightvae-lite-h3-decoder.safetensors"
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lightvae_h3_standalone.lightvae.h3_lightvae_decoder import Decoder
from lightvae_h3_standalone.lightvae.utils import h3_decoded_frames


def _streaming_decoder_forward(self, hidden_states):
    """Run the official H3 decoder forward while moving one block to CUDA."""
    batch, channels, frames, height, width = hidden_states.shape

    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch, frames * height * width, channels
    )

    device = self._h3_device
    self.proj_in.to(device=device, dtype=DTYPE)
    try:
        hidden_states = self.proj_in(hidden_states.to(device=device, dtype=DTYPE))
    finally:
        self.proj_in.to(device="cpu")

    num_patches = hidden_states.shape[1]
    register_tokens = self.register_tokens.to(
        device=device, dtype=DTYPE
    ).expand(batch, -1, -1)
    cls_token = torch.zeros_like(hidden_states[:, :1, :])
    hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
    del register_tokens, cls_token

    grids = [
        2.0 * (
            torch.arange(0.5, size, dtype=torch.float32, device=device) / size
        ) - 1.0
        for size in (frames, height, width)
    ]
    position_ids = torch.stack(
        torch.meshgrid(*grids, indexing="ij"), dim=-1
    ).flatten(0, 2)
    position_ids = position_ids.unsqueeze(0).expand(batch, -1, -1)
    suffix_ids = position_ids.new_zeros(
        (batch, self.num_register_tokens + 1, 3)
    )
    position_ids = torch.cat([position_ids, suffix_ids], dim=1)
    self.rope.to(device=device)
    try:
        rotary_emb = self.rope(position_ids)
    finally:
        self.rope.to(device="cpu")
    del grids, position_ids, suffix_ids

    total_blocks = len(self.transformer_blocks)
    for index, block in enumerate(self.transformer_blocks):
        print(
            f"\r[H3 LightVAE] Decoder block {index + 1}/{total_blocks}",
            end="",
            flush=True,
        )
        block.to(device=device, dtype=DTYPE)
        try:
            # The standalone decoder's reference path performs RMSNorm in FP32.
            block.norm1.to(device=device, dtype=torch.float32)
            block.norm2.to(device=device, dtype=torch.float32)
            hidden_states = block(hidden_states, rotary_emb)
        finally:
            block.to(device="cpu")
    print(flush=True)
    del rotary_emb

    self.norm_out.to(device=device, dtype=DTYPE)
    try:
        hidden_states = self.norm_out(hidden_states)
    finally:
        self.norm_out.to(device="cpu")

    self.proj_out.to(device=device, dtype=DTYPE)
    try:
        hidden_states = self.proj_out(hidden_states)
    finally:
        self.proj_out.to(device="cpu")

    hidden_states = hidden_states[:, :num_patches, :]
    patch_size = self.patch_size
    patch_size_t = self.patch_size_t
    hidden_states = hidden_states.view(
        batch,
        frames,
        height,
        width,
        self.out_channels,
        patch_size_t,
        patch_size,
        patch_size,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(
        batch,
        self.out_channels,
        frames * patch_size_t,
        height * patch_size,
        width * patch_size,
    )


def _streaming_decode_clip(self, latents):
    """Restore H3 latent normalization and run the post-quant projection."""
    mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
    std = self.latents_std.float().view(1, -1, 1, 1, 1)
    latents = (latents.float() * std + mean).to(
        device=self._h3_device, dtype=DTYPE
    )

    self.post_quant_conv.to(device=self._h3_device, dtype=DTYPE)
    try:
        latents = self.post_quant_conv(latents)
    finally:
        self.post_quant_conv.to(device="cpu")
    return self.decoder(latents)


class _H3LightVAEDecoder:
    """ComfyUI carrier for the standalone LightVAE-Lite decoder."""

    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = str(checkpoint)
        self.device = DEVICE


def _register_model_directory():
    """Expose the project's models/vae directory to ComfyUI's VAE browser."""
    if not MODEL_DIR.is_dir():
        return
    try:
        import folder_paths

        registered = folder_paths.get_folder_paths("vae")
        model_dir = str(MODEL_DIR)
        if model_dir not in registered:
            folder_paths.add_model_folder_path("vae", model_dir)
    except (ImportError, KeyError, AttributeError):
        # The node still reports the expected path if ComfyUI's path API differs.
        pass


_register_model_directory()


class H3VideoVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        _register_model_directory()
        try:
            names = [
                name
                for name in folder_paths.get_filename_list("vae")
                if Path(name).name.lower() == DECODER_FILENAME
            ]
        except (KeyError, AttributeError):
            names = []

        if not names and (MODEL_DIR / DECODER_FILENAME).is_file():
            names = [DECODER_FILENAME]
        if not names:
            names = [f"<missing: {MODEL_DIR / DECODER_FILENAME}>"]
        return {"required": {"checkpoint": (sorted(set(names)),)}}

    RETURN_TYPES = ("H3_VIDEO_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE (LightVAE-Lite)"

    def load_vae(self, checkpoint):
        if checkpoint.startswith("<missing:"):
            raise FileNotFoundError(
                "LightVAE-Lite H3 decoder checkpoint was not found. Expected: "
                f"{MODEL_DIR / DECODER_FILENAME}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("H3 LightVAE-Lite decoding requires CUDA.")

        selected = Path(checkpoint)
        if not selected.is_absolute():
            project_checkpoint = MODEL_DIR / selected
            if project_checkpoint.is_file():
                selected = project_checkpoint
            else:
                import folder_paths

                selected = Path(
                    folder_paths.get_full_path_or_raise("vae", checkpoint)
                )
        selected = selected.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Decoder checkpoint not found: {selected}")
        if selected.name.lower() != DECODER_FILENAME:
            raise ValueError(
                f"Expected {DECODER_FILENAME}, received {selected.name}"
            )
        if not LIGHTVAE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Standalone LightVAE source directory not found: {LIGHTVAE_ROOT}"
            )

        print(
            f"[H3 LightVAE] Loading decoder checkpoint: {selected}",
            flush=True,
        )
        model = Decoder(checkpoint=selected)
        model.eval()
        model.requires_grad_(False)

        # Keep checkpoint weights in CPU RAM. The bound forward methods move
        # only the active projection/transformer block to the 4 GB GPU.
        model.to(device="cpu", dtype=DTYPE)
        model.decoder.rope.materialize_buffer("cpu")
        model.decoder._h3_device = DEVICE
        model.decoder.forward = types.MethodType(
            _streaming_decoder_forward, model.decoder
        )
        model._h3_device = DEVICE
        model._decode_clip = types.MethodType(_streaming_decode_clip, model)

        print(
            "[H3 LightVAE] Loaded. "
            f"Retained transformer blocks: {len(model.decoder.transformer_blocks)}; "
            f"tile size: {model.tile_size}; overlap: {model.tile_overlap}.",
            flush=True,
        )
        gc.collect()
        return (_H3LightVAEDecoder(model, selected),)


class H3VideoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("H3_VIDEO_VAE",),
                "video_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE Decode (LightVAE-Lite)"

    @torch.inference_mode()
    def decode(self, vae, video_latent):
        if not isinstance(vae, _H3LightVAEDecoder):
            raise TypeError("Expected the H3 LightVAE-Lite decoder from H3 Video VAE.")
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise TypeError("Expected a ComfyUI LATENT dictionary containing 'samples'.")

        latent = video_latent["samples"]
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24:
            raise ValueError(
                f"Expected H3 video latent [1,24,T,H,W], got {tuple(latent.shape)}"
            )
        if not torch.isfinite(latent).all():
            raise ValueError("H3 video latent contains NaN or Inf.")
        expected_frames = h3_decoded_frames(int(latent.shape[2]))

        # Keep latent storage off the GPU while the tiled decoder processes it.
        latent_cpu = latent.detach().to(device="cpu", dtype=torch.float32)
        print(
            f"[H3 LightVAE] Decoding latent {tuple(latent.shape)} "
            f"to {expected_frames} frames.",
            flush=True,
        )
        video = vae.model(latent_cpu)

        expected_shape = (
            1,
            3,
            expected_frames,
            int(latent.shape[-2]) * 16,
            int(latent.shape[-1]) * 16,
        )
        if tuple(video.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected LightVAE output {tuple(video.shape)}; "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(video).all():
            raise RuntimeError("LightVAE decoded video contains NaN or Inf.")

        # The standalone decoder returns RGB in [-1,1]; ComfyUI IMAGE is [0,1].
        images = (
            (video[0].detach().float() + 1.0)
            .mul_(0.5)
            .clamp_(0.0, 1.0)
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
        )
        del latent_cpu, video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (images,)
