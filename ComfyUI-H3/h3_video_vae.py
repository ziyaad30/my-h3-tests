"""ComfyUI nodes for the MiniMax H3 LightVAE video decoder.

The decoder architecture and checkpoint loader are the project's bundled
LightVAE implementation. Decoder blocks stay in CPU RAM and are moved to CUDA
one at a time for inference, matching the existing GTX 1650 streaming path.
"""
from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path
from types import MethodType

import torch
import folder_paths


# In this repository layout, lightvae_h3_standalone is a sibling of the
# ComfyUI-H3 custom-node package. Add the repository root for that import.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _streaming_decoder_forward(self, hidden_states):
    batch, channels, frames, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch, frames * height * width, channels
    )

    device = torch.device("cuda:0")
    dtype = torch.float16

    self.proj_in.to(device=device, dtype=dtype)
    hidden_states = self.proj_in(hidden_states.to(device=device, dtype=dtype))
    self.proj_in.to(device="cpu")
    num_patches = hidden_states.shape[1]

    register_tokens = self.register_tokens.to(
        device=device, dtype=dtype
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
    ).flatten(0, 2).unsqueeze(0).expand(batch, -1, -1)
    suffix_ids = position_ids.new_zeros(
        (batch, self.num_register_tokens + 1, 3)
    )
    position_ids = torch.cat([position_ids, suffix_ids], dim=1)

    self.rope.to(device=device)
    rotary_emb = self.rope(position_ids)
    del grids, position_ids, suffix_ids
    self.rope.to(device="cpu")

    total_blocks = len(self.transformer_blocks)
    for index, block in enumerate(self.transformer_blocks):
        block.to(device=device, dtype=dtype)
        block.norm1.to(device=device, dtype=torch.float32)
        block.norm2.to(device=device, dtype=torch.float32)
        hidden_states = block(hidden_states, rotary_emb)
        block.to(device="cpu")
        if index == 0 or (index + 1) % 4 == 0 or index + 1 == total_blocks:
            print(
                f"[H3 Video VAE] Decoder block {index + 1}/{total_blocks}",
                flush=True,
            )

    del rotary_emb
    self.norm_out.to(device=device, dtype=dtype)
    hidden_states = self.norm_out(hidden_states)
    self.norm_out.to(device="cpu")

    self.proj_out.to(device=device, dtype=dtype)
    hidden_states = self.proj_out(hidden_states)
    self.proj_out.to(device="cpu")

    hidden_states = hidden_states[:, :num_patches, :]
    hidden_states = hidden_states.view(
        batch, frames, height, width,
        self.out_channels, self.patch_size_t,
        self.patch_size, self.patch_size,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return hidden_states.reshape(
        batch, self.out_channels,
        frames * self.patch_size_t,
        height * self.patch_size,
        width * self.patch_size,
    )


def _streaming_decode_clip(self, latents):
    device = torch.device("cuda:0")
    mean = self.latents_mean.float().view(1, -1, 1, 1, 1)
    std = self.latents_std.float().view(1, -1, 1, 1, 1)
    latents = (latents.float() * std + mean).to(device=device, dtype=torch.float16)

    self.post_quant_conv.to(device=device, dtype=torch.float16)
    latents = self.post_quant_conv(latents)
    self.post_quant_conv.to(device="cpu")
    return self.decoder(latents)


class _H3VideoVAERuntime:
    def __init__(self, checkpoint):
        if not torch.cuda.is_available():
            raise RuntimeError("H3 Video VAE decoding requires CUDA.")
        # Import from the repository's existing verified LightVAE implementation.
        try:
            from lightvae_h3_standalone.lightvae.h3_lightvae_decoder import Decoder
        except ImportError as exc:
            raise ImportError(
                "The H3 LightVAE source package is missing. Keep the "
                "'lightvae_h3_standalone' directory beside the 'ComfyUI-H3' "
                "directory from this repository."
            ) from exc

        self.checkpoint = str(checkpoint)
        self.lock = threading.RLock()
        self.decoder = Decoder(checkpoint=checkpoint)
        self.decoder.eval()
        self.decoder.requires_grad_(False)
        self.decoder.to(device="cpu", dtype=torch.float16)
        self.decoder.decoder.rope.materialize_buffer("cpu")
        self.decoder.decoder.forward = MethodType(
            _streaming_decoder_forward, self.decoder.decoder
        )
        self.decoder._decode_clip = MethodType(
            _streaming_decode_clip, self.decoder
        )
        print(
            f"[H3 Video VAE] Loaded decoder checkpoint: {self.checkpoint}\n"
            f"[H3 Video VAE] Tile size={self.decoder.tile_size}, "
            f"overlap={self.decoder.tile_overlap}, "
            f"retained blocks={len(self.decoder.decoder.transformer_blocks)}",
            flush=True,
        )

    @torch.inference_mode()
    def decode(self, latent):
        if not isinstance(latent, torch.Tensor):
            raise TypeError("H3 Video VAE expects a video latent tensor.")
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24:
            raise ValueError(
                f"Expected H3 video latent [1,24,T,H,W], got {tuple(latent.shape)}"
            )
        if not bool(torch.isfinite(latent).all().item()):
            raise ValueError("H3 video latent contains NaN or Inf.")

        with self.lock:
            try:
                result = self.decoder(
                    latent.detach().to(device="cpu", dtype=torch.float32)
                )
                if result.ndim != 5 or result.shape[0] != 1 or result.shape[1] != 3:
                    raise RuntimeError(
                        f"Unexpected H3 decoder output shape: {tuple(result.shape)}"
                    )
                if not bool(torch.isfinite(result).all().item()):
                    raise RuntimeError("H3 decoded video contains NaN or Inf.")
                # ComfyUI IMAGE layout: [frames, height, width, RGB].
                return (
                    result[0]
                    .permute(1, 2, 3, 0)
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                    .clamp(-1.0, 1.0)
                    .add(1.0)
                    .mul(0.5)
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()


folder_paths.add_model_folder_path(
    "h3_video_vae", str(Path(folder_paths.models_dir) / "vae")
)
folder_paths.folder_names_and_paths["h3_video_vae"] = (
    folder_paths.get_folder_paths("h3_video_vae"),
    {".safetensors"},
)


class H3VideoVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("h3_video_vae")
        if not files:
            files = ["<put H3 video VAE safetensors in models/vae>"]
        return {"required": {"checkpoint": (files,)}}

    RETURN_TYPES = ("H3_VIDEO_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE"

    def load_vae(self, checkpoint):
        if checkpoint.startswith("<put "):
            raise RuntimeError(
                "No H3 video VAE checkpoint found. Place the H3 decoder "
                "safetensors file in ComfyUI/models/vae and refresh ComfyUI."
            )
        path = folder_paths.get_full_path("h3_video_vae", checkpoint)
        if path is None:
            raise FileNotFoundError(f"Could not resolve H3 video VAE: {checkpoint}")
        return (_H3VideoVAERuntime(path),)


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
    TITLE = "H3 Video VAE Decode"

    def decode(self, vae, video_latent):
        if not isinstance(vae, _H3VideoVAERuntime):
            raise TypeError("H3 Video VAE Decode expects H3 Video VAE Loader output.")
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise TypeError("Expected a ComfyUI LATENT dictionary with 'samples'.")
        images = vae.decode(video_latent["samples"])
        return (images,)
