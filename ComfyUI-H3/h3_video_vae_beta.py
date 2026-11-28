"""Beta native-ComfyUI H3 video VAE decoder.

Uses an already-loaded ComfyUI VAE and its normal VAE wrapper for memory
management, latent handling, and output conversion. During decoding only,
the H3 decoder's Comfy Kitchen INT8 Sage attention call is bypassed in favor
of ComfyUI's optimized_attention implementation.

This is intended for testing the Kijai Lite INT8 ConvRot H3 VAE checkpoint
when comfy_kitchen's Sage INT8 kernel is incompatible with the installed
CUDA driver/runtime.
"""
from __future__ import annotations

import types

import torch

from comfy.ldm.modules.attention import optimized_attention


def _h3_optimized_attention(self, query, key, value):
    """Use ComfyUI's regular attention implementation instead of Sage INT8."""
    out = optimized_attention(
        query,
        key,
        value,
        self.heads,
        skip_reshape=True,
    )
    return torch.nan_to_num(out)


class H3VideoVAEDecodeBeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "video_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "diagnostic")
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE Decode (BETA)"

    @torch.inference_mode()
    def decode(self, vae, video_latent):
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise TypeError(
                "Expected a ComfyUI LATENT dictionary containing 'samples'."
            )

        samples = video_latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise TypeError("LATENT['samples'] must be a torch.Tensor.")
        if samples.ndim != 5:
            raise ValueError(
                "Expected a video latent shaped [B,C,T,H,W], "
                f"received {tuple(samples.shape)}."
            )
        if samples.shape[1] != 24:
            raise ValueError(
                "Expected 24-channel MiniMax H3 video latents, "
                f"received {samples.shape[1]} channels."
            )
        if not torch.isfinite(samples).all():
            raise ValueError("Video latent contains NaN or Inf.")

        first_stage = getattr(vae, "first_stage_model", None)
        decoder = getattr(first_stage, "decoder", None)
        blocks = getattr(decoder, "transformer_blocks", None)

        if blocks is None or len(blocks) == 0:
            raise TypeError(
                "The supplied VAE does not expose the expected native "
                "MiniMax H3 decoder.transformer_blocks. Load the H3 video "
                "VAE with ComfyUI's VAE loader and connect that VAE here."
            )

        patched = []
        try:
            for index, block in enumerate(blocks):
                attention = getattr(block, "attn", None)
                if attention is None:
                    continue

                original = getattr(attention, "attention", None)
                if not callable(original):
                    continue

                had_instance_override = "attention" in attention.__dict__
                previous_override = attention.__dict__.get("attention")

                attention.attention = types.MethodType(
                    _h3_optimized_attention, attention
                )
                patched.append(
                    (attention, had_instance_override, previous_override)
                )

            if not patched:
                raise RuntimeError(
                    "Found H3 transformer blocks but no compatible attention "
                    "methods to patch."
                )

            print(
                "[H3 Video VAE BETA] Decoding video latent "
                f"{tuple(samples.shape)} with regular ComfyUI attention; "
                f"patched {len(patched)} decoder blocks. "
                "Quantized decoder weights remain unchanged.",
                flush=True,
            )

            # Keep ComfyUI's own VAE wrapper in charge of model loading,
            # offloading, tiling, dtype selection, and output conversion.
            images = vae.decode(samples)

        finally:
            for attention, had_override, previous_override in reversed(patched):
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
                "Expected ComfyUI video IMAGE output [B,T,H,W,C], "
                f"received {shape}."
            )
        if images.shape[-1] != 3:
            raise RuntimeError(
                f"Expected RGB video output, received {images.shape[-1]} channels."
            )
        if not torch.isfinite(images).all():
            raise RuntimeError("Decoded video contains NaN or Inf.")

        diagnostic = (
            "BETA decode completed using ComfyUI optimized_attention; "
            f"patched {len(patched)} H3 decoder attention modules. "
            "INT8 checkpoint weights were not modified."
        )
        return (images, diagnostic)
