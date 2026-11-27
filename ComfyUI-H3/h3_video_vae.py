"""ComfyUI-native MiniMax H3 video VAE loader and decoder nodes.

The actual model implementation, quantized ConvRot handling, dtype selection,
tiling, and ModelPatcher/offload lifecycle are delegated to ComfyUI's own
comfy.sd.VAE implementation. This avoids a second, incompatible VAE port.
"""
from __future__ import annotations

import folder_paths
import torch


class H3VideoVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = [
            name for name in folder_paths.get_filename_list("vae")
            if "minimax_h3_video_vae" in name.lower()
            or "minimax-h3-video-vae" in name.lower()
        ]
        if not names:
            names = ["<put minimax_h3_video_vae checkpoint in models/vae>"]
        return {"required": {"checkpoint": (names,)}}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE"

    def load_vae(self, checkpoint):
        if checkpoint.startswith("<put "):
            raise RuntimeError(
                "No MiniMax H3 video VAE checkpoint was found. Place "
                "minimax_h3_video_vae_*.safetensors in ComfyUI/models/vae "
                "and refresh ComfyUI."
            )

        path = folder_paths.get_full_path_or_raise("vae", checkpoint)
        # Match ComfyUI's native VAELoader path. This is important for H3's
        # INT8+ConvRot tensor metadata and quantized operation selection.
        import comfy.sd
        import comfy.utils

        state_dict, metadata = comfy.utils.load_torch_file(
            path, return_metadata=True
        )
        vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()

        if getattr(vae, "latent_channels", None) != 24:
            raise RuntimeError(
                f"{checkpoint} did not load as a 24-channel H3 video VAE "
                f"(detected latent_channels={getattr(vae, 'latent_channels', None)})."
            )
        if getattr(vae, "latent_dim", None) != 3 or getattr(vae, "not_video", False):
            raise RuntimeError(
                f"{checkpoint} is not a MiniMax H3 video VAE checkpoint."
            )

        # Preserve ComfyUI's normal disk-backed VAE reload lifecycle.
        vae.patcher.cached_patcher_init = (
            comfy.sd.load_vae_patcher, (path, metadata, None)
        )
        print(
            f"[H3 Video VAE] Loaded through ComfyUI native VAE path: {checkpoint}",
            flush=True,
        )
        return (vae,)


class H3VideoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "video_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE Decode"

    @torch.inference_mode()
    def decode(self, vae, video_latent):
        if not isinstance(video_latent, dict) or "samples" not in video_latent:
            raise TypeError("Expected a ComfyUI LATENT dictionary with 'samples'.")
        latent = video_latent["samples"]
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24:
            raise ValueError(
                f"Expected H3 video latent [1,24,T,H,W], got {tuple(latent.shape)}"
            )
        if getattr(vae, "latent_channels", None) != 24 or getattr(vae, "latent_dim", None) != 3:
            raise TypeError("H3 Video VAE Decode expects the H3 Video VAE loader output.")

        images = vae.decode(latent)
        if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] != 3:
            raise RuntimeError(
                f"Unexpected H3 VAE decoded output: {tuple(images.shape)}"
            )
        # Native MiniMaxH3VideoVAE.decode already returns pixels in [0,1].
        return (images.permute(0, 2, 3, 4, 1).flatten(0, 1).contiguous(),)
