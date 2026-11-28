"""LightVAE-Lite H3 video encoder nodes for ComfyUI.

The encoder architecture and temporal/spatial tiling are provided by the
standalone H3 LightVAE implementation. Encoder layers are staged one at a
time on CUDA, following the working standalone roundtrip implementation.
"""
from __future__ import annotations

import gc
import logging
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger("MiniMax H3")

NODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIGHTVAE_ROOT = NODE_ROOT / "lightvae_h3_standalone"
MODEL_DIR = REPOSITORY_ROOT / "models" / "vae"
ENCODER_FILENAME = "lightvae-lite-h3-encoder.safetensors"
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

if str(NODE_ROOT) not in sys.path:
    sys.path.insert(0, str(NODE_ROOT))

from lightvae_h3_standalone.lightvae.h3_lightvae_encoder import Encoder
from lightvae_h3_standalone.lightvae.utils import latent_frames, padded_frames


def _streaming_encode_clip(self, video):
    """Run one official H3 encoder clip while staging one module at a time."""
    mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    video = ((video + 1.0) / 2.0 - mean) / std

    device = self._h3_device
    started = time.perf_counter()

    def report(stage):
        log.info("[LightVAE Encode] %s | %.1f sec elapsed", stage, time.perf_counter() - started)

    hidden_states = video.to(device=device, dtype=DTYPE)
    report("Input staged on GPU")

    self.encoder.conv_in.to(device=device, dtype=DTYPE)
    try:
        report("Initial convolution")
        hidden_states = self.encoder.conv_in(hidden_states)
    finally:
        self.encoder.conv_in.to(device="cpu")

    for stage_index, down_block in enumerate(self.encoder.down_blocks, start=1):
        report(f"Down stage {stage_index}/{len(self.encoder.down_blocks)}")
        for resnet_index, resnet in enumerate(down_block.resnets, start=1):
            report(f"  Residual block {resnet_index}/{len(down_block.resnets)}")
            resnet.to(device=device, dtype=DTYPE)
            try:
                hidden_states = resnet(hidden_states)
            finally:
                resnet.to(device="cpu")

        if down_block.downsamplers is not None:
            for downsampler_index, downsampler in enumerate(down_block.downsamplers, start=1):
                report(f"  Downsample {downsampler_index}/{len(down_block.downsamplers)}")
                downsampler.to(device=device, dtype=DTYPE)
                try:
                    hidden_states = downsampler(hidden_states)
                finally:
                    downsampler.to(device="cpu")

    report("Output normalization")
    self.encoder.norm_out.to(device=device, dtype=DTYPE)
    try:
        hidden_states = F.silu(self.encoder.norm_out(hidden_states))
    finally:
        self.encoder.norm_out.to(device="cpu")

    report("Output convolution")
    self.encoder.conv_out.to(device=device, dtype=DTYPE)
    try:
        moments = self.encoder.conv_out(hidden_states)
    finally:
        self.encoder.conv_out.to(device="cpu")

    report("Quantization convolution")
    self.quant_conv.to(device=device, dtype=DTYPE)
    try:
        moments = self.quant_conv(moments)
    finally:
        self.quant_conv.to(device="cpu")

    report("Clip complete")
    return moments.to(device="cpu")


class _H3LightVAEEncoder:
    """ComfyUI carrier for the standalone LightVAE-Lite encoder."""

    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = str(checkpoint)
        self.device = DEVICE


def _register_model_directory():
    if not MODEL_DIR.is_dir():
        return
    try:
        import folder_paths

        registered = folder_paths.get_folder_paths("vae")
        model_dir = str(MODEL_DIR)
        if model_dir not in registered:
            folder_paths.add_model_folder_path("vae", model_dir)
    except (ImportError, KeyError, AttributeError):
        pass


_register_model_directory()


class H3LightVAEEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        _register_model_directory()
        try:
            names = [
                name
                for name in folder_paths.get_filename_list("vae")
                if Path(name).name.lower() == ENCODER_FILENAME
            ]
        except (KeyError, AttributeError):
            names = []

        if not names and (MODEL_DIR / ENCODER_FILENAME).is_file():
            names = [ENCODER_FILENAME]
        if not names:
            names = [f"<missing: {MODEL_DIR / ENCODER_FILENAME}>"]
        return {"required": {"checkpoint": (sorted(set(names)),)}}

    RETURN_TYPES = ("H3_VIDEO_VAE_ENCODER",)
    RETURN_NAMES = ("encoder",)
    FUNCTION = "load_encoder"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 LightVAE Encoder Loader"

    def load_encoder(self, checkpoint):
        if checkpoint.startswith("<missing:"):
            raise FileNotFoundError(
                "LightVAE-Lite H3 encoder checkpoint was not found. Expected: "
                f"{MODEL_DIR / ENCODER_FILENAME}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("H3 LightVAE-Lite encoding requires CUDA.")

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
            raise FileNotFoundError(f"Encoder checkpoint not found: {selected}")
        if selected.name.lower() != ENCODER_FILENAME:
            raise ValueError(
                f"Expected {ENCODER_FILENAME}, received {selected.name}"
            )
        if not LIGHTVAE_ROOT.is_dir():
            raise FileNotFoundError(
                f"Standalone LightVAE source directory not found: {LIGHTVAE_ROOT}"
            )

        log.info("Loading LightVAE-Lite H3 encoder checkpoint: %s", selected)
        model = Encoder(checkpoint=selected)
        model.eval()
        model.requires_grad_(False)
        model.to(device="cpu", dtype=DTYPE)
        model.tile_size = 128
        model.tile_overlap = 64
        model._h3_device = DEVICE
        model._encode_clip = types.MethodType(_streaming_encode_clip, model)

        log.info(
            "Loaded LightVAE-Lite H3 encoder: %d down stages, tile=128, overlap=64",
            len(model.encoder.down_blocks),
        )
        gc.collect()
        return (_H3LightVAEEncoder(model, selected),)


class H3VideoVAEEncodeLight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": ("H3_VIDEO_VAE_ENCODER",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/VAE"
    TITLE = "H3 Video VAE Encode (LightVAE-Lite)"

    @torch.inference_mode()
    def encode(self, encoder, images):
        if not isinstance(encoder, _H3LightVAEEncoder):
            raise TypeError(
                "Expected the H3 LightVAE-Lite encoder from H3 LightVAE Encoder Loader."
            )
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise TypeError("Expected ComfyUI IMAGE tensor [T,H,W,C].")
        if images.shape[-1] < 3:
            raise ValueError("H3 video encoding requires at least three RGB channels.")
        if images.shape[0] < 1:
            raise ValueError("At least one input frame is required.")
        if not torch.isfinite(images).all():
            raise ValueError("Input images contain NaN or Inf.")

        pixels = images[..., :3]
        height, width = int(pixels.shape[1]), int(pixels.shape[2])
        if height % 16 or width % 16:
            raise ValueError(
                f"H3 input height and width must be divisible by 16; got {height}x{width}."
            )

        # ComfyUI IMAGE is [T,H,W,C] in [0,1]; H3 encoder expects
        # [B,C,T,H,W] in [-1,1].
        video = (
            pixels.detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .mul(2.0)
            .sub(1.0)
            .contiguous()
        )

        model = encoder.model
        frame_count = int(video.shape[2])
        padded_count = padded_frames("h3", frame_count)
        expected_latent_frames = latent_frames("h3", padded_count)
        log.info(
            "Encoding %d H3 frames (%d padded) at %dx%d with LightVAE-Lite",
            frame_count, padded_count, width, height,
        )

        # Encoder.forward() performs posterior-mean selection and H3
        # diffusion latent normalization itself. Do not normalize a second time.
        samples = model(video)
        if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
            shape = tuple(samples.shape) if isinstance(samples, torch.Tensor) else type(samples).__name__
            raise RuntimeError(f"Unexpected H3 encoder output: {shape}")

        latent_channels = int(model.latent_channels)
        if samples.shape[1] != latent_channels:
            raise RuntimeError(
                f"Expected {latent_channels} normalized latent channels, "
                f"received {samples.shape[1]}."
            )
        if samples.shape[2] != expected_latent_frames:
            raise RuntimeError(
                f"Expected {expected_latent_frames} latent frames, received {samples.shape[2]}."
            )

        samples = samples.contiguous()
        expected_shape = (
            1, latent_channels, expected_latent_frames, height // 16, width // 16
        )
        if tuple(samples.shape) != expected_shape:
            raise RuntimeError(
                f"Expected H3 latent shape {expected_shape}, received {tuple(samples.shape)}."
            )
        if not torch.isfinite(samples).all():
            raise RuntimeError("H3 encoder produced NaN or Inf latents.")

        del video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ({"samples": samples},)
