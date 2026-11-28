"""Experimental H3 FL2VA sampler with first/last-frame conditioning.

This node is intentionally separate from H3Sampler and H3SamplerBeta.
It reuses the existing streamed transformer engine and the repository's
checkpoint-aware LightVAE-Lite encoder for single-frame endpoint latents.
"""
from __future__ import annotations

import gc
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from .h3_engine import NUM_BLOCKS
from .lightvae_h3_standalone.lightvae.h3_lightvae_encoder import Encoder
from .lightvae_h3_standalone.lightvae.utils import split_tiles, stitch_tiles


ENCODER_FILENAME = "lightvae-lite-h3-encoder.safetensors"
ENCODER_DTYPE = torch.float16
ENCODER_DEVICE = torch.device("cuda:0")
ENCODER_TILE_SIZE = 128
ENCODER_TILE_OVERLAP = 64
LATENT_CHANNELS = 24


def _check_interrupted():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


def _streaming_encode_clip(self, video):
    """Run the checkpoint's original single-clip encoder, staging one module at a time."""
    mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    video = ((video + 1.0) / 2.0 - mean) / std
    hidden = video.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
    model = self.encoder

    model.conv_in.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
    try:
        hidden = model.conv_in(hidden)
    finally:
        model.conv_in.to(device="cpu")

    for stage_index, down_block in enumerate(model.down_blocks):
        for resnet_index, resnet in enumerate(down_block.resnets):
            print(
                f"\r[H3 FL2VA] Encoder stage {stage_index + 1}/"
                f"{len(model.down_blocks)} resnet {resnet_index + 1}/"
                f"{len(down_block.resnets)}",
                end="",
                flush=True,
            )
            resnet.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
            try:
                hidden = resnet(hidden)
            finally:
                resnet.to(device="cpu")

        if down_block.downsamplers is not None:
            for downsampler in down_block.downsamplers:
                downsampler.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
                try:
                    hidden = downsampler(hidden)
                finally:
                    downsampler.to(device="cpu")

    print(flush=True)

    model.norm_out.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
    try:
        hidden = F.silu(model.norm_out(hidden))
    finally:
        model.norm_out.to(device="cpu")

    model.conv_out.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
    try:
        hidden = model.conv_out(hidden)
    finally:
        model.conv_out.to(device="cpu")

    self.quant_conv.to(device=ENCODER_DEVICE, dtype=ENCODER_DTYPE)
    try:
        moments = self.quant_conv(hidden)
    finally:
        self.quant_conv.to(device="cpu")

    return moments.to(device="cpu")


def _find_encoder_checkpoint():
    # Reuse the same models/vae directory registered by the H3 VAE nodes.
    from .h3_video_vae import MODEL_DIR, _register_model_directory
    _register_model_directory()

    checkpoint = MODEL_DIR / ENCODER_FILENAME
    if checkpoint.is_file():
        return checkpoint.resolve()

    import folder_paths
    for directory in folder_paths.get_folder_paths("vae"):
        candidate = Path(directory) / ENCODER_FILENAME
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "H3 LightVAE-Lite encoder checkpoint was not found. Expected "
        f"{ENCODER_FILENAME} in the H3 models/vae directory or a ComfyUI VAE folder."
    )


def _load_streaming_encoder():
    if not torch.cuda.is_available():
        raise RuntimeError("H3 endpoint image encoding requires CUDA.")

    checkpoint = _find_encoder_checkpoint()
    print(f"[H3 FL2VA] Loading LightVAE-Lite encoder: {checkpoint}", flush=True)
    encoder = Encoder(checkpoint=checkpoint)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device="cpu", dtype=ENCODER_DTYPE)
    encoder.tile_size = ENCODER_TILE_SIZE
    encoder.tile_overlap = ENCODER_TILE_OVERLAP
    encoder._encode_clip = types.MethodType(_streaming_encode_clip, encoder)

    print(
        "[H3 FL2VA] Encoder ready; CPU-resident weights, "
        f"tile={encoder.tile_size}, overlap={encoder.tile_overlap}.",
        flush=True,
    )
    return encoder


@torch.inference_mode()
def _encode_keyframe(encoder, image, width, height, resize_mode, label):
    """ComfyUI IMAGE [1,H,W,3] -> normalized H3 latent [1,24,1,H/16,W/16]."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TypeError(f"{label} must be a ComfyUI IMAGE tensor [B,H,W,C].")
    if image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError(
            f"{label} must contain one RGB image; got {tuple(image.shape)}."
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(f"{label} contains NaN or Inf.")

    # ComfyUI IMAGE is RGB [0,1]. Match the standalone FL2VA encoder's
    # established preprocessing: first frame stretched, last frame center-fit.
    pixels = (
        image[0].detach().to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    )
    source = Image.fromarray(pixels, mode="RGB")
    if resize_mode == "stretch":
        source = source.resize((width, height), Image.Resampling.LANCZOS)
    elif resize_mode == "center":
        source = ImageOps.fit(
            source, (width, height), method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    else:
        raise ValueError(f"Unsupported keyframe resize mode: {resize_mode}")

    array = np.asarray(source, dtype=np.float32) / 127.5 - 1.0
    video = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()

    heights, height_overlaps = split_tiles(height, encoder.tile_size, encoder.tile_overlap)
    widths, width_overlaps = split_tiles(width, encoder.tile_size, encoder.tile_overlap)
    rows = []
    total_tiles = len(heights) * len(widths)
    tile_number = 0

    print(
        f"[H3 FL2VA] Encoding {label}: {width}x{height}, "
        f"{total_tiles} spatial tile(s), resize={resize_mode}.",
        flush=True,
    )

    for h_slice in heights:
        row = []
        for w_slice in widths:
            _check_interrupted()
            tile_number += 1
            print(
                f"[H3 FL2VA] {label} tile {tile_number}/{total_tiles}",
                flush=True,
            )
            pixel_tile = video[
                :, :, :,
                h_slice.start * 16:h_slice.stop * 16,
                w_slice.start * 16:w_slice.stop * 16,
            ]
            moments = encoder._encode_clip(pixel_tile)
            expected = (
                1, LATENT_CHANNELS * 2, 1,
                h_slice.stop - h_slice.start,
                w_slice.stop - w_slice.start,
            )
            if tuple(moments.shape) != expected:
                raise RuntimeError(
                    f"{label} encoder tile shape {tuple(moments.shape)}; expected {expected}."
                )
            row.append(moments)
            del pixel_tile, moments
        rows.append(row)

    moments = stitch_tiles(
        rows,
        [value // 16 for value in height_overlaps],
        [value // 16 for value in width_overlaps],
    )
    expected_moments = (
        1, LATENT_CHANNELS * 2, 1, height // 16, width // 16
    )
    if tuple(moments.shape) != expected_moments:
        raise RuntimeError(
            f"{label} stitched moments shape {tuple(moments.shape)}; "
            f"expected {expected_moments}."
        )

    posterior_mean = moments[:, :LATENT_CHANNELS].float()
    latent_mean = encoder.latents_mean.float().view(1, LATENT_CHANNELS, 1, 1, 1)
    latent_std = encoder.latents_std.float().view(1, LATENT_CHANNELS, 1, 1, 1)
    latent = ((posterior_mean - latent_mean) / latent_std).contiguous()

    expected_latent = (1, LATENT_CHANNELS, 1, height // 16, width // 16)
    if tuple(latent.shape) != expected_latent:
        raise RuntimeError(
            f"{label} latent shape {tuple(latent.shape)}; expected {expected_latent}."
        )
    if not bool(torch.isfinite(latent).all().item()):
        raise RuntimeError(f"{label} encoded latent contains NaN or Inf.")

    print(f"[H3 FL2VA] {label} latent: {tuple(latent.shape)}", flush=True)
    del video, moments, posterior_mean, latent_mean, latent_std
    return latent


class H3SamplerFL2VABeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 22, "min": 5, "max": 1000, "step": 17}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "steps": ("INT", {"default": 7, "min": 1, "max": 100}),
                "trim_working_set": ("BOOLEAN", {"default": False}),
                "trim_interval_blocks": ("INT", {"default": 10, "min": 1, "max": 50}),
                "video_shift": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "audio_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "lora": ("H3_LORA",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/Sampling"
    TITLE = "H3 Sampler (FL2VA Beta)"

    @torch.inference_mode()
    def sample(
        self, model, conditioning, width, height, frames, fps, steps,
        trim_working_set, trim_interval_blocks, video_shift, audio_shift,
        seed, first_frame=None, last_frame=None, lora=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("H3 sampling requires CUDA; no CUDA device is available.")

        from .h3_model_loader import H3ModelPatcher
        if not isinstance(model, H3ModelPatcher):
            raise TypeError(
                "H3 Sampler (FL2VA Beta) requires the H3 ModelPatcher output "
                "from H3 Model Loader."
            )

        transformer = model.model
        handle = getattr(transformer, "handle", None)
        if (
            handle is None or handle.closed or handle.gguf is None
            or not callable(getattr(transformer, "sample", None))
        ):
            raise RuntimeError("H3 ModelPatcher does not contain an active H3 transformer runtime.")

        gguf = handle.gguf
        gguf_file_size = handle.file_size

        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("H3 width and height must be positive multiples of 32.")
        if frames < 5 or (frames - 5) % 17:
            raise ValueError("H3 frame count must be 5 + 17*n (for example 5, 22, 39).")
        if not np.isfinite(float(fps)) or fps <= 0:
            raise ValueError("FPS must be a positive finite number.")
        if steps < 1:
            raise ValueError("At least one denoising evaluation is required.")
        if video_shift <= 0 or audio_shift <= 0:
            raise ValueError("Video and audio shifts must both be positive.")
        if not 1 <= int(trim_interval_blocks) <= NUM_BLOCKS:
            raise ValueError(f"RAM trim interval must be between 1 and {NUM_BLOCKS} blocks.")

        from comfy.utils import ProgressBar
        progress = ProgressBar(steps * NUM_BLOCKS)

        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise TypeError("Expected non-empty ComfyUI CONDITIONING from H3 Conditioning.")
        entry = conditioning[0]
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            raise TypeError("Malformed CONDITIONING entry; expected [tensor, metadata].")
        text_states = entry[0]
        if not isinstance(text_states, torch.Tensor) or text_states.ndim != 3:
            raise TypeError("H3 conditioning tensor must have shape [1, tokens, 5120].")
        if text_states.shape[0] != 1 or text_states.shape[2] != 5120:
            raise ValueError(
                f"Expected H3 conditioning [1, tokens, 5120], got {tuple(text_states.shape)}"
            )
        text_states = text_states[0].detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if text_states.shape[0] < 1 or not bool(torch.isfinite(text_states).all().item()):
            raise ValueError("H3 text conditioning is empty or contains NaN/Inf.")

        keyframes = []
        encoder = None
        try:
            if first_frame is not None or last_frame is not None:
                encoder = _load_streaming_encoder()
                if first_frame is not None:
                    first_latent = _encode_keyframe(
                        encoder, first_frame, width, height, "stretch", "first frame"
                    )
                    keyframes.append({
                        "resolved_frame_index": 0,
                        "latent": first_latent,
                    })
                    del first_latent

                if last_frame is not None:
                    last_latent = _encode_keyframe(
                        encoder, last_frame, width, height, "center", "last frame"
                    )
                    keyframes.append({
                        "resolved_frame_index": frames - 1,
                        "latent": last_latent,
                    })
                    del last_latent
        finally:
            if encoder is not None:
                del encoder
                gc.collect()
                torch.cuda.empty_cache()

        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        latent_h = height // 16
        latent_w = width // 16
        audio_t = round(frames / float(fps) * 40.0)
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even.")
        if audio_t <= 0:
            raise ValueError("Invalid H3 audio latent temporal length for this FPS/frame count.")

        for keyframe in keyframes:
            latent = keyframe["latent"]
            if tuple(latent.shape) != (1, 24, 1, latent_h, latent_w):
                raise ValueError(
                    "Encoded endpoint latent does not match the requested output canvas: "
                    f"{tuple(latent.shape)}."
                )

        def update_progress(completed, total):
            progress.update_absolute(completed, total)
            if completed % 5 == 0 or completed == total:
                stats = gguf.memory_stats.snapshot()
                mib = 1024.0 * 1024.0
                print(
                    "[H3 FL2VA Beta] "
                    f"peak streamed tile: {stats['peak_vram_bytes'] / mib:.2f} MiB; "
                    f"host staging: {stats['active_host_bytes'] / mib:.2f} MiB active "
                    f"(peak {stats['peak_host_bytes'] / mib:.2f} MiB); "
                    f"GGUF mmap: {gguf_file_size / (1024 ** 2):.1f} MiB",
                    flush=True,
                )
            _check_interrupted()

        try:
            _check_interrupted()
            gguf.memory_stats.reset_peaks()
            import comfy.model_management
            comfy.model_management.load_models_gpu([model], memory_required=0)

            conditioning_mode = (
                "first + last frame" if len(keyframes) == 2
                else "first frame" if keyframes and keyframes[0]["resolved_frame_index"] == 0
                else "last frame" if keyframes
                else "text only"
            )
            print(
                f"[H3 FL2VA Beta] {width}x{height}, {frames} frames at {fps:g} FPS, "
                f"{steps} evaluations, conditioning={conditioning_mode}, "
                f"video shift={video_shift:g}, audio shift={audio_shift:g}, seed={seed}; "
                f"RAM trim={'on' if trim_working_set else 'off'}"
                + (f" every {trim_interval_blocks} blocks" if trim_working_set else ""),
                flush=True,
            )

            video_latent, audio_latent, elapsed, _, _ = transformer.sample(
                text_states=text_states,
                keyframes=keyframes,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                steps=steps,
                seed=seed,
                lora=lora,
                fps=float(fps),
                video_shift=float(video_shift),
                audio_shift=float(audio_shift),
                trim_working_set=bool(trim_working_set),
                trim_interval_blocks=int(trim_interval_blocks),
                progress_callback=update_progress,
            )
            del text_states, keyframes
            gc.collect()
            _check_interrupted()
            print(
                f"[H3 FL2VA Beta] Joint audio/video sampling completed in {elapsed:.2f}s.",
                flush=True,
            )
            return (
                {"samples": video_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
                {"samples": audio_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
            )
        finally:
            gc.collect()
