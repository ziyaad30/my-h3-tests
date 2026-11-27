"""Experimental MiniMax H3 sampler with explicit FPS and flow-shift controls.

This is deliberately separate from H3Sampler. It uses the same patcher-owned
transformer runtime and joint audio/video denoising implementation, while
exposing timing and schedule parameters as ordinary ComfyUI widgets.
"""
import gc

import torch

from .h3_engine import NUM_BLOCKS


def _check_interrupted():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


class H3SamplerBeta:
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
                "video_shift": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "audio_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "lora": ("H3_LORA",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/Sampling"
    TITLE = "H3 Sampler (Beta)"

    @torch.inference_mode()
    def sample(
        self, model, conditioning, width, height, frames, fps, steps,
        video_shift, audio_shift, seed, lora=None
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("H3 sampling requires CUDA; no CUDA device is available.")

        from .h3_model_loader import H3ModelPatcher
        if not isinstance(model, H3ModelPatcher):
            raise TypeError(
                "H3 Sampler (Beta) requires the H3 ModelPatcher output from H3 Model Loader."
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
        if not torch.isfinite(torch.tensor(float(fps))).item() or fps <= 0:
            raise ValueError("FPS must be a positive finite number.")
        if steps < 1:
            raise ValueError("At least one denoising evaluation is required.")
        if video_shift <= 0 or audio_shift <= 0:
            raise ValueError("Video and audio shifts must both be positive.")

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
        text_states = text_states[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if text_states.shape[0] < 1 or not bool(torch.isfinite(text_states).all().item()):
            raise ValueError("H3 text conditioning is empty or contains NaN/Inf.")

        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        latent_h = height // 16
        latent_w = width // 16
        # H3's audio stream is 40 latent time steps per second.
        audio_t = round(frames / float(fps) * 40.0)
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even.")
        if audio_t <= 0:
            raise ValueError("Invalid H3 audio latent temporal length for this FPS/frame count.")

        def update_progress(completed, total):
            progress.update_absolute(completed, total)
            if completed % 5 == 0 or completed == total:
                stats = gguf.memory_stats.snapshot()
                mib = 1024.0 * 1024.0
                print(
                    "[H3 Sampler Beta] "
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

            print(
                f"[H3 Sampler Beta] {width}x{height}, {frames} frames at {fps:g} FPS, "
                f"{steps} evaluations, video shift={video_shift:g}, "
                f"audio shift={audio_shift:g}, seed={seed}",
                flush=True,
            )
            video_latent, audio_latent, elapsed, _, _ = transformer.sample(
                text_states=text_states,
                keyframes=[],
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
                progress_callback=update_progress,
            )
            del text_states
            gc.collect()
            _check_interrupted()
            print(
                f"[H3 Sampler Beta] Joint audio/video sampling completed in {elapsed:.2f}s.",
                flush=True,
            )
            return (
                {"samples": video_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
                {"samples": audio_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
            )
        finally:
            gc.collect()
