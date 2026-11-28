"""Experimental MiniMax H3 sampler using Unsloth-style RES multistep denoising.

This is a separate RES multistep sampler that accepts endpoint latents
pre-encoded by ComfyUI's native H3 VAE Encode node.
"""
import gc

import torch

from .h3_engine import NUM_BLOCKS


def _check_interrupted():
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


class H3SamplerResMultistep:
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
                "print_tensor_stats": ("BOOLEAN", {"default": False}),
                "print_block_progress": ("BOOLEAN", {"default": True}),
                "print_memory_stats": ("BOOLEAN", {"default": False}),
                "print_sampler_diagnostics": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "first_frame_latent": ("LATENT",),
                "last_frame_latent": ("LATENT",),
                "lora": ("H3_LORA",),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/Sampling"
    TITLE = "H3 Sampler (RES Multistep - Native VAE Latents)"

    @torch.inference_mode()
    def sample(
        self, model, conditioning, width, height, frames, fps, steps,
        trim_working_set, trim_interval_blocks,
        video_shift, audio_shift, seed, print_tensor_stats,
        print_block_progress, print_memory_stats, print_sampler_diagnostics,
        first_frame_latent=None, last_frame_latent=None, lora=None
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("H3 sampling requires CUDA; no CUDA device is available.")

        from .h3_model_loader import H3ModelPatcher
        if not isinstance(model, H3ModelPatcher):
            raise TypeError(
                "H3 Sampler (RES Multistep) requires the H3 ModelPatcher output from H3 Model Loader."
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
        if not 1 <= int(trim_interval_blocks) <= NUM_BLOCKS:
            raise ValueError(f"RAM trim interval must be between 1 and {NUM_BLOCKS} blocks.")

        if lora is None and steps < 20:
            print(
                "[H3 sampler warning] No LoRA is connected and RES Multistep is experimental. "
                "For the base H3 checkpoint, use the standard H3 Sampler (Euler) with "
                "the reference 50-step default; low-step RES output is not a validated base-model path.",
                flush=True,
            )

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

        keyframes = []

        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        latent_h = height // 16
        latent_w = width // 16
        # H3's audio stream is 40 latent time steps per second.
        audio_t = round(frames / float(fps) * 40.0)
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even.")
        if audio_t <= 0:
            raise ValueError("Invalid H3 audio latent temporal length for this FPS/frame count.")

        # Endpoint latents must come from the native H3 VAE Encode node.
        # The sampler does not encode IMAGE inputs or load the LightVAE encoder.
        for label, latent_input, frame_index in (
            ("first frame", first_frame_latent, 0),
            ("last frame", last_frame_latent, frames - 1),
        ):
            if latent_input is None:
                continue
            if not isinstance(latent_input, dict) or "samples" not in latent_input:
                raise TypeError(
                    f"{label} input must be a ComfyUI LATENT dictionary containing 'samples'."
                )
            latent = latent_input["samples"]
            expected_shape = (1, 24, 1, latent_h, latent_w)
            if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != expected_shape:
                actual = tuple(latent.shape) if isinstance(latent, torch.Tensor) else type(latent).__name__
                raise ValueError(
                    f"{label} latent must have shape {expected_shape}; received {actual}."
                )
            if not bool(torch.isfinite(latent).all().item()):
                raise ValueError(f"{label} latent contains NaN or Inf.")
            keyframes.append({
                "resolved_frame_index": frame_index,
                "latent": latent.detach().to(
                    device="cpu", dtype=torch.float32
                ).contiguous(),
            })

        def update_progress(completed, total):
            progress.update_absolute(completed, total)
            if completed % 5 == 0 or completed == total:
                stats = gguf.memory_stats.snapshot()
                mib = 1024.0 * 1024.0
                print(
                    "[H3 Sampler RES Multistep - Native VAE Latents] "
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
                f"[H3 Sampler RES Multistep] {width}x{height}, {frames} frames at {fps:g} FPS, "
                f"{steps} evaluations, video shift={video_shift:g}, "
                f"audio shift={audio_shift:g}, seed={seed}; "
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
                sampler="res_multistep",
                print_tensor_stats=bool(print_tensor_stats),
                print_block_progress=bool(print_block_progress),
                print_memory_stats=bool(print_memory_stats),
                print_sampler_diagnostics=bool(print_sampler_diagnostics),
            )
            del text_states, keyframes
            gc.collect()
            _check_interrupted()
            print(
                f"[H3 Sampler RES Multistep] Joint audio/video sampling completed in {elapsed:.2f}s.",
                flush=True,
            )
            return (
                {"samples": video_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
                {"samples": audio_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()},
            )
        finally:
            gc.collect()
