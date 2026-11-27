"""ComfyUI node for streamed joint MiniMax H3 audio/video sampling."""
import gc

import torch

from .h3_engine import (
    NUM_BLOCKS,
    H3InferenceEngine,
    sample as run_h3_sampling,
    validate_checkpoint,
)


def _check_interrupted():
    """Raise ComfyUI's native interruption exception when Cancel is requested."""
    import comfy.model_management

    comfy.model_management.throw_exception_if_processing_interrupted()


class H3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("H3_MODEL",),
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 22, "min": 5, "max": 1000, "step": 17}),
                "steps": ("INT", {"default": 7, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("H3_VIDEO_LATENT", "H3_AUDIO_LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")
    FUNCTION = "sample"
    CATEGORY = "MiniMax H3/Sampling"
    TITLE = "H3 Sampler"

    @torch.inference_mode()
    def sample(self, model, conditioning, width, height, frames, steps, seed):
        if not torch.cuda.is_available():
            raise RuntimeError("H3 sampling requires CUDA; no CUDA device is available.")
        if getattr(model, "closed", True) or getattr(model, "gguf", None) is None:
            raise RuntimeError("H3 model handle is closed or has no mapped GGUF checkpoint.")
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("H3 width and height must be positive multiples of 32.")
        if frames < 5 or (frames - 5) % 17:
            raise ValueError("H3 frame count must be 5 + 17*n (for example 5, 22, 39).")
        if steps < 1:
            raise ValueError("At least one denoising evaluation is required.")

        # Import inside execution so this module remains importable by static
        # tooling without initializing ComfyUI's runtime.
        from comfy.utils import ProgressBar

        total_blocks = steps * NUM_BLOCKS
        progress = ProgressBar(total_blocks)

        # H3Conditioning wraps the unprojected ClipProj tensor in standard ComfyUI CONDITIONING.
        if not isinstance(conditioning, (list, tuple)) or not conditioning:
            raise TypeError("Expected non-empty ComfyUI CONDITIONING from H3 Conditioning.")
        entry = conditioning[0]
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            raise TypeError("Malformed CONDITIONING entry; expected [tensor, metadata].")
        text_states = entry[0]
        if not isinstance(text_states, torch.Tensor) or text_states.ndim != 3:
            raise TypeError("H3 conditioning tensor must have shape [1, tokens, 5120].")
        if text_states.shape[0] != 1 or text_states.shape[2] != 5120:
            raise ValueError(f"Expected H3 conditioning [1, tokens, 5120], got {tuple(text_states.shape)}")
        text_states = text_states[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if text_states.shape[0] < 1 or not bool(torch.isfinite(text_states).all().item()):
            raise ValueError("H3 text conditioning is empty or contains NaN/Inf.")

        latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        latent_h = height // 16
        latent_w = width // 16
        audio_t = round(frames / 24.0 * 40.0)
        if latent_h % 2 or latent_w % 2:
            raise ValueError("H3 latent height and width must be even.")
        if audio_t <= 0:
            raise ValueError("Invalid H3 audio latent temporal length.")

        def update_progress(completed, total):
            # Update first so the UI reflects the just-finished block, then
            # raise ComfyUI's own interruption exception if Cancel was pressed.
            progress.update_absolute(completed, total)
            _check_interrupted()

        engine = None
        try:
            _check_interrupted()
            gguf = model.gguf
            validate_checkpoint(gguf)
            print(
                f"[H3 Sampler] {width}x{height}, {frames} frames, "
                f"{steps} evaluations, seed={seed}",
                flush=True,
            )
            engine = H3InferenceEngine(
                gguf=gguf,
                text_states=text_states,
                keyframes=[],
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                seed=seed,
                lora=None,
            )
            del text_states
            gc.collect()
            _check_interrupted()

            video_latent, audio_latent, elapsed, _, _ = run_h3_sampling(
                engine=engine,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                steps=steps,
                seed=seed,
                progress_callback=update_progress,
            )
            print(
                f"[H3 Sampler] Joint audio/video sampling completed in {elapsed:.2f}s.",
                flush=True,
            )
            return (
                video_latent.detach().to(device="cpu", dtype=torch.float32).contiguous(),
                audio_latent.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            )
        finally:
            # This also runs when ComfyUI raises its native interruption
            # exception, ensuring the active block/engine references are freed.
            if engine is not None:
                del engine
            gc.collect()
            torch.cuda.empty_cache()
