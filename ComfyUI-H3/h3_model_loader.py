"""MiniMax H3 streamed checkpoint loader and ComfyUI ModelPatcher bridge.

The checkpoint remains memory-mapped; transformer weights are streamed from
GGUF row tiles by the H3 inference engine instead of being loaded as a full
torch module.
"""

import os
import threading

import torch
import folder_paths
from comfy.model_patcher import ModelPatcher

# ---------------------------------------------------------------------------
# Streamed H3 transformer checkpoint loader
# ---------------------------------------------------------------------------

# The project GGUF reader is deliberately imported from the existing h3_
# implementation. It memory-maps the checkpoint; it does not materialize
# the transformer weights in RAM or VRAM.
def _load_h3_gguf_reader():
    """Import the GGUF reader bundled with this ComfyUI custom node."""
    from .h3_stream_linear import GGUF
    return GGUF

def _validate_h3_checkpoint(gguf):
    required = [
        "adaln_t_table",
        "rope.inv_freq",
        "condition_proj.weight",
        "condition_proj.bias",
        "token_refiner.final_norm.weight",
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "final_layer.norm.weight",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    ]

    for index in range(2):
        prefix = f"token_refiner.blocks.{index}."
        required.extend(
            prefix + suffix
            for suffix in (
                "norm1.weight",
                "norm2.weight",
                "attn.q_norm.weight",
                "attn.k_norm.weight",
                "attn.qkv_proj.weight",
                "attn.out_proj.weight",
                "mlp.fc1.weight",
                "mlp.fc2.weight",
            )
        )

    for index in range(50):
        prefix = f"blocks.{index}."
        required.extend(
            prefix + suffix
            for suffix in (
                "norm1.weight",
                "norm2.weight",
                "attn.q_norm.weight",
                "attn.k_norm.weight",
                "attn.qkv_proj.weight",
                "attn.out_proj.weight",
                "mlp.fc1.weight",
                "mlp.fc2.weight",
                "adaln_proj.linear.weight",
                "adaln_proj.linear.bias",
            )
        )

    missing = [name for name in required if name not in gguf.tensors]
    if missing:
        preview = "\n".join("  " + name for name in missing[:30])
        raise RuntimeError(
            f"Not a complete supported H3 checkpoint: "
            f"{len(missing)} required tensors are missing:\n{preview}"
        )

    rope = gguf.tensors["rope.inv_freq"]
    if tuple(rope.dims) != (16,):
        raise RuntimeError(
            f"Unexpected rope.inv_freq dimensions: {rope.dims}; expected (16,)"
        )

    print(
        f"[H3 Model Loader] Checkpoint preflight passed: "
        f"{len(gguf.tensors):,} tensors; 50 transformer blocks; "
        f"2 token-refiner blocks.",
        flush=True,
    )


class H3ModelHandle:
    """Owns one read-only, memory-mapped H3 GGUF checkpoint."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.file_size = os.path.getsize(self.path)
        self.gguf = None
        self.closed = False
        self.execution_lock = threading.RLock()

        reader_class = _load_h3_gguf_reader()
        try:
            print(f"[H3 Model Loader] Opening GGUF: {self.path}", flush=True)
            self.gguf = reader_class(self.path)
            _validate_h3_checkpoint(self.gguf)
        except Exception:
            self.close()
            raise

    def close(self):
        with self.execution_lock:
            if self.closed:
                return
            self.closed = True
            gguf = self.gguf
            self.gguf = None
            if gguf is not None:
                gguf.close()
            print("[H3 Model Loader] Checkpoint mapping released.", flush=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class H3Transformer(torch.nn.Module):
    """Streamed H3 transformer owned directly by ComfyUI's ModelPatcher.

    Transformer weights remain in the read-only GGUF mapping and are streamed
    by H3InferenceEngine. This module intentionally has no Parameter tensors,
    so ComfyUI must not move or materialize the checkpoint as a torch model.
    """

    def __init__(self, handle):
        super().__init__()
        self.handle = handle
        self.working_set_bytes = int(handle.gguf.stream_working_set_bytes)
        # Let ModelPatcher initialize this to its configured offload_device.
        self.device = None

    def get_dtype(self):
        return torch.float32

    @torch.inference_mode()
    def sample(
        self,
        *,
        text_states,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        steps,
        seed,
        progress_callback=None,
        keyframes=None,
        lora=None,
    ):
        """Execute joint H3 denoising through this ComfyUI-managed transformer."""
        import gc

        from .h3_engine import (
            H3InferenceEngine,
            sample as run_h3_sampling,
            validate_checkpoint,
        )

        with self.handle.execution_lock:
            if self.handle.closed or self.handle.gguf is None:
                raise RuntimeError("H3 transformer checkpoint is closed.")

            engine = None
            try:
                validate_checkpoint(self.handle.gguf)
                engine = H3InferenceEngine(
                    gguf=self.handle.gguf,
                    text_states=text_states,
                    keyframes=[] if keyframes is None else keyframes,
                    latent_t=latent_t,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    audio_t=audio_t,
                    seed=seed,
                    lora=lora,
                )
                return run_h3_sampling(
                    engine=engine,
                    latent_t=latent_t,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    audio_t=audio_t,
                    steps=steps,
                    seed=seed,
                    progress_callback=progress_callback,
                )
            finally:
                if engine is not None:
                    del engine
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "H3 is a streamed joint audio/video transformer. Execute it with "
            "H3 Sampler, which calls the managed transformer's sample() method."
        )


# Register a dedicated dropdown without changing ComfyUI's normal GGUF lists.
folder_paths.folder_names_and_paths["h3_transformer_gguf"] = (
    folder_paths.get_folder_paths("diffusion_models"),
    {".gguf"},
)


class H3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_files = folder_paths.get_filename_list("h3_transformer_gguf")
        if not model_files:
            model_files = ["<put H3 transformer GGUF in models/diffusion_models>"]

        return {
            "required": {
                "checkpoint": (model_files,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MiniMax H3/Model"
    TITLE = "H3 Model Loader"

    def load_model(self, checkpoint):
        if checkpoint.startswith("<put "):
            raise RuntimeError(
                "No H3 transformer GGUF was found. Place the H3 checkpoint "
                "in ComfyUI/models/diffusion_models and refresh ComfyUI."
            )

        path = folder_paths.get_full_path("h3_transformer_gguf", checkpoint)
        if path is None:
            raise FileNotFoundError(
                f"Could not resolve H3 transformer checkpoint: {checkpoint}"
            )

        handle = H3ModelHandle(path)

        # The standard ComfyUI ModelPatcher owns the H3 transformer module.
        # The GGUF reader remains only the transformer's streamed checkpoint source.
        import comfy.model_management

        transformer = H3Transformer(handle)
        patcher = ModelPatcher(
            transformer,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
            size=transformer.working_set_bytes,
        )
        print(
            f"[H3 Model Loader] Registered streamed MODEL with ComfyUI "
            f"(GGUF mmap: {handle.file_size / (1024 ** 3):.2f} GiB; "
            f"maximum managed FP32 weight tile: "
            f"{transformer.working_set_bytes / (1024 ** 2):.2f} MiB).",
            flush=True,
        )
        return (patcher,)
