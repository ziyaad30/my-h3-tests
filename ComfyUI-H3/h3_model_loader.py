"""MiniMax H3 transformer checkpoint loader.

This is intentionally a checkpoint-reader stage, not yet a ComfyUI ModelPatcher.
It memory-maps the H3 GGUF and validates its architecture without materializing
the transformer weights. ModelPatcher integration belongs in the subsequent
H3 model implementation once its streamed block modules are patchable.
"""

import os

import folder_paths

# ---------------------------------------------------------------------------
# Streamed H3 transformer checkpoint loader
# ---------------------------------------------------------------------------

# The project GGUF reader is deliberately imported from the existing h3_
# implementation. It memory-maps the checkpoint; it does not materialize
# the transformer weights in RAM or VRAM.
def _load_h3_gguf_reader():
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    parent = here
    for _ in range(4):
        candidates.append(parent)
        parent = os.path.dirname(parent)

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "h3_stream_linear.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            break
    else:
        raise ImportError(
            "Could not locate the project's h3_stream_linear.py. "
            "Keep the current h3_ implementation modules in the project "
            "root alongside the ComfyUI-H3 folder."
        )

    from h3_stream_linear import GGUF
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
        self.gguf = None
        self.closed = False

        reader_class = _load_h3_gguf_reader()
        try:
            print(f"[H3 Model Loader] Opening GGUF: {self.path}", flush=True)
            self.gguf = reader_class(self.path)
            _validate_h3_checkpoint(self.gguf)
        except Exception:
            self.close()
            raise

    def close(self):
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

    RETURN_TYPES = ("H3_MODEL",)
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
        return (handle,)
