"""
MiniMax H3 ComfyUI text-conditioning node.

The encoder path intentionally matches extract_qwen_conditioning() in
h3_inference.py: custom llama.cpp layer-output binding, BOS token, 512-token
context, Qwen hidden width 2560, and the H3 ClipProj-v3.1 projection to 5120.
"""

import atexit
import ctypes
import gc
import os
import threading

import numpy as np
import torch
from safetensors.torch import load_file

from llama_cpp import Llama
import llama_cpp.llama_cpp as binding

import folder_paths


# Register the H3-specific model folders with ComfyUI.
# Qwen GGUF files live alongside other text encoders, but use their own
# extension-filtered key so ordinary text-encoder loaders are unaffected.
folder_paths.folder_names_and_paths["h3_qwen_gguf"] = (
    folder_paths.get_folder_paths("text_encoders"),
    {".gguf"},
)

# ClipProj is not a built-in ComfyUI model category.
_clip_projection_dir = os.path.join(folder_paths.models_dir, "clip_projections")
folder_paths.add_model_folder_path("clip_projections", _clip_projection_dir)
_clip_projection_paths = folder_paths.get_folder_paths("clip_projections")
folder_paths.folder_names_and_paths["clip_projections"] = (
    _clip_projection_paths,
    {".safetensors"},
)


_ENCODER_LOCK = threading.RLock()
_CACHED_LLM = None
_CACHED_QWEN_PATH = None


def _close_cached_encoder():
    global _CACHED_LLM, _CACHED_QWEN_PATH
    with _ENCODER_LOCK:
        llm = _CACHED_LLM
        _CACHED_LLM = None
        _CACHED_QWEN_PATH = None
        if llm is not None:
            try:
                llm.close()
            finally:
                del llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


atexit.register(_close_cached_encoder)


def _load_projection(path):
    weights = load_file(path, device="cpu")
    required = ("W", "mean_in", "std_in", "mean_out", "std_out", "sink_out")
    missing = [key for key in required if key not in weights]
    if missing:
        raise RuntimeError(f"ClipProj is missing tensors: {missing}")

    matrix = weights["W"].float().contiguous()
    mean_in = weights["mean_in"].float().reshape(-1).contiguous()
    std_in = weights["std_in"].float().reshape(-1).contiguous()
    mean_out = weights["mean_out"].float().reshape(-1).contiguous()
    std_out = weights["std_out"].float().reshape(-1).contiguous()
    sink_out = weights["sink_out"].float().reshape(-1).contiguous()
    del weights

    if tuple(matrix.shape) != (2560, 5120):
        raise RuntimeError(f"Unexpected ClipProj W shape: {tuple(matrix.shape)}")
    if mean_in.numel() != 2560 or std_in.numel() != 2560:
        raise RuntimeError("Invalid ClipProj input statistics")
    if mean_out.numel() != 5120 or std_out.numel() != 5120 or sink_out.numel() != 5120:
        raise RuntimeError("Invalid ClipProj output statistics")
    if not bool(torch.all(std_in != 0).item()):
        raise RuntimeError("ClipProj std_in contains zero values")

    return matrix, mean_in, std_in, mean_out, std_out, sink_out


def _get_encoder(qwen_path, unload_after_encode, llama_class):
    global _CACHED_LLM, _CACHED_QWEN_PATH

    if unload_after_encode:
        _close_cached_encoder()
        return llama_class(
            model_path=qwen_path,
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        ), True

    if _CACHED_LLM is not None and _CACHED_QWEN_PATH != qwen_path:
        _close_cached_encoder()

    if _CACHED_LLM is None:
        _CACHED_LLM = llama_class(
            model_path=qwen_path,
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )
        _CACHED_QWEN_PATH = qwen_path

    return _CACHED_LLM, False


def _encode(prompt, qwen_path, projection_path, unload_after_encode):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if not os.path.isfile(qwen_path):
        raise FileNotFoundError(f"Qwen GGUF not found: {qwen_path}")
    if not os.path.isfile(projection_path):
        raise FileNotFoundError(f"ClipProj not found: {projection_path}")
    if not hasattr(binding, "llama_get_layer_output"):
        raise RuntimeError(
            "This node requires the project's modified llama_cpp build "
            "with llama_get_layer_output."
        )

    projection = _load_projection(projection_path)
    matrix, mean_in, std_in, mean_out, std_out, sink_out = projection

    llm, is_temporary = _get_encoder(qwen_path, unload_after_encode, Llama)
    try:
        # Match the standalone production path: do not reset a newly-created
        # llama.cpp context before its first evaluation.
        tokens = llm.tokenize(prompt.strip().encode("utf-8"), add_bos=True)
        if not tokens or len(tokens) > 512:
            raise ValueError(
                f"Prompt must tokenize to 1–512 tokens; got {len(tokens)}."
            )

        # Temporary diagnostics: report the exact Python/native state around
        # evaluation and the custom C call without changing the inference path.
        ctx_object = llm._ctx.ctx
        try:
            ctx_address = ctypes.cast(ctx_object, ctypes.c_void_p).value
        except (TypeError, ctypes.ArgumentError):
            ctx_address = None

        native_fn = binding.llama_get_layer_output
        print("\\n[H3 DEBUG] ---- Qwen layer-output diagnostics ----", flush=True)
        print(f"[H3 DEBUG] model_path={qwen_path!r}", flush=True)
        print(f"[H3 DEBUG] llama_object_type={type(llm)!r}", flush=True)
        print(f"[H3 DEBUG] context_object_type={type(ctx_object)!r}", flush=True)
        print(f"[H3 DEBUG] context_object={ctx_object!r}", flush=True)
        print(f"[H3 DEBUG] context_address={ctx_address!r}", flush=True)
        print(f"[H3 DEBUG] token_count={len(tokens)}", flush=True)
        print(f"[H3 DEBUG] token_head={tokens[:8]!r}", flush=True)
        print(f"[H3 DEBUG] native_function={native_fn!r}", flush=True)
        print(f"[H3 DEBUG] native_argtypes={getattr(native_fn, 'argtypes', '<missing>')!r}", flush=True)
        print(f"[H3 DEBUG] native_restype={getattr(native_fn, 'restype', '<missing>')!r}", flush=True)

        print("[H3 DEBUG] calling llm.eval(tokens)...", flush=True)
        eval_result = llm.eval(tokens)
        print(f"[H3 DEBUG] llm.eval completed; return={eval_result!r}", flush=True)

        token_count = ctypes.c_int32(0)
        embedding_width = ctypes.c_int32(0)
        print("[H3 DEBUG] calling llama_get_layer_output...", flush=True)
        pointer = native_fn(
            ctx_object,
            ctypes.byref(token_count),
            ctypes.byref(embedding_width),
        )

        try:
            pointer_address = ctypes.cast(pointer, ctypes.c_void_p).value
        except (TypeError, ctypes.ArgumentError):
            pointer_address = None

        print(f"[H3 DEBUG] returned_pointer_type={type(pointer)!r}", flush=True)
        print(f"[H3 DEBUG] returned_pointer={pointer!r}", flush=True)
        print(f"[H3 DEBUG] returned_pointer_address={pointer_address!r}", flush=True)
        print(f"[H3 DEBUG] returned_token_count={token_count.value}", flush=True)
        print(f"[H3 DEBUG] returned_embedding_width={embedding_width.value}", flush=True)
        print("[H3 DEBUG] ------------------------------------------\\n", flush=True)

        if not pointer:
            raise RuntimeError(
                "llama_get_layer_output returned NULL. See the preceding "
                "[H3 DEBUG] diagnostics in the ComfyUI console."
            )
        if token_count.value != len(tokens):
            raise RuntimeError(
                f"Qwen token mismatch: {token_count.value} vs {len(tokens)}."
            )
        if embedding_width.value != 2560:
            raise RuntimeError(
                f"Expected Qwen hidden width 2560, got {embedding_width.value}."
            )

        count = token_count.value * embedding_width.value
        raw = ctypes.cast(
            pointer,
            ctypes.POINTER(ctypes.c_float * count),
        ).contents
        hidden = torch.tensor(list(raw), dtype=torch.float32).reshape(
            token_count.value, embedding_width.value
        )

        # In persistent mode, clear the evaluated sequence only after the
        # layer output has been copied out of llama.cpp's internal buffer.
        if not is_temporary:
            llm.reset()

        normalized = (hidden - mean_in) / std_in
        conditioning = (normalized @ matrix) * std_out + mean_out
        conditioning[0, :] = sink_out

        if tuple(conditioning.shape) != (len(tokens), 5120):
            raise RuntimeError(
                f"Unexpected H3 conditioning shape: {tuple(conditioning.shape)}"
            )
        if not bool(torch.isfinite(conditioning).all().item()):
            raise RuntimeError("H3 conditioning contains NaN or Inf.")

        return (conditioning.contiguous().cpu(),)

    except Exception:
        # A failed inference should not leave a potentially invalid cached
        # llama.cpp context around for the next ComfyUI execution.
        if not is_temporary:
            _close_cached_encoder()
        raise
    finally:
        del projection, matrix, mean_in, std_in, mean_out, std_out, sink_out
        if is_temporary:
            try:
                llm.close()
            finally:
                del llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


class H3TextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        qwen_files = folder_paths.get_filename_list("h3_qwen_gguf")
        projection_files = folder_paths.get_filename_list("clip_projections")

        if not qwen_files:
            qwen_files = ["<put Qwen GGUF in models/text_encoders>"]
        if not projection_files:
            projection_files = ["<put ClipProj in models/clip_projections>"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "qwen_model": (qwen_files,),
                "clip_projection": (projection_files,),
                "unload_after_encode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("H3_TEXT_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MiniMax H3/Text Encoder"
    TITLE = "H3 Text Encoder"

    def encode(self, prompt, qwen_model, clip_projection, unload_after_encode=True):
        if qwen_model.startswith("<put ") or clip_projection.startswith("<put "):
            raise RuntimeError(
                "H3 model files were not found. Place Qwen GGUF in "
                "ComfyUI/models/text_encoders and ClipProj in "
                "ComfyUI/models/clip_projections, then refresh ComfyUI."
            )

        qwen_path = folder_paths.get_full_path("h3_qwen_gguf", qwen_model)
        projection_path = folder_paths.get_full_path(
            "clip_projections", clip_projection
        )
        if qwen_path is None:
            raise FileNotFoundError(f"Could not resolve Qwen model: {qwen_model}")
        if projection_path is None:
            raise FileNotFoundError(
                f"Could not resolve ClipProj file: {clip_projection}"
            )

        with _ENCODER_LOCK:
            return _encode(
                prompt,
                qwen_path,
                projection_path,
                unload_after_encode,
            )


NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
}
