"""MiniMax H3 streamed checkpoint loader and ComfyUI ModelPatcher bridge.

The checkpoint remains memory-mapped; transformer weights are streamed from
GGUF row tiles by the H3 inference engine instead of being loaded as a full
torch module.
"""

import os
import threading

import torch
import folder_paths
from comfy.model_patcher import ModelPatcherDynamic

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


class H3TransformerRuntime(torch.nn.Module):
    """Lightweight runtime object owned by ComfyUI's ModelPatcher.

    Transformer weights remain in the read-only GGUF mapping and are streamed
    by H3InferenceEngine. This module intentionally has no Parameter tensors,
    so ComfyUI must not move or materialize the checkpoint as a torch model.
    """

    def __init__(self, handle):
        super().__init__()
        self.handle = handle
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
        """Execute joint H3 denoising through the transformer owned by this patcher."""
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


class _H3HostTileOwner:
    """Ephemeral owner record required by ComfyUI's pinned-memory API."""

    def __init__(self, pin_state):
        self._pin_state = pin_state
        self._pins = {}
        self.weight = None
        self.bias = None


class H3ModelPatcher(ModelPatcherDynamic):
    """ComfyUI-managed lifecycle wrapper for the streamed H3 runtime."""

    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        fast_disk=False,
    ):
        super().__init__(
            model,
            load_device,
            offload_device,
            size=0,
            weight_inplace_update=weight_inplace_update,
            fast_disk=fast_disk,
        )

    @property
    def handle(self):
        return self.model.handle

    # ModelPatcherDynamic supplies ComfyUI's dynamic-model lifecycle,
    # including prompt tracking and dynamic VRAM/RAM eviction hooks. H3
    # overrides the weight-loading hooks because its GGUF tiles are streamed
    # by TiledGGUFLinear rather than represented as torch Parameters.
    def model_size(self):
        # ComfyUI manages the maximum simultaneously resident FP32 weight tile,
        # not the complete disk-backed GGUF checkpoint.
        return int(self.handle.gguf.stream_working_set_bytes)

    def loaded_size(self):
        # Live CUDA weight tile bytes; this changes during each projection.
        return int(self.handle.gguf.memory_stats.snapshot()["active_vram_bytes"])

    def loaded_ram_size(self):
        # Live raw + decoded host staging buffers. The mmap is disk-backed and
        # is reported separately rather than being mislabelled as resident RAM.
        return int(self.handle.gguf.memory_stats.snapshot()["active_host_bytes"])

    def model_dtype(self):
        return torch.float32

    def model_patches_to(self, device):
        # H3 GGUF tensors are not ordinary ModelPatcher weight patches.
        return None

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        # Use ComfyUI's actual dynamic loader. Besides registering the patcher
        # with the dynamic manager, this initializes its VBAR and managed
        # HostBuffer pools. The runtime has no torch Parameters, so its dynamic
        # module scan stages no model weights; H3 supplies tiles on demand.
        with self.handle.execution_lock:
            if self.handle.closed or self.handle.gguf is None:
                raise RuntimeError("Cannot activate a closed H3 checkpoint.")
            super().partially_load(
                device_to,
                extra_memory=extra_memory,
                force_patch_weights=force_patch_weights,
            )
            self.model.dynamic_pins[self.load_device]["active"] = True
        return 0

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # Delegate VBAR/back-up eviction to ComfyUI's dynamic patcher.
        with self.handle.execution_lock:
            return super().partially_unload(
                device_to,
                memory_to_free=memory_to_free,
                force_patch_weights=force_patch_weights,
            )

    def partially_unload_ram(self, ram_to_unload, subsets=None):
        # Use ComfyUI's pin eviction implementation for H3's HostBuffer tiles.
        if subsets is None:
            freed = super().partially_unload_ram(ram_to_unload)
        else:
            freed = super().partially_unload_ram(ram_to_unload, subsets=subsets)
        if freed and self.handle.gguf is not None:
            self.handle.gguf.memory_stats.remove_host(freed)
        return freed

    def stage_host_tile(self, values):
        # Allocate decoded FP32 tiles through ComfyUI's pinned HostBuffer
        # manager so RAM pressure and pin eviction use the native APIs.
        import comfy.pinned_memory

        pin_state = self.model.dynamic_pins[self.load_device]
        if not pin_state["hostbufs_initialized"]:
            raise RuntimeError(
                "H3 HostBuffers are not initialized. H3 ModelPatcher must be "
                "loaded by comfy.model_management.load_models_gpu() first."
            )

        owner = _H3HostTileOwner(pin_state)
        size = int(values.nbytes)
        allocated = comfy.pinned_memory.pin_memory(
            owner, subset="weights", size=size
        )
        pin = comfy.pinned_memory.get_pin(owner, subset="weights")
        if not allocated or pin is None:
            raise RuntimeError(
                "ComfyUI could not allocate a managed pinned host tile for H3. "
                "Check ComfyUI pinned-memory limits and available system RAM."
            )

        self.handle.gguf.memory_stats.add_host(size)
        try:
            target = pin.view(torch.float32).reshape(values.shape)
            target.copy_(torch.from_numpy(values))
        except Exception:
            self.partially_unload_ram(size, subsets=["weights"])
            raise

        return {"owner": owner, "tensor": target, "size": size}

    def release_host_tile(self, lease):
        owner = lease["owner"]
        if "pin" in owner._pins.get("weights", {}):
            self.partially_unload_ram(
                lease["size"], subsets=["weights"]
            )

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        # ModelPatcherDynamic's implementation avoids model.to() on the
        # runtime, unloads VBAR pages and evicts managed HostBuffer tiles.
        with self.handle.execution_lock:
            super().unpatch_model(
                device_to=None, unpatch_weights=unpatch_weights
            )
            if unpatch_weights:
                self.model.dynamic_pins[self.load_device]["active"] = False
            if device_to is not None:
                self.model.device = torch.device(device_to)
        return self.model

    def patch_model(
        self,
        device_to=None,
        lowvram_model_memory=0,
        load_weights=True,
        force_patch_weights=False,
    ):
        # Delegate the no-weight patch operation to ComfyUI's dynamic
        # patcher. The streamed GGUF is never passed through model.to().
        if load_weights:
            raise RuntimeError(
                "H3 weights are streamed; load through load_models_gpu(), "
                "not ModelPatcher.patch_model()."
            )
        return super().patch_model(
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            load_weights=False,
            force_patch_weights=force_patch_weights,
        )

    def detach(self, unpatch_all=True):
        # Preserve the mmap for graph reuse when ComfyUI evicts this patcher.
        with self.handle.execution_lock:
            result = super().detach(unpatch_all=unpatch_all)
            # detach(unpatch_all=False) skips unpatch_model; still mark the
            # dynamic model inactive and move its logical device to offload.
            self.model.dynamic_pins[self.load_device]["active"] = False
            self.model.device = self.offload_device
            return result


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

        # Return a ComfyUI dynamic ModelPatcher so the prompt tracker and
        # dynamic model manager use their native lifecycle for this runtime.
        import comfy.model_management

        runtime = H3TransformerRuntime(handle)
        patcher = H3ModelPatcher(
            runtime,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        print(
            f"[H3 Model Loader] Registered streamed MODEL with ComfyUI "
            f"(GGUF mmap: {handle.file_size / (1024 ** 3):.2f} GiB; "
            f"maximum managed FP32 weight tile: "
            f"{runtime.handle.gguf.stream_working_set_bytes / (1024 ** 2):.2f} MiB).",
            flush=True,
        )
        return (patcher,)
