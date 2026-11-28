"""ComfyUI-facing MiniMax H3 joint FL2VA inference core."""
from __future__ import annotations
import ctypes
import gc
import math
import time
import numpy as np
import torch
from .h3_stream_linear import DEFAULT_GGUF, GGUF, TiledGGUFLinear, TYPE_Q2_K, TYPE_Q3_K
from .h3_adaln import StreamingAdaLN, VIDEO, TEXT, AUDIO
from .h3_transformer_ops import (
    H3TransformerBlock, H3TokenRefiner, H3PatchProjection, H3OutputHeads,
    read_checkpoint_vector, rms_norm,
)
from .h3_scheduler import H3FlowScheduler
from .h3_lora import H3TurboLoRA

# Verified H3 architecture constants
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 5376
TEXT_INPUT_SIZE = 5120
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
VIDEO_PATCH_WIDTH = 96
NUM_BLOCKS = 50

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
DEFAULT_FPS = 24.0
VISUAL_COND_TIMESTEP = 0.999

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
PATCH_SIZE = (1, 2, 2)

# Conservative settings for the GTX 1650 4 GB.
CUDA_ROW_TILE = 128
CPU_ROW_TILE = 128
TOKEN_TILE = 2
QUERY_CHUNK = 16
KEY_CHUNK = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def print_stats(name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().float()
    require(bool(torch.isfinite(value).all().item()), f"{name} contains NaN/Inf")
    print(
        f"{name:<24} shape={tuple(value.shape)!s:<25} "
        f"min={value.min().item(): .5f} max={value.max().item(): .5f} "
        f"mean={value.mean().item(): .5f} std={value.std().item(): .5f}",
        flush=True,
    )


def report_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    mib = 1024.0 * 1024.0
    print(f"\n{label}")
    print(f"CUDA allocated: {torch.cuda.memory_allocated() / mib:.2f} MiB")
    print(f"CUDA reserved:  {torch.cuda.memory_reserved() / mib:.2f} MiB")
    print(f"CUDA peak alloc:{torch.cuda.max_memory_allocated() / mib:.2f} MiB")
    print(f"CUDA peak reserv:{torch.cuda.max_memory_reserved() / mib:.2f} MiB")

def trim_process_working_set() -> None:
    """Ask Windows to remove pageable pages from this process's working set."""
    if __import__("os").name != "nt":
        raise OSError("Windows working-set trimming is only available on Windows.")

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)

    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE

    empty_working_set = psapi.EmptyWorkingSet
    empty_working_set.argtypes = [wintypes.HANDLE]
    empty_working_set.restype = wintypes.BOOL

    if not empty_working_set(get_current_process()):
        error_code = ctypes.get_last_error()
        raise OSError(
            error_code,
            ctypes.FormatError(error_code).strip() or "EmptyWorkingSet failed",
        )


def report_system_memory(label: str) -> None:
    """Report Windows physical RAM and this process's resident working set."""
    print(f"\n========== SYSTEM RAM: {label} ==========", flush=True)
    if __import__("os").name != "nt":
        print("Windows RAM counters unavailable on this platform.", flush=True)
        return

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    mib = 1024.0 ** 2
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        print("GlobalMemoryStatusEx failed.", flush=True)
        return

    print(
        f"Physical RAM: {status.ullAvailPhys / mib:.0f} MiB available / "
        f"{status.ullTotalPhys / mib:.0f} MiB total "
        f"({status.dwMemoryLoad}% in use)",
        flush=True,
    )
    from ctypes import wintypes

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

    # Declare the Windows API signatures explicitly. Without argtypes and
    # restype, ctypes can mis-handle pointer-sized HANDLE values.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)

    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE

    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL

    process = get_current_process()
    ok = get_process_memory_info(
        process, ctypes.byref(counters), counters.cb
    )
    if ok:
        print(
            f"Process working set: {counters.WorkingSetSize / mib:.0f} MiB "
            f"(peak {counters.PeakWorkingSetSize / mib:.0f} MiB)",
            flush=True,
        )
        print(
            f"Process pagefile usage: {counters.PagefileUsage / mib:.0f} MiB "
            f"(peak {counters.PeakPagefileUsage / mib:.0f} MiB)",
            flush=True,
        )
    else:
        error_code = ctypes.get_last_error()
        print(
            "GetProcessMemoryInfo failed; Windows error:",
            error_code,
            ctypes.FormatError(error_code).strip(),
            flush=True,
        )


# ---------------------------------------------------------------------------

# Official H3 video/audio patch and positional layouts
# ---------------------------------------------------------------------------

def patchify_video(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 5:
        raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(latent.shape)}")

    batch, channels, full_t, full_h, full_w = latent.shape
    pt, ph, pw = PATCH_SIZE
    if channels != VIDEO_CHANNELS:
        raise ValueError(f"Expected 24 video latent channels, got {channels}")
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError("Video latent dimensions are not patch divisible")

    t, h, w = full_t // pt, full_h // ph, full_w // pw
    value = latent.reshape(batch, channels, t, pt, h, ph, w, pw)
    value = torch.einsum("nctrhpwq->nthwcrpq", value)
    return value.reshape(batch * t * h * w, channels * pt * ph * pw)


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4 or tuple(latent.shape[:3]) != (1, 32, 2):
        raise ValueError(f"Expected audio [1,32,2,T], got {tuple(latent.shape)}")
    return latent[0].permute(1, 2, 0).reshape(-1, 32)


def unpatchify_video(
    rows: torch.Tensor,
    latent_t: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    pt, ph, pw = PATCH_SIZE
    value = rows.reshape(-1, latent_t, patch_h, patch_w, 24, pt, ph, pw)
    value = torch.einsum("nthwcrpq->nctrhpwq", value)
    return value.reshape(-1, 24, latent_t * pt, patch_h * ph, patch_w * pw)


def unpack_audio(rows: torch.Tensor) -> torch.Tensor:
    if rows.shape[0] % 2:
        raise ValueError("Packed stereo audio row count must be even")
    temporal = rows.shape[0] // 2
    return rows.reshape(2, temporal, 32).permute(2, 0, 1).unsqueeze(0)


def axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    count = dim // patch
    return (
        torch.arange(count, dtype=torch.float64) * (ratio / count)
        + (1.0 - ratio) / 2.0
    ) * 32.0


def frame_grid(latent_h: int, latent_w: int):
    area = math.sqrt(latent_h * latent_w)
    h_axis = axis_from_sqrt_area(latent_h, 2, area)
    w_axis = axis_from_sqrt_area(latent_w, 2, area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    coordinates = torch.stack(
        (hh.reshape(-1), ww.reshape(-1)), dim=-1
    )
    return coordinates, w_axis


def video_time_grid(latent_t: int, origin: float, fps: float = DEFAULT_FPS) -> torch.Tensor:
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps}")
    temporal_scale = DEFAULT_FPS / float(fps)
    spans = torch.tensor(
        [
            FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] * temporal_scale
            for index in range(latent_t)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        (torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0))
    )


def video_grid(latent_t: int, frame_coordinates: torch.Tensor, origin: float, fps: float = DEFAULT_FPS):
    result = torch.empty(
        latent_t, frame_coordinates.shape[0], 3, dtype=torch.float64
    )
    result[:, :, 0] = video_time_grid(latent_t, origin, fps)[:, None]
    result[:, :, 1:] = frame_coordinates[None]
    return result.reshape(-1, 3)


def audio_grid(
    audio_t: int,
    origin: float,
    width_low: float,
    width_high: float,
) -> torch.Tensor:
    result = torch.zeros(audio_t * 2, 3, dtype=torch.float64)
    result[:, 0] = (origin + torch.arange(audio_t, dtype=torch.float64)).repeat(2)
    result[:audio_t, 2] = width_low
    result[audio_t:, 2] = width_high
    return result


def build_packed_layout(
    text_length: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    keyframes: list[dict],
    fps: float = DEFAULT_FPS,
) -> dict:
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps}")
    frame_coordinates, width_axis = frame_grid(latent_h, latent_w)
    frame_rows = frame_coordinates.shape[0]
    origin = float(text_length)

    text_positions = torch.zeros(text_length, 3, dtype=torch.float64)
    text_positions[:, 0] = torch.arange(text_length, dtype=torch.float64)

    positions = [text_positions]
    segments = [(0, text_length, "text")]
    keyframe_segments = []
    row = text_length

    # FL2VA keyframe conditioning is inserted directly after text. Each
    # still keyframe reserves its actual one-latent-frame row count.
    for keyframe in keyframes:
        latent = keyframe["latent"]
        frame_index = int(keyframe["resolved_frame_index"])
        if latent.shape[2] != 1:
            raise ValueError("This driver expects one-frame keyframe latents")
        if tuple(latent.shape[-2:]) != (latent_h, latent_w):
            raise ValueError("Keyframe latent spatial size differs from target")

        count = latent.shape[2] * frame_rows
        start, end = row, row + count
        segments.append((start, end, "cond"))
        keyframe_segments.append((start, end))
        positions.append(
            video_grid(
                latent.shape[2],
                frame_coordinates,
                origin + FRAME_RESCALE * frame_index * (DEFAULT_FPS / float(fps)),
                fps,
            )
        )
        row = end

    audio_start = row
    audio_end = audio_start + audio_t * 2
    segments.append((audio_start, audio_end, "audio"))
    positions.append(
        audio_grid(
            audio_t,
            origin,
            float(width_axis[0]),
            float(width_axis[-1]),
        )
    )
    row = audio_end

    video_start = row
    video_count = latent_t * frame_rows
    video_end = video_start + video_count
    segments.append((video_start, video_end, "video"))
    positions.append(video_grid(latent_t, frame_coordinates, origin, fps))
    row = video_end

    position_ids = torch.cat(positions, dim=0)
    require(position_ids.shape == (row, 3), "Packed position layout mismatch")

    return {
        "segments": segments,
        "keyframe_segments": keyframe_segments,
        "position_ids": position_ids,
        "seq_len": row,
        "audio_start": audio_start,
        "audio_end": audio_end,
        "video_start": video_start,
        "video_end": video_end,
        "audio_count": audio_end - audio_start,
        "video_count": video_end - video_start,
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "audio_t": audio_t,
    }


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


# ---------------------------------------------------------------------------
# Checkpoint preflight
# ---------------------------------------------------------------------------

def validate_checkpoint(gguf: GGUF) -> None:
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

    for refiner_index in range(2):
        prefix = f"token_refiner.blocks.{refiner_index}."
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

    for block_index in range(NUM_BLOCKS):
        prefix = f"blocks.{block_index}."
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
        raise KeyError(
            f"Checkpoint is missing {len(missing)} required tensors:\n{preview}"
        )

    core_matrix_names = [
        f"blocks.{index}.{suffix}"
        for index in range(NUM_BLOCKS)
        for suffix in (
            "attn.qkv_proj.weight",
            "attn.out_proj.weight",
            "mlp.fc1.weight",
            "mlp.fc2.weight",
        )
    ]
    q2k_core = [
        name for name in core_matrix_names
        if gguf.tensors[name].ggml_type == TYPE_Q2_K
    ]
    q3k_core = [
        name for name in core_matrix_names
        if gguf.tensors[name].ggml_type == TYPE_Q3_K
    ]
    if q2k_core:
        print(
            "[H3 checkpoint warning] "
            f"{len(q2k_core)}/{len(core_matrix_names)} core block matrices use Q2_K. "
            "This very-low-bit H3 quantization is documented to produce broken video/audio. "
            "Use Q3_K or a higher-quality quantization when possible; streamed Q3_K is supported.",
            flush=True,
        )
    elif q3k_core:
        print(
            f"[H3 checkpoint] {len(q3k_core)}/{len(core_matrix_names)} core block matrices use Q3_K.",
            flush=True,
        )

    inv_freq = read_checkpoint_vector(gguf, "rope.inv_freq")
    require(tuple(inv_freq.shape) == (16,),
            f"Expected rope.inv_freq shape (16,), got {tuple(inv_freq.shape)}")

    print(
        f"Checkpoint preflight passed: {len(gguf.tensors):,} tensors, "
        f"{NUM_BLOCKS} transformer blocks."
    )


# ---------------------------------------------------------------------------
# Joint FL2VA engine
# ---------------------------------------------------------------------------

class H3InferenceEngine:
    def __init__(
        self,
        gguf: GGUF,
        text_states: torch.Tensor,
        keyframes: list[dict],
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        seed: int,
        lora: H3TurboLoRA | None = None,
        transformer_blocks: torch.nn.ModuleList | None = None,
        fps: float = DEFAULT_FPS,
        trim_working_set: bool = False,
        trim_interval_blocks: int = 10,
        print_tensor_stats: bool = False,
        print_block_progress: bool = True,
        print_memory_stats: bool = False,
    ):
        self.gguf = gguf
        self.print_tensor_stats = bool(print_tensor_stats)
        self.print_block_progress = bool(print_block_progress)
        self.print_memory_stats = bool(print_memory_stats)
        if transformer_blocks is None or len(transformer_blocks) != NUM_BLOCKS:
            raise ValueError(
                f"H3 requires {NUM_BLOCKS} registered transformer block modules."
            )
        self.transformer_blocks = transformer_blocks
        self.seed = int(seed)
        self.lora = lora
        self.trim_working_set = bool(trim_working_set)
        self.trim_interval_blocks = int(trim_interval_blocks)
        if not 1 <= self.trim_interval_blocks <= NUM_BLOCKS:
            raise ValueError(
                f"RAM trim interval must be between 1 and {NUM_BLOCKS} blocks."
            )
        if self.trim_working_set:
            print(
                "[H3 RAM Trim] Enabled: request Windows working-set trimming "
                f"every {self.trim_interval_blocks} transformer blocks.",
                flush=True,
            )
        # Do not preload all 50 adapters: only the active block's factors are
        # moved to the execution device, then released before the next block.
        for block in self.transformer_blocks:
            block.lora = None
        self.row_tile = CUDA_ROW_TILE

        self.layout = build_packed_layout(
            text_length=text_states.shape[0],
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            keyframes=keyframes,
            fps=fps,
        )

        self.position_ids = self.layout["position_ids"]
        self.inv_freq = read_checkpoint_vector(gguf, "rope.inv_freq").float()

        self.adaln = StreamingAdaLN(gguf, row_tile=512)

        # These projections are CPU-streamed. Only one transformer block's
        # working activations are moved to CUDA at a time.
        self.video_projection = H3PatchProjection(
            gguf, "video", row_tile=CPU_ROW_TILE, token_tile=TOKEN_TILE
        )
        self.audio_projection = H3PatchProjection(
            gguf, "audio", row_tile=CPU_ROW_TILE, token_tile=TOKEN_TILE
        )
        self.output_heads = H3OutputHeads(
            gguf, row_tile=CPU_ROW_TILE, token_tile=TOKEN_TILE
        )

        print("\nProjecting Qwen conditioning through H3 condition_proj...")
        condition_projection = TiledGGUFLinear(
            gguf,
            "condition_proj.weight",
            device="cpu",
            row_tile=CPU_ROW_TILE,
            token_tile=TOKEN_TILE,
        )
        condition_bias = read_checkpoint_vector(gguf, "condition_proj.bias").float()
        text = condition_projection(text_states.float().cpu()) + condition_bias
        del condition_projection, condition_bias
        self._stats("Condition projection", text)

        print("\nRunning the two checkpoint token-refiner blocks...")
        for refiner_index in range(2):
            refiner = H3TokenRefiner(
                gguf,
                block_index=refiner_index,
                row_tile=CPU_ROW_TILE,
                token_tile=TOKEN_TILE,
                lora=self.lora,
            )
            text = refiner(text)
            del refiner
            print(f"Token refiner {refiner_index + 1}/2 complete", flush=True)

        final_refiner_norm = read_checkpoint_vector(
            gguf, "token_refiner.final_norm.weight"
        ).float()
        self.text_embed = rms_norm(text, final_refiner_norm).contiguous()
        del text, final_refiner_norm
        self._stats("Refined text", self.text_embed)

        # Keyframe rows are fixed context. Encode and project them once;
        # they are never updated by the sampler.
        self.keyframe_embeds = []
        for index, keyframe in enumerate(keyframes):
            rows = patchify_video(keyframe["latent"].float().cpu())

            # Official visual condition noise augmentation: every keyframe
            # restarts the same seed stream.
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            noise = torch.randn(
                rows.shape,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            rows = VISUAL_COND_TIMESTEP * rows + (
                1.0 - VISUAL_COND_TIMESTEP
            ) * noise

            embedded = self.video_projection(rows)
            start, end = self.layout["keyframe_segments"][index]
            require(embedded.shape[0] == end - start,
                    "Keyframe embedding row count does not match packed layout")
            self.keyframe_embeds.append(embedded.contiguous())
            print(
                f"Keyframe {index + 1}/{len(keyframes)} projected: "
                f"{tuple(embedded.shape)}",
                flush=True,
            )
            del rows, noise, embedded

        self.latent_t = latent_t
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.audio_t = audio_t

        print("\nPacked FL2VA layout")
        for start, end, kind in self.layout["segments"]:
            print(f"  {kind:<6} [{start:>6}, {end:>6})  rows={end-start}")
        print("  total sequence rows:", self.layout["seq_len"])

    def _stats(self, name: str, tensor: torch.Tensor) -> None:
        if self.print_tensor_stats:
            print_stats(name, tensor)

    def _build_timestep_indices(
        self,
        sigma_video: float,
        sigma_audio: float,
    ):
        t_video = 1.0 - float(sigma_video)
        t_audio = 1.0 - float(sigma_audio)
        t_condition = max(t_video, VISUAL_COND_TIMESTEP)

        timestep_values = {t_video, t_audio}
        if self.keyframe_embeds:
            timestep_values.add(t_condition)

        unique_timesteps = sorted(timestep_values)
        timestep_lookup = {
            value: index for index, value in enumerate(unique_timesteps)
        }

        tag_parts = []
        index_parts = []

        for start, end, kind in self.layout["segments"]:
            if kind == "text":
                tag, timestep_value = TEXT, t_video
            elif kind == "cond":
                tag, timestep_value = VIDEO, t_condition
            elif kind == "audio":
                tag, timestep_value = AUDIO, t_audio
            elif kind == "video":
                tag, timestep_value = VIDEO, t_video
            else:
                raise RuntimeError(f"Unknown packed segment: {kind}")

            count = end - start
            tag_parts.append(np.full(count, tag, dtype=np.int64))
            index_parts.append(
                np.full(count, timestep_lookup[timestep_value], dtype=np.int64)
            )

        token_tags = np.concatenate(tag_parts)
        timestep_indices = np.concatenate(index_parts)

        require(token_tags.shape == (self.layout["seq_len"],),
                "Packed modality tag count mismatch")
        require(timestep_indices.shape == (self.layout["seq_len"],),
                "Packed timestep index count mismatch")

        final_indices = np.concatenate(
            (
                np.full(
                    self.layout["audio_count"],
                    timestep_lookup[t_audio],
                    dtype=np.int64,
                ),
                np.full(
                    self.layout["video_count"],
                    timestep_lookup[t_video],
                    dtype=np.int64,
                ),
            )
        )

        return (
            np.asarray(unique_timesteps, dtype=np.float32),
            timestep_indices,
            token_tags,
            final_indices,
        )

    @torch.inference_mode()
    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        sigma_video: float,
        sigma_audio: float,
        step_index: int,
        total_steps: int,
        progress_callback=None,
    ):
        require(tuple(video_latent.shape) == (
            1, 24, self.latent_t, self.latent_h, self.latent_w
        ), f"Unexpected target video latent: {tuple(video_latent.shape)}")
        require(tuple(audio_latent.shape) == (
            1, 32, 2, self.audio_t
        ), f"Unexpected target audio latent: {tuple(audio_latent.shape)}")

        video_rows = patchify_video(video_latent)
        audio_rows = pack_audio(audio_latent)

        self._stats("Audio latent entering", audio_latent)
        self._stats("Audio packed rows", audio_rows)

        video_embed = self.video_projection(video_rows)
        audio_embed = self.audio_projection(audio_rows)
        self._stats("Audio patch embedding", audio_embed)

        hidden = torch.empty(
            self.layout["seq_len"],
            HIDDEN_SIZE,
            dtype=torch.float32,
            device="cpu",
        )

        text_length = self.text_embed.shape[0]
        hidden[:text_length] = self.text_embed

        for embedded, (start, end) in zip(
            self.keyframe_embeds,
            self.layout["keyframe_segments"],
        ):
            hidden[start:end] = embedded

        audio_start = self.layout["audio_start"]
        audio_end = self.layout["audio_end"]
        video_start = self.layout["video_start"]
        video_end = self.layout["video_end"]

        hidden[audio_start:audio_end] = audio_embed
        hidden[video_start:video_end] = video_embed

        del video_rows, audio_rows, video_embed, audio_embed
        require(bool(torch.isfinite(hidden).all().item()),
                "Packed hidden states contain NaN/Inf")

        (
            unique_timesteps,
            timestep_indices,
            token_tags,
            final_timestep_indices,
        ) = self._build_timestep_indices(sigma_video, sigma_audio)

        print(
            f"\nStep {step_index}/{total_steps} | "
            f"sigma_v={sigma_video:.7f} | sigma_a={sigma_audio:.7f} | "
            f"sequence={hidden.shape[0]}",
            flush=True,
        )

        start_time = time.perf_counter()

        for block_index in range(NUM_BLOCKS):
            modulation_table, adaln_indices = self.adaln.project_block(
                block_index=block_index,
                timesteps=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                device="cpu",
            )

            block = self.transformer_blocks[block_index]
            # Keep detailed branch/gate output focused on the three blocks
            # implicated by the first full trace.
            block.print_residual_diagnostics = (
                self.print_tensor_stats and block_index in (1, 5, 18)
            )
            # Inspect only blocks implicated by the supplied trace:
            # B2 (first video growth), B6 (first audio growth), and B19
            # (the large isolated video outlier).
            block.print_internal_diagnostics = (
                self.print_tensor_stats and block_index in (1, 5, 18)
            )

            # Stage only the active block's small norm buffers on CUDA.
            # Projection matrices remain mmap-backed and are decoded by row
            # tile; all inactive block buffers remain on CPU.
            block.to(device="cuda:0")
            try:
                # Load only this block's LoRA factors. Keeping all 50 adapters
                # resident would defeat the memory-bounded streamed execution.
                if self.lora is not None:
                    block.lora = self.lora.load_block(
                        "blocks", block.block_index, block.device
                    )
                    block.lora.print_diagnostics = block.print_internal_diagnostics

                # hidden stays in CPU RAM between blocks. The active block
                # computes on CUDA and returns its output to CPU.
                hidden = block(
                    hidden,
                    modulation_table,
                    adaln_indices,
                    self.position_ids,
                    self.inv_freq,
                )
            finally:
                block.lora = None
                # Return the block's norm buffers and projection device to CPU
                # before the next block is staged.
                block.to(device="cpu")

            del modulation_table, adaln_indices
            torch.cuda.synchronize()

            # Report each completed transformer block to ComfyUI. The node's
            # callback also checks the global interrupt flag, so Cancel is
            # honoured at the next block boundary rather than after all 50.
            if progress_callback is not None:
                completed_blocks = (
                    (step_index - 1) * NUM_BLOCKS + block_index + 1
                )
                progress_callback(
                    completed_blocks,
                    total_steps * NUM_BLOCKS,
                )

            if self.print_block_progress and (block_index == 0 or (block_index + 1) % 5 == 0):
                print(
                    f"Step {step_index}/{total_steps} | "
                    f"Block {block_index + 1}/{NUM_BLOCKS}",
                    flush=True,
                )

            if block_index in (0, 1, 5, 18, 49):
                self._stats(
                    f"Audio hidden after B{block_index + 1}",
                    hidden[audio_start:audio_end],
                )
                self._stats(
                    f"Video hidden after B{block_index + 1}",
                    hidden[video_start:video_end],
                )

            completed_in_step = block_index + 1
            report_memory = completed_in_step % 5 == 0
            if report_memory:
                gc.collect()
                torch.cuda.empty_cache()

            trim_now = self.trim_working_set and (
                completed_in_step % self.trim_interval_blocks == 0
                or completed_in_step == NUM_BLOCKS
            )
            if trim_now:
                try:
                    trim_process_working_set()
                    print(
                        "[H3 RAM Trim] Windows working-set trim requested "
                        f"after block {completed_in_step}/{NUM_BLOCKS}. "
                        "Windows may retain file-backed pages in its system cache.",
                        flush=True,
                    )
                except OSError as exc:
                    self.trim_working_set = False
                    print(
                        "[H3 RAM Trim] Request failed; disabling further trims: "
                        f"{exc}",
                        flush=True,
                    )

            if self.print_memory_stats and (report_memory or trim_now):
                memory_label = (
                    f"step {step_index}/{total_steps} | "
                    f"after transformer block {completed_in_step}/{NUM_BLOCKS}"
                )
                report_system_memory(memory_label)
                report_cuda_memory(f"CUDA MEMORY: {memory_label}")

        transformer_elapsed = time.perf_counter() - start_time
        if self.print_block_progress:
            print(f"50-block transformer elapsed: {transformer_elapsed:.2f}s")

        # The final layer is applied only to the target audio+video rows.
        # The packed target order is audio first, video second.
        final_modulation_table, final_indices = self.adaln.project_final(
            timesteps=unique_timesteps,
            timestep_indices=final_timestep_indices,
            device="cpu",
        )
        selected_final_modulation = final_modulation_table.index_select(
            0, final_indices
        )

        target_hidden = hidden[audio_start:video_end]
        audio_count = self.layout["audio_count"]
        video_count = self.layout["video_count"]
        self._stats("Audio hidden before head", target_hidden[:audio_count])
        self._stats("Video hidden before head", target_hidden[audio_count:audio_count + video_count])
        video_all, audio_all = self.output_heads(
            target_hidden,
            selected_final_modulation,
        )

        self._stats("Video output head raw", video_all[audio_count:])
        self._stats("Audio output head raw", audio_all[:audio_count])
        audio_prediction_rows = audio_all[:audio_count]
        video_prediction_rows = video_all[audio_count:]

        require(video_prediction_rows.shape == (
            self.layout["video_count"], VIDEO_PATCH_WIDTH
        ), f"Unexpected video head shape: {tuple(video_prediction_rows.shape)}")
        require(audio_prediction_rows.shape == (
            audio_count, AUDIO_CHANNELS
        ), f"Unexpected audio head shape: {tuple(audio_prediction_rows.shape)}")

        video_prediction = unpatchify_video(
            video_prediction_rows,
            self.latent_t,
            self.latent_h // 2,
            self.latent_w // 2,
        )
        audio_prediction = unpack_audio(audio_prediction_rows)
        self._stats("Video prediction", video_prediction)
        self._stats("Audio prediction", audio_prediction)

        # H3OutputHeads returns the positive data-ward head values.
        # ComfyUI negates these before CONST.calculate_denoised (x - sigma*v).
        # h3_scheduler.py uses the equivalent direct update (x + sigma*v).
        require(bool(torch.isfinite(video_prediction).all().item()),
                "Video output head contains NaN/Inf")
        require(bool(torch.isfinite(audio_prediction).all().item()),
                "Audio output head contains NaN/Inf")

        del (
            hidden,
            target_hidden,
            video_all,
            audio_all,
            audio_prediction_rows,
            video_prediction_rows,
            final_modulation_table,
            final_indices,
            selected_final_modulation,
        )

        return video_prediction, audio_prediction


# ---------------------------------------------------------------------------
# Joint video/audio flow sampling
# ---------------------------------------------------------------------------

def _h3_simple_sigmas(shift: float, steps: int) -> torch.Tensor:
    """Unsloth reference 1000-entry shifted sigma table, sampled at H3 step indices."""
    if steps < 1:
        raise ValueError("steps must be at least one")
    table_size = 1000
    base = torch.arange(1, table_size + 1, dtype=torch.float32) / table_size
    table = float(shift) * base / (1.0 + (float(shift) - 1.0) * base)
    stride = table_size / int(steps)
    values = [float(table[-(1 + int(index * stride))]) for index in range(int(steps))] + [0.0]
    return torch.tensor(values, dtype=torch.float32, device="cpu")


def _h3_res_multistep_coefficients(sigmas: torch.Tensor, index: int):
    """Second-order RES coefficients; None denotes the reference Euler boundary steps."""
    sigma_next = sigmas[index + 1]
    if index == 0 or float(sigma_next) == 0.0:
        return None
    t_fn = lambda value: value.log().neg()
    phi1_fn = lambda value: torch.expm1(value) / value
    phi2_fn = lambda value: (phi1_fn(value) - 1.0) / value
    t = t_fn(sigmas[index])
    t_old = t
    t_next = t_fn(sigma_next)
    t_prev = t_fn(sigmas[index - 1])
    h = t_next - t
    c2 = (t_prev - t_old) / h
    phi1_value = phi1_fn(-h)
    phi2_value = phi2_fn(-h)
    b1 = torch.nan_to_num(phi1_value - phi2_value / c2, nan=0.0)
    b2 = torch.nan_to_num(phi2_value / c2, nan=0.0)
    return h, b1, b2


def _h3_res_multistep_update(sample, denoised, old_denoised, sigmas, index):
    coefficients = None if old_denoised is None else _h3_res_multistep_coefficients(sigmas, index)
    sigma = sigmas[index]
    sigma_next = sigmas[index + 1]
    if coefficients is None:
        derivative = (sample - denoised) / sigma
        return sample + derivative * (sigma_next - sigma)
    h, b1, b2 = coefficients
    return (-h).exp() * sample + h * (b1 * denoised + b2 * old_denoised)


@torch.inference_mode()
def sample(
    engine: H3InferenceEngine,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    steps: int,
    seed: int,
    video_shift: float = VIDEO_SHIFT,
    audio_shift: float = AUDIO_SHIFT,
    progress_callback=None,
    sampler: str = "euler",
    print_sampler_diagnostics: bool = False,
    print_tensor_stats: bool = False,
):
    if sampler not in ("euler", "res_multistep"):
        raise ValueError(f"Unsupported H3 sampler: {sampler!r}")
    if sampler == "res_multistep":
        video_sigmas = _h3_simple_sigmas(video_shift, steps)
        audio_model_sigmas = time_shift_sigma(video_sigmas[:-1].numpy(), video_shift, audio_shift)
        audio_sigmas = torch.cat((
            torch.from_numpy(np.asarray(audio_model_sigmas, dtype=np.float32)),
            torch.zeros(1, dtype=torch.float32),
        ))
        video_timesteps = 1.0 - video_sigmas[:-1]
        audio_timesteps = 1.0 - audio_sigmas[:-1]
    else:
        video_scheduler = H3FlowScheduler(shift=video_shift)
        audio_scheduler = H3FlowScheduler(shift=audio_shift)
        video_scheduler.set_timesteps(steps + 1, device="cpu")
        audio_scheduler.set_timesteps(steps + 1, device="cpu")
        video_sigmas = video_scheduler.sigmas
        audio_sigmas = audio_scheduler.sigmas
        video_timesteps = video_scheduler.timesteps
        audio_timesteps = audio_scheduler.timesteps

    require(video_sigmas.numel() == steps + 1, "Video scheduler evaluation count mismatch")
    require(audio_sigmas.numel() == steps + 1, "Audio scheduler evaluation count mismatch")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video_latent = torch.randn((1, 24, latent_t, latent_h, latent_w), generator=generator, dtype=torch.float32, device="cpu")
    audio_latent = torch.randn((1, 32, 2, audio_t), generator=generator, dtype=torch.float32, device="cpu")

    print("\n========== JOINT H3 FL2VA DENOISING ==========")
    print("Sampler:", sampler)
    print("Video latent:", tuple(video_latent.shape))
    print("Audio latent:", tuple(audio_latent.shape))
    print("Evaluations:", steps)
    print("Transformer blocks per evaluation:", NUM_BLOCKS)
    print("Total transformer block executions:", steps * NUM_BLOCKS)
    print("Video shift:", video_shift)
    print("Audio shift:", audio_shift)
    print("Seed:", seed)
    if sampler == "res_multistep":
        print("Video sigma grid:", video_sigmas.tolist())
        print("Audio sigma grid:", audio_sigmas.tolist())
    if print_tensor_stats:
        print_stats("Initial video noise", video_latent)
        print_stats("Initial audio noise", audio_latent)

    # Cheap live-tensor invariants: catch patch packing/unpacking errors before
    # spending 50 streamed transformer blocks per evaluation.
    video_roundtrip = unpatchify_video(
        patchify_video(video_latent), latent_t, latent_h // 2, latent_w // 2
    )
    audio_roundtrip = unpack_audio(pack_audio(audio_latent))
    video_roundtrip_error = (video_roundtrip - video_latent).abs().max().item()
    audio_roundtrip_error = (audio_roundtrip - audio_latent).abs().max().item()
    require(video_roundtrip_error == 0.0,
            f"Video patchify round-trip failed: max error {video_roundtrip_error}")
    require(audio_roundtrip_error == 0.0,
            f"Audio pack round-trip failed: max error {audio_roundtrip_error}")
    if print_sampler_diagnostics:
        print(
            "[H3 diagnostics] Video patchify/unpatchify and audio pack/unpack: PASS "
            f"(max errors {video_roundtrip_error:g}, {audio_roundtrip_error:g})",
            flush=True,
        )
    require(bool(torch.all(video_sigmas[:-1] > video_sigmas[1:]).item()),
            "Video sigma schedule is not strictly descending")
    require(bool(torch.all(audio_sigmas[:-1] > audio_sigmas[1:]).item()),
            "Audio sigma schedule is not strictly descending")
    require(float(video_sigmas[-1]) == 0.0 and float(audio_sigmas[-1]) == 0.0,
            "Video/audio sigma schedules must terminate at zero")

    old_video_denoised = None
    old_audio_denoised = None
    audio_carried = None
    audio_scale = float(video_shift) / float(audio_shift)
    start_time = time.perf_counter()

    for index in range(steps):
        sigma_video = float(video_sigmas[index].item())
        sigma_audio = float(audio_sigmas[index].item())
        mapped_audio_sigma = time_shift_sigma(sigma_video, video_shift, audio_shift)
        require(abs(float(mapped_audio_sigma) - sigma_audio) < 1e-6,
                "Video/audio sigma schedules do not match the H3 mapping")
        print(f"\n========== DENOISING STEP {index + 1}/{steps} ==========", flush=True)
        video_prediction, audio_prediction = engine.forward(
            video_latent=video_latent,
            audio_latent=audio_latent,
            sigma_video=sigma_video,
            sigma_audio=sigma_audio,
            step_index=index + 1,
            total_steps=steps,
            progress_callback=progress_callback,
        )

        if print_sampler_diagnostics:
            print(
                "[H3 diagnostics] Prediction/latent std ratio: "
                f"video={video_prediction.float().std().item() / max(video_latent.float().std().item(), 1e-12):.5f}, "
                f"audio={audio_prediction.float().std().item() / max(audio_latent.float().std().item(), 1e-12):.5f}",
                flush=True,
            )

        if sampler == "euler":
            video_latent = video_scheduler.step(video_prediction, video_timesteps[index], video_latent).prev_sample
            audio_latent = audio_scheduler.step(audio_prediction, audio_timesteps[index], audio_latent).prev_sample
        else:
            sigma_v = video_sigmas[index]
            sigma_a = audio_sigmas[index]
            x_video = video_latent.float()
            video_denoised = x_video + video_prediction.float() * sigma_v
            video_latent = _h3_res_multistep_update(
                x_video, video_denoised, old_video_denoised, video_sigmas, index
            ).to(video_latent.dtype)
            old_video_denoised = video_denoised

            x_audio = audio_latent.float()
            if audio_carried is None or index == 0 or audio_carried.shape != x_audio.shape:
                audio_carried = x_audio * (sigma_v / sigma_a)
            dy = (1.0 - audio_scale) * x_audio + (
                1.0 + (audio_scale - 1.0) * sigma_a
            ) * audio_prediction.float().neg()
            audio_denoised = audio_carried - dy * sigma_v
            audio_next_carried = _h3_res_multistep_update(
                audio_carried, audio_denoised, old_audio_denoised, video_sigmas, index
            )
            old_audio_denoised = audio_denoised
            sigma_v_next = video_sigmas[index + 1]
            sigma_a_next = audio_sigmas[index + 1]
            if float(sigma_v_next) == 0.0:
                audio_latent = (audio_next_carried / audio_scale).to(audio_latent.dtype)
            else:
                audio_latent = (audio_next_carried * (sigma_a_next / sigma_v_next)).to(audio_latent.dtype)
            audio_carried = audio_next_carried

        require(bool(torch.isfinite(video_latent).all().item()), f"Video latent became non-finite at step {index + 1}")
        require(bool(torch.isfinite(audio_latent).all().item()), f"Audio latent became non-finite at step {index + 1}")
        if print_tensor_stats:
            print_stats("Updated video latent", video_latent)
            print_stats("Updated audio latent", audio_latent)
        del video_prediction, audio_prediction
        gc.collect()

    elapsed = time.perf_counter() - start_time
    print(f"\nJoint denoising elapsed: {elapsed:.2f}s")
    return video_latent, audio_latent, elapsed, None, None
