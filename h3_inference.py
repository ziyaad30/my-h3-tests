#!/usr/bin/env python3
"""
Standalone MiniMax H3 FL2VA inference.

The transformer path uses only the current h3_ modules:
    h3_stream_linear.py
    h3_adaln.py
    h3_transformer_ops.py
    h3_scheduler.py

The only minimax_ import is the existing LightVAE-Lite keyframe encoder.
No ComfyUI, Diffusers, or obsolete minimax_h3_transformer/minimax_h3_kernels
modules are imported.

The model sequence follows the official H3 order:
    [text | fixed keyframe video rows | target audio rows | target video rows]

Audio latents remain in the joint transformer sequence. This script saves
both generated latent streams. Audio waveform decoding is intentionally not
implemented because this project does not have an audio VAE checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from llama_cpp import Llama
import llama_cpp.llama_cpp as llama_binding

from h3_stream_linear import DEFAULT_GGUF, GGUF, TiledGGUFLinear
from h3_adaln import StreamingAdaLN, VIDEO, TEXT, AUDIO
from h3_transformer_ops import (
    H3TransformerBlock,
    H3TokenRefiner,
    H3PatchProjection,
    H3OutputHeads,
    read_checkpoint_vector,
    rms_norm,
)
from h3_scheduler import H3FlowScheduler
from minimax_h3_lightvae_decoder import decode_video_latents


# ---------------------------------------------------------------------------
# Paths and verified H3 constants
# ---------------------------------------------------------------------------

QWEN_MODEL_DEFAULT = Path(
    "C:/Users/User/PycharmProjects/anotherVideo/lingbot_models/"
    "models--bartowski--Qwen_Qwen3-VL-4B-Instruct-GGUF/"
    "snapshots/30cab06d3c6ade289ddc97a9d7c75c2b847ff32a/"
    "Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
)

CLIP_PROJECTION_DEFAULT = Path(
    "C:/Users/User/PycharmProjects/pythonProject/models/"
    "clip_projections/mmh3-4b-ClipProj-v3.1.safetensors"
)

HIDDEN_SIZE = 5376
TEXT_INPUT_SIZE = 5120
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
VIDEO_PATCH_WIDTH = 96
NUM_BLOCKS = 50

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
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


# ---------------------------------------------------------------------------
# Qwen3-VL -> MiniMax H3 ClipProj conditioning
# ---------------------------------------------------------------------------

def extract_qwen_conditioning(
    prompt: str,
    qwen_path: Path,
    clip_projection_path: Path,
) -> torch.Tensor:
    """
    Preserve the project's existing Qwen3-VL layer-output + ClipProj path.

    Output:
        CPU float32 [text_tokens, 5120]

    This requires the project's custom llama_get_layer_output binding.
    """

    require(qwen_path.is_file(), f"Qwen model not found: {qwen_path}")
    require(
        clip_projection_path.is_file(),
        f"ClipProj not found: {clip_projection_path}",
    )
    require(
        hasattr(llama_binding, "llama_get_layer_output"),
        "Custom llama_get_layer_output binding is missing from llama_cpp.",
    )

    print("\nLoading MiniMax H3 ClipProj...", flush=True)
    weights = load_file(str(clip_projection_path), device="cpu")

    required = ("W", "mean_in", "std_in", "mean_out", "std_out", "sink_out")
    for name in required:
        require(name in weights, f"ClipProj is missing tensor: {name}")

    matrix = weights["W"].float()
    mean_in = weights["mean_in"].float().reshape(-1)
    std_in = weights["std_in"].float().reshape(-1)
    mean_out = weights["mean_out"].float().reshape(-1)
    std_out = weights["std_out"].float().reshape(-1)
    sink_out = weights["sink_out"].float().reshape(-1)

    require(tuple(matrix.shape) == (2560, 5120),
            f"Unexpected ClipProj W shape: {tuple(matrix.shape)}")
    require(mean_in.numel() == 2560 and std_in.numel() == 2560,
            "Invalid ClipProj input statistics")
    require(
        mean_out.numel() == 5120
        and std_out.numel() == 5120
        and sink_out.numel() == 5120,
        "Invalid ClipProj output statistics",
    )
    require(bool(torch.all(std_in != 0).item()), "ClipProj std_in contains zeros")

    print("Loading Qwen3-VL-4B...", flush=True)
    llm = Llama(
        model_path=str(qwen_path),
        n_ctx=512,
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False,
    )

    try:
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        require(0 < len(tokens) <= 512,
                f"Prompt token count outside Qwen context: {len(tokens)}")
        print(f"Prompt tokens: {len(tokens)}")
        print("Running Qwen inference...", flush=True)

        llm.eval(tokens)

        token_count = ctypes.c_int32(0)
        embedding_width = ctypes.c_int32(0)
        pointer = llama_binding.llama_get_layer_output(
            llm._ctx.ctx,
            ctypes.byref(token_count),
            ctypes.byref(embedding_width),
        )

        require(bool(pointer), "llama_get_layer_output returned NULL")
        require(token_count.value == len(tokens),
                f"Qwen token mismatch: {token_count.value} vs {len(tokens)}")
        require(embedding_width.value == 2560,
                f"Unexpected Qwen hidden dimension: {embedding_width.value}")

        count = token_count.value * embedding_width.value
        raw = ctypes.cast(
            pointer,
            ctypes.POINTER(ctypes.c_float * count),
        ).contents

        hidden = torch.from_numpy(
            np.fromiter(raw, dtype=np.float32, count=count)
            .reshape(token_count.value, embedding_width.value)
            .copy()
        )

        print_stats("Qwen layer output", hidden)

        normalized = (hidden - mean_in) / std_in
        projected = normalized @ matrix
        conditioning = projected * std_out + mean_out
        conditioning[0, :] = sink_out

        require(tuple(conditioning.shape) == (len(tokens), TEXT_INPUT_SIZE),
                f"Unexpected ClipProj output: {tuple(conditioning.shape)}")
        require(bool(torch.isfinite(conditioning).all().item()),
                "ClipProj conditioning contains NaN/Inf")
        require(torch.equal(conditioning[0], sink_out),
                "ClipProj sink-token replacement failed")

        print_stats("H3 text conditioning", conditioning)
        return conditioning.contiguous()

    finally:
        llm.close()
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Existing LightVAE-Lite single-frame keyframe path
# ---------------------------------------------------------------------------

def encode_keyframes(args) -> list[dict]:
    keyframes: list[dict] = []

    if args.first_image is None and args.last_image is None:
        print("\nNo keyframe images supplied; running text-to-audio/video.")
        return keyframes

    # This is the only minimax_ module used by the inference driver, and it
    # is exclusively the already-existing LightVAE-Lite VAE encoder.
    from minimax_h3_lightvae_encoder import (
        load_streaming_encoder,
        load_keyframe_image,
        encode_keyframe_latent,
    )

    print("\nLoading existing H3 LightVAE-Lite keyframe encoder...", flush=True)
    encoder = load_streaming_encoder()

    try:
        if args.first_image is not None:
            image = load_keyframe_image(
                args.first_image, args.width, args.height, "stretch"
            )
            latent = encode_keyframe_latent(encoder, image).detach().cpu().float()
            require(latent.shape == (1, 24, 1, args.height // 16, args.width // 16),
                    f"Unexpected first keyframe latent: {tuple(latent.shape)}")
            keyframes.append({"resolved_frame_index": 0, "latent": latent})
            print("First keyframe latent:", tuple(latent.shape))
            del image, latent

        if args.last_image is not None:
            image = load_keyframe_image(
                args.last_image, args.width, args.height, "center"
            )
            latent = encode_keyframe_latent(encoder, image).detach().cpu().float()
            require(latent.shape == (1, 24, 1, args.height // 16, args.width // 16),
                    f"Unexpected last keyframe latent: {tuple(latent.shape)}")
            keyframes.append({
                "resolved_frame_index": args.frames - 1,
                "latent": latent,
            })
            print("Last keyframe latent:", tuple(latent.shape))
            del image, latent

    finally:
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    return keyframes


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


def video_time_grid(latent_t: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
            for index in range(latent_t)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        (torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0))
    )


def video_grid(latent_t: int, frame_coordinates: torch.Tensor, origin: float):
    result = torch.empty(
        latent_t, frame_coordinates.shape[0], 3, dtype=torch.float64
    )
    result[:, :, 0] = video_time_grid(latent_t, origin)[:, None]
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
) -> dict:
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
                origin + FRAME_RESCALE * frame_index,
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
    positions.append(video_grid(latent_t, frame_coordinates, origin))
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
    ):
        self.gguf = gguf
        self.seed = int(seed)
        self.row_tile = CUDA_ROW_TILE

        self.layout = build_packed_layout(
            text_length=text_states.shape[0],
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            keyframes=keyframes,
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
        print_stats("Condition projection", text)

        print("\nRunning the two checkpoint token-refiner blocks...")
        for refiner_index in range(2):
            refiner = H3TokenRefiner(
                gguf,
                block_index=refiner_index,
                row_tile=CPU_ROW_TILE,
                token_tile=TOKEN_TILE,
            )
            text = refiner(text)
            del refiner
            print(f"Token refiner {refiner_index + 1}/2 complete", flush=True)

        final_refiner_norm = read_checkpoint_vector(
            gguf, "token_refiner.final_norm.weight"
        ).float()
        self.text_embed = rms_norm(text, final_refiner_norm).contiguous()
        del text, final_refiner_norm
        print_stats("Refined text", self.text_embed)

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
    ):
        require(tuple(video_latent.shape) == (
            1, 24, self.latent_t, self.latent_h, self.latent_w
        ), f"Unexpected target video latent: {tuple(video_latent.shape)}")
        require(tuple(audio_latent.shape) == (
            1, 32, 2, self.audio_t
        ), f"Unexpected target audio latent: {tuple(audio_latent.shape)}")

        video_rows = patchify_video(video_latent)
        audio_rows = pack_audio(audio_latent)

        video_embed = self.video_projection(video_rows)
        audio_embed = self.audio_projection(audio_rows)

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

            block = H3TransformerBlock(
                self.gguf,
                block_index=block_index,
                device="cuda:0",
                row_tile=CUDA_ROW_TILE,
                token_tile=TOKEN_TILE,
                query_chunk=QUERY_CHUNK,
                key_chunk=KEY_CHUNK,
            )

            # hidden stays in CPU RAM between blocks. The block moves its
            # active working tensors to CUDA and returns the result to CPU.
            hidden = block(
                hidden,
                modulation_table,
                adaln_indices,
                self.position_ids,
                self.inv_freq,
            )

            del block, modulation_table, adaln_indices
            torch.cuda.synchronize()

            if block_index == 0 or (block_index + 1) % 5 == 0:
                print(
                    f"Step {step_index}/{total_steps} | "
                    f"Block {block_index + 1}/{NUM_BLOCKS}",
                    flush=True,
                )

            if (block_index + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        transformer_elapsed = time.perf_counter() - start_time
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
        video_all, audio_all = self.output_heads(
            target_hidden,
            selected_final_modulation,
        )

        audio_count = self.layout["audio_count"]
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

@torch.inference_mode()
def sample(
    engine: H3InferenceEngine,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    steps: int,
    seed: int,
):
    video_scheduler = H3FlowScheduler(shift=VIDEO_SHIFT)
    audio_scheduler = H3FlowScheduler(shift=AUDIO_SHIFT)

    # set_timesteps receives sigma points, so steps+1 points produce
    # exactly the requested number of model evaluations.
    video_scheduler.set_timesteps(steps + 1, device="cpu")
    audio_scheduler.set_timesteps(steps + 1, device="cpu")

    require(video_scheduler.num_inference_steps == steps,
            "Video scheduler evaluation count mismatch")
    require(audio_scheduler.num_inference_steps == steps,
            "Audio scheduler evaluation count mismatch")

    generator = torch.Generator(device="cpu").manual_seed(seed)

    video_latent = torch.randn(
        (1, 24, latent_t, latent_h, latent_w),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    audio_latent = torch.randn(
        (1, 32, 2, audio_t),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )

    print("\n========== JOINT H3 FL2VA DENOISING ==========")
    print("Video latent:", tuple(video_latent.shape))
    print("Audio latent:", tuple(audio_latent.shape))
    print("Evaluations:", steps)
    print("Transformer blocks per evaluation:", NUM_BLOCKS)
    print("Total transformer block executions:", steps * NUM_BLOCKS)
    print("Video shift:", VIDEO_SHIFT)
    print("Audio shift:", AUDIO_SHIFT)
    print("Seed:", seed)

    start_time = time.perf_counter()

    for index in range(steps):
        sigma_video = float(video_scheduler.sigmas[index].item())
        sigma_audio = float(audio_scheduler.sigmas[index].item())

        mapped_audio_sigma = time_shift_sigma(
            sigma_video, VIDEO_SHIFT, AUDIO_SHIFT
        )
        require(
            abs(mapped_audio_sigma - sigma_audio) < 1e-6,
            "Video/audio sigma schedules do not match the official H3 mapping: "
            f"{mapped_audio_sigma} vs {sigma_audio}",
        )

        print(
            f"\n========== DENOISING STEP {index + 1}/{steps} ==========",
            flush=True,
        )

        video_prediction, audio_prediction = engine.forward(
            video_latent=video_latent,
            audio_latent=audio_latent,
            sigma_video=sigma_video,
            sigma_audio=sigma_audio,
            step_index=index + 1,
            total_steps=steps,
        )

        video_latent = video_scheduler.step(
            video_prediction,
            video_scheduler.timesteps[index],
            video_latent,
        ).prev_sample

        audio_latent = audio_scheduler.step(
            audio_prediction,
            audio_scheduler.timesteps[index],
            audio_latent,
        ).prev_sample

        require(bool(torch.isfinite(video_latent).all().item()),
                f"Video latent became non-finite at step {index + 1}")
        require(bool(torch.isfinite(audio_latent).all().item()),
                f"Audio latent became non-finite at step {index + 1}")

        print_stats("Updated video latent", video_latent)
        print_stats("Updated audio latent", audio_latent)

        del video_prediction, audio_prediction
        gc.collect()

    elapsed = time.perf_counter() - start_time
    print(f"\nJoint denoising elapsed: {elapsed:.2f}s")

    return video_latent, audio_latent, elapsed, video_scheduler, audio_scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone streamed MiniMax H3 FL2VA inference"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gguf", type=Path, default=Path(DEFAULT_GGUF))
    parser.add_argument("--qwen-model", type=Path, default=QWEN_MODEL_DEFAULT)
    parser.add_argument(
        "--clip-projection", type=Path, default=CLIP_PROJECTION_DEFAULT
    )
    parser.add_argument("--first-image", type=Path, default=None)
    parser.add_argument("--last-image", type=Path, default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("h3_output"))
    parser.add_argument("--decoder-checkpoint", type=Path, default=Path("C:/Users/User/PycharmProjects/pythonProject/models/vae/lightvae-lite-h3-decoder.safetensors"))
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--no-decode", action="store_true", help="Save latents only; skip video decoding")
    return parser.parse_args()


def main():
    args = parse_args()

    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(args.gguf.is_file(), f"H3 GGUF not found: {args.gguf}")
    require(args.qwen_model.is_file(), f"Qwen model not found: {args.qwen_model}")
    require(
        args.clip_projection.is_file(),
        f"ClipProj not found: {args.clip_projection}",
    )
    require(args.width > 0 and args.height > 0, "Invalid output resolution")
    require(
        args.width % 32 == 0 and args.height % 32 == 0,
        "H3 width and height must be multiples of 32",
    )
    require(
        args.frames >= 5 and (args.frames - 5) % 17 == 0,
        "H3 frame count must be 5 + 17*n",
    )
    require(args.steps >= 1, "At least one denoising evaluation is required")
    require(args.fps > 0, "FPS must be positive")
    if not args.no_decode:
        require(args.decoder_checkpoint.is_file(), f"LightVAE decoder checkpoint not found: {args.decoder_checkpoint}")

    if args.first_image is not None:
        require(args.first_image.is_file(), f"First image not found: {args.first_image}")
    if args.last_image is not None:
        require(args.last_image.is_file(), f"Last image not found: {args.last_image}")

    latent_t = (
        2 if args.frames <= 5
        else ((args.frames - 5) // 17) * 5 + 2
    )
    latent_h = args.height // 16
    latent_w = args.width // 16
    audio_t = round(args.frames / 24.0 * 40.0)

    require(latent_h % 2 == 0 and latent_w % 2 == 0,
            "H3 latent height and width must be even")
    require(audio_t > 0, "Invalid audio latent temporal length")

    print("\n========== MINIMAX H3 STANDALONE FL2VA ==========")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GGUF:", args.gguf)
    print("Resolution:", f"{args.width}x{args.height}")
    print("Pixel frames:", args.frames)
    print("Video latent:", (1, 24, latent_t, latent_h, latent_w))
    print("Audio latent:", (1, 32, 2, audio_t))
    print("Denoising evaluations:", args.steps)
    print("Prompt:", args.prompt)

    # Run Qwen first, then close it before loading VAE/transformer work.
    text_states = extract_qwen_conditioning(
        args.prompt,
        args.qwen_model,
        args.clip_projection,
    )
    gc.collect()
    torch.cuda.empty_cache()

    keyframes = encode_keyframes(args)
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with GGUF(str(args.gguf)) as gguf:
        validate_checkpoint(gguf)

        engine = H3InferenceEngine(
            gguf=gguf,
            text_states=text_states,
            keyframes=keyframes,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            seed=args.seed,
        )

        del text_states, keyframes
        gc.collect()

        video_latent, audio_latent, sampling_elapsed, video_scheduler, audio_scheduler = sample(
            engine=engine,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            steps=args.steps,
            seed=args.seed,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / "h3_fl2va_latents.pt"

        torch.save(
            {
                "video_latent": video_latent.detach().cpu(),
                "audio_latent": audio_latent.detach().cpu(),
                "prompt": args.prompt,
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "steps": args.steps,
                "video_shift": VIDEO_SHIFT,
                "audio_shift": AUDIO_SHIFT,
                "video_sigmas": video_scheduler.sigmas.cpu(),
                "audio_sigmas": audio_scheduler.sigmas.cpu(),
                "keyframe_count": len(engine.keyframe_embeds),
            },
            output_path,
        )

        del engine

    total_elapsed = time.perf_counter() - start

    print("\n========== H3 FL2VA COMPLETE ==========")
    print("Saved latent package:", output_path.resolve())
    print_stats("Final video latent", video_latent)
    print_stats("Final audio latent", audio_latent)
    print(f"Sampling elapsed: {sampling_elapsed:.2f}s")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    report_cuda_memory("CUDA memory summary")
    if not args.no_decode:
        decoded = decode_video_latents(
            video_latent.detach().cpu(),
            output_dir=args.output_dir / "decoded_video",
            checkpoint=args.decoder_checkpoint,
            fps=args.fps,
        )
        print("Decoded PNG frames:", decoded["frames_dir"].resolve())
        if decoded["mp4_path"] is not None:
            print("Decoded MP4:", decoded["mp4_path"].resolve())
    else:
        print("Video decoding skipped by --no-decode.")
    print("Audio latent was generated jointly but was not waveform-decoded.")


if __name__ == "__main__":
    main()
