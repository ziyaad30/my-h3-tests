"""MiniMax H3 transformer implementation.
Automatically extracted from the original inference file.
"""

import argparse

import ctypes

import gc

import math

import time

from pathlib import Path

import numpy as np

import torch

import torch.nn.functional as F

from minimax_h3_kernels import (
    GGUFFile,
    GGUF_PATH,
    StreamedLinear,
)

HIDDEN = 5376

HEADS = 56

HEAD_DIM = 128

ATTN_DIM = HEADS * HEAD_DIM

FFN_DIM = 14336

MODALITIES = 3

ADALN_GROUPS = 6

ADALN_RANK = 8

ROPE_DIM = 96

EPS = 1e-5

NUM_BLOCKS = 50

WEIGHT_CHUNK_ROWS = 256

QUERY_TILE = 64

KEY_TILE = 256

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

FRAME_RESCALE = 5.0 / 3.0

SIGMA_SHIFT_VIDEO = 12.0

SIGMA_SHIFT_AUDIO = 3.0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def print_stats(name, tensor):
    x = tensor.detach().float()

    print(
        f"{name:<25}"
        f"shape={tuple(x.shape)!s:<22}"
        f"min={x.min().item():>11.5f} "
        f"max={x.max().item():>11.5f} "
        f"mean={x.mean().item():>11.5f} "
        f"std={x.std(unbiased=False).item():>11.5f} "
        f"finite={bool(torch.isfinite(x).all().item())}"
    )


def rms_norm(x, weight, eps=EPS):
    variance = x.float().square().mean(
        dim=-1,
        keepdim=True,
    )

    return (
            x.float()
            * torch.rsqrt(variance + eps)
            * weight.float()
    )


def read_gpu_tensor(gguf, name, device):
    return torch.from_numpy(
        gguf.read_tensor(name)
    ).to(
        device=device,
        dtype=torch.float32,
    )


def apply_rope(x, cos, sin):
    rotary = x[..., :ROPE_DIM]
    remainder = x[..., ROPE_DIM:]

    x1, x2 = rotary.chunk(2, dim=-1)

    rotated = torch.cat(
        (-x2, x1),
        dim=-1,
    )

    rotary = (
            rotary * cos[:, None, :]
            + rotated * sin[:, None, :]
    )

    return torch.cat(
        (rotary, remainder),
        dim=-1,
    )


def make_rope(position_ids, inv_freq):
    positions = position_ids.float()

    require(
        positions.ndim == 2 and positions.shape[1] == 3,
        f"Expected position IDs [sequence,3], got {tuple(positions.shape)}",
    )

    require(
        inv_freq.numel() == 16,
        f"Expected 16 RoPE frequencies, got {inv_freq.numel()}",
    )

    frequencies = (
            positions.unsqueeze(-1)
            * inv_freq.float().view(1, 1, -1)
    )

    frequencies = torch.cat(
        (
            frequencies[:, 0],
            frequencies[:, 1],
            frequencies[:, 2],
        ),
        dim=-1,
    )

    frequencies = torch.cat(
        (frequencies, frequencies),
        dim=-1,
    )

    return frequencies.cos(), frequencies.sin()


@torch.inference_mode()
def tiled_full_attention(q, k, v):
    sequence = q.shape[0]

    require(
        q.shape == k.shape == v.shape,
        "Q, K and V must have identical shapes",
    )

    require(
        q.ndim == 3
        and q.shape[1] == HEADS
        and q.shape[2] == HEAD_DIM,
        f"Invalid attention shape: {tuple(q.shape)}",
    )

    output = torch.empty_like(q)

    scale = HEAD_DIM ** -0.5

    for q_start in range(0, sequence, QUERY_TILE):
        q_end = min(
            q_start + QUERY_TILE,
            sequence,
        )

        q_tile = q[q_start:q_end].transpose(0, 1)

        query_count = q_end - q_start

        running_max = torch.full(
            (HEADS, query_count, 1),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        running_sum = torch.zeros_like(running_max)

        accumulator = torch.zeros(
            (HEADS, query_count, HEAD_DIM),
            device=q.device,
            dtype=torch.float32,
        )

        for k_start in range(0, sequence, KEY_TILE):
            k_end = min(
                k_start + KEY_TILE,
                sequence,
            )

            k_tile = k[k_start:k_end].transpose(0, 1)
            v_tile = v[k_start:k_end].transpose(0, 1)

            scores = torch.matmul(
                q_tile,
                k_tile.transpose(-2, -1),
            ) * scale

            block_max = scores.amax(
                dim=-1,
                keepdim=True,
            )

            new_max = torch.maximum(
                running_max,
                block_max,
            )

            old_scale = torch.exp(
                running_max - new_max
            )

            probabilities = torch.exp(
                scores - new_max
            )

            accumulator = (
                    accumulator * old_scale
                    + torch.matmul(probabilities, v_tile)
            )

            running_sum = (
                    running_sum * old_scale
                    + probabilities.sum(
                dim=-1,
                keepdim=True,
            )
            )

            running_max = new_max

        result = accumulator / running_sum

        output[q_start:q_end] = result.transpose(0, 1)

    return output


@torch.inference_mode()
def streamed_attention(linear, prefix, x, rope=None):
    qkv = linear(
        f"{prefix}.attn.qkv_proj.weight",
        x,
    )

    require(
        qkv.shape[-1] == 3 * ATTN_DIM,
        f"Invalid QKV width at {prefix}: {qkv.shape[-1]}",
    )

    # Validated single-file checkpoint ordering: [Q; K; V].
    q, k, v = qkv.split(
        ATTN_DIM,
        dim=-1,
    )

    sequence = x.shape[0]

    q = q.reshape(sequence, HEADS, HEAD_DIM)
    k = k.reshape(sequence, HEADS, HEAD_DIM)
    v = v.reshape(sequence, HEADS, HEAD_DIM)

    q_norm = read_gpu_tensor(
        linear.gguf,
        f"{prefix}.attn.q_norm.weight",
        x.device,
    )

    k_norm = read_gpu_tensor(
        linear.gguf,
        f"{prefix}.attn.k_norm.weight",
        x.device,
    )

    q = rms_norm(q, q_norm)
    k = rms_norm(k, k_norm)

    del q_norm, k_norm

    if rope is not None:
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

    attended = tiled_full_attention(q, k, v)

    attended = attended.reshape(
        sequence,
        ATTN_DIM,
    )

    del qkv, q, k, v

    return linear(
        f"{prefix}.attn.out_proj.weight",
        attended,
    )


@torch.inference_mode()
def streamed_swiglu(linear, prefix, x):
    projected = linear(
        f"{prefix}.mlp.fc1.weight",
        x,
    )

    require(
        projected.shape[-1] == 2 * FFN_DIM,
        f"Invalid FFN width at {prefix}: {projected.shape[-1]}",
    )

    # Validated checkpoint ordering: [gate; value].
    gate, value = projected.split(
        FFN_DIM,
        dim=-1,
    )

    activated = value * F.silu(gate)

    del projected, gate, value

    return linear(
        f"{prefix}.mlp.fc2.weight",
        activated,
    )


class MiniMaxH3TokenRefiner:

    def __init__(self, gguf, device):
        self.gguf = gguf
        self.device = torch.device(device)

        self.linear = StreamedLinear(
            gguf,
            device=self.device,
            chunk_rows=WEIGHT_CHUNK_ROWS,
        )

        self.norms = []

        for index in range(2):
            prefix = f"token_refiner.blocks.{index}"

            norm1 = self._read(
                f"{prefix}.norm1.weight"
            )

            norm2 = self._read(
                f"{prefix}.norm2.weight"
            )

            self.norms.append((norm1, norm2))

        self.final_norm = self._read(
            "token_refiner.final_norm.weight"
        )

    def _read(self, name):
        return read_gpu_tensor(
            self.gguf,
            name,
            self.device,
        )

    @torch.inference_mode()
    def forward(self, hidden_states, diagnostics=False):
        x = hidden_states.to(
            device=self.device,
            dtype=torch.float32,
        )

        if diagnostics:
            print_stats("Refiner input", x)

        for index in range(2):
            prefix = f"token_refiner.blocks.{index}"

            norm1, norm2 = self.norms[index]

            normalized = rms_norm(
                x,
                norm1,
            )

            attention_output = streamed_attention(
                self.linear,
                prefix,
                normalized,
                rope=None,
            )

            x = x + attention_output

            del normalized, attention_output

            if diagnostics:
                print_stats(
                    f"Refiner {index} attention",
                    x,
                )

            normalized = rms_norm(
                x,
                norm2,
            )

            ff_output = streamed_swiglu(
                self.linear,
                prefix,
                normalized,
            )

            x = x + ff_output

            del normalized, ff_output

            if diagnostics:
                print_stats(
                    f"Refiner {index} FFN",
                    x,
                )

        x = rms_norm(
            x,
            self.final_norm,
        )

        if diagnostics:
            print_stats("Refiner final norm", x)

        return x


class MiniMaxH3TransformerBlock:

    def __init__(self, gguf, block_index=0, device="cuda"):
        self.gguf = gguf
        self.block_index = int(block_index)
        self.device = torch.device(device)

        self.prefix = f"blocks.{self.block_index}"

        self.linear = StreamedLinear(
            gguf,
            device=self.device,
            chunk_rows=WEIGHT_CHUNK_ROWS,
        )

        self.norm1 = self._read(
            f"{self.prefix}.norm1.weight"
        )

        self.norm2 = self._read(
            f"{self.prefix}.norm2.weight"
        )

        self.q_norm = self._read(
            f"{self.prefix}.attn.q_norm.weight"
        )

        self.k_norm = self._read(
            f"{self.prefix}.attn.k_norm.weight"
        )

        self.adaln_weight = self._read(
            f"{self.prefix}.adaln_proj.linear.weight"
        ).T.contiguous()

        self.adaln_bias = self._read(
            f"{self.prefix}.adaln_proj.linear.bias"
        ).reshape(-1)

        self.inv_freq = self._read(
            "rope.inv_freq"
        ).reshape(-1)

        table = self._read(
            "adaln_t_table"
        )

        require(
            tuple(self.adaln_weight.shape)
            == (
                MODALITIES * ADALN_GROUPS * HIDDEN,
                ADALN_RANK,
            ),
            f"Invalid AdaLN weight shape: {tuple(self.adaln_weight.shape)}",
        )

        require(
            self.adaln_bias.numel()
            == MODALITIES * ADALN_GROUPS * HIDDEN,
            "Invalid AdaLN bias size",
        )

        require(
            self.inv_freq.numel() == 16,
            "Invalid RoPE frequency count",
        )

        require(
            tuple(table.shape) == (ADALN_RANK, 1025),
            f"Invalid timestep table shape: {tuple(table.shape)}",
        )

        self.timestep_table = table.T.contiguous()

    def _read(self, name):
        return read_gpu_tensor(
            self.gguf,
            name,
            self.device,
        )

    def timestep_basis(self, timesteps):
        timesteps = torch.as_tensor(
            timesteps,
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)

        require(
            timesteps.numel() > 0,
            "At least one timestep is required",
        )

        require(
            bool(torch.isfinite(timesteps).all().item()),
            "Timesteps contain NaN/Inf",
        )

        require(
            bool(torch.all(
                (timesteps >= 0.0) & (timesteps <= 1.0)
            ).item()),
            "Timesteps must be in [0,1]",
        )

        position = timesteps * (
                self.timestep_table.shape[0] - 1
        )

        lower = position.floor().long().clamp(
            max=self.timestep_table.shape[0] - 2
        )

        fraction = (
                position - lower.float()
        ).unsqueeze(-1)

        lower_values = self.timestep_table.index_select(
            0,
            lower,
        )

        upper_values = self.timestep_table.index_select(
            0,
            lower + 1,
        )

        return torch.lerp(
            lower_values,
            upper_values,
            fraction,
        )

    def modulation(self, timesteps):
        basis = self.timestep_basis(timesteps)

        result = F.linear(
            basis,
            self.adaln_weight,
            self.adaln_bias,
        )

        return result.reshape(
            basis.shape[0],
            MODALITIES,
            ADALN_GROUPS,
            HIDDEN,
        )

    @torch.inference_mode()
    def forward(
            self,
            hidden_states,
            timesteps,
            timestep_indices,
            token_tags,
            position_ids,
            diagnostics=False,
    ):
        x = hidden_states.to(
            device=self.device,
            dtype=torch.float32,
        )

        sequence = x.shape[0]

        require(
            x.shape == (sequence, HIDDEN),
            f"Expected [sequence,{HIDDEN}], got {tuple(x.shape)}",
        )

        timesteps = torch.as_tensor(
            timesteps,
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)

        timestep_ids = torch.as_tensor(
            timestep_indices,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        tags = torch.as_tensor(
            token_tags,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        positions = torch.as_tensor(
            position_ids,
            device=self.device,
            dtype=torch.float32,
        )

        require(
            tags.shape == (sequence,),
            "Invalid token_tags shape",
        )

        require(
            timestep_ids.shape == (sequence,),
            "Invalid timestep_indices shape",
        )

        require(
            positions.shape == (sequence, 3),
            "Invalid position_ids shape",
        )

        require(
            bool(torch.all(
                (tags >= 0) & (tags < MODALITIES)
            ).item()),
            "Invalid modality tag",
        )

        require(
            bool(torch.all(
                (timestep_ids >= 0)
                & (timestep_ids < timesteps.numel())
            ).item()),
            "Invalid timestep index",
        )

        modulation = self.modulation(timesteps)

        modulation = modulation.reshape(
            -1,
            ADALN_GROUPS,
            HIDDEN,
        )

        # Official index: timestep_index * 3 + modality_tag.
        adaln_indices = (
                timestep_ids * MODALITIES + tags
        )

        selected = modulation.index_select(
            0,
            adaln_indices,
        )

        shift_attn = selected[:, 0]
        scale_attn = selected[:, 1]
        gate_attn = selected[:, 2]

        shift_mlp = selected[:, 3]
        scale_mlp = selected[:, 4]
        gate_mlp = selected[:, 5]

        cos, sin = make_rope(
            positions,
            self.inv_freq,
        )

        if diagnostics:
            print_stats("Block input", x)
            print_stats("AdaLN attention gate", gate_attn)
            print_stats("AdaLN MLP gate", gate_mlp)

        # Attention branch.
        residual = x

        normalized = rms_norm(
            x,
            self.norm1,
        )

        normalized = (
                normalized * (1.0 + scale_attn)
                + shift_attn
        )

        if diagnostics:
            print_stats("Attention normalized", normalized)

        attention_output = streamed_attention(
            self.linear,
            self.prefix,
            normalized,
            rope=(cos, sin),
        )

        if diagnostics:
            print_stats("Attention projection", attention_output)

        x = residual + gate_attn * attention_output

        del residual, normalized, attention_output

        if diagnostics:
            print_stats("After attention residual", x)

        # Feed-forward branch.
        residual = x

        normalized = rms_norm(
            x,
            self.norm2,
        )

        normalized = (
                normalized * (1.0 + scale_mlp)
                + shift_mlp
        )

        if diagnostics:
            print_stats("MLP normalized", normalized)

        ff_output = streamed_swiglu(
            self.linear,
            self.prefix,
            normalized,
        )

        if diagnostics:
            print_stats("MLP projection", ff_output)

        x = residual + gate_mlp * ff_output

        del residual, normalized, ff_output
        del modulation, selected

        require(
            bool(torch.isfinite(x).all().item()),
            f"Block {self.block_index} produced non-finite values",
        )

        if diagnostics:
            print_stats("Block output", x)

        return x


def time_shift_sigma(sigma, from_shift, to_shift):
    base = sigma / (
            from_shift + sigma * (1.0 - from_shift)
    )

    return (
            to_shift * base
            / (1.0 + (to_shift - 1.0) * base)
    )


def patchify_video(latent):
    # [B,C,T,H,W] -> [T*(H/2)*(W/2), C*4]
    b, c, t, h, w = latent.shape

    require(
        b == 1,
        "H3 supports batch size 1",
    )

    require(
        h % 2 == 0 and w % 2 == 0,
        "Video latent H/W must be divisible by 2",
    )

    x = latent.reshape(
        b, c, t, 1, h // 2, 2, w // 2, 2
    )

    x = torch.einsum(
        "nctrhpwq->nthwcrpq",
        x,
    )

    return x.reshape(
        b * t * (h // 2) * (w // 2),
        c * 4,
    )


def unpatchify_video(rows, latent_t, latent_h, latent_w):
    x = rows.reshape(
        1,
        latent_t,
        latent_h // 2,
        latent_w // 2,
        24,
        1,
        2,
        2,
    )

    x = torch.einsum(
        "nthwcrpq->nctrhpwq",
        x,
    )

    return x.reshape(
        1,
        24,
        latent_t,
        latent_h,
        latent_w,
    )


def pack_audio(latent):
    # [1,32,2,T] -> [2*T,32], channel-major.
    require(
        latent.ndim == 4 and latent.shape[0] == 1,
        f"Invalid audio latent shape: {tuple(latent.shape)}",
    )

    return latent[0].permute(
        1, 2, 0
    ).reshape(
        latent.shape[2] * latent.shape[3],
        latent.shape[1],
    )


def unpack_audio(rows):
    require(
        rows.ndim == 2 and rows.shape[0] % 2 == 0,
        f"Invalid packed audio rows: {tuple(rows.shape)}",
    )

    audio_t = rows.shape[0] // 2

    return (
        rows.reshape(2, audio_t, rows.shape[-1])
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def axis_from_sqrt_area(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    count = dim // patch

    return (
            torch.arange(count, dtype=torch.float64)
            * (ratio / count)
            + (1.0 - ratio) / 2.0
    ) * 32.0


def frame_grid(latent_h, latent_w):
    area = math.sqrt(latent_h * latent_w)

    h_axis = axis_from_sqrt_area(
        latent_h,
        2,
        area,
    )

    w_axis = axis_from_sqrt_area(
        latent_w,
        2,
        area,
    )

    hh, ww = torch.meshgrid(
        h_axis,
        w_axis,
        indexing="ij",
    )

    frame = torch.stack(
        [
            hh.reshape(-1),
            ww.reshape(-1),
        ],
        dim=-1,
    )

    return frame, w_axis


def video_t_spans(latent_t):
    return [
        FRAME_RESCALE * FRAME_PER_TOKEN[i % 5]
        for i in range(latent_t)
    ]


def video_t_grid(latent_t, origin):
    spans = torch.tensor(
        video_t_spans(latent_t),
        dtype=torch.float64,
    )

    return float(origin) + torch.cat(
        [
            torch.zeros(1, dtype=torch.float64),
            spans[:-1].cumsum(0),
        ]
    )


def audio_grid(cursor, audio_t, w_low, w_high):
    positions = torch.zeros(
        audio_t * 2,
        3,
        dtype=torch.float64,
    )

    positions[:, 0] = (
            cursor
            + torch.arange(
        audio_t,
        dtype=torch.float64,
    ).repeat(2)
    )

    positions[:audio_t, 2] = w_low
    positions[audio_t:, 2] = w_high

    return positions


def video_grid(latent_t, frame, cursor):
    positions = torch.empty(
        latent_t,
        frame.shape[0],
        3,
        dtype=torch.float64,
    )

    positions[:, :, 0] = video_t_grid(
        latent_t,
        cursor,
    )[:, None]

    positions[:, :, 1:] = frame[None]

    return positions.reshape(-1, 3)


def build_h3_layout(text_length, latent_t, latent_h, latent_w, audio_t):
    frame, w_grid = frame_grid(
        latent_h,
        latent_w,
    )

    frame_rows = frame.shape[0]

    # Official unconditioned t2va/fl2va ordering.
    segments = [
        ("text", text_length),
    ]

    text_positions = torch.zeros(
        text_length,
        3,
        dtype=torch.float64,
    )

    text_positions[:, 0] = torch.arange(
        text_length,
        dtype=torch.float64,
    )

    positions = [text_positions]

    cursor = float(text_length)

    # Audio segment.
    segments.append(
        ("audio", audio_t * 2)
    )

    positions.append(
        audio_grid(
            cursor,
            audio_t,
            float(w_grid[0]),
            float(w_grid[-1]),
        )
    )

    # Video segment.
    video_rows = latent_t * frame_rows

    segments.append(
        ("video", video_rows)
    )

    positions.append(
        video_grid(
            latent_t,
            frame,
            cursor,
        )
    )

    position_ids = torch.cat(
        positions,
        dim=0,
    )

    absolute_segments = []

    offset = 0

    for kind, count in segments:
        absolute_segments.append(
            (offset, offset + count, kind)
        )

        offset += count

    require(
        offset == position_ids.shape[0],
        "Packed layout length mismatch",
    )

    return position_ids, absolute_segments


def build_h3_timesteps(sigma_video, device):
    require(
        0.0 < sigma_video <= 1.0,
        "Video sigma must be in (0,1]",
    )

    sigma_audio = time_shift_sigma(
        sigma_video,
        SIGMA_SHIFT_VIDEO,
        SIGMA_SHIFT_AUDIO,
    )

    t_video = 1.0 - sigma_video
    t_audio = 1.0 - sigma_audio

    unique_values = sorted(
        set((t_video, t_audio))
    )

    timesteps = torch.tensor(
        unique_values,
        device=device,
        dtype=torch.float32,
    )

    timestep_lookup = {
        value: index
        for index, value in enumerate(unique_values)
    }

    return (
        timesteps,
        timestep_lookup,
        t_video,
        t_audio,
    )


def build_modality_indices(segments, timesteps, timestep_lookup, device):
    # H3 tags: video=0, text=1, audio=2.
    tag_for_kind = {
        "video": 0,
        "text": 1,
        "audio": 2,
    }

    sequence = sum(
        end - start
        for start, end, _ in segments
    )

    tags = torch.empty(
        sequence,
        device=device,
        dtype=torch.long,
    )

    timestep_indices = torch.empty(
        sequence,
        device=device,
        dtype=torch.long,
    )

    for start, end, kind in segments:
        if kind in ("text", "video"):
            timestep = timesteps["video"]
        elif kind == "audio":
            timestep = timesteps["audio"]
        else:
            raise RuntimeError(
                f"Unsupported packed segment: {kind}"
            )

        tags[start:end] = tag_for_kind[kind]

        timestep_indices[start:end] = (
            timestep_lookup[timestep]
        )

    return tags, timestep_indices


def interpolate_h3_curve(table, timesteps):
    position = timesteps.clamp(
        0.0,
        1.0,
    ) * (table.shape[0] - 1)

    lower = position.floor().long().clamp(
        max=table.shape[0] - 2
    )

    fraction = (
            position - lower.float()
    ).unsqueeze(1)

    return torch.lerp(
        table.index_select(0, lower),
        table.index_select(0, lower + 1),
        fraction,
    )


def final_layer_modulation(gguf, timesteps, device):
    # Curve checkpoint stores already-computed time embedding basis.
    table = read_gpu_tensor(
        gguf,
        "adaln_t_table",
        device,
    )

    require(
        tuple(table.shape) == (ADALN_RANK, 1025),
        f"Unexpected AdaLN table shape: {tuple(table.shape)}",
    )

    table = table.T.contiguous()

    basis = interpolate_h3_curve(
        table,
        timesteps,
    )

    weight = read_gpu_tensor(
        gguf,
        "final_layer.adaln_proj.linear.weight",
        device,
    ).T.contiguous()

    bias = read_gpu_tensor(
        gguf,
        "final_layer.adaln_proj.linear.bias",
        device,
    ).reshape(-1)

    require(
        tuple(weight.shape) == (2 * HIDDEN, ADALN_RANK),
        f"Unexpected final AdaLN weight: {tuple(weight.shape)}",
    )

    require(
        bias.numel() == 2 * HIDDEN,
        "Unexpected final AdaLN bias size",
    )

    modulation = F.linear(
        basis,
        weight,
        bias,
    ).reshape(
        len(timesteps),
        2,
        HIDDEN,
    )

    del table, basis, weight, bias

    return modulation


__all__ = ['ADALN_GROUPS', 'ADALN_RANK', 'ATTN_DIM', 'EPS', 'FFN_DIM', 'FRAME_PER_TOKEN', 'FRAME_RESCALE', 'HEADS',
           'HEAD_DIM', 'HIDDEN', 'KEY_TILE', 'MODALITIES', 'MiniMaxH3TokenRefiner', 'MiniMaxH3TransformerBlock',
           'NUM_BLOCKS', 'QUERY_TILE', 'ROPE_DIM', 'SIGMA_SHIFT_AUDIO', 'SIGMA_SHIFT_VIDEO', 'WEIGHT_CHUNK_ROWS',
           'apply_rope', 'audio_grid', 'axis_from_sqrt_area', 'build_h3_layout', 'build_h3_timesteps',
           'build_modality_indices', 'final_layer_modulation', 'frame_grid', 'interpolate_h3_curve', 'make_rope',
           'pack_audio', 'patchify_video', 'print_stats', 'read_gpu_tensor', 'require', 'rms_norm',
           'streamed_attention', 'streamed_swiglu', 'tiled_full_attention', 'time_shift_sigma', 'unpack_audio',
           'unpatchify_video', 'video_grid', 'video_t_grid', 'video_t_spans']
