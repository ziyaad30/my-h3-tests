# !/usr/bin/env python3
"""
Standalone MiniMax H3 transformer block operations.

Uses:
    h3_stream_linear.GGUF
    h3_stream_linear.TiledGGUFLinear
    h3_adaln.StreamingAdaLN

TiledGGUFLinear is callable:
    output = projection(input)

It does NOT provide a .linear() method.

H3 dimensions:
    hidden size:       5376
    heads:             56
    head dimension:    128
    attention width:   7168
    rotary dimensions: 96
    MLP width:         14336
"""

from __future__ import annotations

import gc
import math
import os
import sys

import numpy as np
import torch

from h3_adaln import (
    StreamingAdaLN,
    apply_adaln_indexed,
    apply_adaln_gated_residual,
    VIDEO,
    TEXT,
    AUDIO,
)
from h3_stream_linear import GGUF, TiledGGUFLinear

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_GGUF = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models"
    r"\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
INNER_SIZE = NUM_HEADS * HEAD_DIM

ROTARY_DIM = 96
ROTARY_AXIS_DIM = 16

MLP_HIDDEN_SIZE = 14336

RMS_EPS = 1e-6

DEFAULT_ROW_TILE = 8
DEFAULT_TOKEN_TILE = 2

DEFAULT_QUERY_CHUNK = 16
DEFAULT_KEY_CHUNK = 128


# ============================================================
# GENERAL VALIDATION
# ============================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf")


# ============================================================
# RMS NORMALIZATION
# ============================================================

def rms_norm(x, weight, eps=RMS_EPS):
    if x.shape[-1] != weight.numel():
        raise ValueError(
            f"RMSNorm width mismatch: "
            f"{x.shape[-1]} != {weight.numel()}"
        )

    x32 = x.float()
    w32 = weight.float()

    variance = x32.square().mean(dim=-1, keepdim=True)

    return (
            x32 * torch.rsqrt(variance + eps)
    ) * w32


# ============================================================
# CHECKPOINT VECTOR READER
# ============================================================

def read_checkpoint_vector(gguf, tensor_name):
    """
    Read a one-dimensional F32, F16 or BF16 GGUF tensor.

    The GGUF reader supplied for this project exposes the
    read-only mmap as gguf.mm and absolute tensor offsets.
    """

    if tensor_name not in gguf.tensors:
        raise KeyError(
            f"Checkpoint tensor missing: {tensor_name}"
        )

    tensor = gguf.tensors[tensor_name]

    if len(tensor.dims) != 1:
        raise ValueError(
            f"{tensor_name} is not a vector: {tensor.dims}"
        )

    count = tensor.dims[0]

    if tensor.ggml_type == 0:
        byte_count = count * 4
        raw = bytes(
            gguf.mm[tensor.absolute_offset: tensor.absolute_offset + byte_count]
        )
        values = np.frombuffer(raw, dtype="<f4").copy()

    elif tensor.ggml_type == 1:
        byte_count = count * 2
        raw = bytes(
            gguf.mm[tensor.absolute_offset: tensor.absolute_offset + byte_count]
        )
        values = (
            np.frombuffer(raw, dtype="<f2")
            .astype(np.float32)
        )

    elif tensor.ggml_type == 30:
        byte_count = count * 2
        raw = bytes(
            gguf.mm[tensor.absolute_offset: tensor.absolute_offset + byte_count]
        )

        words = np.frombuffer(
            raw,
            dtype="<u2",
        ).astype(np.uint32)

        values = (words << 16).view("<f4").copy()

    else:
        raise ValueError(
            f"Unsupported vector GGML type "
            f"{tensor.ggml_type}: {tensor_name}"
        )

    if values.size != count:
        raise RuntimeError(
            f"Incorrect vector size for {tensor_name}"
        )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            f"Non-finite values in {tensor_name}"
        )

    return torch.from_numpy(values)


# ============================================================
# THREE-AXIS PARTIAL ROTARY EMBEDDING
# ============================================================

def build_rope(position_ids, inv_freq):
    """
    position_ids: [tokens, 3]
    inv_freq:     [16]

    Three axes each contribute 16 frequencies.
    The resulting 48 frequencies are duplicated to 96
    rotary dimensions. The final 32 head dimensions are
    not rotated.
    """

    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError(
            "position_ids must have shape [tokens, 3]"
        )

    if inv_freq.ndim != 1:
        raise ValueError("inv_freq must be one-dimensional")

    if inv_freq.numel() != ROTARY_AXIS_DIM:
        raise ValueError(
            f"Expected {ROTARY_AXIS_DIM} rotary frequencies, "
            f"got {inv_freq.numel()}"
        )

    positions = position_ids.float()
    frequencies = inv_freq.float()

    axis_angles = []

    for axis in range(3):
        angle = (
                positions[:, axis:axis + 1]
                * frequencies.unsqueeze(0)
        )
        axis_angles.append(angle)

    angles = torch.cat(axis_angles, dim=-1)

    angles = torch.cat(
        [angles, angles],
        dim=-1,
    )

    return torch.cos(angles), torch.sin(angles)


def rotate_half(x):
    first = x[..., :ROTARY_DIM // 2]
    second = x[..., ROTARY_DIM // 2:ROTARY_DIM]

    return torch.cat(
        [-second, first],
        dim=-1,
    )


def apply_rope(x, cos, sin):
    """
    x:   [tokens, heads, 128]
    cos: [tokens, 96]
    sin: [tokens, 96]

    Only the first 96 dimensions are rotated.
    """

    if x.ndim != 3:
        raise ValueError("RoPE input must be [tokens, heads, dim]")

    if x.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"Expected head dimension {HEAD_DIM}, "
            f"got {x.shape[-1]}"
        )

    rotary = x[..., :ROTARY_DIM]
    untouched = x[..., ROTARY_DIM:]

    cos = cos[:, None, :].to(
        device=x.device,
        dtype=x.dtype,
    )

    sin = sin[:, None, :].to(
        device=x.device,
        dtype=x.dtype,
    )

    rotated = (
            rotary * cos
            + rotate_half(rotary) * sin
    )

    return torch.cat(
        [rotated, untouched],
        dim=-1,
    )


# ============================================================
# SWIGLU
# ============================================================

def swiglu(x):
    if x.shape[-1] % 2:
        raise ValueError(
            "SwiGLU input width must be divisible by two"
        )

    gate, value = x.chunk(2, dim=-1)

    return torch.nn.functional.silu(gate) * value


# ============================================================
# STREAMED FULL SELF-ATTENTION
# ============================================================

@torch.inference_mode()
def streamed_attention(
        q,
        k,
        v,
        query_chunk=DEFAULT_QUERY_CHUNK,
        key_chunk=DEFAULT_KEY_CHUNK,
):
    """
    Full non-causal self-attention.

    Inputs:
        q, k, v: [tokens, heads, head_dim]

    Attention scores are computed in query/key chunks.
    No full sequence-by-sequence attention matrix is retained.
    """

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K and V shapes must match")

    if q.ndim != 3:
        raise ValueError(
            "Attention tensors must have shape [tokens, heads, dim]"
        )

    if query_chunk < 1 or key_chunk < 1:
        raise ValueError("Attention chunk sizes must be positive")

    tokens, heads, dim = q.shape

    if tokens == 0:
        raise ValueError("Attention received zero tokens")

    scale = 1.0 / math.sqrt(dim)

    # [heads, tokens, head_dim]
    q = q.permute(1, 0, 2).contiguous().float()
    k = k.permute(1, 0, 2).contiguous().float()
    v = v.permute(1, 0, 2).contiguous().float()

    output = torch.empty_like(q)

    for q_start in range(0, tokens, query_chunk):
        q_end = min(q_start + query_chunk, tokens)

        q_part = q[:, q_start:q_end]

        query_count = q_end - q_start

        running_max = torch.full(
            (heads, query_count, 1),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )

        running_sum = torch.zeros_like(running_max)

        running_output = torch.zeros(
            (heads, query_count, dim),
            device=q.device,
            dtype=torch.float32,
        )

        for k_start in range(0, tokens, key_chunk):
            k_end = min(k_start + key_chunk, tokens)

            k_part = k[:, k_start:k_end]
            v_part = v[:, k_start:k_end]

            scores = torch.matmul(
                q_part,
                k_part.transpose(-1, -2),
            ) * scale

            chunk_max = scores.amax(
                dim=-1,
                keepdim=True,
            )

            new_max = torch.maximum(
                running_max,
                chunk_max,
            )

            old_scale = torch.exp(
                running_max - new_max
            )

            probabilities = torch.exp(
                scores - new_max
            )

            running_sum = (
                    running_sum * old_scale
                    + probabilities.sum(dim=-1, keepdim=True, )
            )

            running_output = (
                    running_output * old_scale
                    + torch.matmul(probabilities, v_part)
            )

            running_max = new_max

        output[:, q_start:q_end] = (
                running_output / running_sum.clamp_min(torch.finfo(torch.float32).tiny)
        )

    return output.permute(1, 0, 2).contiguous()


# ============================================================
# H3 TRANSFORMER BLOCK
# ============================================================

class H3TransformerBlock:
    """
    One streamed H3 transformer block.

    The block's matrix weights are never loaded as full tensors.
    Each projection uses TiledGGUFLinear directly as a callable.
    """

    def __init__(
            self,
            gguf,
            block_index,
            device="cpu",
            row_tile=2048,
            token_tile=DEFAULT_TOKEN_TILE,
            query_chunk=DEFAULT_QUERY_CHUNK,
            key_chunk=DEFAULT_KEY_CHUNK,
            lora=None,
    ):
        self.gguf = gguf
        self.block_index = int(block_index)

        # TiledGGUFLinear expects a string for its device argument.
        self.device = torch.device(device)

        self.row_tile = int(row_tile)
        self.token_tile = int(token_tile)

        self.query_chunk = int(query_chunk)
        self.key_chunk = int(key_chunk)

        if self.row_tile < 1 or self.token_tile < 1:
            raise ValueError("Projection tile sizes must be positive")

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")

        prefix = f"blocks.{self.block_index}."

        self.qkv_proj = self._linear(
            prefix + "attn.qkv_proj.weight"
        )

        self.out_proj = self._linear(
            prefix + "attn.out_proj.weight"
        )

        self.fc1 = self._linear(
            prefix + "mlp.fc1.weight"
        )

        self.fc2 = self._linear(
            prefix + "mlp.fc2.weight"
        )

        # The norm vectors are small and remain resident.
        self.norm1_weight = self._load_vector(
            prefix + "norm1.weight"
        )

        self.norm2_weight = self._load_vector(
            prefix + "norm2.weight"
        )

        self.q_norm_weight = self._load_vector(
            prefix + "attn.q_norm.weight"
        )

        self.k_norm_weight = self._load_vector(
            prefix + "attn.k_norm.weight"
        )

        self._validate_checkpoint_shapes()
        self.lora = (
            lora.load_block("blocks", self.block_index, self.device)
            if lora is not None else None
        )

        self.norm1_weight = self.norm1_weight.to(self.device)
        self.norm2_weight = self.norm2_weight.to(self.device)

        self.q_norm_weight = self.q_norm_weight.to(self.device)
        self.k_norm_weight = self.k_norm_weight.to(self.device)

    def _linear(self, tensor_name):
        if tensor_name not in self.gguf.tensors:
            raise KeyError(
                f"Checkpoint projection missing: {tensor_name}"
            )

        # IMPORTANT:
        # TiledGGUFLinear has __call__(x), not .linear(x).
        # Its device argument must be a string.
        return TiledGGUFLinear(
            self.gguf,
            tensor_name,
            device=str(self.device),
            row_tile=self.row_tile,
            token_tile=self.token_tile,
        )

    def _load_vector(self, tensor_name):
        return read_checkpoint_vector(
            self.gguf,
            tensor_name,
        )

    def _validate_checkpoint_shapes(self):
        expected = {
            "norm1": (self.norm1_weight, HIDDEN_SIZE),
            "norm2": (self.norm2_weight, HIDDEN_SIZE),
            "q_norm": (self.q_norm_weight, HEAD_DIM),
            "k_norm": (self.k_norm_weight, HEAD_DIM),
        }

        for name, (tensor, expected_size) in expected.items():
            if tensor.numel() != expected_size:
                raise ValueError(
                    f"{name} has {tensor.numel()} values; "
                    f"expected {expected_size}"
                )

        projection_shapes = {
            "qkv": (
                self.qkv_proj.input_width,
                self.qkv_proj.output_rows,
                HIDDEN_SIZE,
                INNER_SIZE * 3,
            ),
            "out": (
                self.out_proj.input_width,
                self.out_proj.output_rows,
                INNER_SIZE,
                HIDDEN_SIZE,
            ),
            "fc1": (
                self.fc1.input_width,
                self.fc1.output_rows,
                HIDDEN_SIZE,
                MLP_HIDDEN_SIZE * 2,
            ),
            "fc2": (
                self.fc2.input_width,
                self.fc2.output_rows,
                MLP_HIDDEN_SIZE,
                HIDDEN_SIZE,
            ),
        }

        for name, values in projection_shapes.items():
            input_width, output_rows, expected_in, expected_out = values

            if input_width != expected_in or output_rows != expected_out:
                raise ValueError(
                    f"{name} projection shape mismatch: "
                    f"({input_width}, {output_rows}), "
                    f"expected ({expected_in}, {expected_out})"
                )

    @torch.inference_mode()
    def __call__(
            self,
            hidden_states,
            modulation_table,
            adaln_indices,
            position_ids,
            inv_freq,
    ):
        """
        hidden_states:
            [tokens, 5376]

        modulation_table:
            [entries, 6, 5376]

        adaln_indices:
            [tokens]

        position_ids:
            [tokens, 3]

        inv_freq:
            [16]
        """

        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape [tokens, hidden]"
            )

        token_count, hidden_width = hidden_states.shape

        if hidden_width != HIDDEN_SIZE:
            raise ValueError(
                f"Hidden width {hidden_width} != {HIDDEN_SIZE}"
            )

        if position_ids.shape != (token_count, 3):
            raise ValueError(
                f"Invalid position_ids shape: {tuple(position_ids.shape)}"
            )

        if adaln_indices.numel() != token_count:
            raise ValueError(
                "AdaLN index count must equal token count"
            )

        original_device = hidden_states.device

        x = hidden_states.to(
            device=self.device,
            dtype=torch.float32,
        )

        modulation_table = modulation_table.to(
            device=self.device,
            dtype=torch.float32,
        )

        adaln_indices = adaln_indices.to(
            device=self.device,
            dtype=torch.long,
        )

        position_ids = position_ids.to(
            device=self.device,
            dtype=torch.float32,
        )

        inv_freq = inv_freq.to(
            device=self.device,
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # ATTENTION NORMALIZATION AND ADALN
        # ----------------------------------------------------

        normalized = rms_norm(
            x,
            self.norm1_weight,
        )

        normalized = apply_adaln_indexed(
            normalized,
            modulation_table,
            adaln_indices,
            shift_group=0,
            scale_group=1,
            token_tile=self.token_tile,
        )

        # ----------------------------------------------------
        # QKV PROJECTION
        # ----------------------------------------------------

        # TiledGGUFLinear returns CPU FP32 output.
        # Move only the projected activation to the compute device.
        qkv_cpu = self.qkv_proj(normalized)

        qkv = qkv_cpu.to(
            device=self.device,
            dtype=torch.float32,
        )

        del qkv_cpu

        if qkv.shape != (token_count, INNER_SIZE * 3):
            raise RuntimeError(
                f"Unexpected QKV shape: {tuple(qkv.shape)}"
            )
        if self.lora is not None:
            self.lora.add_qkv(normalized, qkv)

        qkv = qkv.reshape(
            token_count,
            3,
            NUM_HEADS,
            HEAD_DIM,
        )

        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        del qkv

        # ----------------------------------------------------
        # Q/K RMS NORMALIZATION
        # ----------------------------------------------------

        q = rms_norm(
            q,
            self.q_norm_weight,
        )

        k = rms_norm(
            k,
            self.k_norm_weight,
        )

        # ----------------------------------------------------
        # THREE-AXIS PARTIAL ROPE
        # ----------------------------------------------------

        cos, sin = build_rope(
            position_ids,
            inv_freq,
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ----------------------------------------------------
        # FULL NON-CAUSAL SELF-ATTENTION
        # ----------------------------------------------------

        attention = streamed_attention(
            q,
            k,
            v,
            query_chunk=self.query_chunk,
            key_chunk=self.key_chunk,
        )

        del q, k, v, cos, sin

        attention = attention.reshape(
            token_count,
            INNER_SIZE,
        )

        # ----------------------------------------------------
        # ATTENTION OUTPUT PROJECTION
        # ----------------------------------------------------

        attention_cpu = self.out_proj(attention)

        attention_output = attention_cpu.to(
            device=self.device,
            dtype=torch.float32,
        )
        if self.lora is not None:
            self.lora.add_linear("out_proj", attention, attention_output)

        del attention_cpu, attention

        # ----------------------------------------------------
        # ATTENTION GATED RESIDUAL
        # ----------------------------------------------------

        x = apply_adaln_gated_residual(
            x,
            attention_output,
            modulation_table,
            adaln_indices,
            gate_group=2,
            token_tile=self.token_tile,
        )

        del attention_output

        # ----------------------------------------------------
        # MLP NORMALIZATION AND ADALN
        # ----------------------------------------------------

        normalized = rms_norm(
            x,
            self.norm2_weight,
        )

        normalized = apply_adaln_indexed(
            normalized,
            modulation_table,
            adaln_indices,
            shift_group=3,
            scale_group=4,
            token_tile=self.token_tile,
        )

        # ----------------------------------------------------
        # MLP FC1
        # ----------------------------------------------------

        fc1_cpu = self.fc1(normalized)

        fc1_output = fc1_cpu.to(
            device=self.device,
            dtype=torch.float32,
        )

        del fc1_cpu

        if fc1_output.shape != (
                token_count,
                MLP_HIDDEN_SIZE * 2,
        ):
            raise RuntimeError(
                f"Unexpected FC1 shape: {tuple(fc1_output.shape)}"
            )
        if self.lora is not None:
            self.lora.add_linear("fc1", normalized, fc1_output)
        del normalized

        # ----------------------------------------------------
        # SWIGLU
        # ----------------------------------------------------

        mlp_hidden = swiglu(fc1_output)

        del fc1_output

        # ----------------------------------------------------
        # MLP FC2
        # ----------------------------------------------------

        fc2_cpu = self.fc2(mlp_hidden)

        mlp_output = fc2_cpu.to(
            device=self.device,
            dtype=torch.float32,
        )
        if self.lora is not None:
            self.lora.add_linear("fc2", mlp_hidden, mlp_output)

        del fc2_cpu, mlp_hidden

        # ----------------------------------------------------
        # MLP GATED RESIDUAL
        # ----------------------------------------------------

        output = apply_adaln_gated_residual(
            x,
            mlp_output,
            modulation_table,
            adaln_indices,
            gate_group=5,
            token_tile=self.token_tile,
        )

        del x, mlp_output, modulation_table, adaln_indices

        check_finite("Transformer block output", output)

        return output.to(original_device)


# ============================================================
# OPERATOR SELF-TESTS
# ============================================================

def test_rms_norm(device):
    x = torch.randn(
        4,
        HIDDEN_SIZE,
        device=device,
    )

    weight = torch.ones(
        HIDDEN_SIZE,
        device=device,
    )

    output = rms_norm(x, weight)

    require(
        output.shape == x.shape,
        "RMSNorm returned incorrect shape",
    )

    check_finite("RMSNorm", output)


def test_rope(device):
    tokens = 7

    positions = torch.zeros(
        tokens,
        3,
        device=device,
    )

    positions[:, 0] = torch.arange(
        tokens,
        device=device,
    )

    inv_freq = torch.exp(
        -torch.arange(
            ROTARY_AXIS_DIM,
            device=device,
            dtype=torch.float32,
        ) / ROTARY_AXIS_DIM
    )

    x = torch.randn(
        tokens,
        NUM_HEADS,
        HEAD_DIM,
        device=device,
    )

    cos, sin = build_rope(
        positions,
        inv_freq,
    )

    output = apply_rope(
        x,
        cos,
        sin,
    )

    require(
        output.shape == x.shape,
        "RoPE returned incorrect shape",
    )

    # The final 32 dimensions must remain unchanged.
    require(
        torch.equal(
            output[..., ROTARY_DIM:],
            x[..., ROTARY_DIM:],
        ),
        "RoPE modified the unrotated head dimensions",
    )

    check_finite("Three-axis partial RoPE", output)


def test_swiglu(device):
    x = torch.randn(
        4,
        MLP_HIDDEN_SIZE * 2,
        device=device,
    )

    output = swiglu(x)

    require(
        output.shape == (4, MLP_HIDDEN_SIZE),
        "SwiGLU returned incorrect shape",
    )

    check_finite("SwiGLU", output)


def test_attention(device):
    tokens = 19

    q = torch.randn(
        tokens,
        NUM_HEADS,
        HEAD_DIM,
        device=device,
    )

    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = streamed_attention(
        q,
        k,
        v,
        query_chunk=5,
        key_chunk=7,
    )

    require(
        output.shape == q.shape,
        "Attention returned incorrect shape",
    )

    check_finite("Streamed full attention", output)


def run_operator_tests():
    print("\nH3 TRANSFORMER OPERATOR TESTS")
    print("=" * 44)

    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda:0")

    for device_name in devices:
        device = torch.device(device_name)

        tests = [
            ("RMSNorm", test_rms_norm),
            ("three-axis partial RoPE", test_rope),
            ("SwiGLU", test_swiglu),
            ("streamed full attention", test_attention),
        ]

        for name, test in tests:
            test(device)
            print(f"{device}: {name} PASSED")

        if device.type == "cuda":
            torch.cuda.synchronize(device)


# ============================================================
# REAL CHECKPOINT BLOCK TEST
# ============================================================

@torch.inference_mode()
def test_checkpoint_block():
    print("\nH3 REAL CHECKPOINT BLOCK TEST")
    print("=" * 44)

    if not os.path.isfile(DEFAULT_GGUF):
        raise FileNotFoundError(
            f"H3 checkpoint not found: {DEFAULT_GGUF}"
        )

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Device: {device}")
    print("Opening GGUF checkpoint...", flush=True)

    with GGUF(DEFAULT_GGUF) as gguf:
        print(
            f"GGUF tensors: {gguf.tensor_count}",
            flush=True,
        )

        # inspect_h3_output_tensors(gguf)
        #
        # available_blocks = {
        #     int(name.split(".")[1])
        #     for name in gguf.tensors
        #     if name.startswith("blocks.")
        #        and len(name.split(".")) > 2
        #        and name.split(".")[1].isdigit()
        # }

        # ----------------------------------------------------
        # ROPE FREQUENCIES
        # ----------------------------------------------------

        inv_freq = read_checkpoint_vector(
            gguf,
            "rope.inv_freq",
        )

        if inv_freq.numel() != ROTARY_AXIS_DIM:
            raise RuntimeError(
                f"Unexpected rope.inv_freq size: "
                f"{inv_freq.numel()}"
            )

        # ----------------------------------------------------
        # REAL CHECKPOINT ADALN PROJECTION
        # ----------------------------------------------------

        adaln = StreamingAdaLN(
            gguf,
            row_tile=512,
        )

        timesteps = torch.tensor(
            [0.5],
            dtype=torch.float32,
        )

        timestep_indices = torch.tensor(
            [0, 0],
            dtype=torch.long,
        )

        token_tags = torch.tensor(
            [VIDEO, TEXT],
            dtype=torch.long,
        )

        modulation_table, adaln_indices = (
            adaln.project_block(
                0,
                timesteps,
                timestep_indices,
                token_tags,
                device=str(device),
                dtype=torch.float32,
            )
        )

        print(
            "Block modulation:",
            tuple(modulation_table.shape),
            flush=True,
        )

        print(
            "AdaLN indices:",
            tuple(adaln_indices.shape),
            flush=True,
        )

        # ----------------------------------------------------
        # REAL CHECKPOINT TRANSFORMER BLOCK
        # ----------------------------------------------------

        block = H3TransformerBlock(
            gguf,
            block_index=0,
            device=str(device),
            row_tile=DEFAULT_ROW_TILE,
            token_tile=DEFAULT_TOKEN_TILE,
            query_chunk=DEFAULT_QUERY_CHUNK,
            key_chunk=DEFAULT_KEY_CHUNK,
        )

        hidden_states = torch.randn(
            2,
            HIDDEN_SIZE,
            dtype=torch.float32,
        )

        position_ids = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        output = block(
            hidden_states,
            modulation_table,
            adaln_indices,
            position_ids,
            inv_freq,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        require(
            tuple(output.shape) == (2, HIDDEN_SIZE),
            f"Incorrect block output shape: {tuple(output.shape)}",
        )

        check_finite(
            "Real checkpoint block output",
            output,
        )

        print(
            "Real checkpoint block output:",
            tuple(output.shape),
            flush=True,
        )

        print(
            "REAL CHECKPOINT BLOCK PASSED",
            flush=True,
        )

        del block
        del adaln
        del modulation_table
        del adaln_indices
        del hidden_states
        del output

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# REAL 50-BLOCK STREAMED TRANSFORMER TEST
# ============================================================


@torch.inference_mode()
def test_full_transformer_stack():
    print("\nH3 REAL 50-BLOCK STREAMED TRANSFORMER TEST")
    print("=" * 48, flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")

    device = torch.device("cuda:0")

    print("Device:", device, flush=True)
    print("Checkpoint:", DEFAULT_GGUF, flush=True)

    hidden_size = HIDDEN_SIZE
    block_count = 50

    # Three packed tokens: video, text and audio.
    token_tags_np = np.array(
        [VIDEO, TEXT, AUDIO],
        dtype=np.int64,
    )

    timestep_indices_np = np.array(
        [0, 0, 0],
        dtype=np.int64,
    )

    timesteps_np = np.array(
        [0.5],
        dtype=np.float32,
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    hidden = torch.randn(
        3,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )

    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Initial hidden states are invalid")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with GGUF(DEFAULT_GGUF) as gguf:

        inspect_h3_refiner_tensors(gguf)

        available_blocks = {
            int(name.split(".")[1])
            for name in gguf.tensors
            if name.startswith("blocks.") and len(name.split(".")) > 2 and name.split(".")[1].isdigit()
        }

        missing = sorted(
            set(range(block_count)) - available_blocks
        )

        if missing:
            raise RuntimeError(
                f"Checkpoint missing blocks: {missing}"
            )

        print("Checkpoint block inventory: PASSED", flush=True)

        # Use the actual checkpoint-backed AdaLN implementation.
        adaln = StreamingAdaLN(
            gguf,
            row_tile=512,
        )

        # Same RoPE frequency source used by the single-block test.
        inv_freq = read_checkpoint_vector(gguf, "rope.inv_freq")

        if not isinstance(inv_freq, torch.Tensor):
            inv_freq = torch.as_tensor(
                inv_freq,
                dtype=torch.float32,
            )

        inv_freq = inv_freq.to(
            device=device,
            dtype=torch.float32,
        )

        print(
            "RoPE inverse frequencies:",
            tuple(inv_freq.shape),
            flush=True,
        )

        for block_index in range(block_count):

            print(
                f"Block {block_index + 1:02d}/{block_count}",
                end=" ",
                flush=True,
            )

            # Calculate the actual modulation for this block.
            modulation, adaln_indices = adaln.project_block(
                block_index=block_index,
                timesteps=timesteps_np,
                timestep_indices=timestep_indices_np,
                token_tags=token_tags_np,
                device=device,
            )

            if not isinstance(modulation, torch.Tensor):
                modulation = torch.as_tensor(
                    modulation,
                    dtype=torch.float32,
                    device=device,
                )

            if not isinstance(adaln_indices, torch.Tensor):
                adaln_indices = torch.as_tensor(
                    adaln_indices,
                    dtype=torch.long,
                    device=device,
                )

            modulation = modulation.to(
                device=device,
                dtype=torch.float32,
            )

            adaln_indices = adaln_indices.to(
                device=device,
                dtype=torch.long,
            )

            if modulation.shape != (
                    3,
                    6,
                    hidden_size,
            ):
                raise RuntimeError(
                    f"Block {block_index}: invalid modulation "
                    f"shape {tuple(modulation.shape)}"
                )

            if adaln_indices.numel() != 3:
                raise RuntimeError(
                    f"Block {block_index}: invalid AdaLN indices "
                    f"{tuple(adaln_indices.shape)}"
                )

            # Use the verified positional __call__ API.
            block = H3TransformerBlock(
                gguf,
                block_index=block_index,
                device=str(device),
                row_tile=DEFAULT_ROW_TILE,
                token_tile=DEFAULT_TOKEN_TILE,
                query_chunk=DEFAULT_QUERY_CHUNK,
                key_chunk=DEFAULT_KEY_CHUNK,
            )

            output = block(
                hidden,
                modulation,
                adaln_indices,
                positions,
                inv_freq,
            )

            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Block {block_index} returned "
                    f"{type(output).__name__}"
                )

            if tuple(output.shape) != (3, hidden_size):
                raise RuntimeError(
                    f"Block {block_index}: output shape "
                    f"{tuple(output.shape)}"
                )

            if not torch.isfinite(output).all():
                raise FloatingPointError(
                    f"Block {block_index} produced NaN/Inf"
                )

            hidden = output.contiguous()

            del output
            del block
            del modulation
            del adaln_indices

            if (block_index + 1) % 5 == 0:
                torch.cuda.synchronize(device)

                allocated = (
                        torch.cuda.memory_allocated(device)
                        / 1024 ** 2
                )

                peak = (
                        torch.cuda.max_memory_allocated(device)
                        / 1024 ** 2
                )

                print(
                    f"PASSED | VRAM {allocated:.0f} MiB"
                    f" | peak {peak:.0f} MiB",
                    flush=True,
                )
            else:
                print("PASSED", flush=True)

            gc.collect()

        torch.cuda.synchronize(device)

        print("\n50-BLOCK EXECUTION COMPLETE", flush=True)
        print("Output shape:", tuple(hidden.shape), flush=True)
        print(
            "Output finite:",
            bool(torch.isfinite(hidden).all().item()),
            flush=True,
        )

        print(
            "Peak allocated VRAM:",
            f"{torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f} MiB",
            flush=True,
        )

        print(
            "Peak reserved VRAM:",
            f"{torch.cuda.max_memory_reserved(device) / 1024 ** 2:.1f} MiB",
            flush=True,
        )

    del hidden
    del inv_freq
    del adaln
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "\nREAL 50-BLOCK STREAMED TRANSFORMER TEST PASSED",
        flush=True,
    )


def inspect_h3_refiner_tensors(gguf):
    import re

    pattern = re.compile(
        r"token_refiner|video_patch_proj|audio_patch_proj",
        re.IGNORECASE,
    )

    print("\nH3 REFINER CHECKPOINT INVENTORY")
    print("=" * 60)

    count = 0

    for name, tensor in sorted(gguf.tensors.items()):
        if not pattern.search(name):
            continue

        print(
            f"{name}\n"
            f"    dims={tensor.dims} | "
            f"ggml_type={tensor.ggml_type} | "
            f"elements={tensor.elements} | "
            f"offset={tensor.offset}"
        )
        count += 1

    print(f"\nTotal matching tensors: {count}")


@torch.inference_mode()
def streamed_refiner_attention(
        q,
        k,
        v,
        query_chunk=64,
        key_chunk=128,
):
    """
    Memory-bounded bidirectional attention for H3 token refiner.

    Inputs:
        q, k, v: [sequence, heads, head_dim]

    Output:
        [sequence, heads, head_dim]

    Uses online softmax. No causal mask and no RoPE.
    """
    import math
    import torch

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must have shape [S, H, D]")

    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(
            f"Q/K/V shape mismatch: {q.shape}, {k.shape}, {v.shape}"
        )

    sequence, heads, head_dim = q.shape

    if sequence == 0:
        return torch.empty_like(q)

    if query_chunk < 1 or key_chunk < 1:
        raise ValueError("Chunk sizes must be positive")

    scale = 1.0 / math.sqrt(head_dim)

    # [H, S, D]
    q = q.transpose(0, 1).contiguous()
    k = k.transpose(0, 1).contiguous()
    v = v.transpose(0, 1).contiguous()

    output = torch.empty_like(q)

    for q_start in range(0, sequence, query_chunk):
        q_end = min(q_start + query_chunk, sequence)
        q_block = q[:, q_start:q_end, :]

        q_length = q_end - q_start

        running_max = torch.full(
            (heads, q_length, 1),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        running_sum = torch.zeros_like(running_max)

        running_output = torch.zeros(
            (heads, q_length, head_dim),
            dtype=torch.float32,
            device=q.device,
        )

        for k_start in range(0, sequence, key_chunk):
            k_end = min(k_start + key_chunk, sequence)

            k_block = k[:, k_start:k_end, :]
            v_block = v[:, k_start:k_end, :]

            scores = torch.matmul(
                q_block.float(),
                k_block.float().transpose(-1, -2),
            ) * scale

            block_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, block_max)

            old_scale = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max)

            running_output.mul_(old_scale)
            running_output.add_(
                torch.matmul(probabilities, v_block.float())
            )

            running_sum.mul_(old_scale)
            running_sum.add_(
                probabilities.sum(dim=-1, keepdim=True)
            )

            running_max = new_max

        output[:, q_start:q_end, :] = (
                running_output / running_sum
        ).to(output.dtype)

    return output.transpose(0, 1).contiguous()


@torch.inference_mode()
def test_streamed_refiner_attention():
    import torch

    print("\nH3 STREAMED REFINER ATTENTION TEST")
    print("=" * 44)

    for device in ["cpu"] + (
            ["cuda:0"] if torch.cuda.is_available() else []
    ):
        torch.manual_seed(1234)

        sequence = 257
        heads = 56
        head_dim = 128

        q = torch.randn(
            sequence, heads, head_dim,
            device=device,
            dtype=torch.float32,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        actual = streamed_refiner_attention(
            q, k, v,
            query_chunk=32,
            key_chunk=64,
        )

        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            is_causal=False,
        )

        reference = reference.squeeze(0).transpose(0, 1)

        max_error = (actual - reference).abs().max().item()

        assert actual.shape == q.shape
        assert torch.isfinite(actual).all()
        assert max_error < 1e-4, (
            f"{device}: attention mismatch, max error={max_error}"
        )

        print(
            f"{device}: PASSED | "
            f"shape={tuple(actual.shape)} | "
            f"max error={max_error:.2e}"
        )

        del q, k, v, actual, reference

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("STREAMED REFINER ATTENTION PASSED")


class H3TokenRefiner:
    """
    Checkpoint-backed H3 token refiner.

    Streams all large GGUF projections through TiledGGUFLinear.
    Keeps the refiner activations on CPU to avoid unnecessary VRAM use.
    """

    def __init__(self, gguf, block_index, row_tile=8, token_tile=16, lora=None):
        self.gguf = gguf
        self.block_index = block_index
        self.prefix = f"token_refiner.blocks.{block_index}"
        self.row_tile = row_tile
        self.token_tile = token_tile

        self.norm1 = self._vector("norm1.weight")
        self.norm2 = self._vector("norm2.weight")
        self.q_norm = self._vector("attn.q_norm.weight")
        self.k_norm = self._vector("attn.k_norm.weight")

        self.qkv = self._linear("attn.qkv_proj.weight")
        self.attn_out = self._linear("attn.out_proj.weight")
        self.fc1 = self._linear("mlp.fc1.weight")
        self.fc2 = self._linear("mlp.fc2.weight")
        self.lora = (
            lora.load_block("token_refiner", self.block_index, "cpu")
            if lora is not None else None
        )

    def _vector(self, suffix):
        value = read_checkpoint_vector(
            self.gguf,
            f"{self.prefix}.{suffix}",
        )
        return torch.as_tensor(value, dtype=torch.float32, device="cpu")

    def _linear(self, suffix):
        return TiledGGUFLinear(
            self.gguf,
            f"{self.prefix}.{suffix}",
            device="cpu",
            row_tile=self.row_tile,
            token_tile=self.token_tile,
        )

    @torch.inference_mode()
    def __call__(self, hidden):
        hidden = hidden.to(device="cpu", dtype=torch.float32)

        if hidden.ndim != 2 or hidden.shape[-1] != 5376:
            raise ValueError(
                f"Expected [tokens, 5376], got {tuple(hidden.shape)}"
            )

        # Attention: pre-norm -> QKV -> Q/K norm -> bidirectional attention.
        normalized = rms_norm(hidden, self.norm1)

        qkv = self.qkv(normalized)
        if self.lora is not None:
            self.lora.add_qkv(normalized, qkv)
        qkv = qkv.reshape(hidden.shape[0], 3, 56, 128)

        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        q = rms_norm(q, self.q_norm)
        k = rms_norm(k, self.k_norm)

        attended = streamed_refiner_attention(
            q,
            k,
            v,
            query_chunk=32,
            key_chunk=64,
        )

        attended = attended.reshape(hidden.shape[0], 7168)
        attention_input = attended
        attended = self.attn_out(attention_input)
        if self.lora is not None:
            self.lora.add_linear("out_proj", attention_input, attended)

        hidden = hidden + attended

        # MLP: pre-norm -> SwiGLU -> output projection -> residual.
        normalized = rms_norm(hidden, self.norm2)
        expanded = self.fc1(normalized)
        if self.lora is not None:
            self.lora.add_linear("fc1", normalized, expanded)
        activated = swiglu(expanded)
        projected = self.fc2(activated)
        if self.lora is not None:
            self.lora.add_linear("fc2", activated, projected)

        hidden = hidden + projected

        return hidden


@torch.inference_mode()
def test_real_token_refiner():
    import torch

    print("\nH3 REAL CHECKPOINT TOKEN REFINER TEST")
    print("=" * 48)

    torch.manual_seed(1234)

    device = "cpu"

    with GGUF(DEFAULT_GGUF) as gguf:
        hidden = torch.randn(
            3, 5376,
            dtype=torch.float32,
            device=device,
        )

        for block_index in range(2):
            print(f"Refiner block {block_index + 1}/2...", flush=True)

            block = H3TokenRefiner(
                gguf,
                block_index=block_index,
                row_tile=8,
                token_tile=16,
            )

            hidden = block(hidden)

            assert hidden.shape == (3, 5376)
            assert torch.isfinite(hidden).all()

            print(
                f"Block {block_index + 1} PASSED | "
                f"shape={tuple(hidden.shape)}"
            )

            del block

        final_norm = read_checkpoint_vector(
            gguf,
            "token_refiner.final_norm.weight",
        )
        final_norm = torch.as_tensor(
            final_norm,
            dtype=torch.float32,
            device="cpu",
        )

        hidden = rms_norm(hidden, final_norm)

        assert hidden.shape == (3, 5376)
        assert torch.isfinite(hidden).all()

        print("Final norm PASSED")
        print(f"Output shape: {tuple(hidden.shape)}")
        print(f"Output finite: {torch.isfinite(hidden).all().item()}")

    print("\nREAL TOKEN REFINER TEST PASSED")


class H3PatchProjection:
    """Streamed checkpoint-backed video or audio patch projection."""

    def __init__(self, gguf, modality, row_tile=8, token_tile=16):
        if modality not in ("video", "audio"):
            raise ValueError("modality must be 'video' or 'audio'")

        self.modality = modality
        self.input_width = 96 if modality == "video" else 32

        prefix = f"{modality}_patch_proj"

        self.projection = TiledGGUFLinear(
            gguf,
            f"{prefix}.weight",
            device="cpu",
            row_tile=row_tile,
            token_tile=token_tile,
        )

        bias = read_checkpoint_vector(gguf, f"{prefix}.bias")
        self.bias = torch.as_tensor(
            bias, dtype=torch.float32, device="cpu"
        )

        if self.bias.shape != (5376,):
            raise ValueError(
                f"{modality} bias has unexpected shape: {tuple(self.bias.shape)}"
            )

    @torch.inference_mode()
    def __call__(self, patches):
        patches = torch.as_tensor(
            patches, dtype=torch.float32, device="cpu"
        )

        if patches.ndim != 2 or patches.shape[1] != self.input_width:
            raise ValueError(
                f"{self.modality} patches must have shape "
                f"[N, {self.input_width}], got {tuple(patches.shape)}"
            )

        output = self.projection(patches)
        output = output + self.bias

        if output.shape != (patches.shape[0], 5376):
            raise RuntimeError(
                f"Unexpected {self.modality} projection output: "
                f"{tuple(output.shape)}"
            )

        if not torch.isfinite(output).all():
            raise FloatingPointError(
                f"{self.modality} projection produced non-finite values"
            )

        return output


@torch.inference_mode()
def test_real_patch_projections():
    print("\nH3 REAL VIDEO/AUDIO PATCH PROJECTION TEST")
    print("=" * 50)

    with GGUF(DEFAULT_GGUF) as gguf:
        video_projection = H3PatchProjection(gguf, "video")
        audio_projection = H3PatchProjection(gguf, "audio")

        video_patches = torch.randn(4, 96, dtype=torch.float32)
        audio_patches = torch.randn(4, 32, dtype=torch.float32)

        video_tokens = video_projection(video_patches)
        audio_tokens = audio_projection(audio_patches)

        assert video_tokens.shape == (4, 5376)
        assert audio_tokens.shape == (4, 5376)

        assert torch.isfinite(video_tokens).all()
        assert torch.isfinite(audio_tokens).all()

        print(
            f"Video projection PASSED | "
            f"{tuple(video_patches.shape)} -> {tuple(video_tokens.shape)}"
        )
        print(
            f"Audio projection PASSED | "
            f"{tuple(audio_patches.shape)} -> {tuple(audio_tokens.shape)}"
        )

        del video_projection, audio_projection

    print("REAL VIDEO/AUDIO PATCH PROJECTIONS PASSED")


def inspect_h3_output_tensors(gguf):
    print("\nH3 VIDEO/AUDIO OUTPUT TENSOR INVENTORY")
    print("=" * 60)

    keywords = (
        "video", "audio", "final", "output",
        "head", "norm", "unpatch", "patch",
    )

    count = 0

    for name, tensor in sorted(gguf.tensors.items()):
        # Transformer blocks and token refiner were already inspected.
        if name.startswith("blocks.") or name.startswith("token_refiner."):
            continue

        if not any(word in name.lower() for word in keywords):
            continue

        print(
            f"{name}\n"
            f"    dims={tensor.dims} | "
            f"ggml_type={tensor.ggml_type} | "
            f"elements={tensor.elements}"
        )
        count += 1

    print(f"\nTotal matching tensors: {count}")


class H3OutputHeads:
    """
    Checkpoint-backed H3 video/audio output heads.

    Input:
        hidden:             [N, 5376]
        final_modulation:   [N, 2, 5376]
                            modulation[:, 0] = shift
                            modulation[:, 1] = scale

    Output:
        video: [N, 96]
        audio: [N, 32]
    """

    def __init__(self, gguf, row_tile=8, token_tile=16):
        self.final_norm_weight = torch.as_tensor(
            read_checkpoint_vector(
                gguf, "final_layer.norm.weight"
            ),
            dtype=torch.float32,
            device="cpu",
        )

        self.video_out = TiledGGUFLinear(
            gguf,
            "final_layer.video_out.weight",
            device="cpu",
            row_tile=row_tile,
            token_tile=token_tile,
        )

        self.audio_out = TiledGGUFLinear(
            gguf,
            "final_layer.audio_out.weight",
            device="cpu",
            row_tile=row_tile,
            token_tile=token_tile,
        )

        self.video_bias = torch.as_tensor(
            read_checkpoint_vector(
                gguf, "final_layer.video_out.bias"
            ),
            dtype=torch.float32,
            device="cpu",
        )

        self.audio_bias = torch.as_tensor(
            read_checkpoint_vector(
                gguf, "final_layer.audio_out.bias"
            ),
            dtype=torch.float32,
            device="cpu",
        )

        assert self.final_norm_weight.shape == (5376,)
        assert self.video_bias.shape == (96,)
        assert self.audio_bias.shape == (32,)

    @torch.inference_mode()
    def __call__(self, hidden, final_modulation):
        hidden = hidden.to(device="cpu", dtype=torch.float32)
        final_modulation = final_modulation.to(
            device="cpu", dtype=torch.float32
        )

        if hidden.ndim != 2 or hidden.shape[1] != 5376:
            raise ValueError(
                f"Expected hidden [N,5376], got {tuple(hidden.shape)}"
            )

        expected = (hidden.shape[0], 2, 5376)
        if tuple(final_modulation.shape) != expected:
            raise ValueError(
                f"Expected final modulation {expected}, "
                f"got {tuple(final_modulation.shape)}"
            )

        normalized = rms_norm(hidden, self.final_norm_weight)

        shift = final_modulation[:, 0, :]
        scale = final_modulation[:, 1, :]

        conditioned = normalized * (1.0 + scale) + shift

        video = self.video_out(conditioned) + self.video_bias
        audio = self.audio_out(conditioned) + self.audio_bias

        if video.shape != (hidden.shape[0], 96):
            raise RuntimeError(f"Unexpected video output: {video.shape}")

        if audio.shape != (hidden.shape[0], 32):
            raise RuntimeError(f"Unexpected audio output: {audio.shape}")

        if not torch.isfinite(video).all():
            raise FloatingPointError("Non-finite video output")

        if not torch.isfinite(audio).all():
            raise FloatingPointError("Non-finite audio output")

        return video, audio


# import numpy as np
# import torch
#
# from h3_stream_linear import GGUF
# from h3_adaln import StreamingAdaLN
#
#
# MODEL_PATH = (
#     r"C:\Users\User\PycharmProjects\pythonProject\models"
#     r"\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
# )
#
# HIDDEN_SIZE = 5376
#
# print("H3 REAL FINAL ADALN + VIDEO/AUDIO OUTPUT TEST")
# print("=" * 52)
#
# # Open the real checkpoint.
# with GGUF(MODEL_PATH) as gguf:
#     adaln = StreamingAdaLN(gguf)
#
#     # Three packed tokens, all using the same diffusion timestep.
#     hidden = torch.randn(3, HIDDEN_SIZE, dtype=torch.float32)
#
#     timesteps = np.array([0.5], dtype=np.float32)
#     timestep_indices = np.array([0, 0, 0], dtype=np.int64)
#
#     # Real final-layer AdaLN checkpoint projection.
#     modulation_table, returned_indices = adaln.project_final(
#         timesteps=timesteps,
#         timestep_indices=timestep_indices,
#         device="cpu",
#     )
#
#     assert tuple(modulation_table.shape) == (1, 2, HIDDEN_SIZE)
#     assert np.array_equal(
#         np.asarray(returned_indices),
#         timestep_indices,
#     )
#
#     print("Final AdaLN checkpoint projection PASSED")
#
#     # Select the correct modulation for each token.
#     selected = modulation_table.index_select(
#         0,
#         torch.as_tensor(returned_indices, dtype=torch.long),
#     )
#
#     assert tuple(selected.shape) == (3, 2, HIDDEN_SIZE)
#
#     # Real final norm and real video/audio output projections.
#     heads = H3OutputHeads(gguf)
#     video, audio = heads(hidden, selected)
#
#     assert tuple(video.shape) == (3, 96)
#     assert tuple(audio.shape) == (3, 32)
#     assert torch.isfinite(video).all()
#     assert torch.isfinite(audio).all()
#
#     print("Final indexed modulation PASSED")
#     print("Video output PASSED:", tuple(video.shape))
#     print("Audio output PASSED:", tuple(audio.shape))
#     print("Both outputs finite: True")
#
# print("\nREAL FINAL ADALN + VIDEO/AUDIO OUTPUT TEST PASSED")


@torch.inference_mode()
def test_real_output_heads():
    print("\nH3 REAL VIDEO/AUDIO OUTPUT HEAD TEST")
    print("=" * 48)

    with GGUF(DEFAULT_GGUF) as gguf:
        heads = H3OutputHeads(gguf)

        hidden = torch.randn(3, 5376, dtype=torch.float32)

        # Synthetic modulation for testing the output heads.
        # The actual final AdaLN projection is tested separately.
        modulation = torch.zeros(
            3, 2, 5376, dtype=torch.float32
        )

        video, audio = heads(hidden, modulation)

        assert video.shape == (3, 96)
        assert audio.shape == (3, 32)
        assert torch.isfinite(video).all()
        assert torch.isfinite(audio).all()

        print(f"Video output PASSED: {tuple(video.shape)}")
        print(f"Audio output PASSED: {tuple(audio.shape)}")
        print("Both outputs finite: True")

    print("\nREAL VIDEO/AUDIO OUTPUT HEAD TEST PASSED")


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        run_operator_tests()
        test_checkpoint_block()

        print("\n" + "=" * 44)
        print("ALL TRANSFORMER OPERATOR TESTS PASSED")
        print("=" * 44)

    except Exception as exc:
        print(
            f"\nTRANSFORMER TEST FAILED: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise

    # test_full_transformer_stack()
    # test_streamed_refiner_attention()
    # test_real_token_refiner()
    # test_real_patch_projections()
    # test_real_output_heads()


if __name__ == "__main__":
    main()
