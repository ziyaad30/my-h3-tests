"""
MiniMax H3 standalone GGUF streaming core.

This file is a clean implementation. It does not import or depend on
any previous project Python modules, ComfyUI, Diffusers, or a third-party
GGUF reader.

Responsibilities:
  * Parse GGUF metadata and tensor directory.
  * Memory-map the checkpoint read-only.
  * Decode F32, F16, BF16 and Q2_K.
  * Decode only requested matrix rows.
  * Perform linear projections with BOTH output-row and token tiling,
    so the full weight matrix and full activation are not placed on GPU.

GGUF matrix convention:
  GGUF ne[0] is the contiguous input dimension.
  The remaining dimensions are flattened into output rows.

Q2_K layout and dequantization follow ggml's block_q2_K definition.
"""

from __future__ import annotations

import argparse
import gc
import math
import mmap
import os
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hardware / file defaults
# ---------------------------------------------------------------------------

DEFAULT_GGUF = (
    r"C:\Users\User\PycharmProjects\pythonProject\models"
    r"\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q2_K = 10
TYPE_BF16 = 30

TYPE_NAMES = {
    TYPE_F32: "F32",
    TYPE_F16: "F16",
    TYPE_Q2_K: "Q2_K",
    TYPE_BF16: "BF16",
}

QK_K = 256
Q2_K_BYTES = 84

# Conservative defaults for a 4 GB GPU.
DEFAULT_OUTPUT_ROW_TILE = 8
DEFAULT_TOKEN_TILE = 16


# ---------------------------------------------------------------------------
# Binary reader helpers
# ---------------------------------------------------------------------------

def _read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(
            f"GGUF ended unexpectedly: wanted {size} bytes, got {len(data)}"
        )
    return data


def _u32(stream) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _u64(stream) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _gguf_string(stream) -> str:
    length = _u64(stream)
    if length > (1 << 30):
        raise ValueError(f"Unreasonable GGUF string length: {length}")
    return _read_exact(stream, length).decode("utf-8", errors="strict")


# GGUF metadata value types.
_META_FORMATS = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    7: "<?",   # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}


def _skip_meta_value(stream, kind: int) -> None:
    if kind == 8:  # STRING
        length = _u64(stream)
        stream.seek(length, os.SEEK_CUR)
        return

    if kind == 9:  # ARRAY
        element_type = _u32(stream)
        count = _u64(stream)
        if count > (1 << 32):
            raise ValueError(f"Unreasonable metadata array length: {count}")
        for _ in range(count):
            _skip_meta_value(stream, element_type)
        return

    fmt = _META_FORMATS.get(kind)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF metadata value type: {kind}")
    stream.seek(struct.calcsize(fmt), os.SEEK_CUR)


def _read_meta_scalar(stream, kind: int) -> Any:
    if kind == 8:
        return _gguf_string(stream)

    if kind == 9:
        _skip_meta_value(stream, kind)
        return None

    fmt = _META_FORMATS.get(kind)
    if fmt is None:
        raise ValueError(f"Unsupported GGUF metadata value type: {kind}")
    return struct.unpack(fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"Invalid GGUF alignment: {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _storage_size(shape: tuple[int, ...], type_id: int) -> int:
    elements = math.prod(shape)

    if type_id == TYPE_F32:
        return elements * 4
    if type_id in (TYPE_F16, TYPE_BF16):
        return elements * 2
    if type_id == TYPE_Q2_K:
        if elements % QK_K:
            raise ValueError(
                f"Q2_K tensor element count is not divisible by {QK_K}: {shape}"
            )
        return (elements // QK_K) * Q2_K_BYTES

    raise NotImplementedError(
        f"Unsupported GGUF tensor type {type_id} ({TYPE_NAMES.get(type_id, 'unknown')})"
    )


# ---------------------------------------------------------------------------
# Q2_K dequantization
# ---------------------------------------------------------------------------

def dequantize_q2_k(raw: memoryview | bytes | bytearray) -> np.ndarray:
    """
    Decode GGML Q2_K blocks to float32.

    One 84-byte block stores 256 values:
      scales[16], qs[64], d[2], dmin[2].

    This implements the same 2-bit extraction and scale/minute-scale
    ordering as ggml's dequantize_row_q2_K.
    """
    packed = np.frombuffer(raw, dtype=np.uint8)

    if packed.size % Q2_K_BYTES:
        raise ValueError(
            f"Q2_K byte count {packed.size} is not divisible by {Q2_K_BYTES}"
        )

    blocks = packed.reshape(-1, Q2_K_BYTES)
    nblocks = blocks.shape[0]

    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]

    d = (
        blocks[:, 80:82].copy().view("<f2").reshape(nblocks).astype(np.float32)
    )
    dmin = (
        blocks[:, 82:84].copy().view("<f2").reshape(nblocks).astype(np.float32)
    )

    out = np.empty((nblocks, QK_K), dtype=np.float32)

    # The C reference advances through two 128-value halves. Within
    # each half, four 2-bit planes are extracted from the same 32 bytes.
    for half in range(2):
        qbase = half * 32
        outbase = half * 128

        for group in range(4):
            shift = group * 2
            scale_base = half * 8 + group * 2
            out_base = outbase + group * 32

            scale0 = scales[:, scale_base]
            q0 = ((qs[:, qbase:qbase + 16] >> shift) & 0x03).astype(np.float32)

            out[:, out_base:out_base + 16] = (
                d[:, None] * ((scale0 & 0x0F).astype(np.float32))[:, None] * q0
                - dmin[:, None] * ((scale0 >> 4).astype(np.float32))[:, None]
            )

            scale1 = scales[:, scale_base + 1]
            q1 = (
                (qs[:, qbase + 16:qbase + 32] >> shift) & 0x03
            ).astype(np.float32)

            out[:, out_base + 16:out_base + 32] = (
                d[:, None] * ((scale1 & 0x0F).astype(np.float32))[:, None] * q1
                - dmin[:, None] * ((scale1 >> 4).astype(np.float32))[:, None]
            )

    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Independent GGUF mmap reader
# ---------------------------------------------------------------------------

class GGUFMap:
    """Read-only GGUF parser with bounded tensor-row access."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(Path(path).expanduser().resolve())
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self._file = open(self.path, "rb")
        self._map = mmap.mmap(
            self._file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.file_size = len(self._map)
        self.version = None
        self.alignment = 32
        self.data_offset = None
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self.closed = False

        try:
            self._parse_directory()
        except Exception:
            self.close()
            raise

    def _parse_directory(self) -> None:
        f = self._file
        f.seek(0)

        if _read_exact(f, 4) != b"GGUF":
            raise ValueError("File does not begin with GGUF magic")

        self.version = _u32(f)
        if self.version not in (2, 3):
            raise ValueError(
                f"Unsupported GGUF version {self.version}; expected v2 or v3"
            )

        tensor_count = _u64(f)
        metadata_count = _u64(f)

        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise ValueError("Unreasonable GGUF directory counts")

        for _ in range(metadata_count):
            key = _gguf_string(f)
            kind = _u32(f)

            if key == "general.alignment" and kind in (4, 10):
                value = _read_meta_scalar(f, kind)
                self.alignment = int(value)
                self.metadata[key] = value
            else:
                self.metadata[key] = _read_meta_scalar(f, kind)

        if self.alignment <= 0 or self.alignment > (1 << 20):
            raise ValueError(f"Invalid GGUF alignment: {self.alignment}")

        for _ in range(tensor_count):
            name = _gguf_string(f)
            ndim = _u32(f)

            if not 1 <= ndim <= 4:
                raise ValueError(f"Invalid dimensions for tensor {name}: {ndim}")

            # GGUF dimensions are stored with ne[0] as the contiguous
            # dimension. Do not reverse this tuple here.
            shape = tuple(_u64(f) for _ in range(ndim))
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"Invalid tensor dimensions for {name}: {shape}")

            type_id = _u32(f)
            relative_offset = _u64(f)
            byte_count = _storage_size(shape, type_id)

            if name in self.tensors:
                raise ValueError(f"Duplicate GGUF tensor name: {name}")

            self.tensors[name] = {
                "shape": shape,
                "type": type_id,
                "offset": relative_offset,
                "bytes": byte_count,
            }

        self.data_offset = _align(f.tell(), self.alignment)

        intervals = []
        for name, info in self.tensors.items():
            start = self.data_offset + info["offset"]
            end = start + info["bytes"]

            if start < self.data_offset or end > self.file_size:
                raise ValueError(f"Tensor data outside file bounds: {name}")

            intervals.append((start, end, name))

        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    f"Overlapping GGUF tensor data: {previous[2]} and {current[2]}"
                )

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("GGUF map is closed")

    def tensor_info(self, name: str) -> dict[str, Any]:
        self._ensure_open()
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"Tensor not found in GGUF: {name}") from None

    def matrix_dimensions(self, name: str) -> tuple[int, int]:
        """
        Return (input_width, output_rows) for a GGUF matrix.

        The first GGUF dimension is contiguous/input width.
        Remaining dimensions flatten into output rows.
        """
        info = self.tensor_info(name)
        shape = info["shape"]
        if len(shape) < 2:
            raise ValueError(f"Tensor is not a matrix: {name}")
        return int(shape[0]), int(math.prod(shape[1:]))

    def _raw_view(self, name: str, byte_start: int, byte_end: int):
        self._ensure_open()
        info = self.tensor_info(name)
        tensor_start = self.data_offset + info["offset"]
        start = tensor_start + byte_start
        end = tensor_start + byte_end
        if not (tensor_start <= start <= end <= tensor_start + info["bytes"]):
            raise ValueError(f"Invalid byte range for {name}")
        return memoryview(self._map)[start:end]

    @staticmethod
    def _decode_plain(raw: memoryview, type_id: int) -> np.ndarray:
        if type_id == TYPE_F32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        if type_id == TYPE_F16:
            return np.frombuffer(raw, dtype="<f2").astype(np.float32)
        if type_id == TYPE_BF16:
            u16 = np.frombuffer(raw, dtype="<u2")
            return (u16.astype(np.uint32) << 16).view(np.float32).copy()
        raise NotImplementedError(f"Not a plain tensor type: {type_id}")

    def read_vector_or_small_tensor(self, name: str) -> np.ndarray:
        """Read a non-matrix/small tensor. Do not use for Q2_K matrices."""
        info = self.tensor_info(name)
        start = self.data_offset + info["offset"]
        end = start + info["bytes"]
        raw = memoryview(self._map)[start:end]
        try:
            if info["type"] == TYPE_Q2_K:
                values = dequantize_q2_k(raw)
            else:
                values = self._decode_plain(raw, info["type"])
            return values.reshape(info["shape"])
        finally:
            raw.release()

    def read_matrix_rows(
        self, name: str, first_row: int, last_row: int
    ) -> np.ndarray:
        """Decode only [first_row:last_row] output rows."""
        input_width, output_rows = self.matrix_dimensions(name)
        info = self.tensor_info(name)

        if not (0 <= first_row < last_row <= output_rows):
            raise ValueError(
                f"Invalid row interval {first_row}:{last_row} for {name} "
                f"with {output_rows} rows"
            )

        type_id = info["type"]

        if type_id == TYPE_Q2_K:
            if input_width % QK_K:
                raise ValueError(
                    f"Q2_K input width {input_width} is not block aligned: {name}"
                )
            blocks_per_row = input_width // QK_K
            first_block = first_row * blocks_per_row
            block_count = (last_row - first_row) * blocks_per_row
            byte_start = first_block * Q2_K_BYTES
            byte_end = byte_start + block_count * Q2_K_BYTES
        else:
            bytes_per_value = {
                TYPE_F32: 4,
                TYPE_F16: 2,
                TYPE_BF16: 2,
            }.get(type_id)
            if bytes_per_value is None:
                raise NotImplementedError(f"Unsupported matrix type: {type_id}")
            row_bytes = input_width * bytes_per_value
            byte_start = first_row * row_bytes
            byte_end = last_row * row_bytes

        raw = self._raw_view(name, byte_start, byte_end)
        try:
            if type_id == TYPE_Q2_K:
                decoded = dequantize_q2_k(raw)
            else:
                decoded = self._decode_plain(raw, type_id)

            return decoded.reshape(last_row - first_row, input_width)
        finally:
            raw.release()

    def inventory(self) -> dict[str, Any]:
        type_counts = Counter()
        type_bytes = Counter()
        for info in self.tensors.values():
            label = TYPE_NAMES.get(info["type"], str(info["type"]))
            type_counts[label] += 1
            type_bytes[label] += info["bytes"]
        return {
            "path": self.path,
            "version": self.version,
            "tensor_count": len(self.tensors),
            "type_counts": dict(type_counts),
            "type_bytes": dict(type_bytes),
            "file_bytes": self.file_size,
        }

    def close(self):
        if self.closed:
            return
        self._map.close()
        self._file.close()
        self.closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------------
# Two-dimensional tiled streamed linear
# ---------------------------------------------------------------------------

class TiledGGUFLinear:
    """
    Linear projection from a GGUF matrix without full weight/activation GPU
    materialization.

    Input:
        CPU or GPU tensor [tokens, input_width].

    Output:
        CPU float32 tensor [tokens, output_width] by default.

    GPU working set is bounded by:
        token_tile * input_width
        + output_row_tile * input_width
        + token_tile * output_row_tile

    The output activation is retained on CPU. The caller decides whether
    to keep it there, transfer it, or consume it in further tiles.
    """

    def __init__(
        self,
        gguf: GGUFMap,
        device: str | torch.device = "cuda:0",
        output_row_tile: int = DEFAULT_OUTPUT_ROW_TILE,
        token_tile: int = DEFAULT_TOKEN_TILE,
    ):
        self.gguf = gguf
        self.device = torch.device(device)
        self.output_row_tile = int(output_row_tile)
        self.token_tile = int(token_tile)

        if self.output_row_tile < 1 or self.token_tile < 1:
            raise ValueError("Tile sizes must be positive")

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch CUDA is unavailable")

    @torch.inference_mode()
    def __call__(
        self,
        weight_name: str,
        x: torch.Tensor,
        bias: np.ndarray | torch.Tensor | None = None,
        output_device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        input_width, output_width = self.gguf.matrix_dimensions(weight_name)

        if x.ndim != 2 or x.shape[1] != input_width:
            raise ValueError(
                f"{weight_name}: expected [tokens,{input_width}], "
                f"received {tuple(x.shape)}"
            )

        tokens = int(x.shape[0])
        if tokens == 0:
            return torch.empty(
                (0, output_width),
                dtype=torch.float32,
                device=output_device,
            )

        output_device = torch.device(output_device)
        output = torch.empty(
            (tokens, output_width),
            dtype=torch.float32,
            device=output_device,
        )

        # Keep the input on its original device until each token tile is
        # transferred. This avoids a full-sequence GPU copy.
        for token_start in range(0, tokens, self.token_tile):
            token_end = min(token_start + self.token_tile, tokens)

            x_tile = x[token_start:token_end].to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=False,
            )
            x_t = x_tile.transpose(0, 1).contiguous()

            for row_start in range(0, output_width, self.output_row_tile):
                row_end = min(
                    row_start + self.output_row_tile, output_width
                )

                w_np = self.gguf.read_matrix_rows(
                    weight_name, row_start, row_end
                )
                w_gpu = torch.from_numpy(w_np).to(
                    device=self.device,
                    dtype=torch.float32,
                )

                # [output_tile,input] @ [input,token_tile]
                result = torch.mm(w_gpu, x_t).transpose(0, 1).contiguous()

                output[token_start:token_end, row_start:row_end].copy_(
                    result.to(device=output_device)
                )

                del w_np, w_gpu, result

            del x_tile, x_t

        if bias is not None:
            if isinstance(bias, np.ndarray):
                b = torch.from_numpy(bias.astype(np.float32, copy=False))
            else:
                b = bias.detach().to(dtype=torch.float32, device="cpu")

            b = b.reshape(-1)
            if b.numel() != output_width:
                raise ValueError(
                    f"Bias width mismatch: expected {output_width}, got {b.numel()}"
                )

            output.add_(b.to(output_device).unsqueeze(0))

        return output


# ---------------------------------------------------------------------------
# Self-tests: synthetic Q2_K layout + tiled linear against PyTorch reference
# ---------------------------------------------------------------------------

def _pack_known_q2k_block() -> bytes:
    """
    Create a deterministic legal-format Q2_K block.

    Scales are all 1 (low nibble), minima are 0 (high nibble).
    d=1, dmin=0. Quantized values cycle 0,1,2,3.
    """
    scales = bytes([0x01] * 16)
    quant = bytearray(64)

    for i in range(64):
        quant[i] = (
            (i % 4)
            | (((i + 1) % 4) << 2)
            | (((i + 2) % 4) << 4)
            | (((i + 3) % 4) << 6)
        )

    return (
        scales
        + bytes(quant)
        + struct.pack("<e", 1.0)
        + struct.pack("<e", 0.0)
    )


def run_self_test(device: str = "cpu") -> None:
    block = _pack_known_q2k_block()
    decoded = dequantize_q2_k(block)

    if decoded.shape != (256,):
        raise AssertionError(f"Wrong Q2_K output shape: {decoded.shape}")

    # Independent expected sequence for this constructed block.
    expected = np.empty(256, dtype=np.float32)
    q = np.frombuffer(block[16:80], dtype=np.uint8)
    for half in range(2):
        for group in range(4):
            for lane in range(16):
                expected[half * 128 + group * 32 + lane] = (
                    (q[half * 32 + lane] >> (2 * group)) & 3
                )
                expected[half * 128 + group * 32 + 16 + lane] = (
                    (q[half * 32 + 16 + lane] >> (2 * group)) & 3
                )

    np.testing.assert_array_equal(decoded, expected)

    # Test the tiled matrix multiplication algorithm on a temporary
    # in-memory GGUF-like object, without requiring a checkpoint file.
    class _SyntheticGGUF:
        def __init__(self):
            self.weight = np.arange(11 * 7, dtype=np.float32).reshape(11, 7) / 31

        def matrix_dimensions(self, name):
            assert name == "synthetic.weight"
            return 7, 11

        def read_matrix_rows(self, name, start, end):
            assert name == "synthetic.weight"
            return self.weight[start:end].copy()

    synthetic = _SyntheticGGUF()
    linear = TiledGGUFLinear(
        synthetic,  # type: ignore[arg-type]
        device=device,
        output_row_tile=3,
        token_tile=2,
    )

    rng = np.random.default_rng(20261127)
    x_np = rng.standard_normal((5, 7)).astype(np.float32)
    x = torch.from_numpy(x_np)

    actual = linear("synthetic.weight", x, output_device="cpu")
    reference = torch.from_numpy(x_np) @ torch.from_numpy(synthetic.weight).T

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    print("Q2_K synthetic block decode: PASS")
    print("2D tiled linear vs torch reference: PASS")
    print(f"Self-test device: {device}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone MiniMax H3 GGUF mmap and tiled-linear core"
    )
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-row-tile", type=int, default=DEFAULT_OUTPUT_ROW_TILE)
    parser.add_argument("--token-tile", type=int, default=DEFAULT_TOKEN_TILE)
    parser.add_argument("--show-tensors", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.device)
        return

    with GGUFMap(args.gguf) as gguf:
        report = gguf.inventory()

        print("\n========== STANDALONE H3 GGUF CORE ==========")
        print(f"File:       {report['path']}")
        print(f"GGUF:       v{report['version']}")
        print(f"File size:  {report['file_bytes'] / 1024**3:.3f} GiB")
        print(f"Tensors:    {report['tensor_count']}")

        print("\n--- Tensor storage ---")
        for name in sorted(report["type_counts"]):
            size = report["type_bytes"][name] / 1024**3
            print(
                f"{name:<8} {report['type_counts'][name]:>5} "
                f"{size:>10.3f} GiB"
            )

        print("\n--- Important matrix dimensions (input, output rows) ---")
        for name in (
            "blocks.0.attn.qkv_proj.weight",
            "blocks.0.attn.out_proj.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.0.mlp.fc2.weight",
            "condition_proj.weight",
        ):
            if name in gguf.tensors:
                print(f"{name}: {gguf.matrix_dimensions(name)}")

        if args.show_tensors:
            for name in sorted(gguf.tensors):
                info = gguf.tensor_info(name)
                print(
                    f"{name:<68} "
                    f"shape={info['shape']} "
                    f"type={TYPE_NAMES.get(info['type'], info['type'])}"
                )

        print("\nGGUF parsed and memory-mapped successfully.")
        print(
            "No large Q2_K tensor was materialized. "
            "Use TiledGGUFLinear for bounded projection."
        )

    gc.collect()


if __name__ == "__main__":
    main()
