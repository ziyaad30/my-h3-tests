
#!/usr/bin/env python3
"""
Standalone MiniMax H3 streamed GGUF linear engine.

Q2_K decoding:
    - Numba parallel decoder.
    - Four CPU threads by default.
    - Independent 256-element blocks decoded in parallel.

Memory:
    - GGUF weights remain memory-mapped.
    - Only one output-row tile is decoded at a time.
    - No full transformer matrices or blocks are cached.
    - No pinned-memory staging buffers.

CUDA projections:
    - CUDA row tile minimum: 2048.
    - Activations transferred once per projection.
    - Weight tiles transferred once.
    - FP32 weights and FP32 matmul retained.
    - Output remains on CUDA.

CPU projections:
    - Output remains on CPU.
    - Requested row tile is retained.

Supported GGML types:
    0  F32
    1  F16
    10 Q2_K
    30 BF16
"""

import argparse
import mmap
import os
import struct
import threading
import time

import numpy as np
import torch
from numba import config, njit, prange, set_num_threads


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_GGUF = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models"
    r"\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q2_K = 10
TYPE_Q3_K = 11
TYPE_BF16 = 30

Q2K_BLOCK_ELEMENTS = 256
Q2K_BLOCK_BYTES = 84
Q3K_BLOCK_ELEMENTS = 256
Q3K_BLOCK_BYTES = 110

# Measured on the GTX 1650:
# 2048 was fastest in both streamed linear and transformer
# block tests. Eight-block propagation also remained stable.
CUDA_MIN_ROW_TILE = 128

# Four threads was the fastest measured setting on the
# user's 8-logical-processor CPU.
Q2K_NUM_THREADS = min(4, config.NUMBA_NUM_THREADS)

if Q2K_NUM_THREADS < 1:
    raise RuntimeError("Numba has no available worker threads")

set_num_threads(Q2K_NUM_THREADS)


GGUF_METADATA_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


# ============================================================
# GGUF TENSOR DESCRIPTION
# ============================================================

class Tensor:
    def __init__(self, name, dims, ggml_type, offset):
        self.name = name
        self.dims = tuple(dims)
        self.ggml_type = ggml_type
        self.offset = offset
        self.absolute_offset = 0

    @property
    def elements(self):
        result = 1

        for dimension in self.dims:
            result *= dimension

        return result

    @property
    def input_width(self):
        if len(self.dims) != 2:
            raise ValueError(f"{self.name} is not a matrix")

        return self.dims[0]

    @property
    def output_rows(self):
        if len(self.dims) != 2:
            raise ValueError(f"{self.name} is not a matrix")

        return self.dims[1]


# ============================================================
# STREAMED MEMORY ACCOUNTING
# ============================================================

class StreamMemoryStats:
    """Tracks live host staging buffers and live CUDA weight tiles."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active_vram_bytes = 0
        self.peak_vram_bytes = 0
        self.active_host_bytes = 0
        self.peak_host_bytes = 0

    def add_vram(self, size):
        size = int(size)
        if size <= 0:
            return
        with self._lock:
            self.active_vram_bytes += size
            self.peak_vram_bytes = max(self.peak_vram_bytes, self.active_vram_bytes)

    def remove_vram(self, size):
        size = int(size)
        if size <= 0:
            return
        with self._lock:
            self.active_vram_bytes -= size
            if self.active_vram_bytes < 0:
                raise RuntimeError("H3 VRAM tile accounting underflow")

    def add_host(self, size):
        size = int(size)
        if size <= 0:
            return
        with self._lock:
            self.active_host_bytes += size
            self.peak_host_bytes = max(self.peak_host_bytes, self.active_host_bytes)

    def remove_host(self, size):
        size = int(size)
        if size <= 0:
            return
        with self._lock:
            self.active_host_bytes -= size
            if self.active_host_bytes < 0:
                raise RuntimeError("H3 host tile accounting underflow")

    def reset_peaks(self):
        with self._lock:
            self.peak_vram_bytes = self.active_vram_bytes
            self.peak_host_bytes = self.active_host_bytes

    def snapshot(self):
        with self._lock:
            return {
                "active_vram_bytes": self.active_vram_bytes,
                "peak_vram_bytes": self.peak_vram_bytes,
                "active_host_bytes": self.active_host_bytes,
                "peak_host_bytes": self.peak_host_bytes,
            }


# ============================================================
# GGUF READER
# ============================================================

class GGUF:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.mm = None
        self.size = 0
        self.version = 0
        self.tensor_count = 0
        self.metadata_count = 0
        self.alignment = 32
        self.data_offset = 0
        self.tensors = {}
        self.memory_stats = StreamMemoryStats()
        self.stream_working_set_bytes = 0

        self.open()
        self.parse()

    def open(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self.file = open(self.path, "rb")
        self.size = os.path.getsize(self.path)

        self.mm = mmap.mmap(
            self.file.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )

    def read(self, n):
        data = self.file.read(n)

        if len(data) != n:
            raise EOFError("Unexpected end of GGUF")

        return data

    def unpack(self, fmt):
        return struct.unpack(
            fmt,
            self.read(struct.calcsize(fmt)),
        )[0]

    def string(self):
        n = self.unpack("<Q")

        if n > self.size:
            raise ValueError("Invalid GGUF string length")

        return self.read(n).decode("utf-8")

    def metadata_value(self, kind):
        if kind in GGUF_METADATA_FORMATS:
            return self.unpack(GGUF_METADATA_FORMATS[kind])

        if kind == 8:
            return self.string()

        if kind == 9:
            element_type = self.unpack("<I")
            count = self.unpack("<Q")

            if count > 100_000_000:
                raise ValueError("Unreasonable GGUF metadata array")

            if count > 1_000_000:
                for _ in range(count):
                    self.metadata_value(element_type)

                return None

            return [
                self.metadata_value(element_type)
                for _ in range(count)
            ]

        raise ValueError(f"Unsupported GGUF metadata type {kind}")

    @staticmethod
    def align(value, alignment):
        return (
            (value + alignment - 1) // alignment
        ) * alignment

    @staticmethod
    def storage_size(t):
        n = t.elements

        if t.ggml_type == TYPE_F32:
            return n * 4

        if t.ggml_type in (TYPE_F16, TYPE_BF16):
            return n * 2

        if t.ggml_type == TYPE_Q2_K:
            if n % Q2K_BLOCK_ELEMENTS:
                raise ValueError(f"Q2_K element count not divisible by 256: {t.name}")
            return (n // Q2K_BLOCK_ELEMENTS) * Q2K_BLOCK_BYTES

        if t.ggml_type == TYPE_Q3_K:
            if n % Q3K_BLOCK_ELEMENTS:
                raise ValueError(f"Q3_K element count not divisible by 256: {t.name}")
            return (n // Q3K_BLOCK_ELEMENTS) * Q3K_BLOCK_BYTES

        raise ValueError(
            f"Unsupported GGML type {t.ggml_type}: {t.name}"
        )

    def parse(self):
        self.file.seek(0)

        if self.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")

        self.version = self.unpack("<I")

        if self.version not in (2, 3):
            raise ValueError(
                f"Unsupported GGUF version {self.version}"
            )

        self.tensor_count = self.unpack("<Q")
        self.metadata_count = self.unpack("<Q")

        metadata = {}

        for _ in range(self.metadata_count):
            key = self.string()
            kind = self.unpack("<I")
            metadata[key] = self.metadata_value(kind)

        self.alignment = int(
            metadata.get("general.alignment", 32) or 32
        )

        if self.alignment <= 0:
            raise ValueError("Invalid GGUF alignment")

        for _ in range(self.tensor_count):
            name = self.string()
            ndims = self.unpack("<I")

            if ndims < 1 or ndims > 4:
                raise ValueError(f"Invalid dimensions: {name}")

            dims = tuple(
                self.unpack("<Q")
                for _ in range(ndims)
            )

            ggml_type = self.unpack("<I")
            offset = self.unpack("<Q")

            if name in self.tensors:
                raise ValueError(f"Duplicate tensor {name}")

            self.tensors[name] = Tensor(
                name,
                dims,
                ggml_type,
                offset,
            )

        self.data_offset = self.align(
            self.file.tell(),
            self.alignment,
        )

        for t in self.tensors.values():
            t.absolute_offset = self.data_offset + t.offset

            end = (
                t.absolute_offset
                + self.storage_size(t)
            )

            if (
                t.absolute_offset < self.data_offset
                or end > self.size
            ):
                raise ValueError(
                    f"Tensor outside file: {t.name}"
                )

        # ComfyUI manages the maximum simultaneously resident weight tile,
        # not the complete disk-backed GGUF. The CUDA path enforces 128 rows.
        for tensor in self.tensors.values():
            if len(tensor.dims) != 2:
                continue
            rows = min(128, tensor.output_rows)
            tile_bytes = rows * tensor.input_width * 4
            self.stream_working_set_bytes = max(
                self.stream_working_set_bytes, tile_bytes
            )

    def read_tensor_rows(self, name, first_row, row_count):
        t = self.tensors[name]

        if len(t.dims) != 2:
            raise ValueError(f"Not a matrix: {name}")

        if first_row < 0 or row_count < 1:
            raise ValueError("Invalid row range")

        if first_row + row_count > t.output_rows:
            raise ValueError("Row range exceeds matrix")

        width = t.input_width

        if t.ggml_type == TYPE_Q2_K:
            if width % Q2K_BLOCK_ELEMENTS:
                raise ValueError(f"Invalid Q2_K input width: {name}")
            bytes_per_row = (width // Q2K_BLOCK_ELEMENTS) * Q2K_BLOCK_BYTES

        elif t.ggml_type == TYPE_Q3_K:
            if width % Q3K_BLOCK_ELEMENTS:
                raise ValueError(f"Invalid Q3_K input width: {name}")
            bytes_per_row = (width // Q3K_BLOCK_ELEMENTS) * Q3K_BLOCK_BYTES

        elif t.ggml_type == TYPE_F32:
            bytes_per_row = width * 4

        elif t.ggml_type in (TYPE_F16, TYPE_BF16):
            bytes_per_row = width * 2

        else:
            raise ValueError(
                f"Unsupported matrix type {t.ggml_type}"
            )

        start = (
            t.absolute_offset
            + first_row * bytes_per_row
        )

        end = start + row_count * bytes_per_row

        # Return a zero-copy view over the read-only mmap. The OS manages
        # mmap page residency; only the decoded tile is H3 host staging.
        return memoryview(self.mm)[start:end]

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None

        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()


# ============================================================
# REFERENCE Q2_K DECODER
# ============================================================

def decode_q2k_reference(raw):
    if len(raw) % Q2K_BLOCK_BYTES:
        raise ValueError("Invalid Q2_K byte count")

    count = len(raw) // Q2K_BLOCK_BYTES

    blocks = np.frombuffer(
        raw,
        dtype=np.uint8,
    ).reshape(count, Q2K_BLOCK_BYTES)

    scales = blocks[:, :16]
    quant = blocks[:, 16:80]

    half_values = np.frombuffer(
        raw,
        dtype="<f2",
    ).reshape(count, 42)

    d = half_values[:, 40].astype(np.float32)
    dmin = half_values[:, 41].astype(np.float32)

    result = np.empty(
        (count, 256),
        dtype=np.float32,
    )

    dst = 0
    scale_index = 0

    for chunk in range(2):
        qbase = chunk * 32

        for shift in (0, 2, 4, 6):
            sc = scales[:, scale_index]
            scale_index += 1

            dl = d * (sc & 15).astype(np.float32)
            ml = dmin * (sc >> 4).astype(np.float32)

            q = (
                quant[:, qbase:qbase + 16] >> shift
            ) & 3

            result[:, dst:dst + 16] = (
                dl[:, None] * q.astype(np.float32)
                - ml[:, None]
            )

            dst += 16

            sc = scales[:, scale_index]
            scale_index += 1

            dl = d * (sc & 15).astype(np.float32)
            ml = dmin * (sc >> 4).astype(np.float32)

            q = (
                quant[:, qbase + 16:qbase + 32] >> shift
            ) & 3

            result[:, dst:dst + 16] = (
                dl[:, None] * q.astype(np.float32)
                - ml[:, None]
            )

            dst += 16

    if dst != 256 or scale_index != 16:
        raise RuntimeError(
            "Q2_K reference decoder layout error"
        )

    return result.reshape(-1)


# ============================================================
# PARALLEL Q2_K NUMBA DECODER
# ============================================================

@njit(cache=True, parallel=True)
def _decode_q2k_numba_kernel(blocks, d, dmin):
    block_count = blocks.shape[0]

    output = np.empty(
        block_count * 256,
        dtype=np.float32,
    )

    for b in prange(block_count):

        for a in range(2):
            for g in range(4):
                shift = g * 2

                for h in range(2):
                    scale_index = (a * 4 + g) * 2 + h

                    scale_byte = blocks[b, scale_index]

                    scale_low = np.float32(
                        scale_byte & 15
                    )

                    scale_high = np.float32(
                        scale_byte >> 4
                    )

                    multiplier = np.float32(
                        d[b] * scale_low
                    )

                    minimum = np.float32(
                        dmin[b] * scale_high
                    )

                    quant_start = 16 + (a * 2 + h) * 16

                    output_start = (
                        (((b * 2 + a) * 4 + g) * 2 + h) * 16
                    )

                    for k in range(16):
                        packed = blocks[
                            b,
                            quant_start + k,
                        ]

                        q = np.float32(
                            (packed >> shift) & 3
                        )

                        value = np.float32(
                            multiplier * q
                        )

                        output[output_start + k] = np.float32(
                            value - minimum
                        )

    return output


def decode_q2k(raw):
    """
    Parallel Numba Q2_K decoder.

    Accepts bytes, bytearray, memoryview or NumPy arrays.
    Returns a flat float32 NumPy array.
    """

    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

    else:
        raw = np.asarray(
            raw,
            dtype=np.uint8,
        ).reshape(-1)

    if raw.size % Q2K_BLOCK_BYTES:
        raise ValueError(
            f"Invalid Q2_K data size: {raw.size} bytes; "
            "expected a multiple of 84."
        )

    block_count = raw.size // Q2K_BLOCK_BYTES

    blocks = np.ascontiguousarray(
        raw.reshape(
            block_count,
            Q2K_BLOCK_BYTES,
        )
    )

    d = np.frombuffer(
        blocks[:, 80:82].copy().tobytes(),
        dtype=np.float16,
    ).astype(np.float32)

    dmin = np.frombuffer(
        blocks[:, 82:84].copy().tobytes(),
        dtype=np.float16,
    ).astype(np.float32)

    return _decode_q2k_numba_kernel(
        blocks,
        d,
        dmin,
    )


# ============================================================
# CUDA Q2_K DECODER
# ============================================================

@torch.inference_mode()
def _decode_q2k_torch(raw, row_count, input_width, device):
    """
    Transfer the compact Q2_K bytes to the target device and decode there.

    The CUDA transformer path must not dequantize every streamed weight tile
    on the CPU. Only the compressed mmap slice is copied from host memory;
    the expanded FP32 tile is produced on the target device.
    """
    if input_width % Q2K_BLOCK_ELEMENTS:
        raise ValueError(f"Q2_K input width is not block-aligned: {input_width}")

    block_count = row_count * (input_width // Q2K_BLOCK_ELEMENTS)
    raw_array = np.frombuffer(raw, dtype=np.uint8)
    expected_bytes = block_count * Q2K_BLOCK_BYTES
    if raw_array.size != expected_bytes:
        raise ValueError(
            f"Q2_K tile byte count mismatch: {raw_array.size} != {expected_bytes}"
        )

    # Copy only compressed bytes out of the read-only mmap before transfer.
    packed = torch.from_numpy(raw_array.copy()).to(device=device)
    blocks = packed.view(block_count, Q2K_BLOCK_BYTES)

    scales = blocks[:, :16].to(torch.int32).view(block_count, 2, 4, 2)
    quant = blocks[:, 16:80].to(torch.int32).view(block_count, 2, 2, 16)

    # Each Q2_K block ends with two FP16 values: d and dmin.
    factors = blocks[:, 80:84].contiguous().view(torch.float16).to(torch.float32)
    d = factors[:, 0].view(block_count, 1, 1, 1)
    dmin = factors[:, 1].view(block_count, 1, 1, 1)

    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.int32, device=device)
    quant_values = (
        (quant.unsqueeze(2) >> shifts.view(1, 1, 4, 1, 1)) & 3
    ).to(torch.float32)

    scale_low = (scales & 15).to(torch.float32)
    scale_high = (scales >> 4).to(torch.float32)
    multipliers = d * scale_low
    minima = dmin * scale_high

    decoded = (
        multipliers.unsqueeze(-1) * quant_values
        - minima.unsqueeze(-1)
    )
    return decoded.reshape(row_count, input_width)


# ============================================================
# Q3_K DECODER (CPU STREAMING PATH)
# ============================================================

def decode_q3k_reference(raw):
    """Straightforward Q3_K reference decoder following ggml dequantize_row_q3_K."""
    if len(raw) % Q3K_BLOCK_BYTES:
        raise ValueError("Invalid Q3_K byte count")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q3K_BLOCK_BYTES)
    output = np.empty((blocks.shape[0], 256), dtype=np.float32)

    for b in range(blocks.shape[0]):
        block = blocks[b]
        d = np.frombuffer(block[108:110].tobytes(), dtype="<f2")[0].astype(np.float32)
        scale_bytes = block[96:108]
        scales = []
        for i in range(16):
            low = int(scale_bytes[i] & 15) if i < 8 else int(scale_bytes[i - 8] >> 4)
            high = (int(scale_bytes[8 + (i % 4)]) >> (2 * (i // 4))) & 3
            scales.append((low | (high << 4)) - 32)

        out_index = 0
        qbase = 0
        mask = 1
        scale_index = 0
        for _chunk in range(2):
            shift = 0
            for _group in range(4):
                for half in range(2):
                    multiplier = np.float32(d * np.float32(scales[scale_index]))
                    scale_index += 1
                    qstart = 32 + qbase + half * 16
                    hstart = half * 16
                    for lane in range(16):
                        low_quant = (int(block[qstart + lane]) >> shift) & 3
                        high_quant = 0 if (int(block[hstart + lane]) & mask) else 4
                        output[b, out_index] = multiplier * np.float32(low_quant - high_quant)
                        out_index += 1
                shift += 2
                mask <<= 1
            qbase += 32
        if out_index != 256 or scale_index != 16:
            raise RuntimeError("Q3_K reference decoder layout error")

    return output.reshape(-1)


@njit(cache=True, parallel=True)
def _decode_q3k_numba_kernel(blocks, d_values):
    output = np.empty(blocks.shape[0] * 256, dtype=np.float32)
    for b in prange(blocks.shape[0]):
        d = d_values[b]
        scales = np.empty(16, dtype=np.int32)
        for i in range(16):
            low = int(blocks[b, 96 + i] & 15) if i < 8 else int(blocks[b, 96 + i - 8] >> 4)
            high = (int(blocks[b, 104 + (i % 4)]) >> (2 * (i // 4))) & 3
            scales[i] = (low | (high << 4)) - 32

        out_index = 0
        qbase = 0
        mask = 1
        scale_index = 0
        for _chunk in range(2):
            shift = 0
            for _group in range(4):
                for half in range(2):
                    multiplier = np.float32(d * np.float32(scales[scale_index]))
                    scale_index += 1
                    qstart = 32 + qbase + half * 16
                    hstart = half * 16
                    for lane in range(16):
                        low_quant = (int(blocks[b, qstart + lane]) >> shift) & 3
                        high_quant = 0 if (int(blocks[b, hstart + lane]) & mask) else 4
                        output[b * 256 + out_index] = multiplier * np.float32(low_quant - high_quant)
                        out_index += 1
                shift += 2
                mask <<= 1
            qbase += 32
    return output


def decode_q3k(raw):
    """Decode GGML Q3_K blocks (256 values / 110 bytes) with the Numba CPU kernel."""
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = np.frombuffer(raw, dtype=np.uint8)
    else:
        raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if raw.size % Q3K_BLOCK_BYTES:
        raise ValueError(f"Invalid Q3_K data size: {raw.size} bytes; expected a multiple of 110.")
    block_count = raw.size // Q3K_BLOCK_BYTES
    blocks = np.ascontiguousarray(raw.reshape(block_count, Q3K_BLOCK_BYTES))
    d_values = np.frombuffer(
        blocks[:, 108:110].copy().tobytes(), dtype="<f2"
    ).astype(np.float32)
    return _decode_q3k_numba_kernel(blocks, d_values)


def test_q3k_decoder():
    rng = np.random.default_rng(1234)
    raw = rng.integers(0, 256, size=Q3K_BLOCK_BYTES * 8, dtype=np.uint8)
    expected = decode_q3k_reference(raw.tobytes())
    actual = decode_q3k(raw)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    print("Q3_K reference/Numba decoder parity: PASSED")


# ============================================================
# OTHER GGML WEIGHT DECODERS
# ============================================================

def decode_weight_tile(raw, ggml_type, row_count, input_width):

    if ggml_type == TYPE_Q2_K:
        return decode_q2k(raw).reshape(row_count, input_width)

    if ggml_type == TYPE_Q3_K:
        return decode_q3k(raw).reshape(row_count, input_width)

    if ggml_type == TYPE_F32:
        return np.frombuffer(
            raw,
            dtype="<f4",
        ).copy().reshape(
            row_count,
            input_width,
        )

    if ggml_type == TYPE_F16:
        return np.frombuffer(
            raw,
            dtype="<f2",
        ).astype(np.float32).reshape(
            row_count,
            input_width,
        )

    if ggml_type == TYPE_BF16:
        words = np.frombuffer(
            raw,
            dtype="<u2",
        ).astype(np.uint32)

        values = (words << 16).view("<f4")

        return values.reshape(
            row_count,
            input_width,
        )

    raise ValueError(
        f"Unsupported type {ggml_type}"
    )


# ============================================================
# STREAMED LINEAR
# ============================================================

class TiledGGUFLinear(torch.nn.Module):

    def __init__(
        self,
        gguf,
        tensor_name,
        device="cpu",
        row_tile=8,
        token_tile=16,
    ):

        super().__init__()

        if tensor_name not in gguf.tensors:
            raise KeyError(
                f"GGUF tensor not found: {tensor_name}"
            )

        if (
            str(device).startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        self.gguf = gguf
        self.name = tensor_name
        self.tensor = gguf.tensors[tensor_name]

        if len(self.tensor.dims) != 2:
            raise ValueError(
                "Linear weight must be a matrix"
            )

        self.input_width = self.tensor.input_width
        self.output_rows = self.tensor.output_rows
        self.ggml_type = self.tensor.ggml_type

        self.device = torch.device(device)

        self.requested_row_tile = int(row_tile)
        self.token_tile = int(token_tile)

        if self.requested_row_tile < 1:
            raise ValueError(
                "row_tile must be positive"
            )

        if self.token_tile < 1:
            raise ValueError(
                "token_tile must be positive"
            )

        if self.ggml_type == TYPE_Q2_K and self.input_width % Q2K_BLOCK_ELEMENTS:
            raise ValueError(f"Q2_K input width is not block-aligned: {self.name}")
        if self.ggml_type == TYPE_Q3_K and self.input_width % Q3K_BLOCK_ELEMENTS:
            raise ValueError(f"Q3_K input width is not block-aligned: {self.name}")

        self.row_tile = (
            max(
                self.requested_row_tile,
                CUDA_MIN_ROW_TILE,
            )
            if self.device.type == "cuda"
            else self.requested_row_tile
        )

    @torch.inference_mode()
    def forward(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(
                x,
                dtype=torch.float32,
            )

        if x.ndim != 2:
            raise ValueError(
                "Input must have shape [tokens, width]"
            )

        if x.shape[1] != self.input_width:
            raise ValueError(
                f"Input width {x.shape[1]} != "
                f"weight width {self.input_width}"
            )

        tokens = x.shape[0]

        if tokens == 0:
            return torch.empty(
                (0, self.output_rows),
                dtype=torch.float32,
                device=self.device,
            )

        activation = x.to(
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        output = torch.empty(
            (tokens, self.output_rows),
            dtype=torch.float32,
            device=self.device,
        )

        for row_start in range(
            0,
            self.output_rows,
            self.row_tile,
        ):

            rows = min(
                self.row_tile,
                self.output_rows - row_start,
            )

            raw = None
            weight_np = None
            weight = None
            raw_bytes = 0
            host_weight_bytes = 0
            vram_weight_bytes = 0

            try:
                raw = self.gguf.read_tensor_rows(
                    self.name,
                    row_start,
                    rows,
                )
                # raw is a memoryview into the mmap, not a copied host buffer.
                raw_bytes = 0

                if self.ggml_type == TYPE_Q2_K and self.device.type == "cuda":
                    # Decode the dominant quantized transformer weights on CUDA.
                    # CPU projections retain the measured Numba decoder path.
                    weight = _decode_q2k_torch(
                        raw,
                        rows,
                        self.input_width,
                        self.device,
                    )
                    vram_weight_bytes = weight.numel() * weight.element_size()
                    self.gguf.memory_stats.add_vram(vram_weight_bytes)

                    if not torch.isfinite(weight).all().item():
                        raise FloatingPointError(
                            f"Non-finite decoded CUDA weight tile in {self.name}"
                        )
                else:
                    weight_np = decode_weight_tile(
                        raw,
                        self.ggml_type,
                        rows,
                        self.input_width,
                    )
                    host_weight_bytes = int(weight_np.nbytes)
                    self.gguf.memory_stats.add_host(host_weight_bytes)

                    if not np.isfinite(weight_np).all():
                        raise FloatingPointError(
                            f"Non-finite weight values in {self.name}"
                        )

                    # CPU-only projections retain CPU decoded tiles. No
                    # per-tile pinned-memory allocation is used.
                    weight = torch.from_numpy(
                        np.ascontiguousarray(weight_np)
                    ).to(
                        device=self.device,
                        dtype=torch.float32,
                    )

                output[
                    :,
                    row_start:row_start + rows,
                ] = torch.matmul(
                    activation,
                    weight.T,
                )
            finally:
                # Release the current decoded tile before reading the next one.
                if weight is not None:
                    del weight
                if vram_weight_bytes:
                    self.gguf.memory_stats.remove_vram(vram_weight_bytes)
                if host_weight_bytes:
                    self.gguf.memory_stats.remove_host(host_weight_bytes)
                if weight_np is not None:
                    del weight_np
                if raw_bytes:
                    self.gguf.memory_stats.remove_host(raw_bytes)
                if raw is not None:
                    del raw

        return output


# ============================================================
# STANDALONE TEST
# ============================================================

DEFAULT_TENSORS = [
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
    "condition_proj.weight",
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gguf",
        default=DEFAULT_GGUF,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=27,
    )

    parser.add_argument(
        "--row-tile",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--token-tile",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--tensor",
        action="append",
        default=None,
    )

    args = parser.parse_args()

    print("=" * 60)
    print("H3 PARALLEL Q2_K STREAMED LINEAR")
    print("=" * 60)

    print(f"Checkpoint: {args.gguf}")
    print(f"Device:     {args.device}")
    print(f"Tokens:     {args.tokens}")
    print(f"Q2_K CPU threads: {Q2K_NUM_THREADS}")
    print(f"CUDA minimum row tile: {CUDA_MIN_ROW_TILE}")
    test_q3k_decoder()

    with GGUF(args.gguf) as gguf:

        passed = 0

        for name in args.tensor or DEFAULT_TENSORS:

            if name not in gguf.tensors:
                print(f"SKIP missing: {name}")
                continue

            t = gguf.tensors[name]

            if len(t.dims) != 2:
                print(f"SKIP non-matrix: {name}")
                continue

            rng = np.random.default_rng(12345)

            x = torch.from_numpy(
                rng.standard_normal(
                    (args.tokens, t.input_width),
                    dtype=np.float32,
                )
            )

            linear = TiledGGUFLinear(
                gguf,
                name,
                device=args.device,
                row_tile=args.row_tile,
                token_tile=args.token_tile,
            )

            if linear.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            y = linear(x)

            if linear.device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            if not torch.isfinite(y).all().item():
                raise FloatingPointError(name)

            print(
                f"{name}: PASSED | "
                f"shape={tuple(y.shape)} | "
                f"device={y.device} | "
                f"row_tile={linear.row_tile} | "
                f"time={elapsed:.4f}s"
            )

            passed += 1

            del x, y, linear

        print(f"\nTESTS PASSED: {passed}")


if __name__ == "__main__":
    main()
