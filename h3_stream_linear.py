
#!/usr/bin/env python3
"""
Standalone MiniMax H3 tiled GGUF linear engine.

Fresh implementation. No imports from previous H3 scripts.

Supported GGML types:
    0  F32
    1  F16
    10 Q2_K
    30 BF16

Matrix dimensions in GGUF:
    (input_width, output_rows)

Weight computation:
    Y = X @ W.T

Only one output-row weight tile is decoded at a time.
"""

import argparse
import mmap
import os
import struct
import sys

import numpy as np
import torch


DEFAULT_GGUF = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models"
    r"\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q2_K = 10
TYPE_BF16 = 30

Q2K_BLOCK_ELEMENTS = 256
Q2K_BLOCK_BYTES = 84

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
# GGUF READER
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
        for d in self.dims:
            result *= d
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
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

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

            # Parse arrays without retaining large metadata arrays.
            if count > 1_000_000:
                for _ in range(count):
                    self.metadata_value(element_type)
                return None

            return [
                self.metadata_value(element_type)
                for _ in range(count)
            ]

        raise ValueError(f"Unsupported metadata type {kind}")

    @staticmethod
    def align(value, alignment):
        return ((value + alignment - 1) // alignment) * alignment

    @staticmethod
    def storage_size(t):
        n = t.elements

        if t.ggml_type == TYPE_F32:
            return n * 4

        if t.ggml_type in (TYPE_F16, TYPE_BF16):
            return n * 2

        if t.ggml_type == TYPE_Q2_K:
            if n % Q2K_BLOCK_ELEMENTS:
                raise ValueError(
                    f"Q2_K element count not divisible by 256: {t.name}"
                )
            return (
                n // Q2K_BLOCK_ELEMENTS
            ) * Q2K_BLOCK_BYTES

        raise ValueError(
            f"Unsupported GGML type {t.ggml_type}: {t.name}"
        )

    def parse(self):
        self.file.seek(0)

        if self.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")

        self.version = self.unpack("<I")

        if self.version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {self.version}")

        self.tensor_count = self.unpack("<Q")
        self.metadata_count = self.unpack("<Q")

        metadata = {}

        for _ in range(self.metadata_count):
            key = self.string()
            kind = self.unpack("<I")
            metadata[key] = self.metadata_value(kind)

        self.alignment = int(metadata.get("general.alignment", 32) or 32)

        if self.alignment <= 0:
            raise ValueError("Invalid alignment")

        for _ in range(self.tensor_count):
            name = self.string()
            ndims = self.unpack("<I")

            if ndims < 1 or ndims > 4:
                raise ValueError(f"Invalid dimensions: {name}")

            dims = tuple(
                self.unpack("<Q") for _ in range(ndims)
            )

            ggml_type = self.unpack("<I")
            offset = self.unpack("<Q")

            if name in self.tensors:
                raise ValueError(f"Duplicate tensor {name}")

            self.tensors[name] = Tensor(
                name, dims, ggml_type, offset
            )

        self.data_offset = self.align(
            self.file.tell(),
            self.alignment,
        )

        for t in self.tensors.values():
            t.absolute_offset = self.data_offset + t.offset

            end = t.absolute_offset + self.storage_size(t)

            if t.absolute_offset < self.data_offset or end > self.size:
                raise ValueError(f"Tensor outside file: {t.name}")

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
                raise ValueError(
                    f"Invalid Q2_K input width: {name}"
                )
            bytes_per_row = (
                width // Q2K_BLOCK_ELEMENTS
            ) * Q2K_BLOCK_BYTES

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

        # Copy bytes, never return a view into mmap.
        return bytes(self.mm[start:end])

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None

        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ============================================================
# Q2_K DECODER
# ============================================================

def decode_q2k(raw):
    if len(raw) % Q2K_BLOCK_BYTES:
        raise ValueError("Invalid Q2_K byte count")

    count = len(raw) // Q2K_BLOCK_BYTES

    blocks = np.frombuffer(
        raw, dtype=np.uint8
    ).reshape(count, Q2K_BLOCK_BYTES)

    scales = blocks[:, :16]
    quant = blocks[:, 16:80]

    # FP16 values are stored at byte offsets 80 and 82.
    d = np.frombuffer(
        raw, dtype="<f2"
    ).reshape(count, 42)[:, 40].astype(np.float32)

    dmin = np.frombuffer(
        raw, dtype="<f2"
    ).reshape(count, 42)[:, 41].astype(np.float32)

    result = np.empty((count, 256), dtype=np.float32)

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
        raise RuntimeError("Q2_K decoder layout error")

    return result.reshape(-1)


# ============================================================
# MIXED-PRECISION TILE DECODER
# ============================================================

def decode_weight_tile(raw, ggml_type, row_count, input_width):
    if ggml_type == TYPE_Q2_K:
        values = decode_q2k(raw)
        return values.reshape(row_count, input_width)

    if ggml_type == TYPE_F32:
        return np.frombuffer(
            raw, dtype="<f4"
        ).copy().reshape(row_count, input_width)

    if ggml_type == TYPE_F16:
        return np.frombuffer(
            raw, dtype="<f2"
        ).astype(np.float32).reshape(row_count, input_width)

    if ggml_type == TYPE_BF16:
        words = np.frombuffer(
            raw, dtype="<u2"
        ).astype(np.uint32)

        # BF16 occupies the upper 16 bits of IEEE FP32.
        values = (words << 16).view("<f4")

        return values.reshape(row_count, input_width)

    raise ValueError(f"Unsupported type {ggml_type}")


# ============================================================
# STREAMED LINEAR
# ============================================================

class TiledGGUFLinear:
    def __init__(
        self,
        gguf,
        tensor_name,
        device="cpu",
        row_tile=8,
        token_tile=16,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

        self.gguf = gguf
        self.name = tensor_name
        self.tensor = gguf.tensors[tensor_name]

        if len(self.tensor.dims) != 2:
            raise ValueError("Linear weight must be a matrix")

        self.input_width = self.tensor.input_width
        self.output_rows = self.tensor.output_rows
        self.ggml_type = self.tensor.ggml_type

        self.device = torch.device(device)
        self.row_tile = row_tile
        self.token_tile = token_tile

        if row_tile < 1 or token_tile < 1:
            raise ValueError("Tile sizes must be positive")

    @torch.inference_mode()
    def __call__(self, x):
        """
        x: [tokens, input_width], CPU or CUDA tensor.

        Returns:
            CPU FP32 tensor [tokens, output_rows].

        Only the current weight tile is transferred to CUDA.
        The output is accumulated on CPU.
        """

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        if x.ndim != 2:
            raise ValueError("Input must have shape [tokens, width]")

        if x.shape[1] != self.input_width:
            raise ValueError(
                f"Input width {x.shape[1]} != "
                f"weight width {self.input_width}"
            )

        tokens = x.shape[0]

        # CPU output prevents full output allocation on the GPU.
        output = torch.empty(
            (tokens, self.output_rows),
            dtype=torch.float32,
            device="cpu",
        )

        for row_start in range(
            0, self.output_rows, self.row_tile
        ):
            rows = min(
                self.row_tile,
                self.output_rows - row_start,
            )

            raw = self.gguf.read_tensor_rows(
                self.name,
                row_start,
                rows,
            )

            weight_np = decode_weight_tile(
                raw,
                self.ggml_type,
                rows,
                self.input_width,
            )

            if not np.all(np.isfinite(weight_np)):
                raise FloatingPointError(
                    f"Non-finite weight values in {self.name}"
                )

            # The only weight tensor on the accelerator is this tile.
            weight = torch.from_numpy(
                np.ascontiguousarray(weight_np)
            ).to(self.device)

            for token_start in range(
                0, tokens, self.token_tile
            ):
                token_end = min(
                    token_start + self.token_tile,
                    tokens,
                )

                activation = x[
                    token_start:token_end
                ].to(
                    device=self.device,
                    dtype=torch.float32,
                )

                result = torch.matmul(
                    activation,
                    weight.T,
                )

                output[
                    token_start:token_end,
                    row_start:row_start + rows,
                ] = result.to("cpu")

                del activation
                del result

            del weight
            del weight_np
            del raw

        return output


# ============================================================
# TEST INPUT
# ============================================================

def make_test_input(width, tokens, seed):
    rng = np.random.default_rng(seed)

    data = rng.standard_normal(
        (tokens, width),
        dtype=np.float32,
    )

    return torch.from_numpy(data)


# ============================================================
# MAIN
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

    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--row-tile", type=int, default=8)
    parser.add_argument("--token-tile", type=int, default=2)
    parser.add_argument("--tensor", action="append", default=None)

    args = parser.parse_args()

    if args.rows < 1 or args.tokens < 1:
        parser.error("rows and tokens must be positive")

    print("=" * 60)
    print("STANDALONE H3 STREAMED MIXED-PRECISION LINEAR")
    print("=" * 60)
    print(f"Checkpoint:  {args.gguf}")
    print(f"Device:      {args.device}")
    print(f"Row tile:    {args.row_tile}")
    print(f"Token tile:  {args.token_tile}")
    print(f"Test tokens: {args.tokens}")

    with GGUF(args.gguf) as gguf:
        print(f"\nGGUF version: {gguf.version}")
        print(f"Tensor count: {gguf.tensor_count}")

        names = args.tensor or DEFAULT_TENSORS
        passed = 0

        for name in names:
            if name not in gguf.tensors:
                print(f"\nSKIP missing: {name}")
                continue

            t = gguf.tensors[name]

            if len(t.dims) != 2:
                print(f"\nSKIP non-matrix: {name}")
                continue

            print(f"\n--- {name} ---")
            print(f"GGML type:   {t.ggml_type}")
            print(f"Dimensions:  {t.dims}")
            print(f"Input:       {t.input_width}")
            print(f"Output:      {t.output_rows}")

            # Run a small projection through every output row.
            x = make_test_input(
                t.input_width,
                args.tokens,
                seed=12345,
            )

            linear = TiledGGUFLinear(
                gguf,
                name,
                device=args.device,
                row_tile=args.row_tile,
                token_tile=args.token_tile,
            )

            y = linear(x)

            if tuple(y.shape) != (
                args.tokens,
                t.output_rows,
            ):
                raise RuntimeError(
                    f"Incorrect output shape: {tuple(y.shape)}"
                )

            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    f"Non-finite output from {name}"
                )

            print(f"Output shape: {tuple(y.shape)}")
            print(f"Output min:   {y.min().item():.7f}")
            print(f"Output max:   {y.max().item():.7f}")
            print(f"Output mean:  {y.mean().item():.7f}")
            print(f"Output norm:  {y.norm().item():.7f}")
            print("STREAMED PROJECTION PASSED")

            passed += 1

            del linear, x, y

        print("\n" + "=" * 60)
        print(f"STREAMED LINEAR TESTS PASSED: {passed}")
        print("=" * 60)

        if passed == 0:
            raise RuntimeError("No matrices were tested")


if __name__ == "__main__":
    main()
