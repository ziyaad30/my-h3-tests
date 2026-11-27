
#!/usr/bin/env python3
"""
Standalone MiniMax H3 GGUF Q2_K validation.

No imports from previous H3 project scripts.
No third-party GGUF reader.
No full tensor dequantization.
No full-model loading.

Validates:
  - GGUF v2/v3 header and metadata
  - Tensor directory and offsets
  - Q2_K block layout
  - Scalar Q2_K reference decoder
  - Independent vectorized Q2_K decoder
  - Selected real checkpoint matrix rows
  - Small matrix-vector projection parity

Q2_K block:
  scales: 16 bytes
  qs:     64 bytes
  d:       2 bytes (FP16)
  dmin:    2 bytes (FP16)
  total:  84 bytes
  values: 256
"""

import argparse
import mmap
import os
import struct
import sys
import traceback

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_GGUF = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models"
    r"\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

GGUF_MAGIC = b"GGUF"

GGML_TYPE_Q2_K = 10

Q2K_ELEMENTS = 256
Q2K_BLOCK_BYTES = 84

DEFAULT_TENSORS = [
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
    "condition_proj.weight",
]


# ============================================================
# GGUF METADATA TYPES
# ============================================================

GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12

METADATA_FORMATS = {
    GGUF_UINT8: "<B",
    GGUF_INT8: "<b",
    GGUF_UINT16: "<H",
    GGUF_INT16: "<h",
    GGUF_UINT32: "<I",
    GGUF_INT32: "<i",
    GGUF_FLOAT32: "<f",
    GGUF_BOOL: "<?",
    GGUF_UINT64: "<Q",
    GGUF_INT64: "<q",
    GGUF_FLOAT64: "<d",
}


# ============================================================
# GGUF READER
# ============================================================

class TensorInfo:
    def __init__(self, name, dimensions, tensor_type, offset):
        self.name = name
        self.dimensions = tuple(dimensions)
        self.tensor_type = tensor_type
        self.offset = offset
        self.absolute_offset = None

    def __repr__(self):
        return (
            f"TensorInfo(name={self.name!r}, "
            f"dimensions={self.dimensions}, "
            f"type={self.tensor_type}, "
            f"offset={self.offset})"
        )


class StandaloneGGUF:
    """
    Minimal GGUF reader.

    All returned tensor bytes are copied out of mmap.
    No exported NumPy views of the mmap are retained.
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.file = None
        self.mapping = None
        self.file_size = 0

        self.version = None
        self.tensor_count = 0
        self.metadata_count = 0
        self.alignment = 32
        self.data_offset = None

        self.metadata = {}
        self.tensors = {}

        self._open()
        self._parse()

    def _open(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self.file = open(self.path, "rb")
        self.file_size = os.path.getsize(self.path)

        if self.file_size == 0:
            raise ValueError("GGUF file is empty")

        self.mapping = mmap.mmap(
            self.file.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        )

    def _read(self, size):
        if size < 0:
            raise ValueError("Negative read size")

        data = self.file.read(size)

        if len(data) != size:
            raise EOFError(
                f"Unexpected end of GGUF file: "
                f"wanted {size} bytes, got {len(data)}"
            )

        return data

    def _unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(size))[0]

    def _read_string(self):
        length = self._unpack("<Q")

        if length > self.file_size:
            raise ValueError(
                f"Invalid GGUF string length: {length}"
            )

        raw = self._read(length)

        return raw.decode("utf-8", errors="strict")

    def _read_metadata_value(self, value_type):
        if value_type in METADATA_FORMATS:
            return self._unpack(METADATA_FORMATS[value_type])

        if value_type == GGUF_STRING:
            return self._read_string()

        if value_type == GGUF_ARRAY:
            element_type = self._unpack("<I")
            count = self._unpack("<Q")

            if count > 100_000_000:
                raise ValueError(
                    f"Unreasonable metadata array size: {count}"
                )

            # Preserve small arrays; skip large arrays safely.
            if count > 1_000_000:
                for _ in range(count):
                    self._read_metadata_value(element_type)
                return None

            return [
                self._read_metadata_value(element_type)
                for _ in range(count)
            ]

        raise ValueError(
            f"Unsupported GGUF metadata type: {value_type}"
        )

    @staticmethod
    def _align(value, alignment):
        return (
            (value + alignment - 1) // alignment
        ) * alignment

    def _parse(self):
        self.file.seek(0)

        magic = self._read(4)

        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Invalid GGUF magic: {magic!r}"
            )

        self.version = self._unpack("<I")

        if self.version not in (2, 3):
            raise ValueError(
                f"Unsupported GGUF version: {self.version}"
            )

        self.tensor_count = self._unpack("<Q")
        self.metadata_count = self._unpack("<Q")

        if self.tensor_count > 10_000_000:
            raise ValueError("Unreasonable tensor count")

        if self.metadata_count > 10_000_000:
            raise ValueError("Unreasonable metadata count")

        # Metadata
        for _ in range(self.metadata_count):
            key = self._read_string()
            value_type = self._unpack("<I")
            value = self._read_metadata_value(value_type)
            self.metadata[key] = value

        self.alignment = int(
            self.metadata.get("general.alignment", 32) or 32
        )

        if self.alignment <= 0:
            raise ValueError(
                f"Invalid GGUF alignment: {self.alignment}"
            )

        # Tensor descriptors
        for _ in range(self.tensor_count):
            name = self._read_string()

            n_dims = self._unpack("<I")

            if n_dims < 1 or n_dims > 4:
                raise ValueError(
                    f"Invalid dimension count for {name}: {n_dims}"
                )

            dimensions = [
                self._unpack("<Q")
                for _ in range(n_dims)
            ]

            tensor_type = self._unpack("<I")
            offset = self._unpack("<Q")

            if name in self.tensors:
                raise ValueError(
                    f"Duplicate tensor name: {name}"
                )

            self.tensors[name] = TensorInfo(
                name,
                dimensions,
                tensor_type,
                offset,
            )

        # Tensor data begins at the aligned position after descriptors.
        descriptor_end = self.file.tell()
        self.data_offset = self._align(
            descriptor_end,
            self.alignment,
        )

        if self.data_offset >= self.file_size:
            raise ValueError(
                "Tensor data offset is outside the file"
            )

        # Resolve absolute offsets and validate tensor ranges.
        for tensor in self.tensors.values():
            tensor.absolute_offset = (
                self.data_offset + tensor.offset
            )

            tensor_bytes = self.tensor_storage_bytes(tensor)

            end = tensor.absolute_offset + tensor_bytes

            if tensor.absolute_offset < self.data_offset:
                raise ValueError(
                    f"Invalid tensor offset: {tensor.name}"
                )

            if end > self.file_size:
                raise ValueError(
                    f"Tensor exceeds file bounds: {tensor.name}"
                )

    @staticmethod
    def tensor_elements(tensor):
        result = 1

        for dimension in tensor.dimensions:
            result *= dimension

        return result

    @staticmethod
    def tensor_storage_bytes(tensor):
        elements = StandaloneGGUF.tensor_elements(tensor)

        # Q2_K
        if tensor.tensor_type == GGML_TYPE_Q2_K:
            if elements % Q2K_ELEMENTS:
                raise ValueError(
                    f"Q2_K tensor element count is not "
                    f"divisible by 256: {tensor.name}"
                )

            return (
                elements // Q2K_ELEMENTS
            ) * Q2K_BLOCK_BYTES

        # Other common GGML tensor types.
        # These are included for directory validation only.
        if tensor.tensor_type == 0:   # F32
            return elements * 4

        if tensor.tensor_type == 1:   # F16
            return elements * 2

        if tensor.tensor_type == 30:  # BF16
            return elements * 2

        raise ValueError(
            f"Cannot calculate storage size for GGML "
            f"type {tensor.tensor_type}: {tensor.name}"
        )

    def get_tensor(self, name):
        tensor = self.tensors.get(name)

        if tensor is None:
            raise KeyError(
                f"Tensor not found: {name}"
            )

        size = self.tensor_storage_bytes(tensor)
        start = tensor.absolute_offset
        end = start + size

        # Important: bytes() creates an independent copy.
        # It does not retain an exported mmap pointer.
        return bytes(self.mapping[start:end])

    def get_tensor_rows(self, name, row_start, row_count):
        tensor = self.tensors.get(name)

        if tensor is None:
            raise KeyError(name)

        if tensor.tensor_type != GGML_TYPE_Q2_K:
            raise TypeError(
                f"{name} is not Q2_K"
            )

        if len(tensor.dimensions) != 2:
            raise ValueError(
                f"{name} is not a matrix: {tensor.dimensions}"
            )

        input_width = tensor.dimensions[0]
        output_rows = tensor.dimensions[1]

        if input_width % Q2K_ELEMENTS:
            raise ValueError(
                f"Input width {input_width} is not divisible "
                f"by {Q2K_ELEMENTS}: {name}"
            )

        if row_start < 0 or row_count <= 0:
            raise ValueError("Invalid row range")

        if row_start + row_count > output_rows:
            raise ValueError(
                f"Requested rows exceed output dimension: "
                f"{row_start}+{row_count}>{output_rows}"
            )

        row_bytes = (
            input_width // Q2K_ELEMENTS
        ) * Q2K_BLOCK_BYTES

        start = (
            tensor.absolute_offset
            + row_start * row_bytes
        )

        end = start + row_count * row_bytes

        # Independent copy. Safe after mmap closure.
        raw = bytes(self.mapping[start:end])

        expected = row_count * row_bytes

        if len(raw) != expected:
            raise IOError(
                f"Short tensor read for {name}: "
                f"{len(raw)} != {expected}"
            )

        return raw, input_width, output_rows

    def close(self):
        """
        No NumPy arrays or memoryviews reference the mmap.
        Close it before closing the underlying file.
        """
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None

        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()


# ============================================================
# Q2_K SCALAR REFERENCE DECODER
# ============================================================

def q2k_decode_scalar(raw):
    """
    Scalar implementation following GGML's Q2_K dequantization
    ordering.

    Input:
        bytes containing one or more complete 84-byte blocks.

    Output:
        float32 array containing 256 values per block.
    """

    if len(raw) % Q2K_BLOCK_BYTES:
        raise ValueError(
            "Q2_K byte count is not divisible by 84"
        )

    block_count = len(raw) // Q2K_BLOCK_BYTES

    output = np.empty(
        (block_count, Q2K_ELEMENTS),
        dtype=np.float32,
    )

    for block_index in range(block_count):
        base = block_index * Q2K_BLOCK_BYTES

        scales = raw[base:base + 16]
        quant = raw[base + 16:base + 80]

        d = np.frombuffer(
            raw,
            dtype="<f2",
            count=1,
            offset=base + 80,
        )[0].astype(np.float32)

        dmin = np.frombuffer(
            raw,
            dtype="<f2",
            count=1,
            offset=base + 82,
        )[0].astype(np.float32)

        destination = 0
        scale_index = 0

        for chunk in range(2):
            quant_base = chunk * 32

            for shift in (0, 2, 4, 6):
                # First 16-value group.
                scale_byte = scales[scale_index]
                scale_index += 1

                scale = np.float32(
                    d * np.float32(scale_byte & 0x0F)
                )

                minimum = np.float32(
                    dmin * np.float32(scale_byte >> 4)
                )

                for i in range(16):
                    q = (
                        quant[quant_base + i] >> shift
                    ) & 3

                    output[
                        block_index, destination
                    ] = np.float32(
                        scale * np.float32(q) - minimum
                    )

                    destination += 1

                # Second 16-value group.
                scale_byte = scales[scale_index]
                scale_index += 1

                scale = np.float32(
                    d * np.float32(scale_byte & 0x0F)
                )

                minimum = np.float32(
                    dmin * np.float32(scale_byte >> 4)
                )

                for i in range(16):
                    q = (
                        quant[quant_base + 16 + i] >> shift
                    ) & 3

                    output[
                        block_index, destination
                    ] = np.float32(
                        scale * np.float32(q) - minimum
                    )

                    destination += 1

        if destination != Q2K_ELEMENTS:
            raise RuntimeError(
                f"Scalar decoder produced {destination} values"
            )

        if scale_index != 16:
            raise RuntimeError(
                f"Scalar decoder consumed {scale_index} scales"
            )

    return output.reshape(-1)


# ============================================================
# INDEPENDENT VECTORIZED Q2_K DECODER
# ============================================================

def q2k_decode_vectorized(raw):
    """
    Separate vectorized implementation.

    It uses the same documented GGML block layout but does not
    call or reuse the scalar decoder.
    """

    if len(raw) % Q2K_BLOCK_BYTES:
        raise ValueError(
            "Q2_K byte count is not divisible by 84"
        )

    block_count = len(raw) // Q2K_BLOCK_BYTES

    if block_count == 0:
        return np.empty(0, dtype=np.float32)

    blocks = np.frombuffer(
        raw,
        dtype=np.uint8,
    ).reshape(block_count, Q2K_BLOCK_BYTES)

    # These are views of the independent bytes object, not mmap.
    scales = blocks[:, 0:16]
    quant = blocks[:, 16:80]

    d = np.frombuffer(
        raw,
        dtype="<f2",
    ).reshape(block_count, 42)[:, 40].astype(
        np.float32
    )

    dmin = np.frombuffer(
        raw,
        dtype="<f2",
    ).reshape(block_count, 42)[:, 41].astype(
        np.float32
    )

    output = np.empty(
        (block_count, Q2K_ELEMENTS),
        dtype=np.float32,
    )

    destination = 0
    scale_index = 0

    for chunk in range(2):
        quant_base = chunk * 32

        for shift in (0, 2, 4, 6):
            # First 16-value group.
            scale_byte = scales[:, scale_index]
            scale_index += 1

            scale = (
                d * (scale_byte & 0x0F).astype(np.float32)
            )

            minimum = (
                dmin * (scale_byte >> 4).astype(np.float32)
            )

            q = (
                quant[:, quant_base:quant_base + 16]
                >> shift
            ) & 3

            output[
                :, destination:destination + 16
            ] = (
                scale[:, None] * q.astype(np.float32)
                - minimum[:, None]
            )

            destination += 16

            # Second 16-value group.
            scale_byte = scales[:, scale_index]
            scale_index += 1

            scale = (
                d * (scale_byte & 0x0F).astype(np.float32)
            )

            minimum = (
                dmin * (scale_byte >> 4).astype(np.float32)
            )

            q = (
                quant[:, quant_base + 16:quant_base + 32]
                >> shift
            ) & 3

            output[
                :, destination:destination + 16
            ] = (
                scale[:, None] * q.astype(np.float32)
                - minimum[:, None]
            )

            destination += 16

    if destination != Q2K_ELEMENTS:
        raise RuntimeError(
            f"Vectorized decoder produced {destination} values"
        )

    if scale_index != 16:
        raise RuntimeError(
            f"Vectorized decoder consumed {scale_index} scales"
        )

    return output.reshape(-1)


# ============================================================
# SYNTHETIC DECODER TEST
# ============================================================

def synthetic_decoder_test():
    print("\n--- Synthetic Q2_K decoder test ---")

    rng = np.random.default_rng(123456)

    block_count = 7

    raw = bytearray(
        rng.integers(
            0,
            256,
            size=block_count * Q2K_BLOCK_BYTES,
            dtype=np.uint8,
        ).tobytes()
    )

    # Ensure finite, ordinary FP16 scale values.
    for i in range(block_count):
        base = i * Q2K_BLOCK_BYTES

        raw[base + 80:base + 82] = struct.pack(
            "<e", 0.125 + i * 0.01
        )

        raw[base + 82:base + 84] = struct.pack(
            "<e", 0.0625 + i * 0.005
        )

    raw = bytes(raw)

    scalar = q2k_decode_scalar(raw)
    vectorized = q2k_decode_vectorized(raw)

    if not np.all(np.isfinite(scalar)):
        raise AssertionError(
            "Scalar decoder generated non-finite values"
        )

    if not np.all(np.isfinite(vectorized)):
        raise AssertionError(
            "Vectorized decoder generated non-finite values"
        )

    difference = np.abs(scalar - vectorized)

    maximum_error = float(difference.max())
    mean_error = float(difference.mean())

    print(f"Blocks:       {block_count}")
    print(f"Decoded:      {scalar.size} values")
    print(f"Maximum err:  {maximum_error:.9g}")
    print(f"Mean err:     {mean_error:.9g}")

    np.testing.assert_allclose(
        scalar,
        vectorized,
        rtol=1e-6,
        atol=1e-7,
    )

    print("Synthetic decoder parity: PASSED")


# ============================================================
# REAL CHECKPOINT MATRIX VALIDATION
# ============================================================

def validate_matrix(gguf, tensor_name, rows_to_check):
    tensor = gguf.tensors.get(tensor_name)

    if tensor is None:
        print(f"\nSKIP: {tensor_name} (not present)")
        return None

    if tensor.tensor_type != GGML_TYPE_Q2_K:
        print(
            f"\nSKIP: {tensor_name} "
            f"(GGML type {tensor.tensor_type}, not Q2_K)"
        )
        return None

    if len(tensor.dimensions) != 2:
        print(
            f"\nSKIP: {tensor_name} "
            f"(not a two-dimensional matrix)"
        )
        return None

    input_width = tensor.dimensions[0]
    output_rows = tensor.dimensions[1]

    if input_width % Q2K_ELEMENTS:
        raise ValueError(
            f"{tensor_name}: input width {input_width} "
            f"is not divisible by 256"
        )

    count = min(rows_to_check, output_rows)

    print(f"\n--- {tensor_name} ---")
    print(f"Input width:  {input_width}")
    print(f"Output rows:  {output_rows}")
    print(f"Rows tested:  {count}")

    raw, actual_width, actual_rows = gguf.get_tensor_rows(
        tensor_name,
        0,
        count,
    )

    if actual_width != input_width:
        raise AssertionError("Input width changed during read")

    if actual_rows != output_rows:
        raise AssertionError("Output row count changed during read")

    blocks_per_row = input_width // Q2K_ELEMENTS
    row_bytes = blocks_per_row * Q2K_BLOCK_BYTES

    if len(raw) != count * row_bytes:
        raise AssertionError(
            "Incorrect number of bytes read for matrix rows"
        )

    # Decode the same real checkpoint rows two independent ways.
    scalar = q2k_decode_scalar(raw)
    vectorized = q2k_decode_vectorized(raw)

    expected_values = count * input_width

    if scalar.size != expected_values:
        raise AssertionError(
            f"Scalar output size {scalar.size}, "
            f"expected {expected_values}"
        )

    if vectorized.size != expected_values:
        raise AssertionError(
            f"Vectorized output size {vectorized.size}, "
            f"expected {expected_values}"
        )

    scalar_matrix = scalar.reshape(count, input_width)
    vector_matrix = vectorized.reshape(count, input_width)

    difference = np.abs(
        scalar_matrix - vector_matrix
    )

    maximum_error = float(difference.max())
    mean_error = float(difference.mean())

    if not np.all(np.isfinite(scalar_matrix)):
        raise AssertionError(
            "Scalar decoder generated non-finite values"
        )

    if not np.all(np.isfinite(vector_matrix)):
        raise AssertionError(
            "Vectorized decoder generated non-finite values"
        )

    print(f"Decoded values: {expected_values:,}")
    print(f"Maximum error:  {maximum_error:.9g}")
    print(f"Mean error:     {mean_error:.9g}")
    print(
        f"Value range:    "
        f"{float(scalar_matrix.min()):.6f} "
        f"to {float(scalar_matrix.max()):.6f}"
    )

    np.testing.assert_allclose(
        scalar_matrix,
        vector_matrix,
        rtol=1e-6,
        atol=1e-7,
    )

    # Test a small real matrix-vector projection.
    # This is intentionally small and does not materialize
    # the full model matrix.
    rng = np.random.default_rng(20261127)

    x = rng.standard_normal(
        input_width,
        dtype=np.float32,
    )

    y_scalar = scalar_matrix @ x
    y_vectorized = vector_matrix @ x

    projection_error = np.abs(
        y_scalar - y_vectorized
    )

    projection_max = float(projection_error.max())

    np.testing.assert_allclose(
        y_scalar,
        y_vectorized,
        rtol=1e-5,
        atol=1e-5,
    )

    print(f"Projection max error: {projection_max:.9g}")
    print("Matrix validation: PASSED")

    return {
        "name": tensor_name,
        "rows": count,
        "input_width": input_width,
        "max_decode_error": maximum_error,
        "projection_error": projection_max,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone H3 GGUF Q2_K validator"
    )

    parser.add_argument(
        "--gguf",
        default=DEFAULT_GGUF,
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of output rows to decode per matrix",
    )

    parser.add_argument(
        "--tensor",
        action="append",
        default=None,
        help="Tensor name to validate; can be specified multiple times",
    )

    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be at least 1")

    print("=" * 60)
    print("STANDALONE H3 REAL CHECKPOINT Q2_K VALIDATION")
    print("=" * 60)

    print(f"Checkpoint: {args.gguf}")
    print(f"Rows:       {args.rows}")
    print("Device:     CPU")
    print("Imports:    NumPy and Python standard library only")

    synthetic_decoder_test()

    results = []

    try:
        with StandaloneGGUF(args.gguf) as gguf:
            print("\n--- GGUF ---")
            print(f"Version:       {gguf.version}")
            print(f"File size:     {gguf.file_size:,} bytes")
            print(f"Tensor count:  {gguf.tensor_count}")
            print(f"Metadata:      {gguf.metadata_count}")
            print(f"Alignment:     {gguf.alignment}")
            print(f"Data offset:   {gguf.data_offset:,}")

            type_counts = {}

            for tensor in gguf.tensors.values():
                type_counts[tensor.tensor_type] = (
                    type_counts.get(tensor.tensor_type, 0) + 1
                )

            print("\nTensor storage types:")

            for tensor_type, count in sorted(type_counts.items()):
                print(
                    f"  GGML type {tensor_type:<3} "
                    f"{count:>5} tensors"
                )

            names = args.tensor or DEFAULT_TENSORS

            for name in names:
                result = validate_matrix(
                    gguf,
                    name,
                    args.rows,
                )

                if result is not None:
                    results.append(result)

            print("\n--- Checkpoint validation summary ---")
            print(f"Matrices validated: {len(results)}")

            if not results:
                raise RuntimeError(
                    "No requested Q2_K matrices were validated"
                )

            print("\nGGUF tensor directory and selected matrix reads: OK")

    except Exception:
        print("\nVALIDATION FAILED")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("H3 Q2_K VALIDATION PASSED")
    print("=" * 60)
    print(
        "Scalar and vectorized decoding agree for the "
        "selected real checkpoint rows."
    )
    print(
        "This validates decoder parity, not yet parity "
        "against an independently executed llama.cpp kernel."
    )


if __name__ == "__main__":
    main()
