
import argparse
import math
import mmap
import os
import struct
import time

import numpy as np
import torch


GGUF_PATH = (
    r"C:\Users\User\PycharmProjects\pythonProject\models"
    r"\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

DEFAULT_CHUNK_ROWS = 32
TEST_SEQUENCE_LENGTH = 16

GGUF_F32 = 0
GGUF_F16 = 1
GGUF_Q2_K = 10
GGUF_BF16 = 30

Q2_K_BLOCK_BYTES = 84
Q2_K_BLOCK_ELEMENTS = 256


def _read_exact(f, count):
    data = f.read(count)
    if len(data) != count:
        raise EOFError("Unexpected end of GGUF file")
    return data


def _read_u32(f):
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f):
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f):
    length = _read_u64(f)
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _skip_metadata_value(f, kind):
    sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    if kind == 8:
        f.seek(_read_u64(f), 1)
    elif kind == 9:
        element_type = _read_u32(f)
        count = _read_u64(f)

        for _ in range(count):
            _skip_metadata_value(f, element_type)
    elif kind in sizes:
        f.seek(sizes[kind], 1)
    else:
        raise ValueError(
            f"Unsupported GGUF metadata type: {kind}"
        )


def _read_metadata_value(f, kind):
    formats = {
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

    if kind == 8:
        return _read_string(f)

    if kind == 9:
        element_type = _read_u32(f)
        count = _read_u64(f)

        for _ in range(count):
            _skip_metadata_value(f, element_type)

        return None

    if kind not in formats:
        raise ValueError(
            f"Unsupported GGUF metadata type: {kind}"
        )

    fmt = formats[kind]
    return struct.unpack(
        fmt,
        _read_exact(f, struct.calcsize(fmt))
    )[0]


def _align_up(value, alignment):
    return (
        (value + alignment - 1) // alignment
    ) * alignment


def _tensor_storage_bytes(shape, tensor_type):
    elements = math.prod(shape)

    if tensor_type == GGUF_F32:
        return elements * 4

    if tensor_type in (GGUF_F16, GGUF_BF16):
        return elements * 2

    if tensor_type == GGUF_Q2_K:
        if elements % Q2_K_BLOCK_ELEMENTS:
            raise ValueError(
                f"Q2_K tensor has invalid element count: {shape}"
            )

        return (
            elements // Q2_K_BLOCK_ELEMENTS
        ) * Q2_K_BLOCK_BYTES

    raise NotImplementedError(
        f"Unsupported tensor type ID: {tensor_type}"
    )


class GGUFFile:
    """
    Read-only memory-mapped GGUF reader.

    Supports the tensor types present in the inspected H3
    checkpoint:
        F32
        F16
        Q2_K
        BF16
    """

    def __init__(self, path):
        self.path = os.path.abspath(path)

        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)

        self.file = open(self.path, "rb")
        self.mm = mmap.mmap(
            self.file.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )

        self.file_size = len(self.mm)
        self.tensors = {}
        self.version = None
        self.data_start = None
        self.alignment = 32

        try:
            self._parse()
        except Exception:
            self.close()
            raise

    def _parse(self):
        f = self.file

        if _read_exact(f, 4) != b"GGUF":
            raise ValueError("Invalid GGUF magic")

        self.version = _read_u32(f)
        tensor_count = _read_u64(f)
        metadata_count = _read_u64(f)

        for _ in range(metadata_count):
            key = _read_string(f)
            kind = _read_u32(f)
            value = _read_metadata_value(f, kind)

            if key == "general.alignment":
                self.alignment = int(value)

        if self.alignment <= 0:
            raise ValueError("Invalid GGUF alignment")

        for _ in range(tensor_count):
            name = _read_string(f)
            dimensions = _read_u32(f)

            if dimensions < 1 or dimensions > 4:
                raise ValueError(
                    f"Invalid dimensions for {name}: {dimensions}"
                )

            shape = tuple(
                _read_u64(f)
                for _ in range(dimensions)
            )

            tensor_type = _read_u32(f)
            offset = _read_u64(f)

            storage_bytes = _tensor_storage_bytes(
                shape,
                tensor_type,
            )

            self.tensors[name] = {
                "shape": shape,
                "type": tensor_type,
                "offset": offset,
                "bytes": storage_bytes,
            }

        self.data_start = _align_up(
            f.tell(),
            self.alignment,
        )

        for name, info in self.tensors.items():
            start = self.data_start + info["offset"]
            end = start + info["bytes"]

            if start < self.data_start or end > self.file_size:
                raise ValueError(
                    f"Tensor outside GGUF file: {name}"
                )

    def close(self):
        if getattr(self, "mm", None) is not None:
            self.mm.close()
            self.mm = None

        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def tensor_info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"GGUF tensor not found: {name}")

    def tensor_location(self, name):
        info = self.tensor_info(name)
        start = self.data_start + info["offset"]
        return start, start + info["bytes"]

    def _decode_plain(self, raw, tensor_type):
        if tensor_type == GGUF_F32:
            return np.frombuffer(
                raw,
                dtype="<f4",
            ).astype(np.float32, copy=True)

        if tensor_type == GGUF_F16:
            return np.frombuffer(
                raw,
                dtype="<f2",
            ).astype(np.float32)

        if tensor_type == GGUF_BF16:
            values = np.frombuffer(
                raw,
                dtype="<u2",
            )

            bits = (
                values.astype(np.uint32) << 16
            )

            return bits.view(np.float32).copy()

        raise NotImplementedError(
            f"Not a plain tensor type: {tensor_type}"
        )

    def read_tensor(self, name):
        """
        Read a complete tensor.

        Intended for small vectors, norms, biases and tables.
        Do not use this for large Q2_K matrices.
        """
        info = self.tensor_info(name)
        start, end = self.tensor_location(name)

        raw = memoryview(self.mm)[start:end]

        try:
            if info["type"] == GGUF_Q2_K:
                result = dequantize_q2_k(raw)
                return result.reshape(info["shape"])

            result = self._decode_plain(
                raw,
                info["type"],
            )

            return result.reshape(info["shape"])

        finally:
            raw.release()

    def read_matrix_rows(self, name, row_start, row_end):
        """
        Read and decode a bounded range of output rows.

        GGUF matrix dimensions store the input dimension first.
        """
        info = self.tensor_info(name)

        if len(info["shape"]) < 2:
            raise ValueError(
                f"Not a matrix: {name}"
            )

        input_width = info["shape"][0]
        output_rows = math.prod(info["shape"][1:])

        if not (0 <= row_start < row_end <= output_rows):
            raise ValueError(
                f"Invalid row range: {row_start}:{row_end}"
            )

        tensor_type = info["type"]

        if tensor_type == GGUF_Q2_K:
            blocks_per_row = (
                input_width // Q2_K_BLOCK_ELEMENTS
            )

            if input_width % Q2_K_BLOCK_ELEMENTS:
                raise ValueError(
                    f"Q2_K row width is not block-aligned: {name}"
                )

            first_block = row_start * blocks_per_row
            block_count = (
                row_end - row_start
            ) * blocks_per_row

            byte_start = (
                self.data_start
                + info["offset"]
                + first_block * Q2_K_BLOCK_BYTES
            )

            byte_end = (
                byte_start
                + block_count * Q2_K_BLOCK_BYTES
            )

        else:
            bytes_per_element = {
                GGUF_F32: 4,
                GGUF_F16: 2,
                GGUF_BF16: 2,
            }.get(tensor_type)

            if bytes_per_element is None:
                raise NotImplementedError(
                    f"Unsupported matrix type: {tensor_type}"
                )

            row_bytes = input_width * bytes_per_element

            byte_start = (
                self.data_start
                + info["offset"]
                + row_start * row_bytes
            )

            byte_end = (
                byte_start
                + (row_end - row_start) * row_bytes
            )

        raw = memoryview(self.mm)[byte_start:byte_end]

        try:
            if tensor_type == GGUF_Q2_K:
                return dequantize_q2_k(raw).reshape(
                    row_end - row_start,
                    input_width,
                )

            return self._decode_plain(
                raw,
                tensor_type,
            ).reshape(
                row_end - row_start,
                input_width,
            )

        finally:
            raw.release()


def dequantize_q2_k(raw):
    """
    GGML Q2_K decoder.

    Each 84-byte block represents 256 values:
        16 scale/min bytes
        64 quantization bytes
        FP16 d
        FP16 dmin

    The output is FP32.
    """
    blocks = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    if blocks.size % Q2_K_BLOCK_BYTES:
        raise ValueError("Invalid Q2_K byte count")

    blocks = blocks.reshape(-1, Q2_K_BLOCK_BYTES)
    count = blocks.shape[0]

    scales = blocks[:, :16]
    quant = blocks[:, 16:80]

    d = (
        blocks[:, 80:82]
        .copy()
        .view("<f2")
        .reshape(count)
        .astype(np.float32)
    )

    dmin = (
        blocks[:, 82:84]
        .copy()
        .view("<f2")
        .reshape(count)
        .astype(np.float32)
    )

    output = np.empty(
        (count, Q2_K_BLOCK_ELEMENTS),
        dtype=np.float32,
    )

    for half in range(2):
        qbase = half * 32

        for group in range(4):
            shift = group * 2
            scale_index = half * 8 + group * 2
            output_index = half * 128 + group * 32

            scale = scales[:, scale_index]
            q = (
                quant[:, qbase:qbase + 16] >> shift
            ) & 3

            output[:, output_index:output_index + 16] = (
                d[:, None]
                * (scale & 15)[:, None]
                * q
                - dmin[:, None]
                * (scale >> 4)[:, None]
            )

            scale = scales[:, scale_index + 1]
            q = (
                quant[:, qbase + 16:qbase + 32] >> shift
            ) & 3

            output[
                :,
                output_index + 16:output_index + 32
            ] = (
                d[:, None]
                * (scale & 15)[:, None]
                * q
                - dmin[:, None]
                * (scale >> 4)[:, None]
            )

    return output


class StreamedLinear:
    """
    Linear layer backed directly by a memory-mapped GGUF tensor.

    Weight rows are decoded in bounded chunks. The full weight
    matrix is never materialized in RAM or VRAM.

    Input:
        [sequence_length, input_width]

    Output:
        [sequence_length, output_width]
    """

    def __init__(
        self,
        gguf,
        device="cuda",
        chunk_rows=DEFAULT_CHUNK_ROWS,
    ):
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("CUDA is not available")

        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")

        self.gguf = gguf
        self.device = torch.device(device)
        self.chunk_rows = int(chunk_rows)

    @torch.inference_mode()
    def __call__(
        self,
        weight_name,
        x,
        bias_name=None,
        output_device=None,
    ):
        info = self.gguf.tensor_info(weight_name)

        if len(info["shape"]) < 2:
            raise ValueError(
                f"Weight is not a matrix: {weight_name}"
            )

        input_width = info["shape"][0]
        output_width = math.prod(info["shape"][1:])

        if x.ndim != 2:
            raise ValueError(
                "Input must have shape [sequence, features]"
            )

        if x.shape[1] != input_width:
            raise ValueError(
                f"{weight_name}: expected input width "
                f"{input_width}, got {x.shape[1]}"
            )

        sequence_length = x.shape[0]

        if output_device is None:
            output_device = x.device

        output_device = torch.device(output_device)

        # The input is transferred once, not once per weight chunk.
        x_gpu = x.to(
            device=self.device,
            dtype=torch.float32,
        )

        x_transposed = x_gpu.transpose(0, 1).contiguous()

        # Output placement is caller-controlled. CPU output can
        # reduce VRAM usage for large intermediate projections.
        output = torch.empty(
            (sequence_length, output_width),
            dtype=torch.float32,
            device=output_device,
        )

        for row_start in range(
            0,
            output_width,
            self.chunk_rows,
        ):
            row_end = min(
                row_start + self.chunk_rows,
                output_width,
            )

            weight_np = self.gguf.read_matrix_rows(
                weight_name,
                row_start,
                row_end,
            )

            weight_gpu = torch.from_numpy(
                weight_np
            ).to(
                device=self.device,
                dtype=torch.float32,
            )

            # [chunk_rows, input_width] @ [input_width, sequence]
            # -> [chunk_rows, sequence]
            result = torch.mm(
                weight_gpu,
                x_transposed,
            )

            # Store as [sequence, chunk_rows].
            output[:, row_start:row_end].copy_(
                result.transpose(0, 1)
            )

            del weight_gpu
            del weight_np
            del result

        if bias_name is not None:
            bias_np = self.gguf.read_tensor(bias_name)
            bias_gpu = torch.from_numpy(
                bias_np.reshape(-1)
            ).to(
                device=output_device,
                dtype=torch.float32,
            )

            if bias_gpu.numel() != output_width:
                raise ValueError(
                    f"Bias width mismatch for {weight_name}"
                )

            output.add_(bias_gpu.unsqueeze(0))

            del bias_gpu
            del bias_np

        return output


def run_self_test():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    targets = [
        "blocks.0.attn.qkv_proj.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
    ]

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    print("\n========== H3 STREAMED KERNEL SELF-TEST ==========")
    print(f"GPU:       {properties.name}")
    print(f"VRAM:      {properties.total_memory / 1024**3:.2f} GiB")
    print(f"PyTorch:   {torch.__version__}")
    print(f"CUDA:      {torch.version.cuda}")

    rng = np.random.default_rng(12345)

    with GGUFFile(GGUF_PATH) as gguf:
        print(f"GGUF:      v{gguf.version}")
        print(f"Data start:{gguf.data_start:>12,}")
        print(f"Tensors:   {len(gguf.tensors):>12,}")

        linear = StreamedLinear(
            gguf,
            device=device,
            chunk_rows=DEFAULT_CHUNK_ROWS,
        )

        for name in targets:
            info = gguf.tensor_info(name)
            input_width = info["shape"][0]
            output_width = math.prod(info["shape"][1:])

            x_np = rng.standard_normal(
                (TEST_SEQUENCE_LENGTH, input_width)
            ).astype(np.float32)

            x = torch.from_numpy(x_np).to(device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            start = time.perf_counter()

            y = linear(
                name,
                x,
                output_device=device,
            )

            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if not torch.isfinite(y).all().item():
                raise RuntimeError(
                    f"Non-finite result: {name}"
                )

            peak_allocated = (
                torch.cuda.max_memory_allocated(device)
                / 1024**2
            )

            peak_reserved = (
                torch.cuda.max_memory_reserved(device)
                / 1024**2
            )

            print(f"\n{name}")
            print(
                f"  Matrix:          "
                f"({output_width:,}, {input_width:,})"
            )
            print(
                f"  Storage:         "
                f"{info['bytes']:,} bytes"
            )
            print(
                f"  Output:          "
                f"{tuple(y.shape)}"
            )
            print(
                f"  Output finite:   True"
            )
            print(
                f"  Output mean:     {y.mean().item():.6f}"
            )
            print(
                f"  Output std:      {y.std().item():.6f}"
            )
            print(
                f"  Peak allocated:  {peak_allocated:.2f} MiB"
            )
            print(
                f"  Peak reserved:   {peak_reserved:.2f} MiB"
            )
            print(
                f"  Elapsed:         {elapsed:.3f} seconds"
            )

            del x, y, x_np

    print("\n========== SELF-TEST COMPLETE ==========")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the four real block-0 streamed GPU projections",
    )
    args = parser.parse_args()

    if not args.self_test:
        parser.print_help()
        return

    run_self_test()


if __name__ == "__main__":
    main()
