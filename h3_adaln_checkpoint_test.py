
#!/usr/bin/env python3
"""Real-checkpoint, tiled AdaLN projection test for MiniMax H3."""

from pathlib import Path

import numpy as np

from h3_stream_linear import GGUF


MODEL = Path(
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

TABLE_NAME = "adaln_t_table"
WEIGHT_NAME = "blocks.0.adaln_proj.linear.weight"
BIAS_NAME = "blocks.0.adaln_proj.linear.bias"

HIDDEN_SIZE = 5376
TABLE_ROWS = 1025
TABLE_WIDTH = 8
OUTPUT_ROWS = 96768
ROW_TILE = 64

# Diagnostic timestep only; this does not establish modality ordering.
TEST_TIMESTEP = 0.5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_f16_vector(path, tensor):
    """Read one contiguous F16 vector using its GGUF byte offset."""
    require(tensor.ggml_type == 1, "Expected F16 bias tensor")
    count = int(np.prod(tensor.dims))

    with path.open("rb") as file:
        file.seek(tensor.absolute_offset)
        raw = file.read(count * 2)

    require(len(raw) == count * 2, "Incomplete bias read")

    return np.frombuffer(raw, dtype="<f2").astype(np.float32)


def main():
    require(MODEL.is_file(), f"Checkpoint not found: {MODEL}")

    gguf = GGUF(str(MODEL))

    try:
        tensors = gguf.tensors
        table = tensors[TABLE_NAME]
        weight = tensors[WEIGHT_NAME]
        bias_tensor = tensors[BIAS_NAME]

        require(tuple(table.dims) == (8, 1025), "Unexpected table dimensions")
        require(table.ggml_type == 0, "Expected F32 timestep table")

        require(
            tuple(weight.dims) == (8, OUTPUT_ROWS),
            "Unexpected AdaLN weight dimensions",
        )
        require(weight.ggml_type == 1, "Expected F16 AdaLN weight")

        require(
            tuple(bias_tensor.dims) == (OUTPUT_ROWS,),
            "Unexpected AdaLN bias dimensions",
        )

        print("H3 REAL-CHECKPOINT ADALN TEST")
        print("=" * 48)

        # Read the small F32 timestep table.
        table_raw = gguf.read_tensor_rows(TABLE_NAME, 0, TABLE_ROWS)

        expected_table_bytes = TABLE_ROWS * TABLE_WIDTH * 4
        require(
            len(table_raw) == expected_table_bytes,
            f"Table byte count mismatch: {len(table_raw)}",
        )

        table_data = np.frombuffer(
            table_raw, dtype="<f4"
        ).reshape(TABLE_ROWS, TABLE_WIDTH)

        require(np.isfinite(table_data).all(), "Non-finite table values")

        # Interpolate one curve at the diagnostic timestep.
        position = TEST_TIMESTEP * (TABLE_ROWS - 1)
        lower = int(position)
        upper = min(lower + 1, TABLE_ROWS - 1)
        fraction = position - lower

        curve = (
            table_data[lower] * (1.0 - fraction)
            + table_data[upper] * fraction
        ).astype(np.float32)

        # Read the bias vector once: approximately 190 KiB on disk.
        bias = read_f16_vector(MODEL, bias_tensor)
        require(np.isfinite(bias).all(), "Non-finite bias values")

        # Stream the 96,768 output rows in small tiles.
        output = np.empty(OUTPUT_ROWS, dtype=np.float32)

        for first in range(0, OUTPUT_ROWS, ROW_TILE):
            count = min(ROW_TILE, OUTPUT_ROWS - first)

            raw = gguf.read_tensor_rows(
                WEIGHT_NAME, first, count
            )

            expected_bytes = count * TABLE_WIDTH * 2
            require(
                len(raw) == expected_bytes,
                f"Weight tile byte count mismatch at row {first}",
            )

            tile = np.frombuffer(
                raw, dtype="<f2"
            ).reshape(count, TABLE_WIDTH).astype(np.float32)

            output[first:first + count] = (
                tile @ curve + bias[first:first + count]
            )

        require(np.isfinite(output).all(), "Non-finite AdaLN output")

        print(f"Table:       {table_data.shape} {table_data.dtype}")
        print(f"Curve:       {curve.shape} {curve.dtype}")
        print(f"Weight:      {tuple(weight.dims)} F16, tiled")
        print(f"Bias:        {bias.shape} F16 -> FP32")
        print(f"Output:      {output.shape} {output.dtype}")
        print(f"Finite:      {bool(np.isfinite(output).all())}")
        print(f"Output range:{output.min(): .6g} .. {output.max(): .6g}")
        print(f"Output mean: {output.mean(): .6g}")
        print(f"Output std:  {output.std(): .6g}")

        # Discover actual block-0 normalization names.
        print("\nBLOCK 0 NORMALIZATION TENSORS")

        norm_names = sorted(
            name for name in tensors
            if name.startswith("blocks.0.")
            and "norm" in name.lower()
        )

        for name in norm_names:
            tensor = tensors[name]
            print(
                f"  {name}: dims={tuple(tensor.dims)}, "
                f"type={tensor.ggml_type}"
            )

        if not norm_names:
            print("  No names containing 'norm' found in block 0.")

        print("\nREAL-CHECKPOINT PROJECTION: PASSED")

    finally:
        gguf.close()


if __name__ == "__main__":
    main()
