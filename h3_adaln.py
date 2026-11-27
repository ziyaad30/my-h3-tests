
#!/usr/bin/env python3
"""Streaming MiniMax H3 AdaLN for pruned curve-basis GGUF checkpoints."""

from pathlib import Path

import numpy as np
import torch

from h3_stream_linear import GGUF


HIDDEN_SIZE = 5376
MODALITIES = 3
MODULATION_GROUPS = 6
FINAL_GROUPS = 2

TABLE_ROWS = 1025
TABLE_WIDTH = 8

BLOCK_OUTPUT_ROWS = MODALITIES * MODULATION_GROUPS * HIDDEN_SIZE
FINAL_OUTPUT_ROWS = FINAL_GROUPS * HIDDEN_SIZE

VIDEO = 0
TEXT = 1
AUDIO = 2

DEFAULT_ROW_TILE = 512
DEFAULT_TOKEN_TILE = 2

MODEL_PATH = Path(
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)


def interpolate_adaln_table(table, timesteps):
    """Interpolate the precomputed 8-dimensional timestep curve."""

    table = np.asarray(table, dtype=np.float32)
    timesteps = np.asarray(
        timesteps, dtype=np.float32
    ).reshape(-1)

    if table.shape != (TABLE_ROWS, TABLE_WIDTH):
        raise ValueError(
            f"Invalid AdaLN table shape: {table.shape}"
        )

    if not np.isfinite(timesteps).all():
        raise ValueError("Non-finite timestep")

    t = np.clip(timesteps, 0.0, 1.0)
    position = t * (TABLE_ROWS - 1)

    # Match the reference interpolation at t=1:
    # use the final interval with interpolation fraction 1.
    lower = np.floor(position).astype(np.int64)
    lower = np.minimum(lower, TABLE_ROWS - 2)

    fraction = (position - lower).astype(np.float32)

    return (
        table[lower] * (1.0 - fraction[:, None])
        + table[lower + 1] * fraction[:, None]
    ).astype(np.float32)


class StreamingAdaLN:
    """
    Streaming checkpoint-backed AdaLN.

    Block projection:
        [N, 8] -> [N, 3, 6, 5376]

    Final projection:
        [N, 8] -> [N, 2, 5376]

    No SiLU is applied to curve-basis inputs.
    """

    def __init__(self, gguf, row_tile=DEFAULT_ROW_TILE):
        if row_tile < 1:
            raise ValueError("row_tile must be positive")

        self.gguf = gguf
        self.row_tile = int(row_tile)

        table_tensor = gguf.tensors["adaln_t_table"]

        if tuple(table_tensor.dims) != (8, 1025):
            raise ValueError(
                f"Unexpected table dimensions: {table_tensor.dims}"
            )

        if table_tensor.ggml_type != 0:
            raise ValueError("Expected F32 adaln_t_table")

        raw = gguf.read_tensor_rows(
            "adaln_t_table", 0, TABLE_ROWS
        )

        expected = TABLE_ROWS * TABLE_WIDTH * 4

        if len(raw) != expected:
            raise RuntimeError("Incomplete AdaLN table read")

        self.table = np.frombuffer(
            raw, dtype="<f4"
        ).reshape(TABLE_ROWS, TABLE_WIDTH).copy()

        if not np.isfinite(self.table).all():
            raise RuntimeError(
                "AdaLN table contains non-finite values"
            )

    @staticmethod
    def _read_f16_bias(path, tensor, expected_rows):
        if tensor.ggml_type != 1:
            raise ValueError(
                f"Expected F16 bias, got GGML type {tensor.ggml_type}"
            )

        count = int(np.prod(tensor.dims))

        if count != expected_rows:
            raise ValueError(
                f"Bias has {count} elements; expected {expected_rows}"
            )

        with Path(path).open("rb") as file:
            file.seek(tensor.absolute_offset)
            raw = file.read(count * 2)

        if len(raw) != count * 2:
            raise RuntimeError("Incomplete AdaLN bias read")

        bias = np.frombuffer(
            raw, dtype="<f2"
        ).astype(np.float32)

        if not np.isfinite(bias).all():
            raise RuntimeError("Non-finite AdaLN bias")

        return bias

    def _project(
        self,
        weight_name,
        bias_name,
        output_rows,
        timesteps,
    ):
        """Project distinct timestep curves using streamed F16 weights."""

        if weight_name not in self.gguf.tensors:
            raise KeyError(weight_name)

        if bias_name not in self.gguf.tensors:
            raise KeyError(bias_name)

        weight_tensor = self.gguf.tensors[weight_name]
        bias_tensor = self.gguf.tensors[bias_name]

        if tuple(weight_tensor.dims) != (
            TABLE_WIDTH, output_rows
        ):
            raise ValueError(
                f"{weight_name}: unexpected dimensions "
                f"{weight_tensor.dims}"
            )

        if weight_tensor.ggml_type != 1:
            raise ValueError(
                f"{weight_name}: expected F16 weights"
            )

        timesteps = np.asarray(
            timesteps, dtype=np.float32
        ).reshape(-1)

        if timesteps.size == 0:
            raise ValueError("No timesteps supplied")

        curves = interpolate_adaln_table(
            self.table, timesteps
        )

        bias = self._read_f16_bias(
            self.gguf.path,
            bias_tensor,
            output_rows,
        )

        projected = np.empty(
            (timesteps.size, output_rows),
            dtype=np.float32,
        )

        for first in range(0, output_rows, self.row_tile):
            count = min(
                self.row_tile,
                output_rows - first,
            )

            raw = self.gguf.read_tensor_rows(
                weight_name, first, count
            )

            expected = count * TABLE_WIDTH * 2

            if len(raw) != expected:
                raise RuntimeError(
                    f"Incomplete weight tile at row {first}"
                )

            weight_tile = np.frombuffer(
                raw, dtype="<f2"
            ).reshape(
                count, TABLE_WIDTH
            ).astype(np.float32)

            # Linear projection of the precomputed curve basis.
            tile_output = weight_tile @ curves.T

            tile_output += bias[
                first:first + count, None
            ]

            projected[:, first:first + count] = (
                tile_output.T
            )

        if not np.isfinite(projected).all():
            raise RuntimeError(
                f"Non-finite projection: {weight_name}"
            )

        return projected

    def project_block(
        self,
        block_index,
        timesteps,
        timestep_indices,
        token_tags,
        device="cpu",
        dtype=torch.float32,
    ):
        """Return block modulation table and per-token modulation indices."""

        timesteps = np.asarray(
            timesteps, dtype=np.float32
        ).reshape(-1)

        timestep_indices = np.asarray(
            timestep_indices, dtype=np.int64
        ).reshape(-1)

        token_tags = np.asarray(
            token_tags, dtype=np.int64
        ).reshape(-1)

        if timestep_indices.shape != token_tags.shape:
            raise ValueError(
                "Token timestep and modality arrays differ in length"
            )

        if timestep_indices.size == 0:
            raise ValueError("Empty packed sequence")

        if timesteps.size == 0:
            raise ValueError("No timesteps supplied")

        if (
            timestep_indices.min() < 0
            or timestep_indices.max() >= timesteps.size
        ):
            raise ValueError("Timestep index out of range")

        if (
            token_tags.min() < 0
            or token_tags.max() >= MODALITIES
        ):
            raise ValueError(
                "Modality IDs must be video=0, text=1, audio=2"
            )

        prefix = f"blocks.{block_index}.adaln_proj.linear"

        projected = self._project(
            prefix + ".weight",
            prefix + ".bias",
            BLOCK_OUTPUT_ROWS,
            timesteps,
        )

        # Reference layout:
        # [timestep, modality, six groups, hidden].
        modulation_table = projected.reshape(
            timesteps.size * MODALITIES,
            MODULATION_GROUPS,
            HIDDEN_SIZE,
        )

        # Reference indexing:
        # timestep_index * 3 + modality_id.
        adaln_indices = (
            timestep_indices * MODALITIES + token_tags
        )

        modulation_table = torch.from_numpy(
            np.ascontiguousarray(modulation_table)
        ).to(device=device, dtype=dtype)

        adaln_indices = torch.from_numpy(
            adaln_indices.astype(np.int64)
        ).to(device=device)

        return modulation_table, adaln_indices

    def project_final(
        self,
        timesteps,
        timestep_indices,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Project final-layer shift/scale.

        Returns:
            modulation_table: [N, 2, 5376]
            timestep_indices: [S]

        Final-layer modulation has no modality dimension.
        """

        timesteps = np.asarray(
            timesteps, dtype=np.float32
        ).reshape(-1)

        timestep_indices = np.asarray(
            timestep_indices, dtype=np.int64
        ).reshape(-1)

        if timesteps.size == 0:
            raise ValueError("No timesteps supplied")

        if timestep_indices.size == 0:
            raise ValueError("Empty packed sequence")

        if (
            timestep_indices.min() < 0
            or timestep_indices.max() >= timesteps.size
        ):
            raise ValueError("Final timestep index out of range")

        prefix = "final_layer.adaln_proj.linear"

        projected = self._project(
            prefix + ".weight",
            prefix + ".bias",
            FINAL_OUTPUT_ROWS,
            timesteps,
        )

        # Final projection chunks:
        # group 0 = shift
        # group 1 = scale
        modulation_table = projected.reshape(
            timesteps.size,
            FINAL_GROUPS,
            HIDDEN_SIZE,
        )

        modulation_table = torch.from_numpy(
            np.ascontiguousarray(modulation_table)
        ).to(device=device, dtype=dtype)

        indices = torch.from_numpy(
            timestep_indices.astype(np.int64)
        ).to(device=device)

        return modulation_table, indices


def _validate_indexed_inputs(
    normed,
    modulation_table,
    indices,
    groups,
    token_tile,
):
    if normed.ndim != 2:
        raise ValueError("Expected normed shape [S, H]")

    if modulation_table.ndim != 3:
        raise ValueError("Expected modulation shape [K, G, H]")

    if indices.ndim != 1:
        raise ValueError("Indices must be one-dimensional")

    if modulation_table.shape[1:] != (
        groups, normed.shape[1]
    ):
        raise ValueError("Modulation dimensions do not match")

    if indices.numel() != normed.shape[0]:
        raise ValueError("Index count does not match sequence length")

    if token_tile < 1:
        raise ValueError("token_tile must be positive")

    if (
        normed.device != modulation_table.device
        or normed.device != indices.device
    ):
        raise ValueError("Inputs must share a device")


def apply_adaln_indexed(
    normed,
    modulation_table,
    adaln_indices,
    shift_group,
    scale_group,
    token_tile=DEFAULT_TOKEN_TILE,
):
    """Apply block AdaLN scale/shift without full-sequence expansion."""

    _validate_indexed_inputs(
        normed,
        modulation_table,
        adaln_indices,
        MODULATION_GROUPS,
        token_tile,
    )

    if not 0 <= shift_group < MODULATION_GROUPS:
        raise ValueError("Invalid shift group")

    if not 0 <= scale_group < MODULATION_GROUPS:
        raise ValueError("Invalid scale group")

    output = torch.empty_like(normed)

    for first in range(0, normed.shape[0], token_tile):
        last = min(first + token_tile, normed.shape[0])

        selected = modulation_table.index_select(
            0, adaln_indices[first:last]
        )

        shift = selected[:, shift_group, :]
        scale = selected[:, scale_group, :]

        output[first:last] = (
            normed[first:last] * (1.0 + scale) + shift
        )

    return output


def apply_adaln_gated_residual(
    hidden_states,
    branch_output,
    modulation_table,
    adaln_indices,
    gate_group,
    token_tile=DEFAULT_TOKEN_TILE,
):
    """Apply block AdaLN gated residual without full-sequence expansion."""

    if hidden_states.shape != branch_output.shape:
        raise ValueError("Residual and branch shapes differ")

    _validate_indexed_inputs(
        hidden_states,
        modulation_table,
        adaln_indices,
        MODULATION_GROUPS,
        token_tile,
    )

    if branch_output.device != hidden_states.device:
        raise ValueError("Branch and residual devices differ")

    if not 0 <= gate_group < MODULATION_GROUPS:
        raise ValueError("Invalid gate group")

    output = torch.empty_like(hidden_states)

    for first in range(0, hidden_states.shape[0], token_tile):
        last = min(first + token_tile, hidden_states.shape[0])

        selected = modulation_table.index_select(
            0, adaln_indices[first:last]
        )

        gate = selected[:, gate_group, :]

        output[first:last] = (
            hidden_states[first:last]
            + gate * branch_output[first:last]
        )

    return output


def apply_final_adaln_indexed(
    normed,
    modulation_table,
    timestep_indices,
    token_tile=DEFAULT_TOKEN_TILE,
):
    """Apply final-layer shift/scale using timestep-only indices."""

    _validate_indexed_inputs(
        normed,
        modulation_table,
        timestep_indices,
        FINAL_GROUPS,
        token_tile,
    )

    output = torch.empty_like(normed)

    for first in range(0, normed.shape[0], token_tile):
        last = min(first + token_tile, normed.shape[0])

        selected = modulation_table.index_select(
            0, timestep_indices[first:last]
        )

        shift = selected[:, 0, :]
        scale = selected[:, 1, :]

        output[first:last] = (
            normed[first:last] * (1.0 + scale) + shift
        )

    return output


def test_indexed_operations():
    torch.manual_seed(1234)

    sequence_length = 7
    hidden_size = 32

    normed = torch.randn(sequence_length, hidden_size)
    hidden = torch.randn(sequence_length, hidden_size)
    branch = torch.randn(sequence_length, hidden_size)

    modulation = torch.randn(6, 6, hidden_size)
    indices = torch.tensor(
        [0, 1, 2, 3, 4, 5, 0],
        dtype=torch.int64,
    )

    selected = modulation.index_select(0, indices)

    for shift_group, scale_group in ((0, 1), (3, 4)):
        expected = (
            normed * (1.0 + selected[:, scale_group])
            + selected[:, shift_group]
        )

        actual = apply_adaln_indexed(
            normed,
            modulation,
            indices,
            shift_group,
            scale_group,
            token_tile=2,
        )

        torch.testing.assert_close(actual, expected)

    for gate_group in (2, 5):
        expected = (
            hidden + selected[:, gate_group] * branch
        )

        actual = apply_adaln_gated_residual(
            hidden,
            branch,
            modulation,
            indices,
            gate_group,
            token_tile=2,
        )

        torch.testing.assert_close(actual, expected)

    final_modulation = torch.randn(2, 2, hidden_size)
    final_indices = torch.tensor(
        [0, 1, 0, 1, 1, 0, 1],
        dtype=torch.int64,
    )

    final_selected = final_modulation.index_select(
        0, final_indices
    )

    expected_final = (
        normed * (1.0 + final_selected[:, 1])
        + final_selected[:, 0]
    )

    actual_final = apply_final_adaln_indexed(
        normed,
        final_modulation,
        final_indices,
        token_tile=2,
    )

    torch.testing.assert_close(actual_final, expected_final)

    print("Block indexed normalization: PASSED")
    print("Block gated residual:        PASSED")
    print("Final indexed normalization: PASSED")


def test_real_checkpoint():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    gguf = GGUF(str(MODEL_PATH))

    try:
        adaln = StreamingAdaLN(
            gguf,
            row_tile=DEFAULT_ROW_TILE,
        )

        timesteps = np.array(
            [0.25, 0.75],
            dtype=np.float32,
        )

        timestep_indices = np.array(
            [0, 0, 0, 1, 1, 1],
            dtype=np.int64,
        )

        token_tags = np.array(
            [VIDEO, TEXT, AUDIO, VIDEO, TEXT, AUDIO],
            dtype=np.int64,
        )

        block_modulation, block_indices = (
            adaln.project_block(
                block_index=0,
                timesteps=timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                device="cpu",
            )
        )

        assert tuple(block_modulation.shape) == (
            6, 6, HIDDEN_SIZE
        )

        assert block_indices.tolist() == [
            0, 1, 2, 3, 4, 5
        ]

        assert torch.isfinite(block_modulation).all()

        final_modulation, final_indices = (
            adaln.project_final(
                timesteps=timesteps,
                timestep_indices=np.array(
                    [0, 1, 0, 1],
                    dtype=np.int64,
                ),
                device="cpu",
            )
        )

        assert tuple(final_modulation.shape) == (
            2, 2, HIDDEN_SIZE
        )

        assert final_indices.tolist() == [0, 1, 0, 1]
        assert torch.isfinite(final_modulation).all()

        print("Block checkpoint projection: PASSED")
        print("  Shape:", tuple(block_modulation.shape))

        print("Final checkpoint projection: PASSED")
        print("  Shape:", tuple(final_modulation.shape))

    finally:
        gguf.close()


def main():
    print("H3 STREAMING ADALN TEST")
    print("=" * 44)

    test_indexed_operations()
    test_real_checkpoint()

    print("\nALL ADALN TESTS PASSED")


if __name__ == "__main__":
    main()
