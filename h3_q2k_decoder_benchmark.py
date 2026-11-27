
#!/usr/bin/env python3
"""
Compare reference and vectorized Q2_K decoders
using the real MiniMax H3 checkpoint.
"""

import time

import numpy as np

from h3_stream_linear import (
    GGUF,
    TYPE_Q2_K,
    decode_q2k,
    decode_q2k_reference,
)


GGUF_PATH = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models"
    r"\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

TENSORS = [
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.attn.out_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.0.mlp.fc2.weight",
]

ROW_TILE = 128
REPEATS = 5


def synchronize_cpu():
    # Makes the timing boundaries explicit.
    pass


def timed_decode(decoder, raw, repeats):
    decoder(raw)

    timings = []

    for _ in range(repeats):
        start = time.perf_counter()
        result = decoder(raw)
        elapsed = time.perf_counter() - start

        timings.append(elapsed)

        del result

    return float(np.median(timings))


def compare_tile(raw, tensor_name):
    reference = decode_q2k_reference(raw)
    vectorized = decode_q2k(raw)

    if reference.shape != vectorized.shape:
        raise AssertionError(
            f"{tensor_name}: output shape mismatch"
        )

    difference = np.abs(reference - vectorized)

    max_error = float(difference.max())
    mean_error = float(difference.mean())

    if not np.array_equal(reference, vectorized):
        if not np.allclose(
            reference,
            vectorized,
            rtol=0.0,
            atol=0.0,
        ):
            raise AssertionError(
                f"{tensor_name}: decoder output mismatch"
            )

    return max_error, mean_error


def main():
    print("=" * 72)
    print("H3 Q2_K DECODER BENCHMARK")
    print("=" * 72)
    print(f"Row tile: {ROW_TILE}")
    print(f"Repeats:  {REPEATS}")

    total_reference = 0.0
    total_vectorized = 0.0

    with GGUF(GGUF_PATH) as gguf:
        for name in TENSORS:
            tensor = gguf.tensors[name]

            if tensor.ggml_type != TYPE_Q2_K:
                raise RuntimeError(
                    f"{name} is not Q2_K"
                )

            width = tensor.input_width
            output_rows = tensor.output_rows

            reference_time = 0.0
            vectorized_time = 0.0
            max_error = 0.0
            mean_error = 0.0
            tile_count = 0

            for row_start in range(
                0,
                output_rows,
                ROW_TILE,
            ):
                rows = min(
                    ROW_TILE,
                    output_rows - row_start,
                )

                raw = gguf.read_tensor_rows(
                    name,
                    row_start,
                    rows,
                )

                tile_max, tile_mean = compare_tile(
                    raw,
                    name,
                )

                max_error = max(max_error, tile_max)
                mean_error = max(mean_error, tile_mean)

                ref_time = timed_decode(
                    decode_q2k_reference,
                    raw,
                    REPEATS,
                )

                fast_time = timed_decode(
                    decode_q2k,
                    raw,
                    REPEATS,
                )

                reference_time += ref_time
                vectorized_time += fast_time
                tile_count += 1

            speedup = (
                reference_time / vectorized_time
                if vectorized_time > 0
                else float("inf")
            )

            total_reference += reference_time
            total_vectorized += vectorized_time

            print(f"\n{name}")
            print("-" * 72)
            print(f"Matrix:               {width:,} x {output_rows:,}")
            print(f"Tiles:                {tile_count}")
            print(f"Reference decoder:    {reference_time:.4f} s")
            print(f"Vectorized decoder:   {vectorized_time:.4f} s")
            print(f"Speedup:              {speedup:.2f}x")
            print(f"Maximum error:        {max_error:.9g}")
            print(f"Mean tile error:      {mean_error:.9g}")
            print("NUMERICAL COMPARISON PASSED")

    print("\n" + "=" * 72)
    print("TOTAL")
    print("=" * 72)
    print(f"Reference:   {total_reference:.4f} s")
    print(f"Vectorized:  {total_vectorized:.4f} s")
    print(
        f"Speedup:     "
        f"{total_reference / total_vectorized:.2f}x"
    )
    print("\nDECODER BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
