
#!/usr/bin/env python3
"""
H3 streamed linear stage profiler.

Profiles the actual checkpoint projection pipeline:

    1. GGUF mmap row reading
    2. Q2_K / F32 / F16 / BF16 decoding
    3. CPU -> CUDA weight transfer
    4. CUDA matrix multiplication

Uses the existing h3_stream_linear.py implementation.

This is a diagnostic script, not an inference implementation.
"""

import argparse
import gc
import time

import numpy as np
import torch

from h3_stream_linear import (
    GGUF,
    decode_weight_tile,
)


DEFAULT_GGUF = (
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

DEVICE = torch.device("cuda:0")
ROW_TILE = 128
TOKEN_COUNT = 27


def synchronize():
    torch.cuda.synchronize(DEVICE)


def timed_profile(gguf, name, tokens, row_tile):
    tensor = gguf.tensors[name]

    width = tensor.input_width
    output_rows = tensor.output_rows

    activation = torch.randn(
        tokens,
        width,
        dtype=torch.float32,
        device=DEVICE,
    )

    output = torch.empty(
        tokens,
        output_rows,
        dtype=torch.float32,
        device=DEVICE,
    )

    read_time = 0.0
    decode_time = 0.0
    transfer_time = 0.0
    matmul_time = 0.0

    total_tiles = (
        output_rows + row_tile - 1
    ) // row_tile

    # Warm up CUDA before collecting measurements.
    warmup_rows = min(row_tile, output_rows)

    warmup_raw = gguf.read_tensor_rows(
        name,
        0,
        warmup_rows,
    )

    warmup_np = decode_weight_tile(
        warmup_raw,
        tensor.ggml_type,
        warmup_rows,
        width,
    )

    warmup_weight = torch.from_numpy(
        np.ascontiguousarray(warmup_np)
    ).to(DEVICE)

    _ = torch.matmul(
        activation,
        warmup_weight.T,
    )

    synchronize()

    del warmup_raw
    del warmup_np
    del warmup_weight

    for row_start in range(
        0,
        output_rows,
        row_tile,
    ):
        rows = min(
            row_tile,
            output_rows - row_start,
        )

        # 1. Read bytes from the memory-mapped checkpoint.
        synchronize()
        start = time.perf_counter()

        raw = gguf.read_tensor_rows(
            name,
            row_start,
            rows,
        )

        read_time += time.perf_counter() - start

        # 2. Decode the quantized weight tile on CPU.
        start = time.perf_counter()

        weight_np = decode_weight_tile(
            raw,
            tensor.ggml_type,
            rows,
            width,
        )

        decode_time += time.perf_counter() - start

        if not np.isfinite(weight_np).all():
            raise FloatingPointError(
                f"Non-finite weights: {name}"
            )

        # 3. Transfer decoded weights to CUDA.
        synchronize()
        start = time.perf_counter()

        weight = torch.from_numpy(
            np.ascontiguousarray(weight_np)
        ).to(
            DEVICE,
            dtype=torch.float32,
        )

        synchronize()
        transfer_time += time.perf_counter() - start

        # 4. Matrix multiplication.
        start = time.perf_counter()

        result = torch.matmul(
            activation,
            weight.T,
        )

        synchronize()
        matmul_time += time.perf_counter() - start

        output[
            :,
            row_start:row_start + rows,
        ] = result

        del raw
        del weight_np
        del weight
        del result

    total = (
        read_time
        + decode_time
        + transfer_time
        + matmul_time
    )

    if not torch.isfinite(output).all().item():
        raise FloatingPointError(
            f"Non-finite output: {name}"
        )

    print(f"\n{name}")
    print("-" * 72)
    print(f"Matrix:             {width:,} x {output_rows:,}")
    print(f"Tokens:             {tokens}")
    print(f"Row tile:           {row_tile}")
    print(f"Weight tiles:       {total_tiles}")

    print(f"GGUF reading:       {read_time:9.4f} s")
    print(f"CPU decoding:       {decode_time:9.4f} s")
    print(f"CPU -> CUDA:        {transfer_time:9.4f} s")
    print(f"CUDA matmul:        {matmul_time:9.4f} s")
    print(f"Measured sum:       {total:9.4f} s")

    if total > 0:
        print("\nStage percentages:")

        for label, value in (
            ("Reading", read_time),
            ("Decoding", decode_time),
            ("Transfer", transfer_time),
            ("Matmul", matmul_time),
        ):
            print(
                f"  {label:12s}: "
                f"{100.0 * value / total:6.2f}%"
            )

    print(f"Output norm:        {output.norm().item():.6f}")

    del activation
    del output

    return {
        "read": read_time,
        "decode": decode_time,
        "transfer": transfer_time,
        "matmul": matmul_time,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gguf",
        default=DEFAULT_GGUF,
    )

    parser.add_argument(
        "--row-tile",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=27,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    if args.row_tile < 1 or args.tokens < 1:
        raise ValueError("Invalid profiling dimensions")

    print("=" * 72)
    print("H3 REAL CHECKPOINT LINEAR STAGE PROFILE")
    print("=" * 72)
    print(f"Checkpoint: {args.gguf}")
    print(f"GPU:        {torch.cuda.get_device_name(DEVICE)}")
    print(f"Row tile:   {args.row_tile}")
    print(f"Tokens:     {args.tokens}")

    torch.cuda.init()

    results = {}

    with GGUF(args.gguf) as gguf:
        for name in TENSORS:
            if name not in gguf.tensors:
                raise KeyError(
                    f"Missing required tensor: {name}"
                )

            results[name] = timed_profile(
                gguf,
                name,
                args.tokens,
                args.row_tile,
            )

            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("COMBINED PROFILE")
    print("=" * 72)

    totals = {
        "read": 0.0,
        "decode": 0.0,
        "transfer": 0.0,
        "matmul": 0.0,
        "total": 0.0,
    }

    for result in results.values():
        for key in totals:
            totals[key] += result[key]

    print(f"GGUF reading:       {totals['read']:.4f} s")
    print(f"CPU decoding:       {totals['decode']:.4f} s")
    print(f"CPU -> CUDA:        {totals['transfer']:.4f} s")
    print(f"CUDA matmul:        {totals['matmul']:.4f} s")
    print(f"Measured sum:       {totals['total']:.4f} s")

    print("\nDominant stage:")

    dominant = max(
        ("read", "decode", "transfer", "matmul"),
        key=lambda key: totals[key],
    )

    print(
        f"{dominant.upper()} "
        f"({100.0 * totals[dominant] / totals['total']:.1f}%)"
    )

    print("\nPROFILE COMPLETE")


if __name__ == "__main__":
    main()
