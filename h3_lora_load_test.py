#!/usr/bin/env python3
"""Test additive loading of multiple compatible H3 Turbo LoRAs.

Each adapter is validated independently by H3TurboLoRA. For every actual
token-refiner and DiT block, this test loads the block-local factors, exercises
all six supported projections, and verifies that applying the adapters
sequentially equals the sum of their individual deltas.

CPU-only: no Qwen, denoising, VAE, or CUDA work. This test supports the exact
checkpoint schema currently implemented by h3_lora.H3TurboLoRA; it does not
claim compatibility with arbitrary H3 LoRA formats.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch

from h3_stream_linear import DEFAULT_GGUF, GGUF
from h3_transformer_ops import (
    H3TokenRefiner,
    H3TransformerBlock,
    HIDDEN_SIZE,
    INNER_SIZE,
    MLP_HIDDEN_SIZE,
)
from h3_lora import H3TurboLoRA


NUM_DIT_BLOCKS = 50
NUM_REFINER_BLOCKS = 2
EXPECTED_TARGETS = {
    "attn.to_q", "attn.to_k", "attn.to_v",
    "attn.to_out.0", "ff.net.0.proj", "ff.net.2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@torch.inference_mode()
def validate_adapter_block(label: str, lora_block) -> None:
    require(lora_block is not None, f"{label}: LoRA was not attached")
    require(set(lora_block.pairs) == EXPECTED_TARGETS,
            f"{label}: unexpected LoRA targets: {sorted(lora_block.pairs)}")
    require(set(lora_block.scales) == EXPECTED_TARGETS,
            f"{label}: scale entries do not match targets")

    for target, (a, b) in lora_block.pairs.items():
        require(a.device.type == "cpu" and b.device.type == "cpu",
                f"{label} {target}: expected CPU factors")
        require(a.dtype == torch.float32 and b.dtype == torch.float32,
                f"{label} {target}: expected FP32 factors")
        require(bool(torch.isfinite(a).all()), f"{label} {target}: A has NaN/Inf")
        require(bool(torch.isfinite(b).all()), f"{label} {target}: B has NaN/Inf")
        require(math.isfinite(lora_block.scales[target]),
                f"{label} {target}: non-finite scale")


@torch.inference_mode()
def projection_deltas(lora_blocks, label: str):
    """Apply each adapter to identical inputs and return per-adapter deltas."""
    tokens = 2
    hidden = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.float32)
    attn_input = torch.randn(tokens, INNER_SIZE, dtype=torch.float32)
    fc2_input = torch.randn(tokens, MLP_HIDDEN_SIZE, dtype=torch.float32)

    per_adapter = []
    for adapter_index, lora_block in enumerate(lora_blocks):
        adapter_label = f"{label} adapter {adapter_index + 1}"

        qkv = torch.zeros(tokens, INNER_SIZE * 3, dtype=torch.float32)
        lora_block.add_qkv(hidden, qkv)

        out_proj = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
        lora_block.add_linear("out_proj", attn_input, out_proj)

        fc1 = torch.zeros(tokens, MLP_HIDDEN_SIZE * 2, dtype=torch.float32)
        lora_block.add_linear("fc1", hidden, fc1)

        fc2 = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
        lora_block.add_linear("fc2", fc2_input, fc2)

        for name, value in (("QKV", qkv), ("out_proj", out_proj),
                            ("FC1", fc1), ("FC2", fc2)):
            require(bool(torch.isfinite(value).all()),
                    f"{adapter_label}: {name} produced NaN/Inf")
            require(bool(torch.count_nonzero(value)),
                    f"{adapter_label}: {name} produced a zero delta")

        per_adapter.append((qkv, out_proj, fc1, fc2))

    return per_adapter


@torch.inference_mode()
def check_additive_combination(label: str, lora_blocks) -> None:
    """Prove sequential multi-LoRA application equals the sum of deltas."""
    require(len(lora_blocks) >= 2, f"{label}: need at least two adapters")
    individual = projection_deltas(lora_blocks, label)

    names = ("QKV", "out_proj", "FC1", "FC2")
    for projection_index, name in enumerate(names):
        expected = torch.stack(
            [item[projection_index] for item in individual], dim=0
        ).sum(dim=0)

        combined = torch.zeros_like(expected)
        if projection_index == 0:
            # Q, K and V are applied through the fused QKV helper.
            tokens = combined.shape[0]
            hidden = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.float32)
            # Recompute all adapters with the same hidden input below.
            combined.zero_()
            for lora_block in lora_blocks:
                lora_block.add_qkv(hidden, combined)
            # The comparison input must match the individual test inputs.
            # projection_deltas returns its input-independent result only, so
            # additive equality is checked in the dedicated shared-input pass.
            del expected
            continue

    # Dedicated shared-input pass: use exactly the same inputs for each
    # individual delta and the sequential combined application.
    tokens = 2
    hidden = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.float32)
    attn_input = torch.randn(tokens, INNER_SIZE, dtype=torch.float32)
    fc2_input = torch.randn(tokens, MLP_HIDDEN_SIZE, dtype=torch.float32)

    def apply_one(lora_block):
        qkv = torch.zeros(tokens, INNER_SIZE * 3, dtype=torch.float32)
        lora_block.add_qkv(hidden, qkv)
        out_proj = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
        lora_block.add_linear("out_proj", attn_input, out_proj)
        fc1 = torch.zeros(tokens, MLP_HIDDEN_SIZE * 2, dtype=torch.float32)
        lora_block.add_linear("fc1", hidden, fc1)
        fc2 = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
        lora_block.add_linear("fc2", fc2_input, fc2)
        return qkv, out_proj, fc1, fc2

    individual_shared = [apply_one(adapter) for adapter in lora_blocks]
    combined = [torch.zeros_like(value) for value in individual_shared[0]]
    for lora_block in lora_blocks:
        lora_block.add_qkv(hidden, combined[0])
        lora_block.add_linear("out_proj", attn_input, combined[1])
        lora_block.add_linear("fc1", hidden, combined[2])
        lora_block.add_linear("fc2", fc2_input, combined[3])

    for index, name in enumerate(names):
        expected = torch.stack(
            [item[index] for item in individual_shared], dim=0
        ).sum(dim=0)
        require(torch.allclose(combined[index], expected, rtol=1e-5, atol=1e-5),
                f"{label}: combined {name} delta differs from sum of individual deltas")


def get_block_lora_blocks(block, adapters, scope: str, index: int):
    """Use the block's first attached adapter plus block-local loads of the rest."""
    first = getattr(block, "lora", None)
    require(first is not None, f"{scope}.{index}: first LoRA was not attached")
    result = [first]
    for adapter in adapters[1:]:
        result.append(adapter.load_block(scope, index, device="cpu"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test additive loading of two or more compatible H3 Turbo LoRAs"
    )
    parser.add_argument("--gguf", type=Path, default=Path(DEFAULT_GGUF))
    parser.add_argument(
        "--lora", type=Path, action="append", required=True, metavar="PATH",
        help="LoRA checkpoint path; repeat this option for each adapter (at least two)",
    )
    parser.add_argument(
        "--lora-scale", type=float, action="append", default=None, metavar="SCALE",
        help="Strength for the corresponding --lora; repeat in the same order. "
             "If omitted, every adapter uses 1.0.",
    )
    args = parser.parse_args()

    require(args.gguf.is_file(), f"H3 GGUF not found: {args.gguf}")
    require(len(args.lora) >= 2, "Provide at least two --lora paths to test combination")
    for path in args.lora:
        require(path.is_file(), f"LoRA checkpoint not found: {path}")

    if args.lora_scale is None:
        scales = [1.0] * len(args.lora)
    else:
        require(len(args.lora_scale) == len(args.lora),
                "Supply exactly one --lora-scale for each --lora, in matching order")
        scales = args.lora_scale

    for scale in scales:
        require(math.isfinite(scale) and scale >= 0,
                "--lora-scale values must be finite and nonnegative")

    print("========== H3 MULTI-LoRA / REAL TRANSFORMER TEST ==========", flush=True)
    print("GGUF:", args.gguf, flush=True)
    print("Device: CPU; no CUDA allocation", flush=True)
    print(f"Adapters: {len(args.lora)}", flush=True)

    adapters = []
    for index, (path, scale) in enumerate(zip(args.lora, scales), start=1):
        print(f"\nValidating adapter {index}: {path}", flush=True)
        adapters.append(H3TurboLoRA(path, strength=scale))

    passed_refiners = 0
    passed_dit = 0

    with GGUF(str(args.gguf)) as gguf:
        print("\nTesting both actual token-refiner blocks...", flush=True)
        for index in range(NUM_REFINER_BLOCKS):
            block = H3TokenRefiner(
                gguf, block_index=index, row_tile=8, token_tile=2,
                lora=adapters[0],
            )
            lora_blocks = get_block_lora_blocks(
                block, adapters, "token_refiner", index
            )
            for adapter_index, lora_block in enumerate(lora_blocks, start=1):
                validate_adapter_block(
                    f"token_refiner.blocks.{index} adapter {adapter_index}",
                    lora_block,
                )
            check_additive_combination(
                f"token_refiner.blocks.{index}", lora_blocks
            )
            passed_refiners += 1
            print(
                f"PASS token_refiner.blocks.{index}: {len(adapters)} adapters, "
                "all six targets, additive deltas",
                flush=True,
            )
            del lora_blocks, block
            gc.collect()

        print("\nTesting all 50 actual DiT transformer blocks...", flush=True)
        for index in range(NUM_DIT_BLOCKS):
            block = H3TransformerBlock(
                gguf,
                block_index=index,
                device="cpu",
                row_tile=8,
                token_tile=2,
                query_chunk=2,
                key_chunk=8,
                lora=adapters[0],
            )
            lora_blocks = get_block_lora_blocks(block, adapters, "blocks", index)
            for adapter_index, lora_block in enumerate(lora_blocks, start=1):
                validate_adapter_block(
                    f"blocks.{index} adapter {adapter_index}", lora_block
                )
            check_additive_combination(f"blocks.{index}", lora_blocks)
            passed_dit += 1
            print(
                f"PASS blocks.{index}: {len(adapters)} adapters, "
                "all six targets, additive deltas",
                flush=True,
            )
            del lora_blocks, block
            gc.collect()

    require(passed_refiners == NUM_REFINER_BLOCKS,
            f"Only {passed_refiners}/{NUM_REFINER_BLOCKS} token refiners passed")
    require(passed_dit == NUM_DIT_BLOCKS,
            f"Only {passed_dit}/{NUM_DIT_BLOCKS} DiT blocks passed")

    print("\n========== RESULT: PASS ==========", flush=True)
    print(f"Adapters: {len(adapters)}")
    print(f"Token refiners: {passed_refiners}/{NUM_REFINER_BLOCKS}")
    print(f"DiT blocks: {passed_dit}/{NUM_DIT_BLOCKS}")
    print("All six LoRA projection targets were tested for every block.")
    print("Sequential combined deltas matched the sum of individual deltas.")
    print("This verifies adapter loading/application, not denoising quality.")
    print("Compatibility is limited to the current H3TurboLoRA checkpoint schema.")


if __name__ == "__main__":
    main()
