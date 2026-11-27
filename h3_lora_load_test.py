#!/usr/bin/env python3
"""Preflight the H3 Turbo LoRA against the actual streamed GGUF transformer.

This test does not run denoising or allocate CUDA memory. It opens the GGUF,
constructs each real token-refiner/DiT block on CPU with the adapter attached,
and checks that all expected LoRA factors are loaded into the corresponding
block objects. It catches checkpoint-name/shape errors and integration errors
in the constructors without starting an expensive inference run.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from h3_stream_linear import DEFAULT_GGUF, GGUF
from h3_transformer_ops import H3TokenRefiner, H3TransformerBlock
from h3_lora import H3TurboLoRA


NUM_DIT_BLOCKS = 50
NUM_REFINER_BLOCKS = 2
EXPECTED_TARGETS = {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_loaded_block(label: str, block) -> None:
    lora_block = getattr(block, "lora", None)
    require(lora_block is not None, f"{label}: transformer block has no attached LoRA")
    require(set(lora_block.pairs) == EXPECTED_TARGETS,
            f"{label}: loaded targets differ: {sorted(lora_block.pairs)}")
    require(set(lora_block.scales) == EXPECTED_TARGETS,
            f"{label}: scale entries differ from loaded targets")

    for target, (a, b) in lora_block.pairs.items():
        require(a.device.type == "cpu" and b.device.type == "cpu",
                f"{label} {target}: expected CPU-resident test factors")
        require(a.dtype == torch.float32 and b.dtype == torch.float32,
                f"{label} {target}: expected FP32 factors")
        require(bool(torch.isfinite(a).all()), f"{label} {target}: A contains NaN/Inf")
        require(bool(torch.isfinite(b).all()), f"{label} {target}: B contains NaN/Inf")
        require(torch.isfinite(torch.tensor(lora_block.scales[target])).item(),
                f"{label} {target}: scale is not finite")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Turbo LoRA loads into all actual streamed H3 transformer blocks"
    )
    parser.add_argument("--gguf", type=Path, default=Path(DEFAULT_GGUF))
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    require(args.gguf.is_file(), f"H3 GGUF not found: {args.gguf}")
    require(args.lora.is_file(), f"LoRA checkpoint not found: {args.lora}")
    require(torch.isfinite(torch.tensor(args.lora_scale)).item()
            and args.lora_scale >= 0,
            "--lora-scale must be finite and nonnegative")

    print("========== H3 TURBO LoRA / TRANSFORMER LOAD TEST ==========", flush=True)
    print("GGUF:", args.gguf)
    print("LoRA:", args.lora)
    print("Strength:", args.lora_scale)
    print("Device: CPU (preflight only; no CUDA allocations)", flush=True)

    # Validate the entire adapter inventory once before touching the GGUF.
    adapter = H3TurboLoRA(args.lora, strength=args.lora_scale)

    passed_refiners = 0
    passed_dit = 0

    with GGUF(str(args.gguf)) as gguf:
        print("\nLoading LoRA into both real token-refiner blocks...", flush=True)
        for index in range(NUM_REFINER_BLOCKS):
            block = H3TokenRefiner(
                gguf,
                block_index=index,
                row_tile=8,
                token_tile=2,
                lora=adapter,
            )
            check_loaded_block(f"token_refiner.blocks.{index}", block)
            passed_refiners += 1
            print(
                f"PASS token_refiner.blocks.{index}: "
                f"{len(block.lora.pairs)} projections attached",
                flush=True,
            )
            del block
            gc.collect()

        print("\nLoading LoRA into all 50 real DiT transformer blocks...", flush=True)
        for index in range(NUM_DIT_BLOCKS):
            block = H3TransformerBlock(
                gguf,
                block_index=index,
                device="cpu",
                row_tile=8,
                token_tile=2,
                query_chunk=2,
                key_chunk=8,
                lora=adapter,
            )
            check_loaded_block(f"blocks.{index}", block)
            passed_dit += 1
            print(
                f"PASS blocks.{index}: "
                f"{len(block.lora.pairs)} projections attached",
                flush=True,
            )
            del block
            gc.collect()

    require(passed_refiners == NUM_REFINER_BLOCKS,
            f"Only {passed_refiners}/{NUM_REFINER_BLOCKS} token refiners passed")
    require(passed_dit == NUM_DIT_BLOCKS,
            f"Only {passed_dit}/{NUM_DIT_BLOCKS} DiT blocks passed")

    print("\n========== RESULT: PASS ==========", flush=True)
    print(f"Token-refiner blocks: {passed_refiners}/{NUM_REFINER_BLOCKS}")
    print(f"DiT transformer blocks: {passed_dit}/{NUM_DIT_BLOCKS}")
    print(f"Total blocks with LoRA attached: {passed_refiners + passed_dit}")
    print("Each block loaded all six LoRA projection pairs on CPU.")
    print("No denoising, prompt encoding, video decoding, or CUDA allocation was run.")


if __name__ == "__main__":
    main()
