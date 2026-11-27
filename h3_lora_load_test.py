#!/usr/bin/env python3
"""Preflight the Turbo LoRA against every real streamed H3 transformer block.

Opens the actual GGUF and adapter, constructs each of the two token refiners
and 50 DiT blocks with LoRA attached, then exercises all six LoRA projections
with small synthetic CPU activations. No Qwen, denoising, VAE, or CUDA work.
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
def check_loaded_block(label: str, block) -> None:
    lora_block = getattr(block, "lora", None)
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
        scale = lora_block.scales[target]
        require(math.isfinite(scale), f"{label} {target}: non-finite scale")

    # Exercise the exact adapter methods called by H3TransformerBlock and
    # H3TokenRefiner. These are deliberately tiny CPU tensors, not a model pass.
    tokens = 2
    hidden = torch.randn(tokens, HIDDEN_SIZE, dtype=torch.float32)
    qkv = torch.zeros(tokens, INNER_SIZE * 3, dtype=torch.float32)
    lora_block.add_qkv(hidden, qkv)
    require(bool(torch.isfinite(qkv).all()), f"{label}: QKV application produced NaN/Inf")
    require(bool(torch.count_nonzero(qkv)), f"{label}: QKV application produced zero delta")

    attn_input = torch.randn(tokens, INNER_SIZE, dtype=torch.float32)
    attn_output = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
    lora_block.add_linear("out_proj", attn_input, attn_output)
    require(bool(torch.isfinite(attn_output).all()), f"{label}: out_proj produced NaN/Inf")
    require(bool(torch.count_nonzero(attn_output)), f"{label}: out_proj produced zero delta")

    fc1_output = torch.zeros(tokens, MLP_HIDDEN_SIZE * 2, dtype=torch.float32)
    lora_block.add_linear("fc1", hidden, fc1_output)
    require(bool(torch.isfinite(fc1_output).all()), f"{label}: FC1 produced NaN/Inf")
    require(bool(torch.count_nonzero(fc1_output)), f"{label}: FC1 produced zero delta")

    fc2_input = torch.randn(tokens, MLP_HIDDEN_SIZE, dtype=torch.float32)
    fc2_output = torch.zeros(tokens, HIDDEN_SIZE, dtype=torch.float32)
    lora_block.add_linear("fc2", fc2_input, fc2_output)
    require(bool(torch.isfinite(fc2_output).all()), f"{label}: FC2 produced NaN/Inf")
    require(bool(torch.count_nonzero(fc2_output)), f"{label}: FC2 produced zero delta")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Turbo LoRA loading/application on all real H3 transformer blocks"
    )
    parser.add_argument("--gguf", type=Path, default=Path(DEFAULT_GGUF))
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    require(args.gguf.is_file(), f"H3 GGUF not found: {args.gguf}")
    require(args.lora.is_file(), f"LoRA checkpoint not found: {args.lora}")
    require(math.isfinite(args.lora_scale) and args.lora_scale >= 0,
            "--lora-scale must be finite and nonnegative")

    print("========== H3 TURBO LoRA / REAL TRANSFORMER TEST ==========", flush=True)
    print("GGUF:", args.gguf, flush=True)
    print("LoRA:", args.lora, flush=True)
    print("Strength:", args.lora_scale, flush=True)
    print("Device: CPU; no CUDA allocation", flush=True)

    adapter = H3TurboLoRA(args.lora, strength=args.lora_scale)
    passed_refiners = 0
    passed_dit = 0

    with GGUF(str(args.gguf)) as gguf:
        print("\nTesting both actual token-refiner blocks...", flush=True)
        for index in range(NUM_REFINER_BLOCKS):
            block = H3TokenRefiner(
                gguf, block_index=index, row_tile=8, token_tile=2, lora=adapter
            )
            check_loaded_block(f"token_refiner.blocks.{index}", block)
            passed_refiners += 1
            print(f"PASS token_refiner.blocks.{index}: load + six projection applications",
                  flush=True)
            del block
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
                lora=adapter,
            )
            check_loaded_block(f"blocks.{index}", block)
            passed_dit += 1
            print(f"PASS blocks.{index}: load + six projection applications", flush=True)
            del block
            gc.collect()

    require(passed_refiners == NUM_REFINER_BLOCKS,
            f"Only {passed_refiners}/{NUM_REFINER_BLOCKS} token refiners passed")
    require(passed_dit == NUM_DIT_BLOCKS,
            f"Only {passed_dit}/{NUM_DIT_BLOCKS} DiT blocks passed")

    print("\n========== RESULT: PASS ==========", flush=True)
    print(f"Token refiners: {passed_refiners}/{NUM_REFINER_BLOCKS}")
    print(f"DiT blocks: {passed_dit}/{NUM_DIT_BLOCKS}")
    print("All six LoRA projection methods were exercised for every block.")
    print("This is an adapter integration preflight, not a denoising-quality test.")


if __name__ == "__main__":
    main()
