#!/usr/bin/env python3
"""Memory-bounded loader for MiniMax-H3 FL2VA Turbo-compatible LoRAs.

Reads only the current token-refiner/DiT block from the safetensors file.
No adapter-wide tensor dictionary is materialized or retained. By default,
the production path enforces the known Turbo filename. The multi-LoRA
preflight may opt into filename-independent validation, but all checkpoint
keys, tensor shapes, rank, alpha metadata and supported key format remain
strictly validated.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open


EXPECTED_FILENAME = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"
RANK = 128
DEFAULT_ALPHA = 8.0
HIDDEN = 5376
ATTENTION = 7168
FFN = 14336

TARGETS = {
    "attn.to_q": (HIDDEN, ATTENTION),
    "attn.to_k": (HIDDEN, ATTENTION),
    "attn.to_v": (HIDDEN, ATTENTION),
    "attn.to_out.0": (ATTENTION, HIDDEN),
    "ff.net.0.proj": (HIDDEN, 2 * FFN),
    "ff.net.2": (FFN, HIDDEN),
}


class H3LoRABlock:
    """One block's LoRA factors, resident only for that block execution."""

    def __init__(self, pairs, scales, device):
        self.pairs = pairs
        self.scales = {name: float(value) for name, value in scales.items()}
        self.device = torch.device(device)
        self.print_diagnostics = False

    @staticmethod
    def _print_delta(label, delta):
        sum_sq = torch.zeros((), device=delta.device, dtype=torch.float32)
        max_abs = torch.zeros((), device=delta.device, dtype=torch.float32)
        count = 0
        for first in range(0, delta.shape[0], 128):
            value = delta[first:first + 128].detach().float()
            sum_sq += value.square().sum()
            max_abs = torch.maximum(max_abs, value.abs().max())
            count += value.numel()
        if count:
            print(
                f"[H3 LoRA delta] {label:<24} "
                f"rms={torch.sqrt(sum_sq / count).item():.5g} "
                f"max_abs={max_abs.item():.5g}",
                flush=True,
            )

    @torch.inference_mode()
    def _delta(self, x, target):
        a, b = self.pairs[target]
        if x.device != self.device:
            raise RuntimeError(
                f"LoRA input for {target} is on {x.device}, expected {self.device}"
            )
        if x.ndim != 2 or x.shape[1] != a.shape[1]:
            raise ValueError(
                f"LoRA input shape mismatch for {target}: "
                f"{tuple(x.shape)} vs [tokens,{a.shape[1]}]"
            )
        hidden = torch.matmul(x, a.transpose(0, 1))
        return torch.matmul(hidden, b.transpose(0, 1)) * self.scales[target]

    @torch.inference_mode()
    def add_qkv(self, x, output):
        expected = (x.shape[0], 3 * ATTENTION)
        if tuple(output.shape) != expected:
            raise ValueError(
                f"Fused QKV output shape {tuple(output.shape)} != {expected}"
            )
        for index, target in enumerate(("attn.to_q", "attn.to_k", "attn.to_v")):
            start = index * ATTENTION
            end = start + ATTENTION
            delta = self._delta(x, target)
            if self.print_diagnostics:
                self._print_delta(target, delta)
            output[:, start:end].add_(delta)
            del delta
        return output

    @torch.inference_mode()
    def add_linear(self, name, x, output):
        target = {
            "out_proj": "attn.to_out.0",
            "fc1": "ff.net.0.proj",
            "fc2": "ff.net.2",
        }.get(name)
        if target is None:
            raise KeyError(f"Unknown H3 LoRA projection: {name}")
        delta = self._delta(x, target)
        if self.print_diagnostics:
            self._print_delta(target, delta)
        if delta.shape != output.shape:
            raise ValueError(
                f"LoRA {name} delta {tuple(delta.shape)} "
                f"does not match projection output {tuple(output.shape)}"
            )
        output.add_(delta)
        return output


class H3TurboLoRA:
    """Strictly validates the H3 Turbo-compatible checkpoint schema."""

    def __init__(
        self,
        path: str | Path,
        strength: float = 1.0,
        *,
        allow_compatible_filename: bool = False,
    ):
        self.path = Path(path)
        self.strength = float(strength)

        if not self.path.is_file():
            raise FileNotFoundError(f"Turbo LoRA not found: {self.path}")
        if not allow_compatible_filename and self.path.name != EXPECTED_FILENAME:
            raise ValueError(
                f"This loader is specifically for {EXPECTED_FILENAME}; "
                f"received {self.path.name}"
            )
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("LoRA strength must be finite and nonnegative")

        self.metadata_alpha = None
        self.rank = RANK
        self._validate_checkpoint()

    @staticmethod
    def _raw_prefix(scope: str, index: int) -> str:
        if scope == "blocks":
            return f"transformer_blocks.{index}"
        if scope == "token_refiner":
            return f"token_refiner.refiner_blocks.{index}"
        raise ValueError(f"Unknown H3 LoRA scope: {scope}")

    def _expected_shapes(self):
        shapes = {}
        for scope, count in (("blocks", 50), ("token_refiner", 2)):
            for index in range(count):
                prefix = self._raw_prefix(scope, index)
                for target, (input_width, output_width) in TARGETS.items():
                    base = f"{prefix}.{target}"
                    shapes[f"{base}.lora_A.default.weight"] = (RANK, input_width)
                    shapes[f"{base}.lora_B.default.weight"] = (output_width, RANK)
        return shapes

    def _validate_checkpoint(self):
        expected = self._expected_shapes()
        with safe_open(str(self.path), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            key_format = metadata.get("key_format")
            if key_format not in (None, "minimax-h3-diffusers"):
                raise ValueError(
                    f"Unsupported Turbo LoRA key_format={key_format!r}"
                )

            raw_alpha = metadata.get("alpha")
            if raw_alpha is not None:
                try:
                    self.metadata_alpha = float(raw_alpha)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid Turbo LoRA metadata alpha: {raw_alpha!r}"
                    ) from exc
                if not math.isfinite(self.metadata_alpha) or self.metadata_alpha <= 0:
                    raise ValueError(f"Invalid Turbo LoRA metadata alpha: {self.metadata_alpha!r}")

            actual = set(checkpoint.keys())
            expected_keys = set(expected)
            pair_prefixes = sorted({name.rsplit(".lora_", 1)[0] for name in expected_keys})
            alpha_keys = {prefix + ".alpha" for prefix in pair_prefixes}
            allowed_keys = expected_keys | (actual & alpha_keys)
            missing = sorted(expected_keys - actual)
            unexpected = sorted(actual - allowed_keys)
            missing_alpha = [prefix for prefix in pair_prefixes
                             if prefix + ".alpha" not in actual and self.metadata_alpha is None]
            if missing_alpha:
                raise ValueError(
                    "Turbo LoRA has no per-layer or metadata alpha for "
                    f"{len(missing_alpha)} pairs; first missing: {missing_alpha[:5]}"
                )
            if missing or unexpected:
                raise ValueError(
                    "Turbo LoRA tensor inventory mismatch: "
                    f"expected {len(expected_keys)} factor tensors, found {len(actual)}; "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )

            for name, shape in expected.items():
                actual_shape = tuple(checkpoint.get_slice(name).get_shape())
                if actual_shape != shape:
                    raise ValueError(
                        f"Turbo LoRA shape mismatch: {name}={actual_shape}, "
                        f"expected={shape}"
                    )

        print(
            f"Turbo-compatible LoRA validated: {self.path.name} | "
            f"312 target pairs | rank={self.rank} | "
            f"alpha source={'per-layer tensors' if any(k.endswith('.alpha') for k in actual) else 'file metadata'} | "
            f"strength={self.strength:g}",
            flush=True,
        )

    @torch.inference_mode()
    def load_block(self, scope: str, index: int, device: str | torch.device):
        prefix = self._raw_prefix(scope, int(index))
        pairs = {}
        scales = {}
        with safe_open(str(self.path), framework="pt", device="cpu") as checkpoint:
            metadata_alpha = (checkpoint.metadata() or {}).get("alpha")
            for target, (input_width, output_width) in TARGETS.items():
                base = f"{prefix}.{target}"
                a_name = f"{base}.lora_A.default.weight"
                b_name = f"{base}.lora_B.default.weight"

                a = checkpoint.get_tensor(a_name).to(
                    device=device, dtype=torch.float32
                )
                b = checkpoint.get_tensor(b_name).to(
                    device=device, dtype=torch.float32
                )

                if tuple(a.shape) != (RANK, input_width):
                    raise RuntimeError(f"Validated LoRA A changed shape: {a_name}")
                if tuple(b.shape) != (output_width, RANK):
                    raise RuntimeError(f"Validated LoRA B changed shape: {b_name}")

                # Official H3 Turbo conversion swaps the source value/gate
                # halves into the fused H3 gate/value FC1 output order.
                if target == "ff.net.0.proj":
                    value, gate = b.chunk(2, dim=0)
                    b = torch.cat((gate, value), dim=0).contiguous()

                alpha_key = base + ".alpha"
                if alpha_key in checkpoint.keys():
                    alpha = float(checkpoint.get_tensor(alpha_key).float().item())
                elif metadata_alpha is not None:
                    alpha = float(metadata_alpha)
                else:
                    raise RuntimeError(f"Validated Turbo LoRA alpha missing: {alpha_key}")
                if not math.isfinite(alpha) or alpha <= 0:
                    raise ValueError(f"Invalid Turbo LoRA alpha for {base}: {alpha}")
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise FloatingPointError(
                        f"Non-finite Turbo LoRA factors in {base}"
                    )
                pairs[target] = (a.contiguous(), b.contiguous())
                scales[target] = alpha / RANK * self.strength

        return H3LoRABlock(pairs, scales, device)



class H3LoRALoader:
    """Load and strictly validate a MiniMax H3 Turbo-compatible LoRA."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        files = folder_paths.get_filename_list("loras")
        if not files:
            files = ["<put H3 Turbo LoRA in models/loras>"]
        return {
            "required": {
                "lora_name": (files,),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("H3_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "MiniMax H3/Model"
    TITLE = "H3 LoRA Loader"

    def load_lora(self, lora_name, strength):
        import folder_paths

        if lora_name.startswith("<put "):
            raise RuntimeError(
                "No H3 LoRA was found. Place the H3 Turbo-compatible "
                "safetensors file in ComfyUI/models/loras and refresh ComfyUI."
            )
        path = folder_paths.get_full_path("loras", lora_name)
        if path is None:
            raise FileNotFoundError(f"Could not resolve H3 LoRA: {lora_name}")

        # The key inventory, factor shapes, rank and alpha are still checked
        # strictly. Filename compatibility is relaxed so users can rename a
        # valid H3 adapter without changing its contents.
        adapter = H3TurboLoRA(
            path,
            strength=float(strength),
            allow_compatible_filename=True,
        )
        return (adapter,)
