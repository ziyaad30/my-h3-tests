
#!/usr/bin/env python3
"""Inspect the real H3 GGUF reader and AdaLN projection API."""

import inspect
from pathlib import Path

from h3_stream_linear import GGUF, TiledGGUFLinear


MODEL = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

NAMES = [
    "adaln_t_table",
    "blocks.0.adaln_proj.linear.weight",
    "blocks.0.adaln_proj.linear.bias",
    "blocks.0.norm1",
    "blocks.0.norm2",
    "blocks.0.attn.q_norm",
    "blocks.0.attn.k_norm",
]


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<signature unavailable>"


def describe_tensor(tensor):
    dims = tuple(tensor.dims)
    return (
        f"dims={dims}, "
        f"type={tensor.ggml_type}, "
        f"elements={tensor.elements}"
    )


def main():
    path = Path(MODEL)

    if not path.is_file():
        raise FileNotFoundError(path)

    print("H3 ADALN RUNTIME API PROBE")
    print("=" * 58)

    print("\nGGUF constructor:")
    print(" ", signature(GGUF))

    print("\nTiledGGUFLinear constructor:")
    print(" ", signature(TiledGGUFLinear))

    print("\nTiledGGUFLinear call:")
    print(" ", signature(TiledGGUFLinear.__call__))

    print("\nGGUF methods:")
    for name in dir(GGUF):
        if name.startswith("_"):
            continue
        member = getattr(GGUF, name)
        if callable(member):
            print(f"  {name}{signature(member)}")

    gguf = GGUF(str(path))

    try:
        print("\nGGUF instance:")
        print("  tensor count:", gguf.tensor_count)
        print("  size:", gguf.size)

        print("\nRequired checkpoint tensors:")

        lookup = {
            tensor.name: tensor
            for tensor in (
                gguf.tensors.values()
                if isinstance(gguf.tensors, dict)
                else gguf.tensors
            )
        }

        for name in NAMES:
            tensor = lookup.get(name)

            if tensor is None:
                print(f"  MISSING: {name}")
            else:
                print(f"  FOUND: {name}")
                print("    ", describe_tensor(tensor))

        print("\nGGUF instance methods:")
        for name in dir(gguf):
            if name.startswith("_"):
                continue

            member = getattr(gguf, name)

            if callable(member):
                print(f"  {name}{signature(member)}")

        print("\nTensor object fields:")
        for name in NAMES:
            tensor = lookup.get(name)

            if tensor is None:
                continue

            print(f"  {name}")
            print("    class:", type(tensor).__name__)
            print("    fields:", sorted(vars(tensor).keys()))

            break

        print("\nProbe complete.")

    finally:
        close = getattr(gguf, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
