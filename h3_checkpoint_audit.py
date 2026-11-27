
#!/usr/bin/env python3
"""
MiniMax H3 FL2VA GGUF metadata audit.

Reads the GGUF tensor directory only.
Does not load or dequantize tensor payloads.

Uses the existing standalone h3_stream_linear.GGUF reader.
"""

import argparse
import re
import sys
from pathlib import Path

from h3_stream_linear import GGUF


DEFAULT_MODEL = (
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

EXPECTED_BLOCKS = 50
EXPECTED_HIDDEN = 5376
EXPECTED_HEAD_DIM = 128
EXPECTED_ADALN_INPUT = 8
EXPECTED_ADALN_OUTPUT = 96768

BLOCK_ADALN_RE = re.compile(
    r"^blocks\.(\d+)\.adaln_proj\.linear\.(weight|bias)$"
)


def get_tensors(gguf):
    if isinstance(gguf.tensors, dict):
        return list(gguf.tensors.values())

    return list(gguf.tensors)


def dimensions(tensor):
    return tuple(int(x) for x in tensor.dims)


def element_count(tensor):
    result = 1

    for dim in dimensions(tensor):
        result *= dim

    return result


def storage_bytes(gguf, tensor):
    try:
        return gguf.storage_size(tensor)
    except Exception:
        return None


def describe_tensor(gguf, tensor):
    dims = dimensions(tensor)
    elements = element_count(tensor)
    size = storage_bytes(gguf, tensor)

    fields = [
        f"type={tensor.ggml_type}",
        f"gguf_dims={dims}",
        f"elements={elements}",
    ]

    # GGUF matrix dimensions are stored as
    # (input_width, output_rows) in this reader.
    # Never access matrix-only properties on vectors.
    if len(dims) == 2:
        fields.append(f"input_width={dims[0]}")
        fields.append(f"output_rows={dims[1]}")

    if size is not None:
        fields.append(f"bytes={size}")

    print(f"  {tensor.name}")
    print("    " + " | ".join(fields))


def print_section(title, tensors, predicate, gguf):
    selected = sorted(
        (t for t in tensors if predicate(t.name)),
        key=lambda t: t.name,
    )

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)

    if not selected:
        print("  NO MATCHING TENSORS")
        return []

    for tensor in selected:
        describe_tensor(gguf, tensor)

    print(f"\n  Count: {len(selected)}")

    return selected


def audit_adaln(lookup, gguf):
    print()
    print("=" * 78)
    print("ADALN STRUCTURAL VALIDATION")
    print("=" * 78)

    ok = True

    table = lookup.get("adaln_t_table")

    if table is None:
        print("ERROR: adaln_t_table is missing")
        ok = False
    else:
        dims = dimensions(table)
        describe_tensor(gguf, table)

        if dims != (8, 1025):
            print(
                f"WARNING: expected GGUF dimensions (8, 1025), "
                f"found {dims}"
            )
            ok = False

    block_ids = set()

    for name in lookup:
        match = BLOCK_ADALN_RE.match(name)

        if match:
            block_ids.add(int(match.group(1)))

    print(f"\nBlocks with AdaLN tensors: {len(block_ids)}")

    if block_ids:
        print(
            f"Block ID range: {min(block_ids)}..{max(block_ids)}"
        )

    missing = []

    for block in range(EXPECTED_BLOCKS):
        prefix = f"blocks.{block}.adaln_proj.linear."

        weight_name = prefix + "weight"
        bias_name = prefix + "bias"

        weight = lookup.get(weight_name)
        bias = lookup.get(bias_name)

        if weight is None:
            missing.append(weight_name)
        else:
            dims = dimensions(weight)

            if dims != (EXPECTED_ADALN_INPUT, EXPECTED_ADALN_OUTPUT):
                print(
                    f"WARNING: {weight_name}: "
                    f"expected GGUF dimensions "
                    f"{(EXPECTED_ADALN_INPUT, EXPECTED_ADALN_OUTPUT)}, "
                    f"found {dims}"
                )
                ok = False

        if bias is None:
            missing.append(bias_name)
        else:
            dims = dimensions(bias)

            if dims != (EXPECTED_ADALN_OUTPUT,):
                print(
                    f"WARNING: {bias_name}: "
                    f"expected dimensions "
                    f"{(EXPECTED_ADALN_OUTPUT,)}, found {dims}"
                )
                ok = False

    if missing:
        print("\nMissing AdaLN tensors:")
        for name in missing:
            print("  " + name)
        ok = False
    else:
        print("\nAll 50 blocks have AdaLN weight and bias.")

    final_names = (
        "final_layer.adaln_proj.linear.weight",
        "final_layer.adaln_proj.linear.bias",
    )

    print("\nFinal-layer AdaLN:")

    for name in final_names:
        tensor = lookup.get(name)

        if tensor is None:
            print(f"  MISSING: {name}")
            ok = False
        else:
            describe_tensor(gguf, tensor)

    return ok


def audit_normalization(tensors):
    print()
    print("=" * 78)
    print("NORMALIZATION TENSOR INVENTORY")
    print("=" * 78)

    selected = [
        t for t in tensors
        if (
            re.match(r"^blocks\.\d+\.(norm1|norm2)", t.name)
            or re.match(r"^blocks\.\d+\.attn\.[qk]_norm", t.name)
        )
    ]

    selected.sort(key=lambda t: t.name)

    if not selected:
        print("No matching normalization tensors found.")
        return

    for tensor in selected:
        print(
            f"  {tensor.name:<65} "
            f"dims={dimensions(tensor)} "
            f"type={tensor.ggml_type}"
        )

    print(f"\nNormalization tensor count: {len(selected)}")


def audit_block_projections(tensors):
    print()
    print("=" * 78)
    print("BLOCK PROJECTION INVENTORY")
    print("=" * 78)

    pattern = re.compile(
        r"^blocks\.\d+\."
        r"(attn\.(qkv_proj|out_proj)|mlp\.(fc1|fc2))"
    )

    selected = sorted(
        (t for t in tensors if pattern.match(t.name)),
        key=lambda t: t.name,
    )

    for tensor in selected:
        print(
            f"  {tensor.name:<65} "
            f"dims={dimensions(tensor)} "
            f"type={tensor.ggml_type}"
        )

    print(f"\nProjection tensor count: {len(selected)}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit MiniMax H3 GGUF tensor metadata"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to the H3 GGUF checkpoint",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    print("=" * 78)
    print("MINIMAX H3 FL2VA GGUF CHECKPOINT AUDIT")
    print("=" * 78)
    print(f"Model: {model_path}")

    if not model_path.is_file():
        print("ERROR: Model file does not exist.")
        return 1

    gguf = None

    try:
        gguf = GGUF(str(model_path))

        print(f"GGUF version:   {gguf.version}")
        print(f"Tensor count:   {gguf.tensor_count}")
        print(f"Metadata count: {gguf.metadata_count}")
        print(f"Alignment:      {gguf.alignment}")
        print(f"Data offset:    {gguf.data_offset}")
        print(f"File size:      {gguf.size:,} bytes")

        tensors = get_tensors(gguf)
        lookup = {t.name: t for t in tensors}

        print_section(
            "GLOBAL ADALN TABLE",
            tensors,
            lambda n: n == "adaln_t_table",
            gguf,
        )

        print_section(
            "BLOCK ADALN WEIGHTS AND BIASES",
            tensors,
            lambda n: bool(BLOCK_ADALN_RE.match(n)),
            gguf,
        )

        print_section(
            "FINAL-LAYER ADALN",
            tensors,
            lambda n: n.startswith(
                "final_layer.adaln_proj.linear."
            ),
            gguf,
        )

        adaln_ok = audit_adaln(lookup, gguf)

        audit_normalization(tensors)
        audit_block_projections(tensors)

        print()
        print("=" * 78)
        print("AUDIT SUMMARY")
        print("=" * 78)
        print(f"Tensor entries: {len(tensors)}")
        print(f"Unique names:   {len(lookup)}")
        print(
            "AdaLN structural validation: "
            + ("PASSED" if adaln_ok else "NEEDS INSPECTION")
        )

        print("\nTensor payloads were not loaded.")

    except Exception as exc:
        print(
            f"\nAUDIT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        if gguf is not None:
            close = getattr(gguf, "close", None)

            if callable(close):
                close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
