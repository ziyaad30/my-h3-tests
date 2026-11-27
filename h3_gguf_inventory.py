
from pathlib import Path
from collections import Counter, defaultdict
from gguf import GGUFReader

PATH = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\diffusion_models\minimax_h3_fl2va_pruned-Q2_K.gguf"
)

if not PATH.is_file():
    raise FileNotFoundError(PATH)

reader = GGUFReader(str(PATH), mode="r")
tensors = reader.tensors

def tensor_type_name(t):
    value = t.tensor_type
    return getattr(value, "name", str(value))

def torch_shape(t):
    return tuple(reversed(tuple(int(x) for x in t.shape)))

def describe(t):
    return f"{torch_shape(t)!s:24} {tensor_type_name(t):16} {t.name}"

print("========== H3 GGUF INVENTORY ==========")
print("File size:", f"{PATH.stat().st_size / (1024**3):.3f} GiB")
print("Tensor count:", len(tensors))

by_type = Counter(tensor_type_name(t) for t in tensors)
print("\n--- Quantization type totals ---")
for name, count in by_type.most_common():
    print(f"{name:20} {count:6}")

lookup = {t.name: t for t in tensors}

# Compare every transformer block's tensor suffixes.
print("\n--- Transformer block consistency ---")
block_maps = {}

for i in range(50):
    prefix = f"blocks.{i}."
    block_maps[i] = {
        t.name[len(prefix):]: t
        for t in tensors
        if t.name.startswith(prefix)
    }

reference = block_maps[0]
print("Block 0 tensor count:", len(reference))

consistent = True

for i in range(1, 50):
    current = block_maps[i]
    missing = sorted(set(reference) - set(current))
    extra = sorted(set(current) - set(reference))

    if missing or extra:
        consistent = False
        print(f"Block {i}: missing={missing}, extra={extra}")

if consistent:
    print("All 50 blocks have identical tensor-name inventories.")

# Compare shape and quantization type for matching suffixes.
print("\n--- Block shape/type consistency ---")
for suffix, ref_tensor in sorted(reference.items()):
    ref_shape = torch_shape(ref_tensor)
    ref_type = tensor_type_name(ref_tensor)

    mismatches = []
    for i in range(1, 50):
        t = block_maps[i].get(suffix)
        if t is None:
            continue
        if torch_shape(t) != ref_shape or tensor_type_name(t) != ref_type:
            mismatches.append(
                (i, torch_shape(t), tensor_type_name(t))
            )

    if mismatches:
        consistent = False
        print(f"MISMATCH {suffix}: reference={ref_shape}/{ref_type}")
        for item in mismatches:
            print("   ", item)

if consistent:
    print("All corresponding transformer tensors have matching shapes and types.")

# Token-refiner inventory.
print("\n--- Token refiner ---")
for i in range(2):
    prefix = f"token_refiner.blocks.{i}."
    selected = [t for t in tensors if t.name.startswith(prefix)]
    print(f"\nBlock {i}: {len(selected)} tensors")
    for t in sorted(selected, key=lambda x: x.name):
        print(" ", describe(t))

# Full inventory for transformer block 0.
print("\n--- Complete transformer block 0 ---")
for t in sorted(
    (t for t in tensors if t.name.startswith("blocks.0.")),
    key=lambda x: x.name,
):
    print(describe(t))

# Global tensors, excluding transformer and refiner blocks.
print("\n--- Global tensors ---")
global_tensors = [
    t for t in tensors
    if not t.name.startswith("blocks.")
    and not t.name.startswith("token_refiner.")
]
for t in sorted(global_tensors, key=lambda x: x.name):
    print(describe(t))

# Estimate raw tensor storage by quantization type.
print("\n--- Raw tensor storage by type ---")
storage = defaultdict(int)

for t in tensors:
    storage[tensor_type_name(t)] += int(t.n_bytes)

for name, amount in sorted(storage.items(), key=lambda x: -x[1]):
    print(f"{name:20} {amount / (1024**3):8.3f} GiB")

print("\n========== RESULT ==========")
print("BLOCK INVENTORY CONSISTENT" if consistent else "BLOCK INVENTORY HAS DIFFERENCES")
print("No tensor data was intentionally dequantized by this inventory script.")
