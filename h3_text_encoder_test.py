import ctypes
import gc
import torch
import torch.nn.functional as F

from pathlib import Path
from llama_cpp import Llama
from llama_cpp import llama_cpp as binding
from safetensors import safe_open
from safetensors.torch import load_file


QWEN_PATH = Path(
    r"C:\Users\User\PycharmProjects\anotherVideo\lingbot_models"
    r"\models--bartowski--Qwen_Qwen3-VL-4B-Instruct-GGUF"
    r"\snapshots\30cab06d3c6ade289ddc97a9d7c75c2b847ff32a"
    r"\Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
)

PROJECTION_PATH = Path(
    r"C:\Users\User\PycharmProjects\pythonProject"
    r"\models\clip_projections\mmh3-4b-ClipProj-v3.1.safetensors"
)

PROMPTS = [
    "A red fox walks through a snowy pine forest at sunrise.",
    "A yellow tram crosses a crowded tropical city street in heavy rain.",
]


def check_files():
    for path in (QWEN_PATH, PROJECTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)


def load_projection():
    with safe_open(
        str(PROJECTION_PATH),
        framework="pt",
        device="cpu",
    ) as f:
        metadata = f.metadata() or {}

    weights = load_file(str(PROJECTION_PATH), device="cpu")

    required = (
        "W",
        "mean_in",
        "std_in",
        "mean_out",
        "std_out",
        "sink_out",
    )

    missing = [key for key in required if key not in weights]
    if missing:
        raise RuntimeError(f"Missing projection tensors: {missing}")

    p = {
        key: weights[key].float().contiguous()
        for key in required
    }

    if tuple(p["W"].shape) != (2560, 5120):
        raise RuntimeError(f"Unexpected W shape: {p['W'].shape}")

    for key in ("mean_in", "std_in"):
        if tuple(p[key].shape) != (2560,):
            raise RuntimeError(f"Unexpected {key} shape: {p[key].shape}")

    for key in ("mean_out", "std_out", "sink_out"):
        if tuple(p[key].shape) != (5120,):
            raise RuntimeError(f"Unexpected {key} shape: {p[key].shape}")

    if torch.any(p["std_in"] == 0):
        raise RuntimeError("std_in contains zero values.")

    print("Projection metadata:", metadata)
    print("Projection W:", tuple(p["W"].shape))

    return p


def extract_hidden(llm, prompt):
    # Ensure the second prompt cannot reuse the first prompt's KV state.
    llm.reset()

    tokens = llm.tokenize(
        prompt.encode("utf-8"),
        add_bos=False,
    )

    if not tokens:
        raise RuntimeError("Tokenizer returned no tokens.")

    llm.eval(tokens)

    n_tokens = ctypes.c_int32()
    n_embd = ctypes.c_int32()

    output = binding.llama_get_layer_output(
        llm._ctx.ctx,
        ctypes.byref(n_tokens),
        ctypes.byref(n_embd),
    )

    if not output:
        raise RuntimeError(
            "llama_get_layer_output returned a null pointer."
        )

    count = n_tokens.value * n_embd.value

    if n_tokens.value != len(tokens):
        raise RuntimeError(
            f"Token count mismatch: tokenizer={len(tokens)}, "
            f"layer output={n_tokens.value}"
        )

    if n_embd.value != 2560:
        raise RuntimeError(
            f"Expected 2560 hidden dimensions, got {n_embd.value}"
        )

    raw = ctypes.cast(
        output,
        ctypes.POINTER(ctypes.c_float * count),
    ).contents

    hidden = torch.tensor(
        list(raw),
        dtype=torch.float32,
    ).reshape(n_tokens.value, n_embd.value)

    if not torch.isfinite(hidden).all():
        raise RuntimeError("Raw hidden states contain NaN or Inf.")

    return tokens, hidden


def project_hidden(hidden, p):
    x = (hidden - p["mean_in"]) / p["std_in"]
    y = x @ p["W"]
    conditioning = y * p["std_out"] + p["mean_out"]

    # H3-specific projection sink replacement.
    conditioning[0, :] = p["sink_out"]

    if not torch.isfinite(conditioning).all():
        raise RuntimeError("Projected conditioning contains NaN or Inf.")

    return conditioning


def report_tensor(label, tensor):
    print(
        f"{label}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():.6f} "
        f"max={tensor.max().item():.6f} "
        f"mean={tensor.mean().item():.6f} "
        f"std={tensor.std().item():.6f} "
        f"rms={tensor.square().mean().sqrt().item():.6f}"
    )


def compare_outputs(a, b, label):
    # Exclude the sink row because it is deliberately replaced
    # with the same sink_out vector for every prompt.
    x = a[1:].mean(dim=0)
    y = b[1:].mean(dim=0)

    cosine = F.cosine_similarity(x, y, dim=0).item()
    delta = x - y

    print(f"\n{label}")
    print(f"  Mean-vector cosine: {cosine:.9f}")
    print(f"  Mean-vector RMS difference: "
          f"{delta.square().mean().sqrt().item():.9f}")
    print(f"  Mean-vector max difference: "
          f"{delta.abs().max().item():.9f}")
    print(f"  Exact tensor equality: {torch.equal(a, b)}")


def main():
    check_files()
    projection = load_projection()

    print("\nLoading Qwen3-VL-4B...")
    print("GPU layers: 0 (CPU encoder test)")

    llm = None
    results = []

    try:
        llm = Llama(
            model_path=str(QWEN_PATH),
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False,
        )

        for index, prompt in enumerate(PROMPTS):
            print(f"\n========== PROMPT {index + 1} ==========")
            print(prompt)

            tokens, hidden = extract_hidden(llm, prompt)
            conditioning = project_hidden(hidden, projection)

            print("Token count:", len(tokens))
            print("Token IDs:", tokens)

            report_tensor("Raw layer output", hidden)
            report_tensor("Projected conditioning", conditioning)

            results.append({
                "tokens": tokens,
                "hidden": hidden,
                "conditioning": conditioning,
            })

        a, b = results

        print("\n========== PROMPT COMPARISON ==========")
        print("Token IDs identical:", a["tokens"] == b["tokens"])
        print(
            "Raw output shapes:",
            tuple(a["hidden"].shape),
            tuple(b["hidden"].shape),
        )
        print(
            "Conditioning shapes:",
            tuple(a["conditioning"].shape),
            tuple(b["conditioning"].shape),
        )

        compare_outputs(
            a["hidden"],
            b["hidden"],
            "RAW HIDDEN STATES",
        )

        compare_outputs(
            a["conditioning"],
            b["conditioning"],
            "PROJECTED CONDITIONING",
        )

        print("\nCONDITIONING AUDIT COMPLETE")

    finally:
        if llm is not None:
            llm.close()
            del llm

        gc.collect()


if __name__ == "__main__":
    main()
