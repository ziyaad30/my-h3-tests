"""MiniMax H3 standalone inference entry point."""

import argparse

import ctypes

import gc

import math

import time

from pathlib import Path

import numpy as np

import torch

import torch.nn.functional as F

from minimax_h3_kernels import (
    GGUFFile,
    GGUF_PATH,
    StreamedLinear,
)

from minimax_h3_transformer import *

QWEN_MODEL = Path(
    r"C:\Users\User\PycharmProjects\anotherVideo\lingbot_models"
    r"\models--bartowski--Qwen_Qwen3-VL-4B-Instruct-GGUF"
    r"\snapshots\30cab06d3c6ade289ddc97a9d7c75c2b847ff32a"
    r"\Qwen_Qwen3-VL-4B-Instruct-Q4_K_M.gguf"
)

CLIP_PROJECTION = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models"
    r"\clip_projections\mmh3-4b-ClipProj-v3.1.safetensors"
)


def extract_qwen_conditioning(prompt):
    from safetensors.torch import load_file
    from llama_cpp import Llama
    import llama_cpp.llama_cpp as binding

    require(
        QWEN_MODEL.is_file(),
        f"Qwen model not found: {QWEN_MODEL}",
    )

    require(
        CLIP_PROJECTION.is_file(),
        f"Clip projection not found: {CLIP_PROJECTION}",
    )

    require(
        hasattr(binding, "llama_get_layer_output"),
        "Custom llama_get_layer_output binding is missing",
    )

    print("\nLoading ClipProj...")

    weights = load_file(
        str(CLIP_PROJECTION),
        device="cpu",
    )

    required = [
        "W",
        "mean_in",
        "std_in",
        "mean_out",
        "std_out",
        "sink_out",
    ]

    for key in required:
        require(
            key in weights,
            f"ClipProj is missing tensor: {key}",
        )

    W = weights["W"].float()

    mean_in = weights["mean_in"].float().reshape(-1)
    std_in = weights["std_in"].float().reshape(-1)

    mean_out = weights["mean_out"].float().reshape(-1)
    std_out = weights["std_out"].float().reshape(-1)

    sink_out = weights["sink_out"].float().reshape(-1)

    require(
        tuple(W.shape) == (2560, 5120),
        f"Unexpected ClipProj W shape: {tuple(W.shape)}",
    )

    require(
        mean_in.numel() == 2560
        and std_in.numel() == 2560,
        "Invalid ClipProj input statistics",
    )

    require(
        mean_out.numel() == 5120
        and std_out.numel() == 5120
        and sink_out.numel() == 5120,
        "Invalid ClipProj output statistics",
    )

    require(
        bool(torch.all(std_in != 0).item()),
        "ClipProj std_in contains zeros",
    )

    print("ClipProj loaded.")
    print("\nLoading Qwen3-VL-4B...")

    llm = Llama(
        model_path=str(QWEN_MODEL),
        n_ctx=512,
        n_batch=512,
        n_gpu_layers=-1,
        verbose=False,
    )

    try:
        tokens = llm.tokenize(
            prompt.encode("utf-8"),
            add_bos=True,
        )

        require(
            0 < len(tokens) <= 512,
            f"Prompt token count outside Qwen context: {len(tokens)}",
        )

        print("Prompt tokens:", len(tokens))
        print("Running Qwen inference...")

        llm.eval(tokens)

        n_tokens = ctypes.c_int32(0)
        n_embd = ctypes.c_int32(0)

        output = binding.llama_get_layer_output(
            llm._ctx.ctx,
            ctypes.byref(n_tokens),
            ctypes.byref(n_embd),
        )

        require(
            bool(output),
            "Block-24 extraction returned NULL",
        )

        count = n_tokens.value * n_embd.value

        require(
            n_tokens.value == len(tokens),
            f"Qwen token mismatch: {n_tokens.value} vs {len(tokens)}",
        )

        require(
            n_embd.value == 2560,
            f"Unexpected Qwen hidden dimension: {n_embd.value}",
        )

        raw = ctypes.cast(
            output,
            ctypes.POINTER(ctypes.c_float * count),
        ).contents

        h = torch.from_numpy(
            np.fromiter(
                raw,
                dtype=np.float32,
                count=count,
            ).reshape(
                n_tokens.value,
                n_embd.value,
            ).copy()
        )

        require(
            bool(torch.isfinite(h).all().item()),
            "Qwen hidden states contain NaN/Inf",
        )

        print_stats("Qwen block-24 output", h)

        # User-validated MiniMax H3 ClipProj path.
        xn = (h - mean_in) / std_in
        yn = xn @ W
        cond = yn * std_out + mean_out

        cond[0, :] = sink_out

        require(
            tuple(cond.shape) == (len(tokens), 5120),
            f"Unexpected conditioning shape: {tuple(cond.shape)}",
        )

        require(
            bool(torch.isfinite(cond).all().item()),
            "ClipProj conditioning contains NaN/Inf",
        )

        require(
            torch.equal(cond[0], sink_out),
            "Sink token replacement failed",
        )

        print_stats("ClipProj conditioning", cond)

        return cond.contiguous()

    finally:
        llm.close()
        del llm

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_text_test(prompt, timestep):
    require(
        torch.cuda.is_available(),
        "CUDA is unavailable",
    )

    device = torch.device("cuda:0")

    properties = torch.cuda.get_device_properties(device)

    print("\n========== H3 REAL TEXT-CONDITIONING TEST ==========")
    print("GPU:", properties.name)
    print(f"VRAM: {properties.total_memory / 1024 ** 3:.2f} GiB")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Timestep:", timestep)
    print("Prompt:", prompt)

    conditioning = extract_qwen_conditioning(prompt)

    print("\nQwen model closed.")
    print("Loading H3 transformer components...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    with GGUFFile(GGUF_PATH) as gguf:
        linear = StreamedLinear(
            gguf,
            device=device,
            chunk_rows=WEIGHT_CHUNK_ROWS,
        )

        print("\nApplying H3 condition_proj...")

        text = linear(
            "condition_proj.weight",
            conditioning.to(device),
            bias_name="condition_proj.bias",
            output_device=device,
        )

        del conditioning

        print_stats("H3 condition_proj", text)

        print("\nRunning H3 token refiner...")

        refiner = MiniMaxH3TokenRefiner(
            gguf,
            device,
        )

        text = refiner.forward(
            text,
            diagnostics=True,
        )

        del refiner

        tags = torch.ones(
            text.shape[0],
            device=device,
            dtype=torch.long,
        )

        positions = torch.zeros(
            (text.shape[0], 3),
            device=device,
            dtype=torch.float32,
        )

        timestep_values = torch.tensor(
            [timestep],
            device=device,
            dtype=torch.float32,
        )

        timestep_indices = torch.zeros(
            text.shape[0],
            device=device,
            dtype=torch.long,
        )

        print("\nRunning transformer block 0 with refined text...")

        block = MiniMaxH3TransformerBlock(
            gguf,
            block_index=0,
            device=device,
        )

        output = block.forward(
            text,
            timesteps=timestep_values,
            timestep_indices=timestep_indices,
            token_tags=tags,
            position_ids=positions,
            diagnostics=True,
        )

        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        print("\n========== TEXT TEST RESULT ==========")
        print("Text tokens:", text.shape[0])
        print("Conditioning shape:", tuple(text.shape))
        print("Block output shape:", tuple(output.shape))
        print("Output finite:", bool(torch.isfinite(output).all().item()))
        print(
            f"Peak allocated: "
            f"{torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MiB"
        )
        print(
            f"Peak reserved: "
            f"{torch.cuda.max_memory_reserved(device) / 1024 ** 2:.2f} MiB"
        )
        print(f"Elapsed: {elapsed:.3f} seconds")

        print("\n========== TEXT TEST COMPLETE ==========")


def run_adaln_test():
    require(
        torch.cuda.is_available(),
        "CUDA is unavailable",
    )

    device = torch.device("cuda:0")

    print("\n========== H3 MULTI-TIMESTEP ADALN TEST ==========")

    with GGUFFile(GGUF_PATH) as gguf:
        block = MiniMaxH3TransformerBlock(
            gguf,
            block_index=0,
            device=device,
        )

        timesteps = torch.tensor(
            [0.0, 0.5, 1.0],
            device=device,
            dtype=torch.float32,
        )

        modulation = block.modulation(timesteps)

        require(
            tuple(modulation.shape)
            == (3, MODALITIES, ADALN_GROUPS, HIDDEN),
            f"Unexpected modulation shape: {tuple(modulation.shape)}",
        )

        tags = torch.tensor(
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            device=device,
            dtype=torch.long,
        )

        timestep_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            device=device,
            dtype=torch.long,
        )

        indices = timestep_indices * MODALITIES + tags

        selected = modulation.reshape(
            -1,
            ADALN_GROUPS,
            HIDDEN,
        ).index_select(
            0,
            indices,
        )

        require(
            tuple(selected.shape)
            == (9, ADALN_GROUPS, HIDDEN),
            f"Unexpected selected shape: {tuple(selected.shape)}",
        )

        require(
            bool(torch.isfinite(selected).all().item()),
            "Selected modulation contains NaN/Inf",
        )

        print("Timesteps:", timesteps.tolist())
        print("Modality tags:", tags.tolist())
        print("Timestep indices:", timestep_indices.tolist())
        print("Modulation shape:", tuple(modulation.shape))
        print("Selected shape:", tuple(selected.shape))
        print("All values finite:", bool(torch.isfinite(selected).all().item()))

    print("\n========== ADALN TEST COMPLETE ==========")


def run_block_test(tokens, timestep):
    require(
        torch.cuda.is_available(),
        "CUDA is unavailable",
    )

    device = torch.device("cuda:0")

    properties = torch.cuda.get_device_properties(device)

    print("\n========== MINIMAX H3 REAL BLOCK TEST ==========")
    print("GPU:", properties.name)
    print(f"VRAM: {properties.total_memory / 1024 ** 3:.2f} GiB")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Tokens:", tokens)
    print("Block: 0")
    print("Timestep:", timestep)

    rng = np.random.default_rng(12345)

    with GGUFFile(GGUF_PATH) as gguf:
        block = MiniMaxH3TransformerBlock(
            gguf,
            block_index=0,
            device=device,
        )

        x = torch.from_numpy(
            rng.standard_normal(
                (tokens, HIDDEN)
            ).astype(np.float32)
        ).to(device)

        tags = (
                torch.arange(
                    tokens,
                    device=device,
                ) % MODALITIES
        )

        positions = torch.zeros(
            (tokens, 3),
            device=device,
            dtype=torch.float32,
        )

        positions[:, 0] = torch.arange(
            tokens,
            device=device,
        )

        timestep_values = torch.tensor(
            [timestep],
            device=device,
            dtype=torch.float32,
        )

        timestep_indices = torch.zeros(
            tokens,
            device=device,
            dtype=torch.long,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()

        with torch.inference_mode():
            output = block.forward(
                x,
                timesteps=timestep_values,
                timestep_indices=timestep_indices,
                token_tags=tags,
                position_ids=positions,
            )

            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        print("\n--- RESULT ---")
        print("Output shape:", tuple(output.shape))
        print("Output finite:", bool(torch.isfinite(output).all().item()))
        print("Output mean:", output.mean().item())
        print("Output std:", output.std(unbiased=False).item())

        print(
            "Peak allocated:",
            f"{torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MiB",
        )

        print(
            "Peak reserved:",
            f"{torch.cuda.max_memory_reserved(device) / 1024 ** 2:.2f} MiB",
        )

        print(f"Elapsed: {elapsed:.3f} seconds")

    print("\n========== BLOCK TEST COMPLETE ==========")


@torch.inference_mode()
def run_packed_h3_forward(
        gguf,
        text_states,
        video_latent,
        audio_latent,
        sigma_video,
        device,
):
    print("\n========== H3 PACKED FORWARD ==========")

    require(
        tuple(video_latent.shape[:2]) == (1, 24)
        and video_latent.ndim == 5,
        f"Invalid video latent shape: {tuple(video_latent.shape)}",
    )

    require(
        tuple(audio_latent.shape[:3]) == (1, 32, 2)
        and audio_latent.ndim == 4,
        f"Invalid audio latent shape: {tuple(audio_latent.shape)}",
    )

    require(
        text_states.ndim == 2
        and text_states.shape[1] == 5120,
        f"Invalid text conditioning shape: {tuple(text_states.shape)}",
    )

    latent_t = video_latent.shape[2]
    latent_h = video_latent.shape[3]
    latent_w = video_latent.shape[4]
    audio_t = audio_latent.shape[-1]

    text_length = text_states.shape[0]

    print("Text tokens:", text_length)
    print("Video latent:", tuple(video_latent.shape))
    print("Audio latent:", tuple(audio_latent.shape))
    print("Video sigma:", sigma_video)

    # --------------------------------------------------------
    # Timesteps
    # --------------------------------------------------------

    (
        timesteps,
        timestep_lookup,
        t_video,
        t_audio,
    ) = build_h3_timesteps(
        sigma_video,
        device,
    )

    timestep_map = {
        "video": t_video,
        "audio": t_audio,
    }

    print("Video timestep:", t_video)
    print("Audio timestep:", t_audio)

    # --------------------------------------------------------
    # Packed positions and segments
    # --------------------------------------------------------

    position_ids, segments = build_h3_layout(
        text_length,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
    )

    position_ids = position_ids.to(device)

    sequence = position_ids.shape[0]

    tags, timestep_indices = build_modality_indices(
        segments,
        timestep_map,
        timestep_lookup,
        device,
    )

    require(
        tags.shape == (sequence,),
        "Packed modality tag length mismatch",
    )

    require(
        timestep_indices.shape == (sequence,),
        "Packed timestep index length mismatch",
    )

    print("Packed sequence length:", sequence)
    print("Segments:", segments)

    # --------------------------------------------------------
    # Locate target segments
    # --------------------------------------------------------

    segment_lookup = {
        kind: (start, end)
        for start, end, kind in segments
    }

    text_start, text_end = segment_lookup["text"]
    audio_start, audio_end = segment_lookup["audio"]
    video_start, video_end = segment_lookup["video"]

    require(
        text_end - text_start == text_length,
        "Text segment length mismatch",
    )

    # --------------------------------------------------------
    # Input projections
    # --------------------------------------------------------

    linear = StreamedLinear(
        gguf,
        device=device,
        chunk_rows=WEIGHT_CHUNK_ROWS,
    )

    print("\nPatchifying video...")

    video_rows = patchify_video(
        video_latent.to(
            device=device,
            dtype=torch.float32,
        )
    )

    print("Packing audio...")

    audio_rows = pack_audio(
        audio_latent.to(
            device=device,
            dtype=torch.float32,
        )
    )

    require(
        video_rows.shape[0] == video_end - video_start,
        "Video patch count does not match packed layout",
    )

    require(
        audio_rows.shape[0] == audio_end - audio_start,
        "Audio token count does not match packed layout",
    )

    print("\nProjecting video rows...")

    video_embed = linear(
        "video_patch_proj.weight",
        video_rows,
        bias_name="video_patch_proj.bias",
        output_device=device,
    )

    print_stats(
        "Video embeddings",
        video_embed,
    )

    print("\nProjecting audio rows...")

    audio_embed = linear(
        "audio_patch_proj.weight",
        audio_rows,
        bias_name="audio_patch_proj.bias",
        output_device=device,
    )

    print_stats(
        "Audio embeddings",
        audio_embed,
    )

    # --------------------------------------------------------
    # Text projection and refiner
    # --------------------------------------------------------

    print("\nProjecting text conditioning...")

    text_embed = linear(
        "condition_proj.weight",
        text_states.to(
            device=device,
            dtype=torch.float32,
        ),
        bias_name="condition_proj.bias",
        output_device=device,
    )

    del text_states

    print_stats(
        "Condition projection",
        text_embed,
    )

    refiner = MiniMaxH3TokenRefiner(
        gguf,
        device,
    )

    text_embed = refiner.forward(
        text_embed,
    )

    del refiner

    print_stats(
        "Refined text",
        text_embed,
    )

    # --------------------------------------------------------
    # Assemble packed hidden states
    # [text | audio | video]
    # --------------------------------------------------------

    hidden = torch.empty(
        sequence,
        HIDDEN,
        device=device,
        dtype=torch.float32,
    )

    hidden[text_start:text_end] = text_embed
    hidden[audio_start:audio_end] = audio_embed
    hidden[video_start:video_end] = video_embed

    del text_embed, audio_embed
    del video_rows, audio_rows, linear

    require(
        bool(torch.isfinite(hidden).all().item()),
        "Packed embeddings contain NaN/Inf",
    )

    print_stats(
        "Packed hidden states",
        hidden,
    )

    # --------------------------------------------------------
    # Run all 50 blocks
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for block_index in range(NUM_BLOCKS):
        block = MiniMaxH3TransformerBlock(
            gguf,
            block_index=block_index,
            device=device,
        )

        hidden = block.forward(
            hidden,
            timesteps=timesteps,
            timestep_indices=timestep_indices,
            token_tags=tags,
            position_ids=position_ids,
        )

        del block

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        if block_index == 0 or (block_index + 1) % 5 == 0:
            print(
                f"Completed block {block_index + 1}/{NUM_BLOCKS}"
            )

        if block_index in (9, 24, 49):
            print_stats(
                f"After block {block_index + 1}",
                hidden,
            )

        if block_index % 5 == 4:
            gc.collect()

    transformer_elapsed = time.perf_counter() - start_time

    print(
        f"\nTransformer elapsed: {transformer_elapsed:.2f} seconds"
    )

    # --------------------------------------------------------
    # Final layer normalization and AdaLN
    # --------------------------------------------------------

    print("\nRunning final output layer...")

    final_norm = read_gpu_tensor(
        gguf,
        "final_layer.norm.weight",
        device,
    )

    final_modulation = final_layer_modulation(
        gguf,
        timesteps,
        device,
    )

    video_timestep_index = timestep_lookup[t_video]
    audio_timestep_index = timestep_lookup[t_audio]

    # Official final AdaLN groups: shift, scale.
    video_shift = final_modulation[
        video_timestep_index, 0
    ]

    video_scale = final_modulation[
        video_timestep_index, 1
    ]

    audio_shift = final_modulation[
        audio_timestep_index, 0
    ]

    audio_scale = final_modulation[
        audio_timestep_index, 1
    ]

    # --------------------------------------------------------
    # Video output head
    # --------------------------------------------------------

    video_hidden = rms_norm(
        hidden[video_start:video_end],
        final_norm,
    )

    video_hidden = (
            video_hidden * (1.0 + video_scale)
            + video_shift
    )

    video_linear = StreamedLinear(
        gguf,
        device=device,
        chunk_rows=WEIGHT_CHUNK_ROWS,
    )

    video_prediction_rows = video_linear(
        "final_layer.video_out.weight",
        video_hidden,
        bias_name="final_layer.video_out.bias",
        output_device=device,
    )

    require(
        video_prediction_rows.shape[1] == 96,
        f"Unexpected video output width: {video_prediction_rows.shape[1]}",
    )

    # H3 model returns negative output-head prediction.
    video_prediction_rows = -video_prediction_rows

    video_prediction = unpatchify_video(
        video_prediction_rows,
        latent_t,
        latent_h,
        latent_w,
    )

    print_stats(
        "Video prediction",
        video_prediction,
    )

    del video_hidden, video_prediction_rows, video_linear

    # --------------------------------------------------------
    # Audio output head
    # --------------------------------------------------------

    audio_hidden = rms_norm(
        hidden[audio_start:audio_end],
        final_norm,
    )

    audio_hidden = (
            audio_hidden * (1.0 + audio_scale)
            + audio_shift
    )

    audio_linear = StreamedLinear(
        gguf,
        device=device,
        chunk_rows=WEIGHT_CHUNK_ROWS,
    )

    audio_prediction_rows = audio_linear(
        "final_layer.audio_out.weight",
        audio_hidden,
        bias_name="final_layer.audio_out.bias",
        output_device=device,
    )

    require(
        audio_prediction_rows.shape[1] == 32,
        f"Unexpected audio output width: {audio_prediction_rows.shape[1]}",
    )

    audio_prediction_rows = -audio_prediction_rows

    audio_prediction = unpack_audio(
        audio_prediction_rows
    )

    print_stats(
        "Audio prediction",
        audio_prediction,
    )

    require(
        bool(torch.isfinite(video_prediction).all().item()),
        "Video prediction contains NaN/Inf",
    )

    require(
        bool(torch.isfinite(audio_prediction).all().item()),
        "Audio prediction contains NaN/Inf",
    )

    print("\n========== PACKED FORWARD COMPLETE ==========")

    return video_prediction, audio_prediction


def run_packed_test(
        prompt,
        width,
        height,
        frames,
        sigma,
):
    require(
        torch.cuda.is_available(),
        "CUDA is unavailable",
    )

    require(
        width > 0 and height > 0,
        "Width and height must be positive",
    )

    require(
        width % 32 == 0 and height % 32 == 0,
        "Width and height must be multiples of 32",
    )

    require(
        frames >= 5 and (frames - 5) % 17 == 0,
        "H3 frame count must be 5 + 17*n",
    )

    latent_t = (
        2
        if frames <= 5
        else ((frames - 5) // 17) * 5 + 2
    )

    latent_h = height // 16
    latent_w = width // 16

    require(
        latent_h % 2 == 0 and latent_w % 2 == 0,
        "Video latent dimensions must be divisible by 2",
    )

    # H3 audio latent temporal rate is 40 Hz.
    audio_t = round(
        frames / 24.0 * 40.0
    )

    require(
        audio_t > 0,
        "Invalid audio latent temporal length",
    )

    device = torch.device("cuda:0")

    properties = torch.cuda.get_device_properties(device)

    print("\n========== H3 PACKED AUDIO/VIDEO TEST ==========")
    print("GPU:", properties.name)
    print(f"VRAM: {properties.total_memory / 1024 ** 3:.2f} GiB")
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Resolution:", f"{width}x{height}")
    print("Frames:", frames)
    print("Video latent T:", latent_t)
    print("Video latent H/W:", latent_h, latent_w)
    print("Audio latent T:", audio_t)
    print("Sigma:", sigma)
    print("Prompt:", prompt)

    # Qwen completes before H3 transformer allocations.
    conditioning = extract_qwen_conditioning(prompt)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(
        device=device
    ).manual_seed(12345)

    # Random starting latent tensors for a forward-pass test.
    video_noise = torch.randn(
        (1, 24, latent_t, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    audio_noise = torch.randn(
        (1, 32, 2, audio_t),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    start = time.perf_counter()

    with GGUFFile(GGUF_PATH) as gguf:
        video_prediction, audio_prediction = (
            run_packed_h3_forward(
                gguf=gguf,
                text_states=conditioning,
                video_latent=video_noise,
                audio_latent=audio_noise,
                sigma_video=sigma,
                device=device,
            )
        )

    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    print("\n========== PACKED TEST RESULT ==========")

    print(
        "Video prediction:",
        tuple(video_prediction.shape),
    )

    print(
        "Audio prediction:",
        tuple(audio_prediction.shape),
    )

    print(
        "Video finite:",
        bool(torch.isfinite(video_prediction).all().item()),
    )

    print(
        "Audio finite:",
        bool(torch.isfinite(audio_prediction).all().item()),
    )

    print(
        "Peak allocated:",
        f"{torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MiB",
    )

    print(
        "Peak reserved:",
        f"{torch.cuda.max_memory_reserved(device) / 1024 ** 2:.2f} MiB",
    )

    print(f"Elapsed: {elapsed:.2f} seconds")

    print("\nThis was one model forward pass.")
    print("No denoising sampler was run.")
    print("No VAE decoding was run.")
    print("Audio waveform decoding is not required for this test.")

    print("\n========== PACKED TEST COMPLETE ==========")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone streamed MiniMax H3 inference"
    )

    modes = parser.add_mutually_exclusive_group(
        required=True
    )

    modes.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic block-0 test",
    )

    modes.add_argument(
        "--text-test",
        action="store_true",
        help="Run Qwen -> ClipProj -> H3 text refiner -> block 0",
    )

    modes.add_argument(
        "--adaln-test",
        action="store_true",
        help="Test multi-timestep AdaLN selection",
    )

    modes.add_argument(
        "--packed-test",
        action="store_true",
        help="Run one joint audio/video forward pass through all 50 blocks",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a red car driving through a rainy city at night.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if (
            not math.isfinite(args.timestep)
            or not 0.0 <= args.timestep <= 1.0
    ):
        parser.error(
            "--timestep must be finite and in [0,1]"
        )

    if (
            not math.isfinite(args.sigma)
            or not 0.0 < args.sigma <= 1.0
    ):
        parser.error(
            "--sigma must be finite and in (0,1]"
        )

    if args.self_test:
        if args.tokens < 1:
            parser.error("--tokens must be positive")

        run_block_test(
            args.tokens,
            args.timestep,
        )

    elif args.text_test:
        run_text_test(
            args.prompt,
            args.timestep,
        )

    elif args.adaln_test:
        run_adaln_test()

    elif args.packed_test:
        run_packed_test(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            frames=args.frames,
            sigma=args.sigma,
        )


if __name__ == "__main__":
    main()
