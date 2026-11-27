
import gc
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from lightvae_h3_standalone.lightvae.h3_lightvae_encoder import Encoder
from lightvae_h3_standalone.lightvae.utils import (
    latent_frames,
    padded_frames,
)


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\vae"
    r"\lightvae-lite-h3-encoder.safetensors"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16

# Initial encoder feasibility test.
INPUT_SHAPE = (1, 3, 5, 512, 512)

# Smaller spatial tiles to reduce Conv3D activation memory.
TILE_SIZE = 128
TILE_OVERLAP = 64


# ============================================================
# MEMORY-MANAGED ENCODER CLIP FORWARD
# ============================================================

def streaming_encode_clip(self, video):

    # Official H3 RGB normalization.
    mean = video.new_tensor(
        (0.485, 0.456, 0.406)
    ).view(1, 3, 1, 1, 1)

    std = video.new_tensor(
        (0.229, 0.224, 0.225)
    ).view(1, 3, 1, 1, 1)

    video = ((video + 1) / 2 - mean) / std

    hidden_states = video.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    model = self.encoder

    # Input convolution.
    model.conv_in.to(device=DEVICE, dtype=DTYPE)

    hidden_states = model.conv_in(hidden_states)

    model.conv_in.to(device="cpu")

    # Process each down block one module at a time.
    for stage_index, down_block in enumerate(model.down_blocks):

        for resnet_index, resnet in enumerate(down_block.resnets):

            print(
                f"\rEncoder stage {stage_index + 1}/"
                f"{len(model.down_blocks)} "
                f"resnet {resnet_index + 1}/"
                f"{len(down_block.resnets)}",
                end="",
                flush=True,
            )

            resnet.to(device=DEVICE, dtype=DTYPE)

            hidden_states = resnet(hidden_states)

            resnet.to(device="cpu")

        # Downsampling convolution.
        if down_block.downsamplers is not None:

            for downsampler in down_block.downsamplers:

                downsampler.to(
                    device=DEVICE,
                    dtype=DTYPE,
                )

                hidden_states = downsampler(hidden_states)

                downsampler.to(device="cpu")

    print()

    # Output normalization and activation.
    model.norm_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = F.silu(
        model.norm_out(hidden_states)
    )

    model.norm_out.to(device="cpu")

    # Output convolution.
    model.conv_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = model.conv_out(hidden_states)

    model.conv_out.to(device="cpu")

    # Official quantization convolution.
    self.quant_conv.to(device=DEVICE, dtype=DTYPE)

    moments = self.quant_conv(hidden_states)

    self.quant_conv.to(device="cpu")

    # Return CPU moments for the official tiling/stitching code.
    return moments.to(device="cpu")


# ============================================================
# GPU MEMORY REPORTING
# ============================================================

def report_gpu_memory(label):

    torch.cuda.synchronize(DEVICE)

    allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)

    peak_allocated = (
        torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 2)
    )

    print(f"\n{label}")
    print("-" * 48)
    print(f"Current allocated: {allocated:.2f} MB")
    print(f"Current reserved:  {reserved:.2f} MB")
    print(f"Peak allocated:    {peak_allocated:.2f} MB")
    print(f"Peak reserved:     {peak_reserved:.2f} MB")


# ============================================================
# MAIN
# ============================================================

def main():

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    SOURCE = (
        ROOT
        / "lightvae_h3_standalone"
        / "lightvae"
        / "h3_lightvae_encoder.py"
    )

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Official LightVAE encoder source not found: {SOURCE}"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Checkpoint:", CHECKPOINT)

    print("\nLoading official H3 Lite encoder...")

    # Load official checkpoint-aware Lite encoder.
    encoder = Encoder(checkpoint=CHECKPOINT)

    encoder.eval()
    encoder.requires_grad_(False)

    # Keep model weights in CPU RAM as FP16.
    # H3Module preserves FP32 latent normalization buffers.
    encoder.to(device="cpu", dtype=DTYPE)

    # Reduce spatial tile size to limit Conv3D activation memory.
    # The official encoder.forward() handles tile processing
    # and stitching.
    encoder.tile_size = TILE_SIZE
    encoder.tile_overlap = TILE_OVERLAP

    print(
        "Retained residual indices:",
        encoder.config["kept_residual_indices"],
    )

    print("Tile size:", encoder.tile_size)
    print("Tile overlap:", encoder.tile_overlap)

    # Replace only the clip execution method.
    # The official encoder.forward() retains its temporal
    # padding, spatial tiling, stitching and latent normalization.
    encoder._encode_clip = types.MethodType(
        streaming_encode_clip,
        encoder,
    )

    # Synthetic RGB video in CPU RAM.
    torch.manual_seed(42)

    video = torch.rand(
        INPUT_SHAPE,
        dtype=torch.float32,
        device="cpu",
    ) * 2.0 - 1.0

    print("\nSynthetic video shape:", tuple(video.shape))
    print("Video dtype:", video.dtype)
    print(
        "Video value range:",
        video.min().item(),
        video.max().item(),
    )

    # Calculate expected temporal output using official helpers.
    padded_frame_count = padded_frames(
        "h3",
        INPUT_SHAPE[2],
    )

    expected_latent_frames = latent_frames(
        "h3",
        padded_frame_count,
    )

    expected_shape = (
        INPUT_SHAPE[0],
        24,
        expected_latent_frames,
        INPUT_SHAPE[3] // 16,
        INPUT_SHAPE[4] // 16,
    )

    print("\nExpected latent shape:", expected_shape)

    # Clear previous CUDA allocations before measuring.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)

    torch.cuda.reset_peak_memory_stats(DEVICE)

    print("\nGPU memory before encoding:")
    print(
        "Allocated:",
        f"{torch.cuda.memory_allocated(DEVICE) / (1024 ** 2):.2f} MB",
    )
    print(
        "Reserved:",
        f"{torch.cuda.memory_reserved(DEVICE) / (1024 ** 2):.2f} MB",
    )

    print("\nStarting official H3 Lite encoder...")

    with torch.inference_mode():

        latents = encoder(video)

        torch.cuda.synchronize(DEVICE)

        print("\nEncoded latent shape:", tuple(latents.shape))
        print("Encoded latent dtype:", latents.dtype)

        print(
            "Latent value range:",
            latents.min().item(),
            latents.max().item(),
        )

        if tuple(latents.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected latent shape: {tuple(latents.shape)}; "
                f"expected {expected_shape}"
            )

        report_gpu_memory("GPU memory after encoding")

    print("\nEncoder execution completed successfully.")

    del latents, video, encoder

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
