import gc
import types
from pathlib import Path

import torch
from PIL import Image

from lightvae_h3_standalone.lightvae.h3_lightvae_decoder import Decoder
from lightvae_h3_standalone.lightvae.utils import h3_decoded_frames


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\vae"
    r"\lightvae-lite-h3-decoder.safetensors"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16

# Seven latent frames.
# 64 x 64 latent -> 1024 x 1024 pixels.
LATENT_SHAPE = (1, 24, 7, 64, 64)


# ============================================================
# MEMORY-MANAGED OFFLOADED TRANSFORMER FORWARD
# ============================================================

def streaming_forward(self, hidden_states):

    batch, channels, frames, height, width = hidden_states.shape

    # BCTHW -> B(sequence)C
    hidden_states = hidden_states.permute(
        0, 2, 3, 4, 1
    ).reshape(
        batch, frames * height * width, channels
    )

    # Input projection
    self.proj_in.to(device=DEVICE, dtype=DTYPE)

    hidden_states = self.proj_in(
        hidden_states.to(device=DEVICE, dtype=DTYPE)
    )

    self.proj_in.to(device="cpu")

    num_patches = hidden_states.shape[1]

    # Register tokens and zero class token
    register_tokens = self.register_tokens.to(
        device=DEVICE,
        dtype=DTYPE,
    ).expand(batch, -1, -1)

    cls_token = torch.zeros_like(hidden_states[:, :1, :])

    hidden_states = torch.cat(
        [hidden_states, register_tokens, cls_token],
        dim=1,
    )

    del register_tokens, cls_token

    # Official H3 three-axis position coordinates
    grids = [
        2.0 * (
            torch.arange(
                0.5,
                size,
                dtype=torch.float32,
                device=DEVICE,
            ) / size
        ) - 1.0
        for size in (frames, height, width)
    ]

    position_ids = torch.stack(
        torch.meshgrid(*grids, indexing="ij"),
        dim=-1,
    ).flatten(0, 2)

    position_ids = position_ids.unsqueeze(0).expand(
        batch, -1, -1
    )

    suffix_ids = position_ids.new_zeros(
        (batch, self.num_register_tokens + 1, 3)
    )

    position_ids = torch.cat(
        [position_ids, suffix_ids],
        dim=1,
    )

    # Official rotary embedding
    self.rope.to(device=DEVICE)

    rotary_emb = self.rope(position_ids)

    del grids, position_ids, suffix_ids

    self.rope.to(device="cpu")

    # Sequential transformer blocks.
    # RMSNorm weights remain FP32.
    total_blocks = len(self.transformer_blocks)

    for index, block in enumerate(self.transformer_blocks):

        print(
            f"\rDecoder block {index + 1}/{total_blocks}",
            end="",
            flush=True,
        )

        block.to(device=DEVICE, dtype=DTYPE)

        block.norm1.to(device=DEVICE, dtype=torch.float32)
        block.norm2.to(device=DEVICE, dtype=torch.float32)

        hidden_states = block(
            hidden_states,
            rotary_emb,
        )

        block.to(device="cpu")

    print()

    del rotary_emb

    # Output normalization
    self.norm_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = self.norm_out(hidden_states)

    self.norm_out.to(device="cpu")

    # Output projection
    self.proj_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = self.proj_out(hidden_states)

    self.proj_out.to(device="cpu")

    # Remove register and class tokens.
    hidden_states = hidden_states[:, :num_patches, :]

    # Official H3 unpatchification
    hidden_states = hidden_states.view(
        batch,
        frames,
        height,
        width,
        self.out_channels,
        self.patch_size_t,
        self.patch_size,
        self.patch_size,
    )

    hidden_states = hidden_states.permute(
        0, 4, 1, 5, 2, 6, 3, 7
    ).contiguous()

    return hidden_states.reshape(
        batch,
        self.out_channels,
        frames * self.patch_size_t,
        height * self.patch_size,
        width * self.patch_size,
    )


# ============================================================
# MEMORY-MANAGED CLIP DECODING
# ============================================================

def streaming_decode_clip(self, latents):

    mean = self.latents_mean.float().view(
        1, -1, 1, 1, 1
    )

    std = self.latents_std.float().view(
        1, -1, 1, 1, 1
    )

    # Restore official H3 latent distribution.
    latents = (
        latents.float() * std + mean
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # Post-quant convolution
    self.post_quant_conv.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    latents = self.post_quant_conv(latents)

    self.post_quant_conv.to(device="cpu")

    return self.decoder(latents)


# ============================================================
# GPU MEMORY REPORTING
# ============================================================

def report_gpu_memory(label):

    torch.cuda.synchronize(DEVICE)

    allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(DEVICE) / (1024 ** 2)

    print(f"\n{label}")
    print("-" * 48)
    print(f"Current allocated: {allocated:.2f} MB")
    print(f"Current reserved:  {reserved:.2f} MB")
    print(f"Peak allocated:    {peak_allocated:.2f} MB")
    print(f"Peak reserved:     {peak_reserved:.2f} MB")

    return peak_allocated, peak_reserved


# ============================================================
# MAIN
# ============================================================



# ============================================================
# REUSABLE VIDEO-LATENT DECODING + EXPORT
# ============================================================

@torch.inference_mode()
def decode_video_latents(latents, output_dir, checkpoint=None, fps=24):
    """Decode normalized [1,24,T,H,W] latents and export PNG frames + MP4."""
    output_dir = Path(output_dir)
    checkpoint = Path(checkpoint) if checkpoint is not None else CHECKPOINT
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Decoder checkpoint not found: {checkpoint}")
    if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != 24:
        raise ValueError(f"Expected [1,24,T,H,W] video latent, got {tuple(latents.shape)}")
    if not torch.isfinite(latents).all():
        raise ValueError("Video latents contain NaN or Inf.")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")

    expected_frames = h3_decoded_frames(int(latents.shape[2]))
    source = ROOT / "lightvae_h3_standalone" / "lightvae" / "h3_lightvae_decoder.py"
    if not source.is_file():
        raise FileNotFoundError(f"LightVAE decoder source not found: {source}")

    print("\n========== LIGHTVAE-LITE VIDEO DECODING ==========", flush=True)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Decoder checkpoint:", checkpoint)
    print("Input latent:", tuple(latents.shape))
    print("Expected decoded frames:", expected_frames)
    print("Output directory:", output_dir.resolve())

    decoder = Decoder(checkpoint=checkpoint)
    decoder.eval()
    decoder.requires_grad_(False)
    decoder.to(device="cpu", dtype=DTYPE)
    decoder.decoder.rope.materialize_buffer("cpu")
    decoder.decoder.forward = types.MethodType(streaming_forward, decoder.decoder)
    decoder._decode_clip = types.MethodType(streaming_decode_clip, decoder)

    print("Retained decoder blocks:", decoder.config["kept_layers"])
    print("Spatial tile size:", decoder.tile_size)
    print("Spatial tile overlap:", decoder.tile_overlap)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    print("\nStarting LightVAE-Lite decode...", flush=True)
    video = decoder(latents.detach().to(device="cpu", dtype=torch.float32))
    torch.cuda.synchronize(DEVICE)

    expected_shape = (1, 3, expected_frames, int(latents.shape[-2]) * 16, int(latents.shape[-1]) * 16)
    if tuple(video.shape) != expected_shape:
        raise RuntimeError(f"Decoded shape {tuple(video.shape)}; expected {expected_shape}")
    if not torch.isfinite(video).all():
        raise RuntimeError("Decoded video contains NaN or Inf.")

    print("Decoded video shape:", tuple(video.shape))
    print("Decoded value range:", float(video.min()), float(video.max()))

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_cpu = video[0].detach().float().cpu()

    for frame_index in range(video_cpu.shape[1]):
        rgb = ((video_cpu[:, frame_index].clamp(-1, 1) + 1) * 127.5).round()
        rgb = rgb.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(rgb, mode="RGB").save(frames_dir / f"frame_{frame_index + 1:04d}.png")
        print(f"Saved frame {frame_index + 1}/{video_cpu.shape[1]}", flush=True)
        del rgb

    mp4_path = output_dir / "h3_generated.mp4"
    mp4_written = False
    try:
        import cv2
        height, width = int(video_cpu.shape[-2]), int(video_cpu.shape[-1])
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if writer.isOpened():
            for frame_index in range(video_cpu.shape[1]):
                rgb = ((video_cpu[:, frame_index].clamp(-1, 1) + 1) * 127.5).round()
                rgb = rgb.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                del rgb
            writer.release()
            mp4_written = mp4_path.is_file() and mp4_path.stat().st_size > 0
        else:
            writer.release()
            print("OpenCV MP4 writer unavailable; trying imageio.")
    except ImportError:
        print("OpenCV unavailable; trying imageio.")

    if not mp4_written:
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(str(mp4_path), fps=float(fps), codec="libx264") as writer:
                for frame_index in range(video_cpu.shape[1]):
                    rgb = ((video_cpu[:, frame_index].clamp(-1, 1) + 1) * 127.5).round()
                    rgb = rgb.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
                    writer.append_data(rgb)
                    del rgb
                mp4_written = mp4_path.is_file() and mp4_path.stat().st_size > 0
        except ImportError:
            print("Neither OpenCV nor imageio is installed; PNG frames are available.")

    print("\nLightVAE-Lite decoding complete.")
    print("PNG frames:", frames_dir.resolve())
    if mp4_written:
        print("MP4 video:", mp4_path.resolve())
    else:
        print("MP4 was not created; PNG frames were saved.")
    report_gpu_memory("GPU memory after decoding")

    del video_cpu, video, decoder
    gc.collect()
    torch.cuda.empty_cache()
    return {"frames_dir": frames_dir, "mp4_path": mp4_path if mp4_written else None, "frame_count": expected_frames}

def main():

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Decoder checkpoint not found: {CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    SOURCE = (
        ROOT
        / "lightvae_h3_standalone"
        / "lightvae"
        / "h3_lightvae_decoder.py"
    )

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Official LightVAE source not found: {SOURCE}"
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Checkpoint:", CHECKPOINT)
    print("Loading official H3 Lite decoder...")

    # Load official checkpoint-aware decoder.
    decoder = Decoder(checkpoint=CHECKPOINT)

    decoder.eval()
    decoder.requires_grad_(False)

    # Keep model weights in CPU RAM as FP16.
    decoder.to(device="cpu", dtype=DTYPE)

    # Restore FP32 rotary frequency buffer.
    decoder.decoder.rope.materialize_buffer("cpu")

    # Replace execution methods only.
    decoder.decoder.forward = types.MethodType(
        streaming_forward,
        decoder.decoder,
    )

    decoder._decode_clip = types.MethodType(
        streaming_decode_clip,
        decoder,
    )

    print("Retained blocks:", decoder.config["kept_layers"])
    print("Tile size:", decoder.tile_size)
    print("Tile overlap:", decoder.tile_overlap)

    # Synthetic latent input in CPU RAM.
    torch.manual_seed(42)

    latents = torch.randn(
        LATENT_SHAPE,
        dtype=torch.float32,
        device="cpu",
    )

    print("\nSynthetic latent shape:", tuple(latents.shape))
    print("Latent dtype:", latents.dtype)

    # Clear previous CUDA allocations before measuring.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)

    torch.cuda.reset_peak_memory_stats(DEVICE)

    print("\nGPU memory before decode:")
    print(
        "Allocated:",
        f"{torch.cuda.memory_allocated(DEVICE) / (1024 ** 2):.2f} MB",
    )
    print(
        "Reserved:",
        f"{torch.cuda.memory_reserved(DEVICE) / (1024 ** 2):.2f} MB",
    )

    print("\nStarting official H3 temporal decoder...")

    with torch.inference_mode():

        video = decoder(latents)

        torch.cuda.synchronize(DEVICE)

        print("\nDecoded video shape:", tuple(video.shape))
        print("Decoded video dtype:", video.dtype)

        print(
            "Value range:",
            video.min().item(),
            video.max().item(),
        )

        # Calculate expected temporal output using
        # the official H3 frame-count function.
        expected_frames = h3_decoded_frames(LATENT_SHAPE[2])

        expected_shape = (
            1,
            3,
            expected_frames,
            LATENT_SHAPE[3] * 16,
            LATENT_SHAPE[4] * 16,
        )

        print("Expected output shape:", expected_shape)

        if tuple(video.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected output shape: {tuple(video.shape)}; "
                f"expected {expected_shape}"
            )

        report_gpu_memory("GPU memory after decoding")

    print("\nDecoder execution completed successfully.")

    del video, latents, decoder

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()