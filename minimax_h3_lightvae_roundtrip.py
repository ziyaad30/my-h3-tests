"""MiniMax H3 LightVAE-Lite image/video round-trip test.

One source image is resized to the requested canvas and repeated across the
requested frame length. The encoder creates H3 video latents; the decoder
reconstructs one image or a video. Batch items are processed sequentially to
keep GPU memory bounded. Because there is only one source image, batch items
are identical copies, not independently generated samples.
"""

import gc
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from minimax_h3_lightvae_encoder import (
    DEVICE,
    encode_keyframe_latent,
    load_streaming_encoder,
)
from minimax_h3_lightvae_decoder import (
    decode_single_image_latent,
    decode_video_latents,
)


# ============================================================
# TEST SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
INPUT_IMAGE_PATH = ROOT / "input.png"
OUTPUT_DIR = ROOT / "lightvae_roundtrip_test"

# ComfyUI-style generation dimensions.
WIDTH = 320
HEIGHT = 320

# H3 accepts one still frame, or video lengths following 17*n + 5.
# 5 is the shortest valid H3 video; 22 gives a more useful temporal test.
LENGTH = 22

# Process one item at a time by default to protect 4 GB VRAM.
BATCH_SIZE = 1
FPS = 24


def validate_settings():
    if WIDTH <= 0 or HEIGHT <= 0:
        raise ValueError("WIDTH and HEIGHT must be positive.")
    if WIDTH % 16 or HEIGHT % 16:
        raise ValueError(
            f"H3 VAE width and height must be divisible by 16; got {WIDTH}x{HEIGHT}."
        )
    if LENGTH != 1 and (LENGTH < 5 or (LENGTH - 5) % 17):
        raise ValueError(
            f"H3 LENGTH must be 1 for a still image, or 5 + 17*n for video; got {LENGTH}."
        )
    if BATCH_SIZE < 1:
        raise ValueError(f"BATCH_SIZE must be at least 1; got {BATCH_SIZE}.")
    if FPS <= 0:
        raise ValueError(f"FPS must be positive; got {FPS}.")


def load_source_image():
    if not INPUT_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE_PATH}\n"
            "Place one image named input.png beside this script."
        )

    with Image.open(INPUT_IMAGE_PATH) as source:
        image = source.convert("RGB")

    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return image, tensor.contiguous()


def reconstruction_metrics(source_image, image_paths):
    source = np.asarray(source_image, dtype=np.float32) / 127.5 - 1.0
    errors = []
    squared_errors = []

    for path in image_paths:
        with Image.open(path) as decoded_file:
            decoded_image = decoded_file.convert("RGB")
            if decoded_image.size != source_image.size:
                raise RuntimeError(
                    f"Decoded size {decoded_image.size} does not match "
                    f"requested size {source_image.size}: {path}"
                )
            decoded = np.asarray(decoded_image, dtype=np.float32) / 127.5 - 1.0

        difference = decoded - source
        errors.append(float(np.mean(np.abs(difference))))
        squared_errors.append(float(np.mean(np.square(difference))))

    mae = float(np.mean(errors))
    mse = float(np.mean(squared_errors))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse)
    print("\nReconstruction metrics against the repeated source image:")
    print(f"  Frames measured: {len(image_paths)}")
    print(f"  Mean frame MAE:  {mae:.6f}")
    print(f"  Mean frame PSNR: {psnr:.2f} dB")
    print("  Note: this is a static-image-in/video-out VAE test, not a motion test.")


@torch.inference_mode()
def main():
    validate_settings()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; this test uses the streamed CUDA implementations.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_image, still_tensor = load_source_image()

    print("\n========== H3 LIGHTVAE-LITE ROUND TRIP ==========")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Input image:", INPUT_IMAGE_PATH.resolve())
    print("Requested width:", WIDTH)
    print("Requested height:", HEIGHT)
    print("Requested length:", LENGTH)
    print("Batch size:", BATCH_SIZE)
    print("FPS:", FPS)
    print("Batch processing: sequential (one item at a time)")
    print("Output directory:", OUTPUT_DIR.resolve())

    # A single source image is repeated over time. For a still test (LENGTH=1),
    # use the dedicated single-frame encoder path. For video, use the native
    # H3 17-frame-window encoder path.
    encoder = load_streaming_encoder()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    print("\nEncoding source image sequence...")
    if LENGTH == 1:
        latents = encode_keyframe_latent(encoder, still_tensor)
    else:
        video = still_tensor.expand(-1, -1, LENGTH, -1, -1).contiguous()
        print("Encoder input tensor:", tuple(video.shape))
        latents = encoder(video)
        del video

    torch.cuda.synchronize(DEVICE)

    expected_latent_frames = 1 if LENGTH == 1 else ((LENGTH - 5) // 17) * 5 + 2
    expected_latent_shape = (
        1, 24, expected_latent_frames, HEIGHT // 16, WIDTH // 16
    )
    if tuple(latents.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Encoder latent shape {tuple(latents.shape)}; "
            f"expected {expected_latent_shape}"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError("Encoder produced NaN or Inf latents.")

    print("Encoded latent shape:", tuple(latents.shape))
    print("Latent range:", float(latents.min()), float(latents.max()))

    latent_path = OUTPUT_DIR / "video_latent.pt"
    torch.save(latents.detach().cpu(), latent_path)
    print("Saved latent:", latent_path.resolve())

    # Release the encoder before loading the decoder.
    del encoder, still_tensor
    gc.collect()
    torch.cuda.empty_cache()

    all_frame_paths = []
    for batch_index in range(BATCH_SIZE):
        item_dir = OUTPUT_DIR / f"batch_{batch_index + 1:04d}"
        print(
            f"\n========== DECODING BATCH ITEM "
            f"{batch_index + 1}/{BATCH_SIZE} ==========",
            flush=True,
        )

        # Batch entries are copies of the same source-image sequence. Decode
        # one at a time instead of increasing the GPU activation footprint.
        if LENGTH == 1:
            output_path = item_dir / "reconstruction.png"
            decode_single_image_latent(
                latents=latents,
                output_path=output_path,
            )
            all_frame_paths.append(output_path)
            print("Saved image:", output_path.resolve())
        else:
            result = decode_video_latents(
                latents=latents,
                output_dir=item_dir,
                fps=FPS,
            )
            frames_dir = result["frames_dir"]
            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            if len(frame_paths) != LENGTH:
                raise RuntimeError(
                    f"Decoder saved {len(frame_paths)} frames; expected {LENGTH}."
                )
            all_frame_paths.extend(frame_paths)
            print("Saved video frames:", len(frame_paths))
            if result["mp4_path"] is not None:
                print("Saved MP4:", result["mp4_path"].resolve())

    reconstruction_metrics(source_image, all_frame_paths)

    print("\n========== ROUND TRIP COMPLETE ==========")
    print("Input image:", INPUT_IMAGE_PATH.resolve())
    print("Latent:", latent_path.resolve())
    print("Batch items:", BATCH_SIZE)
    print("Frames per item:", LENGTH)
    print("Total reconstructed frames:", len(all_frame_paths))
    print("Output directory:", OUTPUT_DIR.resolve())

    del latents, source_image
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
