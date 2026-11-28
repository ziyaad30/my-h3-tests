"""End-to-end MiniMax H3 LightVAE-Lite video VAE round-trip test.

Runs the standalone encoder and decoder on the same five-frame clip:
RGB [1,3,5,H,W] -> normalized latents [1,24,2,H/16,W/16] -> RGB [1,3,5,H,W].

By default this uses a deterministic synthetic test chart. Set INPUT_FRAME_PATHS
to exactly five image paths to test a real five-frame sample instead.
"""

import gc
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from minimax_h3_lightvae_encoder import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEVICE,
    load_keyframe_image,
    load_streaming_encoder,
)
from minimax_h3_lightvae_decoder import decode_video_latents


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "lightvae_roundtrip_test"

# Supply exactly five image paths, or leave empty for the synthetic chart.
INPUT_FRAME_PATHS = []

EXPECTED_INPUT_FRAMES = 5


def make_synthetic_clip():
    """Create a deterministic spatial test chart, repeated for five frames."""
    x = torch.linspace(0.0, 1.0, CANVAS_WIDTH).view(1, 1, 1, CANVAS_WIDTH)
    y = torch.linspace(0.0, 1.0, CANVAS_HEIGHT).view(1, 1, CANVAS_HEIGHT, 1)

    red = x.expand(1, 1, CANVAS_HEIGHT, CANVAS_WIDTH)
    green = y.expand(1, 1, CANVAS_HEIGHT, CANVAS_WIDTH)
    blue = (((x * 8).floor() % 2) * 0.7 + y * 0.3).expand(
        1, 1, CANVAS_HEIGHT, CANVAS_WIDTH
    )

    chart = torch.cat((red, green, blue), dim=1) * 2.0 - 1.0

    # Add hard-edged regions and thin lines to make spatial blending defects visible.
    chart[:, :, 48:112, 48:112] = torch.tensor(
        [[[[-1.0]], [[1.0]], [[-1.0]]]]
    )
    chart[:, :, 208:272, 208:272] = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]]]]
    )
    chart[:, :, 159:161, :] = 0.0
    chart[:, :, :, 159:161] = 0.0

    return chart.unsqueeze(2).repeat(1, 1, EXPECTED_INPUT_FRAMES, 1, 1).contiguous()


def load_input_clip():
    if not INPUT_FRAME_PATHS:
        print("Input: deterministic synthetic test chart")
        return make_synthetic_clip()

    if len(INPUT_FRAME_PATHS) != EXPECTED_INPUT_FRAMES:
        raise ValueError(
            f"INPUT_FRAME_PATHS must contain exactly {EXPECTED_INPUT_FRAMES} paths."
        )

    frames = [
        load_keyframe_image(path, CANVAS_WIDTH, CANVAS_HEIGHT, "center")
        for path in INPUT_FRAME_PATHS
    ]
    print("Input frames:")
    for path in INPUT_FRAME_PATHS:
        print("  ", Path(path))
    return torch.cat(frames, dim=2).contiguous()


def save_input_frames(video):
    input_dir = OUTPUT_DIR / "input_frames"
    input_dir.mkdir(parents=True, exist_ok=True)

    cpu = video[0].detach().float().cpu()
    for index in range(cpu.shape[1]):
        rgb = ((cpu[:, index].clamp(-1, 1) + 1.0) * 127.5).round()
        array = rgb.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(array, mode="RGB").save(
            input_dir / f"input_{index + 1:04d}.png"
        )
    return input_dir


def report_roundtrip_metrics(video, decoded_frames_dir):
    source = video[0].detach().float().cpu()
    rows = []

    for index in range(EXPECTED_INPUT_FRAMES):
        output_path = decoded_frames_dir / f"frame_{index + 1:04d}.png"
        if not output_path.is_file():
            raise FileNotFoundError(f"Decoder did not export expected frame: {output_path}")

        with Image.open(output_path) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0

        original = source[:, index].permute(1, 2, 0).numpy()
        difference = decoded - original
        mae = float(np.mean(np.abs(difference)))
        mse = float(np.mean(np.square(difference)))
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse)
        rows.append((index + 1, mae, psnr))

    print("\nPer-frame reconstruction metrics (RGB values normalized to [-1,1]):")
    for index, mae, psnr in rows:
        print(f"  Frame {index}: MAE={mae:.6f} | PSNR={psnr:.2f} dB")

    mean_mae = sum(row[1] for row in rows) / len(rows)
    finite_psnr = [row[2] for row in rows if math.isfinite(row[2])]
    mean_psnr = (
        sum(finite_psnr) / len(finite_psnr)
        if finite_psnr
        else math.inf
    )
    print(f"  Mean: MAE={mean_mae:.6f} | PSNR={mean_psnr:.2f} dB")
    print("Note: these metrics quantify pixel error; they do not by themselves diagnose tile seams.")


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; this test uses the streaming CUDA implementations.")

    if CANVAS_WIDTH % 32 or CANVAS_HEIGHT % 32:
        raise ValueError("Canvas dimensions must be divisible by 32.")

    video = load_input_clip()
    expected_input_shape = (1, 3, EXPECTED_INPUT_FRAMES, CANVAS_HEIGHT, CANVAS_WIDTH)
    if tuple(video.shape) != expected_input_shape:
        raise RuntimeError(
            f"Input shape {tuple(video.shape)}; expected {expected_input_shape}"
        )
    if not torch.isfinite(video).all():
        raise ValueError("Input clip contains NaN or Inf.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = save_input_frames(video)

    print("\n========== H3 LIGHTVAE-LITE ROUND TRIP ==========")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Input clip:", tuple(video.shape))
    print("Input range:", float(video.min()), float(video.max()))
    print("Output directory:", OUTPUT_DIR.resolve())

    print("\nLoading standalone LightVAE-Lite encoder...")
    encoder = load_streaming_encoder()

    print("\nEncoding five input frames...")
    latents = encoder(video)
    torch.cuda.synchronize(DEVICE)

    expected_latent_shape = (
        1,
        24,
        2,
        CANVAS_HEIGHT // 16,
        CANVAS_WIDTH // 16,
    )
    if tuple(latents.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Encoder latent shape {tuple(latents.shape)}; expected {expected_latent_shape}"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError("Encoder produced NaN or Inf latents.")

    print("Encoded latent shape:", tuple(latents.shape))
    print("Latent range:", float(latents.min()), float(latents.max()))

    # Release the encoder before loading the decoder to minimize host/GPU memory use.
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    latent_path = OUTPUT_DIR / "roundtrip_latents.pt"
    torch.save(latents.detach().cpu(), latent_path)
    print("Saved normalized latents:", latent_path)

    print("\nDecoding those exact latents with the standalone LightVAE-Lite decoder...")
    result = decode_video_latents(
        latents=latents,
        output_dir=OUTPUT_DIR / "decoded",
        fps=24,
    )
    torch.cuda.synchronize(DEVICE)

    if result["frame_count"] != EXPECTED_INPUT_FRAMES:
        raise RuntimeError(
            f"Decoder returned {result['frame_count']} frames; expected {EXPECTED_INPUT_FRAMES}"
        )

    report_roundtrip_metrics(video, result["frames_dir"])

    print("\n========== ROUND TRIP TEST COMPLETE ==========")
    print("Input frames:", input_dir.resolve())
    print("Decoded frames:", result["frames_dir"].resolve())
    if result["mp4_path"] is not None:
        print("Decoded MP4:", result["mp4_path"].resolve())
    print("Latents:", latent_path.resolve())

    del latents, video
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
