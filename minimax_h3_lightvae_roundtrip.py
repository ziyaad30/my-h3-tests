"""Single-image MiniMax H3 LightVAE-Lite round-trip test.

Matches the ComfyUI still-image workflow:
one RGB image -> one [1,24,1,H/16,W/16] latent -> one RGB image.

Place an image named input.png beside this script, or change INPUT_IMAGE_PATH.
The image dimensions are preserved and must be divisible by 16.
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
from minimax_h3_lightvae_decoder import decode_single_image_latent


ROOT = Path(__file__).resolve().parent
INPUT_IMAGE_PATH = ROOT / "input.png"
OUTPUT_DIR = ROOT / "lightvae_roundtrip_test"
LATENT_PATH = OUTPUT_DIR / "image_latent.pt"
OUTPUT_IMAGE_PATH = OUTPUT_DIR / "reconstruction.png"


def load_rgb_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {path}\n"
            "Place one image named input.png beside this script or edit INPUT_IMAGE_PATH."
        )

    with Image.open(path) as source:
        image = source.convert("RGB")

    width, height = image.size
    if width % 16 or height % 16:
        raise ValueError(
            f"Image dimensions must be divisible by 16 for H3 VAE encoding; "
            f"received {width}x{height}. The image is not resized automatically."
        )

    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return image, tensor.contiguous()


def save_metrics(source_image, decoded_path):
    source = np.asarray(source_image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0

    with Image.open(decoded_path) as decoded_image:
        decoded_image = decoded_image.convert("RGB")
        if decoded_image.size != source_image.size:
            raise RuntimeError(
                f"Decoded size {decoded_image.size} does not match input size {source_image.size}."
            )
        decoded = np.asarray(decoded_image, dtype=np.float32) / 127.5 - 1.0

    difference = decoded - source
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10(4.0 / mse)

    print("\nSingle-image reconstruction metrics (RGB normalized to [-1,1]):")
    print(f"  MAE:  {mae:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print("Note: pixel metrics quantify reconstruction error; inspect the image for visible tile seams.")


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; this test uses the streaming CUDA implementations.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_image, image_tensor = load_rgb_image(INPUT_IMAGE_PATH)
    width, height = source_image.size

    print("\n========== H3 LIGHTVAE-LITE SINGLE-IMAGE ROUND TRIP ==========")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Input image:", INPUT_IMAGE_PATH.resolve())
    print("Input tensor:", tuple(image_tensor.shape))
    print("Input range:", float(image_tensor.min()), float(image_tensor.max()))
    print("Image dimensions:", f"{width}x{height}")
    print("Output directory:", OUTPUT_DIR.resolve())

    print("\nLoading standalone LightVAE-Lite encoder...")
    encoder = load_streaming_encoder()

    print("\nEncoding exactly one image...")
    latents = encode_keyframe_latent(encoder, image_tensor)
    torch.cuda.synchronize(DEVICE)

    expected_latent_shape = (1, 24, 1, height // 16, width // 16)
    if tuple(latents.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Encoder latent shape {tuple(latents.shape)}; expected {expected_latent_shape}"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError("Encoder produced NaN or Inf latents.")

    print("Encoded image latent shape:", tuple(latents.shape))
    print("Latent range:", float(latents.min()), float(latents.max()))

    # Free encoder weights before loading the decoder, as on the 4 GB GPU.
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(latents.detach().cpu(), LATENT_PATH)
    print("Saved image latent:", LATENT_PATH.resolve())

    print("\nDecoding that single image latent...")
    decode_single_image_latent(
        latents=latents,
        output_path=OUTPUT_IMAGE_PATH,
    )
    torch.cuda.synchronize(DEVICE)

    save_metrics(source_image, OUTPUT_IMAGE_PATH)

    print("\n========== SINGLE-IMAGE ROUND TRIP COMPLETE ==========")
    print("Input image:", INPUT_IMAGE_PATH.resolve())
    print("Reconstructed image:", OUTPUT_IMAGE_PATH.resolve())
    print("Image latent:", LATENT_PATH.resolve())

    del latents, image_tensor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
