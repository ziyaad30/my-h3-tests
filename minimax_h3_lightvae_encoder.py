
import gc
import math
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from lightvae_h3_standalone.lightvae.h3_lightvae_encoder import Encoder
from lightvae_h3_standalone.lightvae.utils import (
    split_tiles,
    stitch_tiles,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\vae"
    r"\lightvae-lite-h3-encoder.safetensors"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16

# Match the canvas used by the current FL2VA layout test.
CANVAS_WIDTH = 320
CANVAS_HEIGHT = 320

# FL2VA target video frame count.
TARGET_FRAME_COUNT = 22

# Set these to actual image paths when ready.
# Leave both as None for the synthetic encoder test.
FIRST_IMAGE_PATH = None
LAST_IMAGE_PATH = None

# First frame uses stretch.
# Last frame uses aspect-preserving center crop.
TILE_SIZE = 128
TILE_OVERLAP = 64

LATENT_CHANNELS = 24
PATCH_SIZE = (1, 2, 2)


# ============================================================
# STREAMING ENCODER EXECUTION
# ============================================================

def streaming_encode_clip(self, video):

    mean = video.new_tensor(
        (0.485, 0.456, 0.406)
    ).view(1, 3, 1, 1, 1)

    std = video.new_tensor(
        (0.229, 0.224, 0.225)
    ).view(1, 3, 1, 1, 1)

    video = ((video + 1.0) / 2.0 - mean) / std

    hidden_states = video.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    model = self.encoder

    # Input convolution.
    model.conv_in.to(device=DEVICE, dtype=DTYPE)
    hidden_states = model.conv_in(hidden_states)
    model.conv_in.to(device="cpu")

    # Residual blocks, one at a time.
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

        # Downsampling.
        if down_block.downsamplers is not None:

            for downsampler in down_block.downsamplers:

                downsampler.to(
                    device=DEVICE,
                    dtype=DTYPE,
                )

                hidden_states = downsampler(hidden_states)

                downsampler.to(device="cpu")

    print()

    # Output normalization.
    model.norm_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = F.silu(
        model.norm_out(hidden_states)
    )

    model.norm_out.to(device="cpu")

    # Output convolution.
    model.conv_out.to(device=DEVICE, dtype=DTYPE)

    hidden_states = model.conv_out(hidden_states)

    model.conv_out.to(device="cpu")

    # Quantization convolution.
    self.quant_conv.to(device=DEVICE, dtype=DTYPE)

    moments = self.quant_conv(hidden_states)

    self.quant_conv.to(device="cpu")

    return moments.to(device="cpu")


# ============================================================
# ENCODER INITIALIZATION
# ============================================================

def load_streaming_encoder():

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"LightVAE-Lite encoder checkpoint not found:\n{CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Encoder checkpoint:", CHECKPOINT)

    print("\nLoading H3 LightVAE-Lite encoder...")

    encoder = Encoder(checkpoint=CHECKPOINT)

    encoder.eval()
    encoder.requires_grad_(False)

    # Keep weights in CPU RAM.
    encoder.to(device="cpu", dtype=DTYPE)

    encoder.tile_size = TILE_SIZE
    encoder.tile_overlap = TILE_OVERLAP

    print(
        "Retained residual indices:",
        encoder.config["kept_residual_indices"],
    )

    print("Spatial tile size:", encoder.tile_size)
    print("Spatial tile overlap:", encoder.tile_overlap)

    # Replace only the clip execution method.
    # The checkpoint-aware architecture remains unchanged.
    encoder._encode_clip = types.MethodType(
        streaming_encode_clip,
        encoder,
    )

    return encoder


# ============================================================
# SINGLE-FRAME H3 KEYFRAME ENCODING
# ============================================================

@torch.inference_mode()
def encode_keyframe_latent(encoder, image):

    """
    Encode one RGB keyframe using the H3 LightVAE-Lite encoder.

    Input:
        [1, 3, H, W]
        or
        [1, 3, 1, H, W]

    RGB range:
        [-1, 1]

    Output:
        [1, 24, 1, H/16, W/16]

    This follows the H3 video VAE single-frame posterior-mean
    path. It does not use the separate H3 Image VAE sampling path.
    """

    if image.ndim == 4:
        image = image.unsqueeze(2)

    if image.ndim != 5:
        raise ValueError(
            f"Expected [1,3,H,W] or [1,3,1,H,W], got {tuple(image.shape)}"
        )

    if image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("Keyframe must have batch size 1 and RGB channels.")

    if image.shape[2] != 1:
        raise ValueError(
            f"Keyframe must contain exactly one frame, got {image.shape[2]}"
        )

    height = image.shape[-2]
    width = image.shape[-1]

    if height % 16 or width % 16:
        raise ValueError(
            f"Keyframe dimensions must be divisible by 16: {width}x{height}"
        )

    image = image.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()

    if not torch.isfinite(image).all():
        raise ValueError("Keyframe contains NaN or infinite values.")

    if image.min().item() < -1.001 or image.max().item() > 1.001:
        raise ValueError("Keyframe RGB values must be in [-1, 1].")

    print(
        f"\nEncoding single keyframe: {width}x{height}"
    )

    # The official encoder performs spatial tiling and stitching.
    # We reproduce that part here while bypassing video temporal
    # padding, because this input is one still image.

    height_tiles, height_overlaps = split_tiles(
        height,
        encoder.tile_size,
        encoder.tile_overlap,
    )

    width_tiles, width_overlaps = split_tiles(
        width,
        encoder.tile_size,
        encoder.tile_overlap,
    )

    rows = []

    total_tiles = len(height_tiles) * len(width_tiles)
    tile_number = 0

    for h in height_tiles:

        row = []

        for w in width_tiles:

            tile_number += 1

            print(
                f"\nKeyframe tile {tile_number}/{total_tiles}"
                f" | pixels "
                f"{h.start * 16}:{h.stop * 16},"
                f" {w.start * 16}:{w.stop * 16}"
            )

            pixel_tile = image[
                :,
                :,
                :,
                h.start * 16:h.stop * 16,
                w.start * 16:w.stop * 16,
            ]

            # Crucial: call the streaming encoder directly.
            # Do not call Encoder.forward(), because that method
            # applies the 17-frame video temporal layout.
            moments = encoder._encode_clip(pixel_tile)

            expected_tile_shape = (
                1,
                LATENT_CHANNELS * 2,
                1,
                h.stop - h.start,
                w.stop - w.start,
            )

            if tuple(moments.shape) != expected_tile_shape:
                raise RuntimeError(
                    "Unexpected single-frame encoder tile shape: "
                    f"{tuple(moments.shape)}; "
                    f"expected {expected_tile_shape}"
                )

            row.append(moments)

            del pixel_tile, moments

        rows.append(row)

    # Blend and stitch posterior moments in latent coordinates.
    latent_height_overlaps = [
        value // 16 for value in height_overlaps
    ]

    latent_width_overlaps = [
        value // 16 for value in width_overlaps
    ]

    moments = stitch_tiles(
        rows,
        latent_height_overlaps,
        latent_width_overlaps,
    )

    expected_moments_shape = (
        1,
        LATENT_CHANNELS * 2,
        1,
        height // 16,
        width // 16,
    )

    if tuple(moments.shape) != expected_moments_shape:
        raise RuntimeError(
            f"Stitched moments shape {tuple(moments.shape)}; "
            f"expected {expected_moments_shape}"
        )

    # H3 video VAE uses the posterior mean for this path.
    posterior_mean = moments[:, :LATENT_CHANNELS].float()

    latent_mean = encoder.latents_mean.float().view(
        1, LATENT_CHANNELS, 1, 1, 1
    )

    latent_std = encoder.latents_std.float().view(
        1, LATENT_CHANNELS, 1, 1, 1
    )

    latents = (
        posterior_mean - latent_mean
    ) / latent_std

    expected_latent_shape = (
        1,
        LATENT_CHANNELS,
        1,
        height // 16,
        width // 16,
    )

    if tuple(latents.shape) != expected_latent_shape:
        raise RuntimeError(
            f"Keyframe latent shape {tuple(latents.shape)}; "
            f"expected {expected_latent_shape}"
        )

    if not torch.isfinite(latents).all():
        raise RuntimeError("Encoded keyframe contains non-finite values.")

    print("\nKeyframe encoding complete.")
    print("Latent shape:", tuple(latents.shape))
    print("Latent dtype:", latents.dtype)
    print(
        "Latent range:",
        latents.min().item(),
        latents.max().item(),
    )

    del moments, posterior_mean

    return latents.contiguous()


# ============================================================
# OFFICIAL H3 VIDEO PATCHIFICATION
# ============================================================

def patchify_h3_video(latent):

    """
    Official H3 patch layout:

        [B,24,T,H,W]
          ->
        [B*T*(H/2)*(W/2),96]

    Patch size:
        (1,2,2)
    """

    if latent.ndim != 5:
        raise ValueError("Expected [B,C,T,H,W] latent.")

    b, c, t_full, h_full, w_full = latent.shape

    pt, ph, pw = PATCH_SIZE

    if c != LATENT_CHANNELS:
        raise ValueError(f"Expected 24 latent channels, got {c}")

    if t_full % pt or h_full % ph or w_full % pw:
        raise ValueError("Latent dimensions are not divisible by patch size.")

    t = t_full // pt
    h = h_full // ph
    w = w_full // pw

    x = latent.reshape(
        b,
        c,
        t,
        pt,
        h,
        ph,
        w,
        pw,
    )

    x = torch.einsum(
        "nctrhpwq->nthwcrpq",
        x,
    )

    return x.reshape(
        b * t * h * w,
        c * pt * ph * pw,
    )


# ============================================================
# FL2VA KEYFRAME PAYLOAD
# ============================================================

def build_fl2va_keyframe_payload(
    first_latent=None,
    last_latent=None,
    frame_count=TARGET_FRAME_COUNT,
):

    """
    Construct the two matching FL2VA payload fields.

    keyframes:
        Used by PackedLayout to reserve non-denoised rows.

    cond_video_latents:
        Used by the H3 model to populate those rows.

    The ordering of both lists must remain identical.
    """

    if frame_count < 5 or (frame_count - 5) % 17 != 0:
        raise ValueError(
            "H3 frame count must follow the 17k+5 pattern."
        )

    keyframes = []
    cond_video_latents = []

    if first_latent is not None:

        keyframes.append({
            "resolved_frame_index": 0,
            "latent": first_latent,
        })

        cond_video_latents.append(first_latent)

    if last_latent is not None:

        last_index = frame_count - 1

        keyframes.append({
            "resolved_frame_index": last_index,
            "latent": last_latent,
        })

        cond_video_latents.append(last_latent)

    return {
        "keyframes": keyframes,
        "cond_video_latents": cond_video_latents,
    }


# ============================================================
# IMAGE LOADING AND CANVAS RESIZING
# ============================================================

def load_keyframe_image(path, width, height, resize_mode):

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as source:
        image = source.convert("RGB")

    if resize_mode == "stretch":

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    elif resize_mode == "center":

        image = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )

    else:
        raise ValueError(
            f"Unknown resize mode: {resize_mode}"
        )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    # RGB [0,255] -> RGB [-1,1].
    array = array / 127.5 - 1.0

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0).unsqueeze(2)

    return tensor.contiguous()


# ============================================================
# GPU MEMORY REPORT
# ============================================================

def report_gpu_memory(label):

    torch.cuda.synchronize(DEVICE)

    allocated = (
        torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
    )

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
# MAIN TEST
# ============================================================

def main():

    if CANVAS_WIDTH % 32 or CANVAS_HEIGHT % 32:
        raise ValueError("Canvas dimensions must be multiples of 32.")

    if TARGET_FRAME_COUNT < 5 or (TARGET_FRAME_COUNT - 5) % 17:
        raise ValueError("Invalid H3 target frame count.")

    encoder = load_streaming_encoder()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)

    print("\nStarting FL2VA keyframe encoding.")

    first_latent = None
    last_latent = None

    with torch.inference_mode():

        if FIRST_IMAGE_PATH is not None:

            first_image = load_keyframe_image(
                FIRST_IMAGE_PATH,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                "stretch",
            )

            first_latent = encode_keyframe_latent(
                encoder,
                first_image,
            )

            del first_image

        if LAST_IMAGE_PATH is not None:

            last_image = load_keyframe_image(
                LAST_IMAGE_PATH,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
                "center",
            )

            last_latent = encode_keyframe_latent(
                encoder,
                last_image,
            )

            del last_image

        # No image paths: exercise the real checkpoint with a
        # deterministic synthetic single-frame input.
        if first_latent is None and last_latent is None:

            print(
                "\nNo image paths configured. "
                "Running a synthetic single-keyframe test."
            )

            generator = torch.Generator(
                device="cpu"
            ).manual_seed(42)

            synthetic = torch.rand(
                (1, 3, 1, CANVAS_HEIGHT, CANVAS_WIDTH),
                generator=generator,
                dtype=torch.float32,
            )

            synthetic = synthetic * 2.0 - 1.0

            first_latent = encode_keyframe_latent(
                encoder,
                synthetic,
            )

            del synthetic

        payload = build_fl2va_keyframe_payload(
            first_latent=first_latent,
            last_latent=last_latent,
            frame_count=TARGET_FRAME_COUNT,
        )

        print("\nFL2VA keyframe payload")
        print("-" * 48)

        print("Target frames:", TARGET_FRAME_COUNT)
        print("Keyframe count:", len(payload["keyframes"]))

        for index, entry in enumerate(payload["keyframes"]):

            latent = entry["latent"]
            rows = patchify_h3_video(latent)

            print(
                f"\nKeyframe {index + 1}:"
            )

            print(
                "Resolved frame index:",
                entry["resolved_frame_index"],
            )

            print(
                "Latent shape:",
                tuple(latent.shape),
            )

            print(
                "Patchified shape:",
                tuple(rows.shape),
            )

            expected_rows = (
                (CANVAS_HEIGHT // 16 // 2)
                * (CANVAS_WIDTH // 16 // 2)
            )

            if tuple(rows.shape) != (
                expected_rows,
                96,
            ):
                raise RuntimeError(
                    f"Unexpected patchified keyframe shape: "
                    f"{tuple(rows.shape)}"
                )

        if len(payload["keyframes"]) != len(
            payload["cond_video_latents"]
        ):
            raise RuntimeError(
                "FL2VA keyframe metadata and latent lists differ in length."
            )

        print("\nPayload fields:")
        print("  keyframes")
        print("  cond_video_latents")

        print(
            "\nConditioning rows:",
            sum(
                patchify_h3_video(z).shape[0]
                for z in payload["cond_video_latents"]
            ),
        )

        report_gpu_memory(
            "GPU memory after keyframe encoding"
        )

    print("\nFL2VA keyframe encoder test completed.")

    del payload, first_latent, last_latent, encoder

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
