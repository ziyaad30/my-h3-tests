# h3_audio_vae_test.py
# Standalone MiniMax H3 AudioVAE decoder.
# Architecture ported from Comfy-Org/ComfyUI:
# comfy/ldm/minimax/audio_vae.py
#
# Default:
#   - Loads the existing FP32 safetensors checkpoint.
#   - Strictly validates and loads decoder weights.
#   - Reads the sampler's .pt output.
#   - Finds an unambiguous audio latent tensor [B,32,2,T].
#   - Decodes stereo audio at 32 kHz.
#   - Writes a 16-bit PCM WAV.
#
# CPU is the default to avoid exhausting 4 GB VRAM.

import argparse
import math
import sys
import wave
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_CHECKPOINT = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\vae"
    r"\minimax_h3_audio_vae_fp32.safetensors"
)

DEFAULT_INPUT = Path(
    r"C:\Users\User\PycharmProjects\TTS_Final\h3_sampler_output"
    r"\h3_sample_latents.pt"
)

DEFAULT_OUTPUT = Path(
    r"C:\Users\User\PycharmProjects\TTS_Final\h3_sampler_output"
    r"\h3_decoded_audio.wav"
)

SAMPLE_RATE = 32000
SAMPLES_PER_LATENT = 800


# ---------------------------------------------------------------------
# Exact alias-free activation components used by the official decoder
# ---------------------------------------------------------------------

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(
        kernel_size, beta=beta, periodic=False
    )

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filt /= filt.sum()
    return filt.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2

        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.expand(channels, -1, -1).to(
            device=x.device, dtype=x.dtype
        )
        x = F.conv_transpose1d(
            x, filt, stride=self.stride, groups=channels
        ).mul_(self.ratio)
        return x[..., self.pad_left:-self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(
        self, cutoff=0.5, half_width=0.6, stride=1, kernel_size=12
    ):
        super().__init__()
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride

        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(cutoff, half_width, kernel_size),
        )

    def forward(self, x):
        channels = x.shape[1]
        x = F.pad(
            x, (self.pad_left, self.pad_right), mode="replicate"
        )
        filt = self.filter.expand(channels, -1, -1).to(
            device=x.device, dtype=x.dtype
        )
        return F.conv1d(
            x, filt, stride=self.stride, groups=channels
        )


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def forward(self, x):
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio=2,
        down_ratio=2,
        up_kernel_size=12,
        down_kernel_size=12,
    ):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


# ---------------------------------------------------------------------
# BigVGAN SnakeBeta activation and residual blocks
# ---------------------------------------------------------------------

def snake(x, alpha, beta):
    t = torch.sin(alpha * x)
    return t.mul(t).mul((beta + 1e-9).reciprocal()).add(x)


class SnakeBeta(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(in_features))
        self.beta = nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        alpha = torch.exp(self.alpha.to(
            device=x.device, dtype=x.dtype
        )).view(1, -1, 1)

        beta = torch.exp(self.beta.to(
            device=x.device, dtype=x.dtype
        )).view(1, -1, 1)

        return snake(x, alpha, beta)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilation
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in range(len(dilation))
        ])

        self.num_layers = len(self.convs1) + len(self.convs2)

        self.activations = nn.ModuleList([
            Activation1d(SnakeBeta(channels))
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]

        for c1, c2, a1, a2 in zip(
            self.convs1, self.convs2, acts1, acts2
        ):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt.add_(x)

        return x


# ---------------------------------------------------------------------
# MiniMax H3 BigVGAN decoder
# ---------------------------------------------------------------------

class BigVGAN(nn.Module):
    def __init__(
        self,
        num_mels=2048,
        upsample_initial_channel=1024,
        upsample_rates=(5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=(
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(
            num_mels, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            ]))

        self.resblocks = nn.ModuleList()

        for i in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (i + 1))

            for k, d in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks.append(AMPBlock1(channels, k, d))

        self.activation_post = Activation1d(SnakeBeta(channels))
        self.conv_post = nn.Conv1d(
            channels, 1, 7, 1, padding=3, bias=False
        )

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)

            xs = None

            for j in range(self.num_kernels):
                y = self.resblocks[i * self.num_kernels + j](x)
                xs = y if xs is None else xs + y

            x = xs.div_(self.num_kernels)

        x = self.activation_post(x)
        return self.conv_post(x).clamp_(-1.0, 1.0)


# ---------------------------------------------------------------------
# Top-level decoder and latent normalization
# ---------------------------------------------------------------------

class MiniMaxH3AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec_in_proj = nn.Conv1d(32, 2048, 1)
        self.decoder = BigVGAN(
            num_mels=2048,
            upsample_initial_channel=1024,
        )

        self.register_buffer("latents_mean", torch.empty(32))
        self.register_buffer("latents_std", torch.empty(32))

    def decode_mono(self, z):
        mean = self.latents_mean.view(1, -1, 1).to(
            device=z.device, dtype=z.dtype
        )
        std = self.latents_std.view(1, -1, 1).to(
            device=z.device, dtype=z.dtype
        )

        z = z * std + mean
        z = self.dec_in_proj(z)
        return self.decoder(z)

    @torch.inference_mode()
    def decode(self, z):
        if z.ndim != 4:
            raise ValueError(
                f"Expected [B,32,2,T] audio latents, got {tuple(z.shape)}"
            )

        b, c, stereo, t = z.shape

        if c != 32 or stereo != 2:
            raise ValueError(
                f"Expected [B,32,2,T], got {tuple(z.shape)}"
            )

        # Official implementation folds stereo into the batch dimension.
        # Decode one channel at a time here to reduce peak activation memory.
        batches = []

        for batch_index in range(b):
            channels = []

            for stereo_index in range(2):
                mono = z[
                    batch_index:batch_index + 1,
                    :,
                    stereo_index,
                    :
                ].contiguous()

                channels.append(self.decode_mono(mono))

            batches.append(torch.cat(channels, dim=1))

        return torch.cat(batches, dim=0)


# ---------------------------------------------------------------------
# Strict decoder checkpoint loading
# ---------------------------------------------------------------------

def load_decoder_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"AudioVAE checkpoint not found: {checkpoint_path}"
        )

    expected = model.state_dict()
    expected_names = set(expected.keys())

    with safe_open(
        str(checkpoint_path), framework="pt", device="cpu"
    ) as handle:
        checkpoint_names = list(handle.keys())

        # Detect an optional common prefix from the actual checkpoint keys.
        candidates = [
            key[:-len("dec_in_proj.weight")]
            for key in checkpoint_names
            if key.endswith("dec_in_proj.weight")
        ]

        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely identify the decoder checkpoint "
                f"prefix. Matching dec_in_proj.weight keys: {candidates}"
            )

        prefix = candidates[0]
        mapped = {}

        for name in expected_names:
            source_name = prefix + name
            if source_name in checkpoint_names:
                mapped[name] = source_name

        missing = sorted(expected_names - set(mapped.keys()))

        if missing:
            preview = "\n".join(missing[:30])
            raise RuntimeError(
                "Checkpoint is missing required decoder tensors. "
                f"First missing names:\n{preview}"
            )

        loaded = 0

        with torch.no_grad():
            for name, source_name in mapped.items():
                tensor = handle.get_tensor(source_name)
                target = expected[name]

                if tuple(tensor.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"Shape mismatch for {name}: "
                        f"checkpoint {tuple(tensor.shape)}, "
                        f"model {tuple(target.shape)}"
                    )

                target.copy_(tensor.to(
                    device=target.device,
                    dtype=target.dtype,
                ))

                loaded += 1

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Decoder tensors loaded: {loaded}")
    print("Strict decoder parameter and shape checks: PASSED")

    return model


# ---------------------------------------------------------------------
# Find audio latent tensor in sampler output
# ---------------------------------------------------------------------

def find_audio_latent(obj, path="root"):
    found = []

    if torch.is_tensor(obj):
        shape = tuple(obj.shape)

        if (
            len(shape) == 4
            and shape[1] == 32
            and shape[2] == 2
        ):
            found.append((path, obj))

        elif (
            len(shape) == 3
            and shape[0] == 32
            and shape[1] == 2
        ):
            found.append((path, obj.unsqueeze(0)))

    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(
                find_audio_latent(value, f"{path}.{key}")
            )

    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(
                find_audio_latent(value, f"{path}[{index}]")
            )

    return found


def load_sampler_latents(input_path):
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Sampler output not found: {input_path}"
        )

    # weights_only avoids unpickling arbitrary Python objects.
    data = torch.load(
        input_path,
        map_location="cpu",
        weights_only=True,
    )

    candidates = find_audio_latent(data)

    if not candidates:
        raise RuntimeError(
            "No audio latent tensor with shape [B,32,2,T] "
            "or [32,2,T] was found in the sampler output."
        )

    if len(candidates) > 1:
        audio_named = [
            item for item in candidates
            if "audio" in item[0].lower()
        ]

        if len(audio_named) == 1:
            candidates = audio_named
        else:
            descriptions = [
                f"{name}: {tuple(tensor.shape)}"
                for name, tensor in candidates
            ]
            raise RuntimeError(
                "More than one possible audio latent was found. "
                "Cannot safely guess which one to decode:\n"
                + "\n".join(descriptions)
            )

    name, latent = candidates[0]
    latent = latent.detach().to(dtype=torch.float32).contiguous()

    if not torch.isfinite(latent).all():
        raise RuntimeError("Audio latent contains NaN or Inf values.")

    print(f"Audio latent key: {name}")
    print(f"Audio latent shape: {tuple(latent.shape)}")
    print(
        f"Audio latent range: "
        f"{latent.min().item():.6f} .. {latent.max().item():.6f}"
    )

    return latent


# ---------------------------------------------------------------------
# WAV output
# ---------------------------------------------------------------------

def save_stereo_wav(audio, output_path):
    audio = audio.detach().float().cpu()

    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(
            f"Expected stereo audio [2,L], got {tuple(audio.shape)}"
        )

    audio = audio.clamp(-1.0, 1.0)
    pcm = (audio.transpose(0, 1).contiguous().numpy() * 32767.0)
    pcm = pcm.round().astype("<i2")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    print(f"WAV saved: {output_path}")
    print(f"Duration: {audio.shape[-1] / SAMPLE_RATE:.3f} seconds")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("Channels: stereo")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone MiniMax H3 AudioVAE decoder"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.set_num_threads(max(1, args.threads))

    device = torch.device(args.device)

    print("========== MINIMAX H3 AUDIO VAE ==========")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    latent = load_sampler_latents(args.input)

    print("\nBuilding decoder...")
    model = MiniMaxH3AudioDecoder()

    print("Loading checkpoint...")
    load_decoder_checkpoint(model, args.checkpoint)

    model.eval()
    model.to(device)

    print("\nDecoding stereo audio...")
    latent = latent.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        decoded = model.decode(latent)

    if decoded.shape[0] != 1:
        raise RuntimeError(
            "WAV export currently expects batch size 1, "
            f"but decoder returned batch size {decoded.shape[0]}."
        )

    audio = decoded[0]

    if not torch.isfinite(audio).all():
        raise RuntimeError("Decoded audio contains NaN or Inf values.")

    print(f"Decoded audio shape: {tuple(audio.shape)}")
    print(
        f"Decoded range: "
        f"{audio.min().item():.6f} .. {audio.max().item():.6f}"
    )

    save_stereo_wav(audio, args.output)

    print("\n========== AUDIO DECODE COMPLETE ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)