# h3_audio_vae_debug.py
# Full MiniMax H3 audio-VAE waveform roundtrip:
# WAV -> official DAC-lineage encoder -> normalized [B,32,2,T] latents
#     -> BigVGAN decoder -> reconstructed WAV.
#
# Encoder/decoder architecture follows Comfy-Org/ComfyUI:
# comfy/ldm/minimax/audio_vae.py
#
# Keep this file beside h3_audio_vae_test.py. It reuses that file's
# verified alias-free activation and BigVGAN decoder definitions, avoiding
# a second permanent copy of the large decoder implementation.
#
# CPU is the default. The complete VAE is memory-heavy.

import argparse
import math
import sys
import wave
from pathlib import Path
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from h3_audio_vae_test import BigVGAN


DEFAULT_CHECKPOINT = Path(
    r"C:\Users\User\PycharmProjects\pythonProject\models\vae"
    r"\minimax_h3_audio_vae_fp32.safetensors"
)
SAMPLE_RATE = 32000
HOP_LENGTH = 800


# ---------------------------------------------------------------------
# Encoder-side architecture: direct structural port of official H3 VAE
# ---------------------------------------------------------------------

def snake(x, alpha, beta):
    t = torch.sin(alpha * x)
    return t.mul(t).mul((beta + 1e-9).reciprocal()).add(x)


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, x):
        alpha = self.alpha.to(device=x.device, dtype=x.dtype)
        return snake(x, alpha, alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return y.add_(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim=16, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, d_model=64, strides=(2, 4, 4, 5, 5), d_latent=2048):
        super().__init__()
        block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block.extend([
            Snake1d(d_model),
            nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1),
        ])
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class GeGluMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.act = nn.GELU(approximate="tanh")
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(self.act(self.w0(x)).mul_(self.w1(x)))


class CausalAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.head_dim = in_dim // num_heads
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.empty(in_dim))
        self.v_bias = nn.Parameter(torch.empty(in_dim))
        # The key bias is explicitly zero in the reference implementation.
        self.register_buffer("zero_k_bias", torch.zeros(in_dim))
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        b, n, _ = x.shape
        bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias.to(device=x.device, dtype=x.dtype))
        q, k, v = (
            qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        pooled = F.adaptive_avg_pool1d(
            torch.mean(attended, dim=1), self.out_dim
        )
        return self.proj(pooled)


class AttnProjection(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = GeGluMlp(out_dim, int(out_dim * mlp_ratio))

    def forward(self, x):
        x = self.proj(self.norm3(x)).add_(self.attn(self.norm1(x)))
        return x.add_(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------
# Complete H3 audio VAE
# ---------------------------------------------------------------------

class MiniMaxH3AudioVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self.hop_length = HOP_LENGTH
        self.encoder = Encoder(64, (2, 4, 4, 5, 5), 2048)
        self.pre_block = AttnProjection(2048, 32, num_heads=8)
        self.mean_proj = nn.Conv1d(32, 32, 1)
        # Present in the official checkpoint/module, unused by deterministic
        # inference because encode() uses the posterior mean.
        self.logs_proj = nn.Conv1d(32, 32, 1)
        self.dec_in_proj = nn.Conv1d(32, 2048, 1)
        self.decoder = BigVGAN(num_mels=2048, upsample_initial_channel=1024)
        self.register_buffer("latents_mean", torch.empty(32))
        self.register_buffer("latents_std", torch.empty(32))

    def encode(self, waveform):
        # waveform: [B,2,L], 32 kHz, float in [-1,1]
        b, stereo, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        waveform = F.pad(waveform, (0, right_pad))
        x = waveform.reshape(b * stereo, 1, -1)
        x = self.encoder(x)
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        mean = self.latents_mean.view(1, 32, 1).to(z)
        std = self.latents_std.view(1, 32, 1).to(z)
        z = (z - mean) / std
        return z.reshape(b, stereo, 32, z.shape[-1]).permute(0, 2, 1, 3)

    def decode(self, z):
        # z: [B,32,2,T], normalized latent
        b, channels, stereo, frames = z.shape
        if channels != 32 or stereo != 2:
            raise ValueError(f"Expected [B,32,2,T], got {tuple(z.shape)}")
        x = z.permute(0, 2, 1, 3).reshape(b * stereo, channels, frames)
        mean = self.latents_mean.view(1, 32, 1).to(x)
        std = self.latents_std.view(1, 32, 1).to(x)
        x = x * std + mean
        x = self.dec_in_proj(x)
        x = self.decoder(x)
        return x.reshape(b, stereo, -1)


# ---------------------------------------------------------------------
# Strict checkpoint loading. Deterministic alias-free filters are generated
# by the module and are not required to exist in the checkpoint.
# ---------------------------------------------------------------------

def load_full_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    state = model.state_dict()
    params = dict(model.named_parameters())
    required = set(params)
    required.update((
        "latents_mean",
        "latents_std",
        # ComfyUI uses this checkpoint tensor to identify the H3 audio VAE.
        # The reference module registers it as a buffer, so require it explicitly.
        "pre_block.attn.zero_k_bias",
    ))

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        candidates = [
            key[:-len("dec_in_proj.weight")]
            for key in keys
            if key.endswith("dec_in_proj.weight")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely identify checkpoint prefix from "
                f"dec_in_proj.weight: {candidates}"
            )
        prefix = candidates[0]

        missing = sorted(name for name in required if prefix + name not in keys)
        if missing:
            raise RuntimeError(
                "Full audio VAE checkpoint is missing required tensors "
                f"({len(missing)}). First missing entries:\n"
                + "\n".join(missing[:40])
            )

        loaded = 0
        with torch.no_grad():
            for name in sorted(required):
                source = handle.get_tensor(prefix + name)
                target = state[name]
                if tuple(source.shape) != tuple(target.shape):
                    raise RuntimeError(
                        f"Shape mismatch {name}: checkpoint "
                        f"{tuple(source.shape)} vs model {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
                loaded += 1


    if not torch.isfinite(model.latents_mean).all():
        raise RuntimeError("latents_mean contains NaN/Inf")
    if not torch.isfinite(model.latents_std).all() or not (model.latents_std > 0).all():
        raise RuntimeError("latents_std must be finite and positive")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Full VAE learned tensors loaded: {loaded}")
    print("Encoder, projection, normalization and decoder tensor checks: PASSED")
    return model


# ---------------------------------------------------------------------
# WAV input/output. PCM 8/16/24/32-bit WAV is supported.
# Other sample rates are resampled with scipy when available.
# ---------------------------------------------------------------------

def read_wav(path):
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        comptype = wav.getcomptype()
        raw = wav.readframes(frames)

    if comptype != "NONE":
        raise ValueError(f"Compressed WAV is not supported: {comptype}")
    if channels not in (1, 2):
        raise ValueError(f"Expected mono or stereo WAV, found {channels} channels")

    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float32) / 8388608.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {width * 8}-bit")

    samples = samples.reshape(-1, channels).T
    if channels == 1:
        samples = np.repeat(samples, 2, axis=0)

    if rate != SAMPLE_RATE:
        ratio = Fraction(SAMPLE_RATE, rate)
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                f"Input is {rate} Hz. Install scipy to resample it to "
                "32000 Hz, or provide a 32000 Hz WAV."
            ) from exc
        samples = resample_poly(
            samples, ratio.numerator, ratio.denominator, axis=-1
        ).astype(np.float32)
        rate = SAMPLE_RATE

    samples = np.clip(samples, -1.0, 1.0)
    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    return waveform, rate


def write_wav(path, audio):
    audio = audio.detach().float().cpu().clamp(-1.0, 1.0)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"Expected [2,L] stereo waveform, got {tuple(audio.shape)}")
    pcm = (audio.transpose(0, 1).contiguous().numpy() * 32767.0)
    pcm = pcm.round().astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def waveform_report(label, audio):
    audio = audio.detach().float().cpu()
    print(f"{label}: shape={tuple(audio.shape)}")
    for side, name in enumerate(("L", "R")):
        x = audio[side]
        print(
            f"  {name}: mean={x.mean().item():+.6f}, "
            f"RMS={x.square().mean().sqrt().item():.6f}, "
            f"peak={x.abs().max().item():.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax H3 audio VAE WAV -> latent -> WAV roundtrip"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input PCM WAV")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--latents", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)

    output = args.output or args.input.with_name(args.input.stem + "_h3_roundtrip.wav")
    latent_path = args.latents or args.input.with_name(args.input.stem + "_h3_audio_latents.pt")

    print("========== MINIMAX H3 AUDIO VAE ROUNDTRIP ==========")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Input: {args.input}")

    waveform, sample_rate = read_wav(args.input)
    original_length = waveform.shape[-1]
    print(
        f"Input waveform: {tuple(waveform.shape)} | "
        f"{sample_rate} Hz | {original_length / SAMPLE_RATE:.3f}s"
    )
    waveform_report("Original", waveform[0])

    print("\nBuilding complete audio VAE...")
    model = MiniMaxH3AudioVAE()
    print("Loading complete checkpoint...")
    load_full_checkpoint(model, args.checkpoint)
    model.eval().to(device)

    waveform = waveform.to(device=device, dtype=torch.float32)
    print("\nEncoding original waveform...")
    with torch.inference_mode():
        latents = model.encode(waveform)
    if not torch.isfinite(latents).all():
        raise RuntimeError("Encoder produced NaN/Inf latents")

    print(f"Encoded latent shape: {tuple(latents.shape)}")
    print(
        f"Latent range: {latents.min().item():+.6f} .. "
        f"{latents.max().item():+.6f}"
    )
    print(
        f"Latent global mean/std: {latents.mean().item():+.6f} / "
        f"{latents.std(unbiased=False).item():.6f}"
    )

    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "audio_latent": latents.detach().cpu(),
            "sample_rate": SAMPLE_RATE,
            "original_samples": original_length,
            "source": str(args.input),
        },
        latent_path,
    )
    print(f"Latents saved: {latent_path}")

    print("\nDecoding encoded latents...")
    with torch.inference_mode():
        reconstructed = model.decode(latents)
    reconstructed = reconstructed[0, :, :original_length].detach().cpu()

    if not torch.isfinite(reconstructed).all():
        raise RuntimeError("Decoder produced NaN/Inf waveform")

    waveform_report("Reconstructed", reconstructed)
    write_wav(output, reconstructed)
    print(f"Roundtrip WAV saved: {output}")

    reference = waveform[0, :, :original_length].detach().float()
    recon = reconstructed.to(device)
    error = recon - reference
    mse = error.square().mean().item()
    signal_power = reference.square().mean().item()
    snr = 10.0 * math.log10(signal_power / max(mse, 1e-20))
    correlations = []
    for side in range(2):
        a = reference[side] - reference[side].mean()
        b = recon[side] - recon[side].mean()
        denom = a.norm() * b.norm()
        correlations.append(
            float((torch.dot(a, b) / denom).item()) if denom > 0 else float("nan")
        )

    print("\n========== ROUNDTRIP METRICS ==========")
    print(f"MSE: {mse:.9g}")
    print(f"Waveform SNR: {snr:.3f} dB")
    print(f"Left correlation: {correlations[0]:+.6f}")
    print(f"Right correlation: {correlations[1]:+.6f}")
    print("Listen to the reconstructed WAV; metrics alone do not establish perceptual quality.")
    print("\n========== ROUNDTRIP COMPLETE ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)
