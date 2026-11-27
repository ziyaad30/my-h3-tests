
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class H3SchedulerOutput:
    prev_sample: torch.Tensor


class H3FlowScheduler:
    """
    Standalone implementation of the official MiniMax H3 rectified-flow
    Euler scheduler.

    No Diffusers dependency.

    H3 conventions:
      * t = 1 is clean; t = 0 is pure noise.
      * The transformer predicts data-ward velocity.
      * x0 = xt + sigma * velocity.
      * Video shift = 12.0.
      * Audio shift = 3.0.
      * The sigma grid includes terminal zero.
    """

    def __init__(
        self,
        shift: float = 12.0,
        stochastic_sampling: bool = False,
        shift_provided_sigmas: bool | None = None,
        euler_update_form: str = "blend",
    ):
        if shift <= 0:
            raise ValueError("shift must be positive")

        if euler_update_form not in ("blend", "velocity"):
            raise ValueError(
                "euler_update_form must be 'blend' or 'velocity'"
            )

        if stochastic_sampling and euler_update_form != "blend":
            raise ValueError(
                "stochastic_sampling requires euler_update_form='blend'"
            )

        self.shift = float(shift)
        self.stochastic_sampling = bool(stochastic_sampling)
        self.shift_provided_sigmas = shift_provided_sigmas
        self.euler_update_form = euler_update_form

        self.sigmas = None
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = None
        self._begin_index = None

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = int(begin_index)

    def set_shift(self, shift: float):
        if shift <= 0:
            raise ValueError("shift must be positive")
        self.shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas=None,
    ):
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "num_inference_steps must be at least 2"
                )

            base = torch.linspace(
                1.0,
                0.0,
                int(num_inference_steps),
                dtype=torch.float32,
                device="cpu",
            )

            shifted = (
                self.shift * base
                / (1.0 + (self.shift - 1.0) * base)
            )

            sigmas = torch.unique_consecutive(shifted)

        else:
            sigmas = torch.as_tensor(
                sigmas,
                dtype=torch.float64,
                device="cpu",
            ).flatten()

            if self.shift_provided_sigmas:
                sigmas = (
                    self.shift * sigmas
                    / (1.0 + (self.shift - 1.0) * sigmas)
                )

            sigmas = sigmas.to(torch.float32)

            if (
                sigmas.numel() < 2
                or not bool((sigmas[1:] < sigmas[:-1]).all())
                or sigmas[-1].item() != 0.0
            ):
                raise ValueError(
                    "sigmas must be strictly decreasing and end at zero"
                )

        self.sigmas = sigmas.to(device=device)

        # The final sigma is zero and does not need a model evaluation.
        self.timesteps = 1.0 - self.sigmas[:-1]
        self.num_inference_steps = int(self.timesteps.numel())

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep) -> int:
        if self.timesteps is None:
            raise RuntimeError("Call set_timesteps first")

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        matches = (self.timesteps == timestep).nonzero()

        if len(matches) == 0:
            raise ValueError(
                "Timestep is not part of this scheduler's schedule"
            )

        return int(matches[0].item())

    @staticmethod
    def scale_noise(sample, timestep, noise):
        """
        H3 forward noising:
            xt = t*x0 + (1-t)*noise

        A Python float is deliberately kept as a Python float to preserve
        the official scalar precision behaviour.
        """
        if not isinstance(timestep, torch.Tensor):
            return timestep * sample + (1.0 - timestep) * noise

        timestep = timestep.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)

        return timestep * sample + (1.0 - timestep) * noise

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        if self.sigmas is None:
            raise RuntimeError("Call set_timesteps first")

        if isinstance(timestep, int):
            raise ValueError(
                "Pass a scheduler timestep value, not an integer index"
            )

        if (
            isinstance(timestep, torch.Tensor)
            and not timestep.is_floating_point()
        ):
            raise ValueError("Timestep tensor must be floating point")

        if model_output.shape != sample.shape:
            raise ValueError(
                f"Output/sample shape mismatch: "
                f"{tuple(model_output.shape)} vs {tuple(sample.shape)}"
            )

        if self._step_index is None:
            if self._begin_index is None:
                self._step_index = self.index_for_timestep(timestep)
            else:
                self._step_index = self._begin_index

        i = self._step_index

        if i + 1 >= len(self.sigmas):
            raise RuntimeError("Scheduler has no remaining step")

        sigma = self.sigmas[i].to(
            device=sample.device,
            dtype=torch.float32,
        )
        sigma_next = self.sigmas[i + 1].to(
            device=sample.device,
            dtype=torch.float32,
        )

        # Optional stochastic re-noising mode used by specific adapters.
        if self.stochastic_sampling:
            denoised = (
                sample.to(torch.float32)
                + sigma * model_output.to(torch.float32)
            )

            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=torch.float32,
            )

            previous = (
                (1.0 - sigma_next) * denoised
                + sigma_next * noise
            ).to(sample.dtype)

            self._step_index += 1

            if return_dict:
                return H3SchedulerOutput(previous)

            return (previous,)

        # Equivalent Euler update expressed directly using velocity.
        if self.euler_update_form == "velocity":
            previous = (
                sample.to(torch.float32)
                + (sigma - sigma_next)
                * model_output.to(torch.float32)
            ).to(sample.dtype)

            self._step_index += 1

            if return_dict:
                return H3SchedulerOutput(previous)

            return (previous,)

        # Official H3 blend update.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(
                timestep,
                dtype=sample.dtype,
            )

        sigma_from_timestep = 1.0 - timestep.to(
            device=sample.device,
            dtype=sample.dtype,
        )

        while sigma_from_timestep.ndim < sample.ndim:
            sigma_from_timestep = sigma_from_timestep.unsqueeze(-1)

        # H3 predicts towards clean data, hence the plus sign.
        denoised = sample + sigma_from_timestep * model_output

        compute_dtype = (
            torch.float32
            if sample.dtype in (torch.float16, torch.bfloat16)
            else sample.dtype
        )

        sigma_compute = sigma.to(dtype=compute_dtype)
        sigma_next_compute = sigma_next.to(dtype=compute_dtype)

        ratio = sigma_next_compute / sigma_compute

        previous = (
            ratio * sample.to(dtype=compute_dtype)
            + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
        ).to(sample.dtype)

        self._step_index += 1

        if return_dict:
            return H3SchedulerOutput(previous)

        return (previous,)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_device_tests(device):
    print(f"\nTesting device: {device}")

    video = H3FlowScheduler(shift=12.0)
    audio = H3FlowScheduler(shift=3.0)

    video.set_timesteps(8, device=device)
    audio.set_timesteps(8, device=device)

    check(video.sigmas.numel() == 8, "Video sigma count")
    check(audio.sigmas.numel() == 8, "Audio sigma count")

    check(video.timesteps.numel() == 7, "Video evaluation count")
    check(audio.timesteps.numel() == 7, "Audio evaluation count")

    check(
        torch.all(video.sigmas[:-1] > video.sigmas[1:]).item(),
        "Video sigma monotonicity",
    )
    check(
        torch.all(audio.sigmas[:-1] > audio.sigmas[1:]).item(),
        "Audio sigma monotonicity",
    )

    check(video.sigmas[-1].item() == 0.0, "Video terminal sigma")
    check(audio.sigmas[-1].item() == 0.0, "Audio terminal sigma")

    # Verify exact shifted-grid construction.
    base = torch.linspace(1.0, 0.0, 8, dtype=torch.float32)

    expected_video = 12.0 * base / (1.0 + 11.0 * base)
    expected_audio = 3.0 * base / (1.0 + 2.0 * base)

    torch.testing.assert_close(
        video.sigmas.cpu(),
        expected_video,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        audio.sigmas.cpu(),
        expected_audio,
        rtol=0,
        atol=0,
    )

    print("Video shift 12.0: PASSED")
    print("Audio shift 3.0: PASSED")
    print("Eight sigma points / seven evaluations: PASSED")

    # Verify the official conditioning-noise formula.
    clean = torch.randn(3, 12, device=device)
    noise = torch.randn_like(clean)

    noised = H3FlowScheduler.scale_noise(
        clean,
        0.999,
        noise,
    )

    expected = 0.999 * clean + 0.001 * noise

    torch.testing.assert_close(
        noised,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

    print("Conditioning noise augmentation: PASSED")

    # Verify one deterministic Euler blend step.
    scheduler = H3FlowScheduler(shift=12.0)
    scheduler.set_timesteps(5, device=device)

    sample = torch.randn(4, 16, device=device)
    velocity = torch.randn_like(sample)
    timestep = scheduler.timesteps[0]

    sigma = scheduler.sigmas[0].to(device=device)
    sigma_next = scheduler.sigmas[1].to(device=device)

    denoised = sample + (1.0 - timestep) * velocity
    ratio = sigma_next / sigma

    expected_step = (
        ratio * sample.float()
        + (1.0 - ratio) * denoised.float()
    )

    actual_step = scheduler.step(
        velocity,
        timestep,
        sample,
    ).prev_sample

    torch.testing.assert_close(
        actual_step.float(),
        expected_step,
        rtol=1e-6,
        atol=1e-6,
    )

    print("Data-ward Euler blend update: PASSED")

    # Verify the velocity form gives the same mathematical update.
    velocity_scheduler = H3FlowScheduler(
        shift=12.0,
        euler_update_form="velocity",
    )
    velocity_scheduler.set_timesteps(5, device=device)

    velocity_step = velocity_scheduler.step(
        velocity,
        velocity_scheduler.timesteps[0],
        sample,
    ).prev_sample

    torch.testing.assert_close(
        actual_step.float(),
        velocity_step.float(),
        rtol=2e-5,
        atol=2e-5,
    )

    print("Equivalent velocity update: PASSED")

    # Verify stochastic re-noising is reproducible.
    stochastic = H3FlowScheduler(
        shift=3.0,
        stochastic_sampling=True,
    )
    stochastic.set_timesteps(5, device=device)

    stochastic_sample = torch.randn(2, 8, device=device)
    stochastic_velocity = torch.randn_like(stochastic_sample)
    stochastic_t = stochastic.timesteps[0]

    g1 = torch.Generator(device=device).manual_seed(1234)
    g2 = torch.Generator(device=device).manual_seed(1234)

    result1 = stochastic.step(
        stochastic_velocity,
        stochastic_t,
        stochastic_sample,
        generator=g1,
    ).prev_sample

    stochastic_again = H3FlowScheduler(
        shift=3.0,
        stochastic_sampling=True,
    )
    stochastic_again.set_timesteps(5, device=device)

    result2 = stochastic_again.step(
        stochastic_velocity,
        stochastic_again.timesteps[0],
        stochastic_sample,
        generator=g2,
    ).prev_sample

    torch.testing.assert_close(
        result1,
        result2,
        rtol=0,
        atol=0,
    )

    print("Stochastic re-noising reproducibility: PASSED")
    print(f"Peak CUDA allocation: ", end="")

    if torch.device(device).type == "cuda":
        print(
            f"{torch.cuda.max_memory_allocated(torch.device(device)) / 1048576:.1f} MiB"
        )
    else:
        print("CPU")


def main():
    print("=" * 62)
    print("MINIMAX H3 STANDALONE SCHEDULER SELF-TEST")
    print("=" * 62)

    run_device_tests("cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        run_device_tests("cuda:0")

    print("\n" + "=" * 62)
    print("H3 SCHEDULER TESTS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
