"""Duration calculator for MiniMax H3 frame counts.

Returns both the direct seconds * FPS frame count and the nearest temporal
length accepted by the current H3 sampler (5 + 17*n). The requested frame
count is never clamped to the sampler's present UI maximum.
"""

import math


def calculate_h3_frames(seconds, fps):
    seconds = float(seconds)
    fps = float(fps)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Duration must be a positive finite number of seconds.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be a positive finite number.")
    exact_frames = seconds * fps
    if not math.isfinite(exact_frames) or exact_frames > 9_223_372_036_854_775_807:
        raise ValueError("Requested frame count exceeds the supported integer range.")

    requested_frames = int(math.floor(exact_frames + 0.5))

    # Current H3 sampler accepts temporal lengths of 5 + 17*n frames.
    # Keep the requested count separately and calculate the nearest valid count.
    n = max(0, int(math.floor((requested_frames - 5) / 17.0 + 0.5)))
    h3_frames = 5 + 17 * n
    actual_seconds = h3_frames / fps
    return requested_frames, h3_frames, actual_seconds


class H3SecondsToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 30.0, "min": 0.001, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("requested_frames", "h3_compatible_frames", "actual_seconds")
    FUNCTION = "calculate"
    CATEGORY = "MiniMax H3/Utilities"
    DESCRIPTION = (
        "Converts any positive duration to frames. Also reports the nearest "
        "frame count accepted by the current H3 temporal rule (5 + 17*n), "
        "without limiting the requested frame count."
    )

    def calculate(self, seconds, fps):
        return calculate_h3_frames(seconds, fps)
