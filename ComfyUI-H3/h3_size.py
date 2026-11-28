"""MiniMax H3 resolution calculator node.

Uses the published 16:9 reference size tiers for 32-pixel alignment,
transposes those tiers for 9:16, and calculates other values with the
same megapixel/aspect-ratio ceiling formula used by the H3 size helper.
"""

import math


# Reference H3 size tiers: megapixel label -> (width, height), 16:9.
# The 9:16 entries are the exact transposes of these dimensions.
_TABLE_16_9 = {
    "0.2": (608, 352),
    "0.3": (736, 416),
    "0.4": (864, 480),
    "0.5": (960, 544),
    "0.6": (1056, 608),
    "0.7": (1152, 640),
    "0.8": (1216, 672),
    "0.9": (1280, 736),
    "0.98": (1344, 768),
    "1.0": (1376, 768),
    "1.2": (1504, 832),
    "1.5": (1664, 928),
    "1.8": (1824, 1024),
    "2.0": (1920, 1088),
}

_TABLE = {}
for _mp, (_width, _height) in _TABLE_16_9.items():
    _TABLE[("16:9", _mp)] = (_width, _height)
    _TABLE[("9:16", _mp)] = (_height, _width)


ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "21:9",
    "9:21",
)


def minimax_h3_size(aspect_ratio="16:9", megapixels=0.7, multiple=32):
    """Return H3 width and height for an aspect ratio and megapixel target."""
    aspect_ratio = str(aspect_ratio).strip()
    try:
        ratio_width_text, ratio_height_text = aspect_ratio.split(":")
        ratio_width = float(ratio_width_text)
        ratio_height = float(ratio_height_text)
        megapixels = float(megapixels)
        multiple = int(multiple)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid aspect ratio, megapixel value, or multiple.") from exc

    if (
        not math.isfinite(ratio_width)
        or not math.isfinite(ratio_height)
        or ratio_width <= 0
        or ratio_height <= 0
    ):
        raise ValueError("Aspect-ratio components must be positive finite numbers.")
    if not math.isfinite(megapixels) or megapixels <= 0:
        raise ValueError("Megapixels must be a positive finite number.")
    if multiple <= 0:
        raise ValueError("multiple must be a positive integer.")

    # Preserve the reference tiers for the two standard H3 orientations.
    mp_key = format(megapixels, ".12g")
    key = (aspect_ratio, mp_key)
    if multiple == 32 and key in _TABLE:
        return _TABLE[key]

    target_pixels = megapixels * 1_000_000
    ratio = ratio_width / ratio_height
    raw_height = math.sqrt(target_pixels / ratio)
    raw_width = raw_height * ratio

    width = math.ceil(raw_width / multiple) * multiple
    height = math.ceil(raw_height / multiple) * multiple
    return int(width), int(height)


class H3Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "megapixels": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
                "multiple": (
                    "INT",
                    {"default": 32, "min": 1, "max": 256, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "MiniMax H3/Utilities"
    DESCRIPTION = (
        "Calculate H3 width and height from aspect ratio, megapixels, "
        "and alignment multiple. Uses reference tiers for 16:9 and 9:16 "
        "when multiple is 32."
    )

    def calculate(self, aspect_ratio, megapixels, multiple):
        return minimax_h3_size(aspect_ratio, megapixels, multiple)
