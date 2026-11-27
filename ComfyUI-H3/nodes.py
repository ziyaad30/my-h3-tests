"""ComfyUI node registration for MiniMax H3."""

from .h3_text_encoder import H3TextEncoder

NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
}
