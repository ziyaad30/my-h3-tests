"""ComfyUI node registration for MiniMax H3."""

from .h3_text_encoder import H3Conditioning, H3TextEncoder

NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
    "H3Conditioning": H3Conditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
    "H3Conditioning": "H3 Conditioning",
}
