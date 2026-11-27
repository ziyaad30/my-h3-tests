"""ComfyUI node registration for MiniMax H3."""

from .h3_text_encoder import H3Conditioning, H3TextEncoder
from .h3_model_loader import H3ModelLoader

NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
    "H3Conditioning": H3Conditioning,
    "H3ModelLoader": H3ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
    "H3Conditioning": "H3 Conditioning",
    "H3ModelLoader": "H3 Model Loader",
}
