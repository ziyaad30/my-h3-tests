"""ComfyUI node registration for MiniMax H3."""

from .h3_text_encoder import H3Conditioning, H3TextEncoder
from .h3_model_loader import H3ModelLoader
from .h3_sampler import H3Sampler
from .h3_sampler_beta import H3SamplerBeta
from .h3_lora import H3LoRALoader
from .h3_video_vae import H3VideoVAELoader, H3VideoVAEDecode

NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
    "H3Conditioning": H3Conditioning,
    "H3ModelLoader": H3ModelLoader,
    "H3Sampler": H3Sampler,
    "H3SamplerBeta": H3SamplerBeta,
    "H3LoRALoader": H3LoRALoader,
    "H3VideoVAELoader": H3VideoVAELoader,
    "H3VideoVAEDecode": H3VideoVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
    "H3Conditioning": "H3 Conditioning",
    "H3ModelLoader": "H3 Model Loader",
    "H3Sampler": "H3 Sampler",
    "H3SamplerBeta": "H3 Sampler (Beta)",
    "H3LoRALoader": "H3 LoRA Loader",
    "H3VideoVAELoader": "H3 Video VAE",
    "H3VideoVAEDecode": "H3 Video VAE Decode",
}
