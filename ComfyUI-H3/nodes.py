"""ComfyUI node registration for MiniMax H3."""

from .h3_text_encoder import H3Conditioning, H3TextEncoder
from .h3_model_loader import H3ModelLoader
from .h3_sampler import H3Sampler
from .h3_sampler_beta import H3SamplerBeta
from .h3_sampler_res_multistep import H3SamplerResMultistep
from .h3_sampler_fl2va_beta import H3SamplerFL2VABeta
from .h3_lora import H3LoRALoader
from .h3_video_vae import H3VideoVAELoader, H3VideoVAEDecode
from .h3_video_vae_beta import H3VideoVAEDecodeBeta
from .h3_size import H3Size
from .h3_duration import H3SecondsToFrames

NODE_CLASS_MAPPINGS = {
    "H3TextEncoder": H3TextEncoder,
    "H3Conditioning": H3Conditioning,
    "H3ModelLoader": H3ModelLoader,
    "H3Sampler": H3Sampler,
    "H3SamplerBeta": H3SamplerBeta,
    "H3SamplerResMultistep": H3SamplerResMultistep,
    "H3SamplerFL2VABeta": H3SamplerFL2VABeta,
    "H3LoRALoader": H3LoRALoader,
    "H3VideoVAELoader": H3VideoVAELoader,
    "H3VideoVAEDecode": H3VideoVAEDecode,
    "H3VideoVAEDecodeBeta": H3VideoVAEDecodeBeta,
    "H3Size": H3Size,
    "H3SecondsToFrames": H3SecondsToFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TextEncoder": "H3 Text Encoder",
    "H3Conditioning": "H3 Conditioning",
    "H3ModelLoader": "H3 Model Loader",
    "H3Sampler": "H3 Sampler",
    "H3SamplerBeta": "H3 Sampler (Beta)",
    "H3SamplerResMultistep": "H3 Sampler (RES Multistep Experimental)",
    "H3SamplerFL2VABeta": "H3 Sampler (FL2VA Beta)",
    "H3LoRALoader": "H3 LoRA Loader",
    "H3VideoVAELoader": "H3 Video VAE",
    "H3VideoVAEDecode": "H3 Video VAE Decode",
    "H3VideoVAEDecodeBeta": "H3 Video VAE Decode (BETA)",
    "H3Size": "MiniMax H3 Size",
    "H3SecondsToFrames": "H3 Seconds to Frames",
}
