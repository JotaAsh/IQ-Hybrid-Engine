"""Architecture strategy for standard dense LLMs (LLaMA, Mistral, etc.)."""

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.core.constants import strip_weight
from iq_hybrid.core.types import TensorClass, TensorName

DENSE_TENSOR_CLASS: dict[str, TensorClass] = {
    # Attention projections
    "attn_q": "attn_proj",
    "attn_k": "attn_proj",
    "attn_v": "attn_proj",
    "attn_qkv": "attn_proj",
    "attn_output": "attn_proj",
    "wq": "attn_proj",
    "wk": "attn_proj",
    "wv": "attn_proj",
    "wo": "attn_proj",
    "wqkv": "attn_proj",
    # Feed-forward network
    "ffn_gate": "ffn_gate_up",
    "ffn_up": "ffn_gate_up",
    "ffn_down": "ffn_down",
    "w1": "ffn_gate_up",
    "w3": "ffn_gate_up",
    "w2": "ffn_down",
    # Normalization layers
    "attn_norm": "norms",
    "ffn_norm": "norms",
    "post_attention_norm": "norms",
    "output_norm": "norms",
    "norm": "norms",
    # Global tensors
    "token_embd": "embd",
    "output": "embd",
    "nextn": "mtp",
}


class DenseArchitecture(BaseArchitecture):
    """Architecture strategy for standard dense LLMs."""

    name: str = "dense"

    # Tunable hyperparameters for standard dense models
    depth_alpha_max: float = 0.85
    depth_decay_rate: float = 2.0
    depth_activation_bpw: float = 3.5
    early_layer_cutoff: int = 2
    anchor_boost: float = 1.30

    def classify_tensor(self, name: TensorName) -> TensorClass:
        clean = strip_weight(name)
        parts = clean.split(".")
        suffix = parts[-1]

        if any(term in clean for term in ("norm", "layer_output_scale")):
            return "norms"

        if suffix in DENSE_TENSOR_CLASS:
            return DENSE_TENSOR_CLASS[suffix]

        for pattern, cls in DENSE_TENSOR_CLASS.items():
            if pattern in clean:
                return cls

        return "default"
