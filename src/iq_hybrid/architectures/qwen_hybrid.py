"""Architecture strategy for Qwen 3.5, BailingMoE3, and SSM hybrid models."""

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.core.constants import strip_weight
from iq_hybrid.core.types import TensorClass, TensorName, TierName

QWEN_TENSOR_CLASS: dict[str, TensorClass] = {
    # Qwen 3.5 hybrid
    "attn_gate": "gate",
    "ssm_alpha": "gate",
    "ssm_beta": "gate",
    "attn_q": "attn_proj",
    "attn_k": "attn_proj",
    "attn_v": "attn_proj",
    "attn_qkv": "attn_proj",
    "attn_output": "attn_proj",
    "ffn_gate": "ffn_gate_up",
    "ffn_up": "ffn_gate_up",
    "ffn_down": "ffn_down",
    "ssm_out": "ffn_down",
    "ssm_conv1d": "norms",
    "router": "norms",
    "ssm_dt": "ssm_params",
    "ssm_a": "ssm_params",
    "nextn": "mtp",
    "ffn_gate_exps": "ffn_gate_up",
    "ffn_up_exps": "ffn_gate_up",
    "ffn_down_exps": "ffn_down",
    "ffn_gate_inp": "norms",
    # BailingMoE3 / Ling hybrid
    "ffn_gate_shexp": "ffn_gate_up",
    "ffn_up_shexp": "ffn_gate_up",
    "ffn_down_shexp": "ffn_down",
    "attn_q_a": "attn_proj",
    "attn_q_b": "attn_proj",
    "attn_k_b": "attn_proj",
    "attn_v_b": "attn_proj",
    "attn_kv_a_mqa": "attn_proj",
    "attn_q_a_norm": "norms",
    "attn_kv_a_norm": "norms",
    # Global tensors
    "token_embd": "embd",
    "output": "embd",
    "output_norm": "norms",
    "attn_norm": "norms",
    "post_attention_norm": "norms",
    "ssm_norm": "norms",
    "attn_k_norm": "norms",
    "attn_q_norm": "norms",
}


class QwenHybridArchitecture(BaseArchitecture):
    """Architecture strategy for Qwen 3.5, BailingMoE3, and SSM hybrid models."""

    name: str = "qwen_hybrid"

    # Tunable hyperparameters
    depth_alpha_max: float = 1.10  # Greater strength to protect ffn_down in blk.0-2
    depth_decay_rate: float = 1.80  # Gentler decline to avoid penalizing intermediate layers
    depth_activation_bpw: float = 3.2
    early_layer_cutoff: int = 2
    anchor_boost: float = 1.60

    def classify_tensor(self, name: TensorName) -> TensorClass:
        clean = strip_weight(name)
        parts = clean.split(".")
        suffix = parts[-1]

        if any(term in clean for term in ("norm", "ssm_norm", "conv1d")):
            return "norms"

        if suffix in QWEN_TENSOR_CLASS:
            return QWEN_TENSOR_CLASS[suffix]

        for pattern, cls in QWEN_TENSOR_CLASS.items():
            if pattern in clean:
                return cls

        return "default"

    def get_early_layer_floor(
        self,
        cls: TensorClass,
        layer_idx: int,
        target_bpw: float,
    ) -> TierName | None:
        """Prevent extreme down-scaling (IQ1/IQ2) in early hybrid layers at medium/high budgets."""
        if target_bpw >= 5.0 and layer_idx <= self.early_layer_cutoff:
            if cls == "ffn_down":
                return "IQ4_XS"
            if cls == "ffn_gate_up":
                return "IQ4_XS"
        elif target_bpw >= 4.4 and layer_idx <= 1:
            if cls == "ffn_down":
                return "IQ3_S"
        return None
