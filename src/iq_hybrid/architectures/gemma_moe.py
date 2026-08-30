"""Architecture strategy for Gemma 4 MoE models."""

from typing import Any

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.core.constants import strip_weight
from iq_hybrid.core.types import ImportanceTable, TensorClass, TensorName, TierName

GEMMA_MOE_TENSOR_CLASS: dict[str, TensorClass] = {
    "attn_q": "attn_proj",
    "attn_k": "attn_proj",
    "attn_v": "attn_proj",
    "attn_qkv": "attn_proj",
    "attn_output": "attn_proj",
    "proj": "attn_proj",
    "attn_gate": "gate",
    "inp_gate": "router",
    "ffn_gate_inp": "router",
    "ffn_router": "router",
    "ffn_gate": "ffn_gate_up",
    "ffn_up": "ffn_gate_up",
    "ffn_down": "ffn_down",
    "ffn_gate_exps": "ffn_gate_up",
    "ffn_up_exps": "ffn_gate_up",
    "ffn_down_exps": "ffn_down",
    "token_embd": "embd",
    "per_layer_token_embd": "embd",
    "output": "embd",
}


class GemmaMoEArchitecture(BaseArchitecture):
    """Architecture strategy for Gemma 4 MoE models."""

    name: str = "gemma_moe"

    # Tunable hyperparameters for 42+ layer MoE
    depth_alpha_max: float = 0.60  # Reduced linear bias to avoid unprotecting deep layers
    depth_decay_rate: float = 1.20  # Flatter decay curve for deep layers
    depth_activation_bpw: float = 4.0
    early_layer_cutoff: int = 2
    anchor_boost: float = 1.40

    def classify_tensor(self, name: TensorName) -> TensorClass:
        clean = strip_weight(name)
        parts = clean.split(".")
        suffix = parts[-1]

        if any(
            term in clean
            for term in ("norm", "layer_output_scale", "post_attention_norm", "post_ffw_norm")
        ):
            return "norms"

        if suffix in GEMMA_MOE_TENSOR_CLASS:
            return GEMMA_MOE_TENSOR_CLASS[suffix]

        for pattern, cls in GEMMA_MOE_TENSOR_CLASS.items():
            if pattern in clean:
                return cls

        return "default"

    def get_fixed_tensors(
        self,
        model: dict[str, Any],
        imp_table: ImportanceTable,
        target_size_mib: float,
        wide_ladder: bool = False,
    ) -> dict[TensorName, TierName]:
        """Inherit universal rules and apply Gemma MoE shared expert overrides."""
        fixed = super().get_fixed_tensors(
            model=model,
            imp_table=imp_table,
            target_size_mib=target_size_mib,
            wide_ladder=wide_ladder,
        )
        model_tensors = model.get("tensors", {})

        # Shared MLP experts -> Q6_K
        for name in model_tensors:
            if "shexp" in name:
                fixed[name] = "Q6_K"

        return fixed

    def get_structural_tied_groups(self, tensor_names: list[TensorName]) -> list[list[TensorName]]:
        """Match ffn_gate and ffn_up (both dense and expert) for synchronous mutation."""
        tied: list[list[TensorName]] = []
        visited: set[TensorName] = set()

        for name in tensor_names:
            if name in visited:
                continue

            partner: TensorName | None = None

            # Explicit MoE with _exps suffix
            if "ffn_gate_exps" in name:
                partner = name.replace("ffn_gate_exps", "ffn_up_exps")
            # Standard / dense names (avoiding collision with ffn_gate_inp / routers)
            elif "ffn_gate" in name and "ffn_gate_inp" not in name:
                partner = name.replace("ffn_gate", "ffn_up")

            if partner and partner in tensor_names:
                tied.append([name, partner])
                visited.add(name)
                visited.add(partner)

        return tied
