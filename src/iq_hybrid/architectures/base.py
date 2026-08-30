"""Base architecture definition and centralized quantization rules.

Centralizes universal quantization tier ladders, budget base heuristics,
ladder ceilings, MTP module protection, and dynamic vocab embedding scaling.
Subclasses only need to define tensor classification, structural tied groups,
and architecture-specific overrides.
"""

import math
from abc import ABC, abstractmethod
from typing import Any

from iq_hybrid.core.constants import BITS_IN_MIB, MSE_BPW
from iq_hybrid.core.types import ImportanceTable, TensorClass, TensorName, TierName

# ── Universal Quantization Tier Ladders ───────────────────────────────────────

DEFAULT_LADDERS: dict[TensorClass, list[TierName]] = {
    "embd": ["Q3_K", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
    "gate": ["Q8_0"],
    "router": ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16"],
    "attn_proj": [
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    ],
    "ffn_gate_up": [
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    ],
    "ffn_down": [
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    ],
    "mtp": ["Q8_0"],
    "norms": ["F16"],
    "ssm_params": ["F16"],
    "default": [
        "IQ1_S",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
        "IQ4_XS",
        "IQ4_NL",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    ],
}

DEFAULT_LADDER_BASES: dict[TensorClass, TierName] = {
    "gate": "Q8_0",
    "router": "Q8_0",
    "attn_proj": "Q8_0",
    "ffn_gate_up": "IQ4_XS",
    "ffn_down": "Q6_K",
    "norms": "F16",
    "ssm_params": "F16",
    "mtp": "Q8_0",
    "embd": "Q5_K",
    "default": "IQ4_XS",
}

DEFAULT_LADDER_CEILINGS: dict[TensorClass, TierName] = {
    "gate": "Q8_0",
    "router": "F16",
    "attn_proj": "Q8_0",
    "ffn_gate_up": "Q8_0",
    "ffn_down": "Q8_0",
    "norms": "F16",
    "ssm_params": "F16",
    "mtp": "Q8_0",
    "embd": "Q8_0",
    "default": "Q8_0",
}


# ── Base Architecture Strategy ────────────────────────────────────────────────


class BaseArchitecture(ABC):
    """Abstract base class defining tensor classification and quantization rules."""

    name: str = "base"

    # Architecture-specific ladder overrides (optional in subclasses)
    ladders: dict[TensorClass, list[TierName]] = {}
    ladder_bases: dict[TensorClass, TierName] = {}
    ladder_ceilings: dict[TensorClass, TierName] = {}

    # Tunable hyperparameters per architecture
    depth_alpha_max: float = 0.85
    depth_decay_rate: float = 2.0
    depth_activation_bpw: float = 3.5
    early_layer_cutoff: int = 2
    anchor_boost: float = 1.30

    @abstractmethod
    def classify_tensor(self, name: TensorName) -> TensorClass:
        """Map a tensor name to its semantic class."""
        ...

    def get_tier_ladder(self, cls: TensorClass, wide_ladder: bool = False) -> list[TierName]:
        """Return the sequence of allowable tiers for the tensor class."""
        ladder = self.ladders.get(cls, DEFAULT_LADDERS.get(cls, DEFAULT_LADDERS["default"]))
        if not wide_ladder:
            base_tier = self.get_ladder_base(cls, wide_ladder=False)
            if base_tier in ladder:
                idx = ladder.index(base_tier)
                return ladder[idx:]
        return ladder

    def get_ladder_base(
        self,
        cls: TensorClass,
        wide_ladder: bool = False,
        target_bpw: float = 4.5,
    ) -> TierName:
        """Return the initial baseline tier for a tensor class given budget and ladder width."""
        ladder = self.get_tier_ladder(cls, wide_ladder=True)
        if not wide_ladder:
            return self.ladder_bases.get(cls, DEFAULT_LADDER_BASES.get(cls, ladder[0]))

        # Progressive ladder base for wide-ladder depending on budget
        if target_bpw >= 6.5:
            # High budget (>= 6.5 BPW): never start below Q6_K / Q5_K
            if cls in ("attn_proj", "gate", "mtp"):
                preferred = "Q6_K"
            elif cls in ("ffn_down", "ffn_gate_up", "default"):
                preferred = "Q5_K"
            else:
                preferred = ladder[0]
        elif target_bpw >= 5.4:
            # Medium budget (>= 5.4 BPW): start at IQ4
            if cls in ("attn_proj", "gate", "mtp"):
                preferred = "IQ4_NL"
            elif cls in ("ffn_down", "ffn_gate_up", "default"):
                preferred = "IQ4_XS"
            else:
                preferred = ladder[0]
        elif target_bpw >= 4.4:
            # Standard budget (>= 4.4 BPW): start at IQ3
            if cls in ("attn_proj", "gate"):
                preferred = "IQ3_S"
            elif cls in ("ffn_down", "ffn_gate_up", "default"):
                preferred = "IQ3_XXS"
            else:
                preferred = ladder[0]
        else:
            # Low budget (< 4.4 BPW): full wide-ladder from bottom rung
            preferred = ladder[0]

        if preferred in ladder:
            return preferred
        return ladder[0]

    def get_ladder_ceiling(self, cls: TensorClass) -> TierName:
        """Return the maximum allowable tier (ceiling) for a tensor class."""
        return self.ladder_ceilings.get(cls, DEFAULT_LADDER_CEILINGS.get(cls, "Q8_0"))

    def get_fixed_tensors(
        self,
        model: dict[str, Any],
        imp_table: ImportanceTable,
        target_size_mib: float,
        wide_ladder: bool = False,
    ) -> dict[TensorName, TierName]:
        """Return deterministic tier assignments for non-elastic tensors.

        Handles:
          1. Normalization and SSM parameters -> F16
          2. MTP / NextN modules (e.g. blk.32 / nextn.*) -> Q8_0 weights, F16 norms
          3. Vocab embeddings (token_embd & output) -> dynamically scaled by target BPW
        """
        fixed: dict[TensorName, TierName] = {}
        model_tensors = model.get("tensors", {})

        # 1. Detect any MTP / NextN block layers
        mtp_layers: set[int] = set()
        for name in model_tensors:
            if "nextn" in name:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0].lower() in ("blk", "blk"):
                    try:
                        mtp_layers.add(int(parts[1]))
                    except ValueError:
                        pass

        # 2. Fix normalizations to F16 and MTP layer weights to Q8_0
        for name in model_tensors:
            cls = self.classify_tensor(name)
            if cls in ("norms", "ssm_params") or name.endswith("norm.weight"):
                fixed[name] = "F16"
            elif "nextn" in name:
                fixed[name] = "Q8_0"
            else:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0].lower() in ("blk", "blk"):
                    try:
                        if int(parts[1]) in mtp_layers:
                            fixed[name] = "Q8_0"
                    except ValueError:
                        pass

        # 3. Dynamically scale vocab embeddings (token_embd & output) based on target BPW
        total_elements = sum(v.get("n_elements", 0) for v in model_tensors.values())
        target_bpw = (
            (target_size_mib * BITS_IN_MIB) / max(1, total_elements) if total_elements > 0 else 4.5
        )

        if target_bpw >= 7.5:
            embd_tier: TierName = "Q8_0"
        elif target_bpw >= 6.0:
            embd_tier = "Q6_K"
        elif target_bpw >= 5.4:
            embd_tier = "Q5_K"
        elif target_bpw >= 3.8 or not wide_ladder:
            embd_tier = "IQ4_NL"
        else:
            embd_tier = "Q3_K"

        for name in model_tensors:
            if self.classify_tensor(name) == "embd":
                fixed[name] = embd_tier

        return fixed

    def get_diminishing_exponent(self, cls: TensorClass) -> float:
        """Return class-specific diminishing return exponent on Delta-MSE ratio.

        Higher exponent (>0.6) aggressively penalizes small/high-importance tensors
        from taking expensive Q6/Q8 upgrades when larger matrices are still in lower tiers.
        """
        if cls in ("attn_proj", "gate", "router"):
            return 0.70  # Stronger saturation damping on small attention/gating tensors
        if cls in ("ffn_down", "ffn_gate_up"):
            return 0.35  # Softer damping on massive MLP tensors so they can progress
        return 0.50

    def get_structural_tied_groups(self, tensor_names: list[TensorName]) -> list[list[TensorName]]:
        """Return structural tied tensor pairs (e.g. SwiGLU expert gate/up)."""
        return []

    def is_anchor_layer(self, clean_name: str, layer_idx: int) -> bool:
        """Check if tensor belongs to an anchor layer or full attention block."""
        is_early = layer_idx <= self.early_layer_cutoff
        is_full_attn = any(k in clean_name for k in ("attn_k", "attn_q", "attn_v", "attn_output"))
        return is_early or is_full_attn

    def get_depth_multiplier(
        self,
        rep_name: TensorName,
        cur_tier: TierName,
        target_bpw: float,
        n_layers: int = 32,
    ) -> float:
        """Calculate architecture-specific depth multiplier using template parameters."""
        if target_bpw <= self.depth_activation_bpw:
            return 1.0

        alpha = self.depth_alpha_max * max(
            0.0, min(1.0, (target_bpw - self.depth_activation_bpw) / 3.0)
        )

        cur_bpw = MSE_BPW.get(cur_tier, 4.0)
        if cur_bpw >= 6.5:
            return 1.0
        tier_urgency = max(0.0, min(1.0, (6.5 - cur_bpw) / 4.0))

        parts = rep_name.split(".")
        is_blk = len(parts) >= 2 and parts[0] in ("blk", "BLK")
        if not is_blk:
            return 1.0

        try:
            layer_idx = int(parts[1])
        except ValueError:
            return 1.0

        depth_decay = math.exp(-self.depth_decay_rate * (layer_idx / max(1, n_layers)))
        anchor = self.anchor_boost if self.is_anchor_layer(rep_name, layer_idx) else 1.0

        return 1.0 + (alpha * depth_decay * anchor * tier_urgency)

    def get_early_layer_floor(
        self,
        cls: TensorClass,
        layer_idx: int,
        target_bpw: float,
    ) -> TierName | None:
        """Return a protected baseline tier for critical early layers under high budgets."""
        return None
