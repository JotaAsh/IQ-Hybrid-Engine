"""Constants and GGML block definitions for IQ-Hybrid Engine.

This module centralizes universal quantization tier definitions, BPW constants,
and GGML format properties.
"""

from iq_hybrid.core.types import TensorName, TierName

# ── GGUF type ID mappings ────────────────────────────────────────────────────

GGUF_TYPE_NAMES: dict[int, TierName] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q4_K",
    11: "Q5_K",
    12: "Q6_K",
    13: "Q5_K_M",
    14: "Q4_K_M",
    15: "IQ4_XS",
    16: "IQ4_NL",
    20: "IQ3_XXS",
    24: "IQ2_XXS",
    30: "IQ1_S",
}


# ── Tier ordering and BPW definitions ────────────────────────────────────────

# Ordered from worst to best quality
TIER_ORDER: list[TierName] = [
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "Q3_K",
    "IQ4_XS",
    "IQ4_NL",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "F16",
    "F32",
]

# Exact bits per weight from ggml block structs (ggml_type_sizef * 8).
# Does NOT include GGUF overhead — GGUF_OVERHEAD_FACTOR is applied separately.
MSE_BPW: dict[TierName, float] = {
    "IQ1_S": 1.5625,
    "IQ1_M": 1.75,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.3125,
    "IQ2_S": 2.5,
    "IQ2_M": 2.7,
    "Q2_K": 2.5625,
    "IQ3_XXS": 3.0625,
    "IQ3_XS": 3.3,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "Q3_K": 3.4375,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "Q4_K": 4.58,
    "Q5_K": 5.5,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
    "F16": 16.0,
    "F32": 32.0,
}

TIER_BPW: dict[TierName, float] = MSE_BPW

# Base divisor (8 bits × 1024 bytes × 1024 KB)
BITS_IN_MIB: float = 8 * 1024 * 1024.0
GGUF_OVERHEAD_FACTOR: float = 1.0

# Pre-computed size multipliers per tier (MiB per element, including overhead)
TIER_SIZE_MULTIPLIER: dict[TierName, float] = {
    tier: (bpw / BITS_IN_MIB) * GGUF_OVERHEAD_FACTOR for tier, bpw in TIER_BPW.items()
}

# Quality rank: higher index = better quality (same order as TIER_ORDER)
QUANT_RANK: dict[TierName, int] = {tier: i for i, tier in enumerate(TIER_ORDER)}


# ── Classifier algorithm parameters ─────────────────────────────────────────

# K-quant block alignment size (elements per block)
K_QUANT_BLOCK_SIZE: int = 256

# K-quants use blocks of K_QUANT_BLOCK_SIZE elements
K_QUANTS: set[TierName] = {"Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}

MOE_PAD_TYPES: set[str] = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_down",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
}

# Vectorized formats that trigger GGML_ASSERT crash when imatrix is missing
TIERS_REQUIRING_IMATRIX: set[TierName] = {
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
}

# Depth-factor: maximum alpha for early-layer rescue boost
DEPTH_ALPHA_MAX: float = 0.85

# BPW threshold below which depth-factor is disabled (extreme compression)
DEPTH_ACTIVATION_BPW: float = 3.5

# Decay multiplier: BPW at or above which class-based boost decays to 1.0
HIGH_BPW_THRESHOLD: float = 5.0

# Decay multiplier: denominator range for linear interpolation
DECAY_BPW_RANGE: float = 3.0

# Depth-factor: BPW at or above which tier saturation urgency is zero (Q6_K level)
Q6K_BPW_THRESHOLD: float = 6.5

# Depth-factor: denominator range for tier urgency interpolation
TIER_URGENCY_RANGE: float = 4.0

# Depth-factor: exponential decay rate for layer depth
DEPTH_DECAY_RATE: float = 2.0

# Depth-factor: anchor boost for early layers or full-attention tensors
ANCHOR_BOOST: float = 1.30

# Maximum number of early layers that receive anchor boost
EARLY_LAYER_CUTOFF: int = 2

# Residual BPW threshold for embedding tier selection
RESIDUAL_BPW_THRESHOLD: float = 2.5

# Importance spike detection ratio
SPIKE_IMPORTANCE_RATIO: float = 0.50
SPIKE_PREFERRED_TIER: TierName = "F16"


# ── Tensor name helpers ──────────────────────────────────────────────────────


def strip_weight(name: TensorName) -> TensorName:
    """Remove leading dots and trailing .weight/.bias suffixes from a tensor name."""
    name = name.lstrip(".")
    if name.endswith(".weight"):
        name = name[:-7]
    elif name.endswith(".bias"):
        name = name[:-5]
    return name
