"""Architecture registry and detection factory."""

from typing import Any

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.architectures.dense import DenseArchitecture
from iq_hybrid.architectures.gemma_moe import GemmaMoEArchitecture
from iq_hybrid.architectures.qwen_hybrid import QwenHybridArchitecture

_ARCHITECTURES: list[type[BaseArchitecture]] = [
    GemmaMoEArchitecture,
    QwenHybridArchitecture,
    DenseArchitecture,
]


def detect_architecture_strategy(model: dict[str, Any]) -> BaseArchitecture:
    """Detect and instantiate the appropriate architecture strategy for a model."""
    arch_name = model.get("architecture", "unknown")
    features = model.get("features", {})
    tensors = model.get("tensors", {})
    names = list(tensors.keys())

    # 1. Gemma MoE or specific MoE structures
    if (
        arch_name in ("gemma4", "mellum2")
        or features.get("has_moe")
        or any("layer_output_scale" in n for n in names)
        or (any("exps" in n for n in names) and not any("ssm_" in n for n in names))
    ):
        return GemmaMoEArchitecture()

    # 2. Qwen / SSM / Bailing hybrids
    if (
        arch_name in ("qwen35", "bailingmoe3")
        or features.get("has_ssm")
        or features.get("has_mtp")
        or any("ssm_" in n for n in names)
        or any("nextn" in n for n in names)
    ):
        return QwenHybridArchitecture()

    # 3. Dense default
    return DenseArchitecture()
