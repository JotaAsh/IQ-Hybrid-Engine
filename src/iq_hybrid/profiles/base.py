"""Base abstract class for quantization profile strategies."""

from abc import ABC, abstractmethod

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.core.types import TensorClass, TierName


class BaseProfile(ABC):
    """Abstract base class defining profile budget resolution and naming conventions."""

    name: str = "base"
    description: str = ""
    force_wide_ladder: bool = False
    ladders: dict[TensorClass, list[TierName]] = {}
    bases: dict[TensorClass, TierName] = {}
    ceilings: dict[TensorClass, TierName] = {}

    @abstractmethod
    def calculate_budget_mib(self, total_elements: int, requested_value: str | float) -> float:
        """Calculate target size in MiB based on total elements and requested value."""
        ...

    @abstractmethod
    def get_output_tag(self, target_mib: float) -> str:
        """Return the naming tag to be embedded into the quantized GGUF filename."""
        ...

    def get_tier_ladder(
        self,
        arch: BaseArchitecture,
        cls: TensorClass,
        wide_ladder: bool = False,
    ) -> list[TierName]:
        """Resolve tier ladder for a class, prioritizing profile overrides over architecture defaults."""
        if cls in self.ladders:
            return list(self.ladders[cls])
        if "default" in self.ladders and cls not in (
            "norms",
            "ssm_params",
            "gate",
            "mtp",
            "router",
        ):
            return list(self.ladders["default"])
        return arch.get_tier_ladder(cls, wide_ladder=wide_ladder)

    def get_ladder_base(
        self,
        arch: BaseArchitecture,
        cls: TensorClass,
        wide_ladder: bool = False,
        target_bpw: float = 4.5,
    ) -> TierName:
        """Resolve ladder base for a class, prioritizing profile overrides over architecture defaults."""
        if cls in self.bases:
            return self.bases[cls]
        if "default" in self.bases and cls not in ("norms", "ssm_params", "gate", "mtp"):
            return self.bases["default"]
        return arch.get_ladder_base(cls, wide_ladder=wide_ladder, target_bpw=target_bpw)

    def get_ladder_ceiling(
        self,
        arch: BaseArchitecture,
        cls: TensorClass,
    ) -> TierName:
        """Resolve ladder ceiling for a class, prioritizing profile overrides over architecture defaults."""
        if cls in self.ceilings:
            return self.ceilings[cls]
        return arch.get_ladder_ceiling(cls)
