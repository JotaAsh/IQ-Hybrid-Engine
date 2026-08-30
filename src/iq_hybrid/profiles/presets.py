"""Declarative quantization presets and external profile loaders.

Supports:
  - Built-in presets (quality_ladder, wide_ladder)
  - Custom JSON / YAML external profile recipes
"""

import json
from pathlib import Path
from typing import Any

import yaml

from iq_hybrid.core.constants import BITS_IN_MIB
from iq_hybrid.core.types import TensorClass, TierName
from iq_hybrid.profiles.base import BaseProfile

PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "quality_ladder": {
        "description": "Strict high-quality profile with hard floor at IQ4_XS (no IQ1/IQ2/IQ3)",
        "force_wide_ladder": True,
        "ladders": {
            "attn_proj": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
            "ffn_gate_up": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
            "ffn_down": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
            "default": ["IQ4_XS", "IQ4_NL", "Q5_K", "Q6_K", "Q8_0"],
        },
        "bases": {
            "attn_proj": "IQ4_XS",
            "ffn_gate_up": "IQ4_XS",
            "ffn_down": "IQ4_XS",
            "default": "IQ4_XS",
        },
    },
    "wide_ladder": {
        "description": "Generic wide-ladder profile allowing lower rungs down to IQ1_S for manual MiB targets",
        "force_wide_ladder": True,
    },
}


class DeclarativeProfile(BaseProfile):
    """Profile instantiated from declarative dictionary definitions or external files."""

    def __init__(
        self,
        name: str,
        target_bpw: float | None = None,
        target_size_mib: float | None = None,
        description: str = "",
        force_wide_ladder: bool = False,
        ladders: dict[TensorClass, list[TierName]] | None = None,
        bases: dict[TensorClass, TierName] | None = None,
        ceilings: dict[TensorClass, TierName] | None = None,
    ) -> None:
        self.name = name
        self.target_bpw = target_bpw
        self.target_size_mib = target_size_mib
        self.description = description
        self.force_wide_ladder = force_wide_ladder
        self.ladders = ladders or {}
        self.bases = bases or {}
        self.ceilings = ceilings or {}

    def calculate_budget_mib(self, total_elements: int, requested_value: str | float) -> float:
        if self.target_size_mib is not None:
            return float(self.target_size_mib)
        if self.target_bpw is not None:
            if total_elements <= 0:
                raise ValueError("total_elements must be positive to compute BPW budget")
            return (self.target_bpw * total_elements) / BITS_IN_MIB
        # Fallback to numeric requested_value if profile has no intrinsic BPW/MiB
        try:
            return float(requested_value)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Profile '{self.name}' has no intrinsic target BPW/MiB. An explicit numeric size is required."
            ) from err

    def get_output_tag(self, target_mib: float) -> str:
        if self.target_bpw is not None or self.name in PRESET_DEFINITIONS:
            return self.name
        target_gb = target_mib / 1024.0
        return f"{self.name}-{target_gb:.2f}GB"


def get_preset_profile(name: str) -> DeclarativeProfile:
    """Instantiate a built-in preset profile by name."""
    norm_name = name.strip()
    # Check case-insensitive match
    for k, data in PRESET_DEFINITIONS.items():
        if k.upper() == norm_name.upper():
            return DeclarativeProfile(
                name=k,
                target_bpw=data.get("target_bpw"),
                target_size_mib=data.get("target_size_mib"),
                description=data.get("description", ""),
                force_wide_ladder=data.get("force_wide_ladder", False),
                ladders=data.get("ladders"),
                bases=data.get("bases"),
                ceilings=data.get("ceilings"),
            )
    raise KeyError(f"Unknown preset profile '{name}'. Available: {list(PRESET_DEFINITIONS.keys())}")


def load_profile_from_file(file_path: str | Path) -> DeclarativeProfile:
    """Load a custom profile recipe from a JSON or YAML file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Profile recipe file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Profile file {path} must contain a valid dictionary structure.")

    profile_name = data.get("name", path.stem)
    return DeclarativeProfile(
        name=profile_name,
        target_bpw=data.get("target_bpw"),
        target_size_mib=data.get("target_size_mib") or data.get("size_mib"),
        description=data.get("description", f"Custom profile loaded from {path.name}"),
        force_wide_ladder=data.get("force_wide_ladder", False),
        ladders=data.get("ladders"),
        bases=data.get("bases"),
        ceilings=data.get("ceilings"),
    )
