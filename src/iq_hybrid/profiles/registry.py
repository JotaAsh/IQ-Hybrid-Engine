"""Profile registry and resolution factory."""

from pathlib import Path

from iq_hybrid.profiles.base import BaseProfile
from iq_hybrid.profiles.presets import (
    PRESET_DEFINITIONS,
    get_preset_profile,
    load_profile_from_file,
)
from iq_hybrid.profiles.standard import StandardProfile

AVAILABLE_PRESETS: list[str] = list(PRESET_DEFINITIONS.keys())


def resolve_profile(arg: str | float) -> tuple[BaseProfile, float | str, bool]:
    """Resolve a size/profile argument into (profile_instance, raw_or_computed_val, is_preset).

    Args:
        arg: Profile name (e.g. 'quality_ladder', 'wide_ladder'), file path (.json/.yaml), or numeric MiB (e.g. 6310).

    Returns:
        tuple of (BaseProfile, target_val, is_preset_boolean)
    """
    if isinstance(arg, (int, float)):
        return StandardProfile(), float(arg), False

    str_val = str(arg).strip()

    # 1. Check if argument is a file path
    p = Path(str_val)
    if p.is_file() or str_val.lower().endswith((".json", ".yaml", ".yml")):
        profile = load_profile_from_file(p)
        val = (
            profile.target_bpw
            if profile.target_bpw is not None
            else (profile.target_size_mib or 0.0)
        )
        return profile, val, True

    # 2. Check if argument matches any built-in preset
    for preset_name in PRESET_DEFINITIONS:
        if preset_name.upper() == str_val.upper():
            profile = get_preset_profile(preset_name)
            target_bpw = profile.target_bpw or 0.0
            return profile, target_bpw, True

    # 3. Numeric manual target size in MiB
    try:
        num_val = float(str_val)
        return StandardProfile(), num_val, False
    except ValueError as err:
        raise ValueError(
            f"Unrecognized profile or size '{arg}'. "
            f"Must be a numeric size in MiB, a preset name ({AVAILABLE_PRESETS}), or a path to a .json/.yaml profile recipe."
        ) from err
