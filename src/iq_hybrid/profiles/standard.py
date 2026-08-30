"""Standard manual MiB profile strategy."""

from iq_hybrid.profiles.base import BaseProfile


class StandardProfile(BaseProfile):
    """Standard profile for explicit manual size in MiB."""

    name: str = "standard"
    description: str = "Explicit manual target file size in MiB"
    force_wide_ladder: bool = False

    def calculate_budget_mib(self, total_elements: int, requested_value: str | float) -> float:
        try:
            return float(requested_value)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid manual size value: {requested_value}") from err

    def get_output_tag(self, target_mib: float) -> str:
        target_gb = target_mib / 1024.0
        return f"{target_gb:.2f}GB"
