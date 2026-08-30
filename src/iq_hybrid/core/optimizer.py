"""Compatibility wrapper delegating to modern architecture strategies."""

from typing import Any

from iq_hybrid.core.greedy_engine import compute_stats, run_greedy_optimization
from iq_hybrid.core.types import ImportanceTable, TensorAssignments, TensorName


def optimize_quantization(
    model: dict[str, Any],
    importance_table: ImportanceTable,
    target_size_mib: float,
    tied_groups: list[list[TensorName]] | None = None,
    arch: Any = None,
) -> tuple[TensorAssignments, dict[TensorName, int]]:
    """Legacy wrapper delegating to architecture strategy and greedy engine."""
    if arch is None:
        from iq_hybrid.architectures.registry import detect_architecture_strategy

        arch = detect_architecture_strategy(model)

    return run_greedy_optimization(
        arch=arch,
        importance_table=importance_table,
        tied_groups=tied_groups or [],
        model=model,
        target_size_mib=target_size_mib,
        wide_ladder=False,
    )


__all__ = ["compute_stats", "optimize_quantization", "run_greedy_optimization"]
