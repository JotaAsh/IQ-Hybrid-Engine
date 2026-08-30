"""Greedy imatrix-driven optimization engine.

Optimizes quantization tier assignments for individual tensors and tied groups
to minimize total model degradation within a target file size.
"""

import heapq
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.core.constants import (
    BITS_IN_MIB,
    K_QUANT_BLOCK_SIZE,
    K_QUANTS,
    MOE_PAD_TYPES,
    MSE_BPW,
    TIER_SIZE_MULTIPLIER,
    TIERS_REQUIRING_IMATRIX,
)
from iq_hybrid.core.types import (
    GroupRegistry,
    ImportanceTable,
    TensorAssignments,
    TensorClass,
    TensorName,
    TierName,
)
from iq_hybrid.profiles.base import BaseProfile

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(order=True, slots=True)
class UpgradeItem:
    """Priority queue item. Compared only by neg_utility."""

    neg_utility: float
    group_id: int = field(compare=False)
    next_tier: TierName = field(compare=False)
    cost_delta: float = field(compare=False)


# ── Core math ────────────────────────────────────────────────────────────────


def _size_mib(tier: TierName, n_elements: int) -> float:
    """Tensor size in MiB, with GGUF overhead."""
    if n_elements <= 0:
        return 0.0
    return n_elements * TIER_SIZE_MULTIPLIER.get(tier, 0.0)


@lru_cache(maxsize=128)
def _mse_delta(cur_tier: TierName, next_tier: TierName) -> float:
    """Theoretical MSE reduction when upgrading from *cur_tier* to *next_tier*."""
    cur_bpw = MSE_BPW.get(cur_tier, 4.0)
    next_bpw = MSE_BPW.get(next_tier, 5.0)
    return (2 ** (-2 * cur_bpw)) - (2 ** (-2 * next_bpw))


def _get_next_tier(
    arch: BaseArchitecture,
    cls: TensorClass,
    cur_tier: TierName,
    has_imatrix: bool = True,
    wide_ladder: bool = False,
    profile: BaseProfile | None = None,
) -> TierName | None:
    """Determine the next tier on the architecture/profile class-specific ladder."""
    if profile is not None:
        ladder = profile.get_tier_ladder(arch, cls, wide_ladder=wide_ladder)
        max_tier = profile.get_ladder_ceiling(arch, cls)
    else:
        ladder = arch.get_tier_ladder(cls, wide_ladder=wide_ladder)
        max_tier = arch.get_ladder_ceiling(cls)

    valid_ladder = [t for t in ladder if has_imatrix or (t not in TIERS_REQUIRING_IMATRIX)]

    if not valid_ladder:
        return None

    if cur_tier not in valid_ladder:
        cur_bpw = MSE_BPW.get(cur_tier, 0.0)
        for step in valid_ladder:
            if MSE_BPW.get(step, 0.0) > cur_bpw:
                return step
        return None

    cur_idx = valid_ladder.index(cur_tier)

    max_idx = len(valid_ladder) - 1
    if max_tier in valid_ladder:
        max_idx = valid_ladder.index(max_tier)

    if cur_idx >= max_idx or cur_idx + 1 >= len(valid_ladder):
        return None

    return valid_ladder[cur_idx + 1]


# ── Upgrade queue management ────────────────────────────────────────────────


def _push_upgrade(
    group_id: int,
    group_registry: GroupRegistry,
    assignments: TensorAssignments,
    tensor_importance: dict[TensorName, float],
    upgrade_queue: list[UpgradeItem],
    importance_table: ImportanceTable,
    arch: BaseArchitecture,
    target_bpw: float = 4.5,
    n_layers: int = 32,
    wide_ladder: bool = False,
    profile: BaseProfile | None = None,
) -> None:
    """Evaluate the next possible upgrade for a group and push it onto the queue."""
    g_names, g_elements, g_elements_padded = group_registry[group_id]
    rep_name = g_names[0]
    cur_tier = assignments[rep_name]

    rep_info = importance_table.get(rep_name, {})
    has_imatrix = rep_info.get("importance_mean", 0.0) > 0
    cls = arch.classify_tensor(rep_name)

    next_tier = _get_next_tier(
        arch=arch,
        cls=cls,
        cur_tier=cur_tier,
        has_imatrix=has_imatrix,
        wide_ladder=wide_ladder,
        profile=profile,
    )
    if not next_tier:
        return

    cur_padded = cur_tier in K_QUANTS
    next_padded = next_tier in K_QUANTS

    cur_elements = g_elements_padded if cur_padded else g_elements
    next_elements = g_elements_padded if next_padded else g_elements

    cur_cost = _size_mib(cur_tier, cur_elements)
    next_cost = _size_mib(next_tier, next_elements)
    cost_delta = next_cost - cur_cost

    if cost_delta <= 0:
        cost_delta = 1e-6

    g_importance = sum(tensor_importance.get(n, 0.0) for n in g_names)
    mse_d = _mse_delta(cur_tier, next_tier)

    # Class-specific geometric diminishing return factor based on MSE reduction curve
    diminishing_exp = arch.get_diminishing_exponent(cls)
    if profile is not None:
        base_tier = profile.get_ladder_base(
            arch, cls, wide_ladder=wide_ladder, target_bpw=target_bpw
        )
    else:
        base_tier = arch.get_ladder_base(cls, wide_ladder=wide_ladder, target_bpw=target_bpw)

    base_next = (
        _get_next_tier(
            arch,
            cls,
            base_tier,
            has_imatrix=has_imatrix,
            wide_ladder=wide_ladder,
            profile=profile,
        )
        or next_tier
    )
    base_mse_delta = max(1e-9, _mse_delta(base_tier, base_next))
    diminishing_factor = (min(1.0, mse_d / base_mse_delta)) ** diminishing_exp

    depth_multiplier = arch.get_depth_multiplier(
        rep_name=rep_name,
        cur_tier=cur_tier,
        target_bpw=target_bpw,
        n_layers=n_layers,
    )

    # Smoothed cost delta to prevent massive tensors from being blocked by small ones
    cost_smoothed = (cost_delta**0.85) * (1024 * 1024)

    utility = g_importance * mse_d * diminishing_factor * depth_multiplier / cost_smoothed

    heapq.heappush(
        upgrade_queue,
        UpgradeItem(
            neg_utility=-utility,
            group_id=group_id,
            next_tier=next_tier,
            cost_delta=cost_delta,
        ),
    )


# ── Group construction ──────────────────────────────────────────────────────


def build_groups(
    tied_groups: list[list[TensorName]],
    elastic_names: set[TensorName],
    ne_map: dict[TensorName, int],
    padded_ne_map: dict[TensorName, int],
) -> GroupRegistry:
    """Build group registry for elastic tensors."""
    group_registry: GroupRegistry = {}
    assigned: set[TensorName] = set()

    for g_idx, group in enumerate(tied_groups):
        clean_group = [n for n in group if n in elastic_names]
        if clean_group:
            g_elements = sum(ne_map.get(n, 0) for n in clean_group)
            g_elements_padded = sum(padded_ne_map.get(n, 0) for n in clean_group)
            group_registry[g_idx] = (clean_group, g_elements, g_elements_padded)
            assigned.update(clean_group)

    unassigned = elastic_names - assigned
    next_idx = len(group_registry)

    for name in unassigned:
        group_registry[next_idx] = ([name], ne_map.get(name, 0), padded_ne_map.get(name, 0))
        next_idx += 1

    return group_registry


# ── Statistics ──────────────────────────────────────────────────────────────


def compute_stats(
    assignments: TensorAssignments,
    ne_map: dict[TensorName, int],
    padded_ne_map: dict[TensorName, int] | None = None,
) -> dict[str, Any]:
    """Compute size, count, and BPW statistics for a given assignment map."""
    by_tier_count: dict[str, int] = {}
    by_tier_mib: dict[str, float] = {}
    total_mib = 0.0

    for name, tier in assignments.items():
        n_el = (
            padded_ne_map.get(name, ne_map.get(name, 0))
            if (padded_ne_map and tier in K_QUANTS)
            else ne_map.get(name, 0)
        )
        tier_mib = _size_mib(tier, n_el)

        by_tier_count[tier] = by_tier_count.get(tier, 0) + 1
        by_tier_mib[tier] = by_tier_mib.get(tier, 0.0) + tier_mib
        total_mib += tier_mib

    total_el = sum(ne_map.values())
    bpw = (total_mib * 8 * 1024 * 1024) / total_el if total_el > 0 else 0.0

    return {
        "by_tier_count": by_tier_count,
        "by_tier_mib": by_tier_mib,
        "total_mib": total_mib,
        "total_elements": total_el,
        "bpw": bpw,
    }


# ── Main Greedy Optimizer Engine ────────────────────────────────────────────


def run_greedy_optimization(
    arch: BaseArchitecture,
    importance_table: ImportanceTable,
    tied_groups: list[list[TensorName]],
    model: dict[str, Any],
    target_size_mib: float,
    wide_ladder: bool = False,
    profile: BaseProfile | None = None,
) -> tuple[TensorAssignments, dict[TensorName, int]]:
    """Run pure greedy priority-queue quantization optimization."""
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")

    features = model.get("features", {})
    n_layers = features.get("n_layers", 32)
    model_tensors = model.get("tensors", {})

    ne_map: dict[TensorName, int] = {k: v["n_elements"] for k, v in model_tensors.items()}
    for tname, info in importance_table.items():
        if tname not in ne_map:
            ne_map[tname] = info.get("n_elements", 0)

    # Align MoE dimensions for K-quants
    moe_d_ff = features.get("moe_intermediate_size", 0)
    padded_ne_map = dict(ne_map)
    if moe_d_ff > 0 and moe_d_ff % K_QUANT_BLOCK_SIZE != 0:
        aligned_d_ff = (
            (moe_d_ff + K_QUANT_BLOCK_SIZE - 1) // K_QUANT_BLOCK_SIZE
        ) * K_QUANT_BLOCK_SIZE
        for name, n_el in ne_map.items():
            if any(p in name for p in MOE_PAD_TYPES):
                padded_ne_map[name] = (n_el // moe_d_ff) * aligned_d_ff

    all_names = set(ne_map.keys())

    # 1. Obtain fixed tensors from the architecture strategy
    fixed_assignments = arch.get_fixed_tensors(
        model=model,
        imp_table=importance_table,
        target_size_mib=target_size_mib,
        wide_ladder=wide_ladder,
    )

    # 2. Elastic tensors initialize to ladder base (or architecture early layer floor)
    elastic_names = all_names - set(fixed_assignments.keys())
    assignments: TensorAssignments = dict(fixed_assignments)
    target_bpw = (target_size_mib * BITS_IN_MIB) / max(1, sum(ne_map.values()))

    for name in elastic_names:
        cls = arch.classify_tensor(name)
        if profile is not None:
            base_tier = profile.get_ladder_base(
                arch, cls, wide_ladder=wide_ladder, target_bpw=target_bpw
            )
        else:
            base_tier = arch.get_ladder_base(cls, wide_ladder=wide_ladder, target_bpw=target_bpw)

        # Check for architecture-specific early layer floor
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
            try:
                layer_idx = int(parts[1])
                floor_tier = arch.get_early_layer_floor(cls, layer_idx, target_bpw)
                if floor_tier:
                    if profile is not None:
                        ladder = profile.get_tier_ladder(arch, cls, wide_ladder=wide_ladder)
                    else:
                        ladder = arch.get_tier_ladder(cls, wide_ladder=wide_ladder)
                    if floor_tier in ladder:
                        f_idx = ladder.index(floor_tier)
                        b_idx = ladder.index(base_tier) if base_tier in ladder else 0
                        if f_idx > b_idx:
                            base_tier = floor_tier
            except ValueError:
                pass

        assignments[name] = base_tier

    # 3. Calculate initial baseline cost
    current_size = sum(
        _size_mib(
            tier, padded_ne_map.get(n, ne_map.get(n, 0)) if tier in K_QUANTS else ne_map.get(n, 0)
        )
        for n, tier in assignments.items()
    )

    logger.info(
        "  Initial baseline size: %.1f MiB / Target: %.1f MiB", current_size, target_size_mib
    )

    # 4. Build group registry (combining any user/structural groups, default independent)
    structural_tied = arch.get_structural_tied_groups(list(all_names))
    combined_tied = (
        list(tied_groups) + [g for g in structural_tied if g not in tied_groups]
        if tied_groups
        else structural_tied
    )

    group_registry = build_groups(combined_tied, elastic_names, ne_map, padded_ne_map)

    # 5. Initialize greedy queue
    tensor_importance: dict[TensorName, float] = {
        n: importance_table.get(n, {}).get("importance_mean", 0.0) for n in elastic_names
    }

    target_bpw = (target_size_mib * BITS_IN_MIB) / max(1, sum(ne_map.values()))
    upgrade_queue: list[UpgradeItem] = []

    for group_id in group_registry:
        _push_upgrade(
            group_id=group_id,
            group_registry=group_registry,
            assignments=assignments,
            tensor_importance=tensor_importance,
            upgrade_queue=upgrade_queue,
            importance_table=importance_table,
            arch=arch,
            target_bpw=target_bpw,
            n_layers=n_layers,
            wide_ladder=wide_ladder,
            profile=profile,
        )

    # 6. Greedy upgrade loop (strict ceiling budget)
    effective_target = target_size_mib
    upgrade_count = 0

    while upgrade_queue:
        item = heapq.heappop(upgrade_queue)

        if current_size + item.cost_delta > effective_target:
            continue

        g_names, _, _ = group_registry[item.group_id]
        for name in g_names:
            assignments[name] = item.next_tier

        current_size += item.cost_delta
        upgrade_count += 1

        _push_upgrade(
            group_id=item.group_id,
            group_registry=group_registry,
            assignments=assignments,
            tensor_importance=tensor_importance,
            upgrade_queue=upgrade_queue,
            importance_table=importance_table,
            arch=arch,
            target_bpw=target_bpw,
            n_layers=n_layers,
            wide_ladder=wide_ladder,
            profile=profile,
        )

    logger.info("  Greedy upgrades executed: %d", upgrade_count)
    return assignments, padded_ne_map
