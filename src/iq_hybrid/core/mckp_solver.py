"""Multiple-Choice Knapsack Problem (MCKP) solver for global optimal quantization.

Formulates quantization as a Mixed-Integer Linear Program (MILP) using SciPy:
  Maximize:   Sum_{g, t} (Importance_g * (MSE_base - MSE_t) * x_{g, t})
  Subject to: Sum_t (x_{g, t}) == 1  for each group g
              Sum_{g, t} (Size_MiB_{g, t} * x_{g, t}) <= Target_MiB
              x_{g, t} in {0, 1}
"""

import logging
from typing import Any

import numpy as np

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
from iq_hybrid.core.greedy_engine import build_groups, run_greedy_optimization
from iq_hybrid.core.types import ImportanceTable, TensorAssignments, TensorName, TierName
from iq_hybrid.profiles.base import BaseProfile

logger = logging.getLogger(__name__)


def _size_mib(tier: TierName, n_elements: int) -> float:
    """Calculate tensor size in MiB."""
    if n_elements <= 0:
        return 0.0
    return n_elements * TIER_SIZE_MULTIPLIER.get(tier, 0.0)


def _tier_mse(tier: TierName) -> float:
    """Theoretical quantization MSE for a given tier."""
    bpw = MSE_BPW.get(tier, 4.0)
    return 2.0 ** (-2.0 * bpw)


def run_mckp_optimization(
    arch: BaseArchitecture,
    importance_table: ImportanceTable,
    tied_groups: list[list[TensorName]],
    model: dict[str, Any],
    target_size_mib: float,
    wide_ladder: bool = False,
    profile: BaseProfile | None = None,
) -> tuple[TensorAssignments, dict[TensorName, int]]:
    """Run globally optimal Multiple-Choice Knapsack optimization using MILP.

    Falls back automatically to greedy engine if scipy is unavailable or if MILP fails.
    """
    try:
        from scipy.optimize import LinearConstraint, milp
    except ImportError:
        logger.warning("scipy is not installed. Falling back to greedy optimization engine.")
        return run_greedy_optimization(
            arch=arch,
            importance_table=importance_table,
            tied_groups=tied_groups,
            model=model,
            target_size_mib=target_size_mib,
            wide_ladder=wide_ladder,
            profile=profile,
        )

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
    total_elements = sum(ne_map.values())
    target_bpw = (target_size_mib * BITS_IN_MIB) / max(1, total_elements)

    # 1. Obtain fixed tensors from the architecture strategy
    fixed_assignments = arch.get_fixed_tensors(
        model=model,
        imp_table=importance_table,
        target_size_mib=target_size_mib,
        wide_ladder=wide_ladder,
    )

    fixed_size_mib = sum(
        _size_mib(
            tier, padded_ne_map.get(n, ne_map.get(n, 0)) if tier in K_QUANTS else ne_map.get(n, 0)
        )
        for n, tier in fixed_assignments.items()
    )
    remaining_budget_mib = target_size_mib - fixed_size_mib

    if remaining_budget_mib <= 0:
        logger.warning(
            "Fixed tensors exceed target budget (%.1f MiB >= %.1f MiB). Returning base assignments.",
            fixed_size_mib,
            target_size_mib,
        )
        assignments = dict(fixed_assignments)
        for name in all_names - set(fixed_assignments.keys()):
            cls = arch.classify_tensor(name)
            if profile is not None:
                assignments[name] = profile.get_ladder_base(
                    arch, cls, wide_ladder=wide_ladder, target_bpw=target_bpw
                )
            else:
                assignments[name] = arch.get_ladder_base(
                    cls, wide_ladder=wide_ladder, target_bpw=target_bpw
                )
        return assignments, padded_ne_map

    # 2. Build elastic group registry (combining any user/structural groups, default independent)
    elastic_names = all_names - set(fixed_assignments.keys())
    structural_tied = arch.get_structural_tied_groups(list(all_names))
    combined_tied = (
        list(tied_groups) + [g for g in structural_tied if g not in tied_groups]
        if tied_groups
        else structural_tied
    )
    group_registry = build_groups(combined_tied, elastic_names, ne_map, padded_ne_map)

    # 3. Build candidate tiers and coefficients for each group
    # Variables are indexed as: var_map[(group_id, tier)] -> var_index
    c_list: list[float] = []  # Objective coefficients (to minimize: -utility)
    weight_list: list[float] = []  # Budget weights in MiB
    group_var_indices: list[list[int]] = []
    var_meta: list[tuple[int, TierName]] = []

    for group_id, (g_names, g_elements, g_elements_padded) in group_registry.items():
        rep_name = g_names[0]
        cls = arch.classify_tensor(rep_name)
        rep_info = importance_table.get(rep_name, {})
        has_imatrix = rep_info.get("importance_mean", 0.0) > 0

        if profile is not None:
            ladder = profile.get_tier_ladder(arch, cls, wide_ladder=wide_ladder)
            base_tier = profile.get_ladder_base(
                arch, cls, wide_ladder=wide_ladder, target_bpw=target_bpw
            )
            ceiling_tier = profile.get_ladder_ceiling(arch, cls)
        else:
            ladder = arch.get_tier_ladder(cls, wide_ladder=wide_ladder)
            base_tier = arch.get_ladder_base(cls, wide_ladder=wide_ladder, target_bpw=target_bpw)
            ceiling_tier = arch.get_ladder_ceiling(cls)

        valid_ladder = [t for t in ladder if has_imatrix or (t not in TIERS_REQUIRING_IMATRIX)]

        # Apply early layer floor
        parts = rep_name.split(".")
        if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
            try:
                layer_idx = int(parts[1])
                floor_tier = arch.get_early_layer_floor(cls, layer_idx, target_bpw)
                if floor_tier and floor_tier in valid_ladder:
                    if valid_ladder.index(floor_tier) > valid_ladder.index(base_tier):
                        base_tier = floor_tier
            except ValueError:
                pass

        # Filter valid tier options between base and ceiling
        start_idx = valid_ladder.index(base_tier) if base_tier in valid_ladder else 0
        end_idx = (
            valid_ladder.index(ceiling_tier)
            if ceiling_tier in valid_ladder
            else len(valid_ladder) - 1
        )
        candidate_tiers = valid_ladder[start_idx : end_idx + 1]
        if not candidate_tiers:
            candidate_tiers = [base_tier]

        g_importance = sum(importance_table.get(n, {}).get("importance_sum", 1.0) for n in g_names)
        depth_mult = arch.get_depth_multiplier(
            rep_name=rep_name,
            cur_tier=base_tier,
            target_bpw=target_bpw,
            n_layers=n_layers,
        )

        current_group_vars: list[int] = []
        base_mse = _tier_mse(candidate_tiers[0])

        for tier in candidate_tiers:
            v_idx = len(c_list)
            current_group_vars.append(v_idx)
            var_meta.append((group_id, tier))

            # MSE reduction relative to group base tier
            mse_gain = max(0.0, base_mse - _tier_mse(tier))
            utility = g_importance * mse_gain * depth_mult

            # SciPy milp minimizes c^T x, so c = -utility
            c_list.append(-utility)

            is_padded = tier in K_QUANTS
            g_el = g_elements_padded if is_padded else g_elements
            weight_list.append(_size_mib(tier, g_el))

        group_var_indices.append(current_group_vars)

    # 4. Formulate constraints
    n_vars = len(c_list)
    n_groups = len(group_var_indices)

    # Constraint matrix A shape: (n_groups + 1, n_vars)
    A = np.zeros((n_groups + 1, n_vars), dtype=np.float64)
    lhs = np.zeros(n_groups + 1, dtype=np.float64)
    rhs = np.zeros(n_groups + 1, dtype=np.float64)

    # Group choice constraints: Exactly one tier per group
    for g_i, var_indices in enumerate(group_var_indices):
        for v_i in var_indices:
            A[g_i, v_i] = 1.0
        lhs[g_i] = 1.0
        rhs[g_i] = 1.0

    # Budget row (last constraint)
    A[n_groups, :] = weight_list
    lhs[n_groups] = 0.0
    rhs[n_groups] = remaining_budget_mib

    constraints = LinearConstraint(A, lhs, rhs)
    integrality = np.ones(n_vars, dtype=np.int32)  # All binary 0/1 variables

    logger.info(
        "Solving MCKP with SciPy MILP (%d variables, %d groups, target: %.1f MiB)...",
        n_vars,
        n_groups,
        target_size_mib,
    )

    res = milp(
        c=np.array(c_list, dtype=np.float64),
        integrality=integrality,
        constraints=constraints,
    )

    if not res.success:
        logger.warning(
            "MCKP MILP solver failed (%s). Falling back to greedy engine.", res.status_message
        )
        return run_greedy_optimization(
            arch=arch,
            importance_table=importance_table,
            tied_groups=tied_groups,
            model=model,
            target_size_mib=target_size_mib,
            wide_ladder=wide_ladder,
            profile=profile,
        )

    # 5. Extract solution
    assignments: TensorAssignments = dict(fixed_assignments)
    sol_x = np.round(res.x).astype(int)

    for v_idx, is_selected in enumerate(sol_x):
        if is_selected == 1:
            g_id, chosen_tier = var_meta[v_idx]
            g_names, _, _ = group_registry[g_id]
            for name in g_names:
                assignments[name] = chosen_tier

    total_est = sum(
        _size_mib(
            tier, padded_ne_map.get(n, ne_map.get(n, 0)) if tier in K_QUANTS else ne_map.get(n, 0)
        )
        for n, tier in assignments.items()
    )
    logger.info(
        "MCKP solver converged successfully. Total estimated size: %.1f MiB / Target: %.1f MiB",
        total_est,
        target_size_mib,
    )
    return assignments, padded_ne_map
