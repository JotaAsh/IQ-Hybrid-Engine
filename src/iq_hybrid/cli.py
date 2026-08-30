"""CLI entry point for IQ-Hybrid Engine.

Orchestrates the full pipeline:
  1. Read model metadata and imatrix tensors
  2. Detect architecture and resolve quantization profile
  3. Run greedy / MCKP optimization engine
  4. Generate llama-quantize config flags
  5. Dry-run validation and model execution
"""

import argparse
import json
import logging
import os
import sys

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.architectures.registry import detect_architecture_strategy
from iq_hybrid.core.config import load_config
from iq_hybrid.core.greedy_engine import compute_stats, run_greedy_optimization
from iq_hybrid.core.mckp_solver import run_mckp_optimization
from iq_hybrid.core.types import ImportanceTable, TensorAssignments, TensorName
from iq_hybrid.io.config_writer import format_flags, generate_flags
from iq_hybrid.io.gguf_reader import read_model
from iq_hybrid.io.imatrix_merger import merge_imatrix_files
from iq_hybrid.io.imatrix_reader import (
    build_importance_table,
    read_imatrix,
)
from iq_hybrid.profiles.registry import AVAILABLE_PRESETS, resolve_profile
from iq_hybrid.runner.quantizer import resolve_output_dir, run_dry_run, run_quantization
from iq_hybrid.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ── Display functions ────────────────────────────────────────────────────────


def _show_tier_summary(
    assignments: TensorAssignments,
    imp_table: ImportanceTable,
    ne_map: dict[TensorName, int],
    padded_ne_map: dict[TensorName, int] | None = None,
    verbose: bool = False,
) -> None:
    """Log the tier distribution and top tensors by importance."""
    stats = compute_stats(assignments, ne_map, padded_ne_map)

    logger.info("  Distribution:")
    sorted_tiers = sorted(
        stats["by_tier_count"].keys(),
        key=lambda t: stats["by_tier_mib"][t],
        reverse=True,
    )
    for tier in sorted_tiers:
        count = stats["by_tier_count"][tier]
        mib = stats["by_tier_mib"][tier]
        pct = (mib / stats["total_mib"] * 100) if stats["total_mib"] > 0 else 0
        logger.info("    %-10s: %3d tensors (%6.1f MiB, %4.1f%%)", tier, count, mib, pct)

    logger.info(
        "  Total: %.1f MiB across %d tensors (BPW: %.2f)",
        stats["total_mib"],
        len(assignments),
        stats["bpw"],
    )

    if verbose:
        logger.info("  Top 10 highest-importance tensors:")
        sorted_imp = sorted(
            assignments.keys(),
            key=lambda t: imp_table.get(t, {}).get("importance_sum", 0.0),
            reverse=True,
        )
        for i, name in enumerate(sorted_imp[:10], 1):
            t_info = imp_table.get(name, {})
            imp_mean = t_info.get("importance_mean", 0.0)
            imp_sum = t_info.get("importance_sum", 0.0)
            tier = assignments[name]
            logger.info(
                "    %2d. %-45s -> %-10s (mean=%6.4f, sum=%8.2f)", i, name, tier, imp_mean, imp_sum
            )


def _show_ladders(arch: BaseArchitecture) -> None:
    """Print the tier ladders for all tensor classes in the strategy."""
    logger.info("=== Tier Ladders for Architecture: %s ===", arch.name)
    classes = ["embd", "gate", "router", "attn_proj", "ffn_gate_up", "ffn_down", "norms", "default"]
    for cls in classes:
        standard = arch.get_tier_ladder(cls, wide_ladder=False)
        wide = arch.get_tier_ladder(cls, wide_ladder=True)
        base = arch.get_ladder_base(cls, wide_ladder=False)
        ceiling = arch.get_ladder_ceiling(cls)
        logger.info("  Class: %-12s (Base: %-8s Ceiling: %-8s)", cls, base, ceiling)
        logger.info("    Standard: %s", " -> ".join(standard))
        logger.info("    Wide:     %s", " -> ".join(wide))


# ── Parser construction ──────────────────────────────────────────────────────


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Construct CLI argument parser using configuration defaults."""
    presets_help = ", ".join(AVAILABLE_PRESETS)
    parser = argparse.ArgumentParser(
        description="IQ-Hybrid Engine: imatrix-driven selective hybrid quantization for GGUF models",
    )
    parser.add_argument("--model", default=None, help="BF16 GGUF model path")
    parser.add_argument(
        "--imatrix",
        action="append",
        default=None,
        help="Imatrix GGUF path (can be specified multiple times)",
    )
    parser.add_argument(
        "--imatrix-method",
        choices=["add", "max", "mean"],
        default=cfg.get("imatrix_method", "add"),
        help="Merge strategy for multiple imatrix: 'add' (sum activations, default), 'max' (conservative), or 'mean'",
    )
    parser.add_argument(
        "--size",
        default=None,
        help=f"Target file size in MiB (e.g. 6400) or preset profile ({presets_help})",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=f"Quantization profile preset ({presets_help}) or path to .json/.yaml profile recipe",
    )
    parser.add_argument(
        "--solver",
        choices=["greedy", "mckp"],
        default=cfg.get("solver", "greedy"),
        help="Optimization engine: greedy (fast heuristic) or mckp (global optimal MILP with SciPy)",
    )
    parser.add_argument("--output", default=None, help="Output GGUF path")
    parser.add_argument("--run", action="store_true", help="Execute quantization")
    parser.add_argument("--show-config", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=cfg.get("verbose", False),
        help="Detailed output",
    )
    parser.add_argument(
        "--wide-ladder",
        action="store_true",
        default=cfg.get("wide_ladder", False),
        help="Enable full tier ladder range down to ultra-low bitrates for non-critical tensors",
    )
    parser.add_argument(
        "--show-ladders", action="store_true", help="Print class tier ladders and exit"
    )
    parser.add_argument(
        "--quantizer-path",
        default=None,
        help="Path to llama-quantize binary (overrides .env and PATH)",
    )
    return parser


def main() -> None:
    """Entry point for IQ-Hybrid Engine."""
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args()

    # Initialize logging before any output
    setup_logging(verbose=args.verbose)

    if args.show_ladders:
        sample_arch = detect_architecture_strategy({"architecture": "unknown"})
        _show_ladders(sample_arch)
        return

    if not args.model or not args.imatrix:
        parser.print_usage()
        logger.error("--model and --imatrix are required")
        sys.exit(1)

    n_imatrix = len(args.imatrix)
    if n_imatrix > 3:
        logger.error("A maximum of 3 imatrix files is allowed (received %d).", n_imatrix)
        sys.exit(1)

    logger.info("=== IQ-Hybrid Engine ===")
    logger.info("Model:   %s", args.model)
    if n_imatrix == 1:
        logger.info("Imatrix: %s", args.imatrix[0])
    else:
        logger.info("Imatrix: %d files (merge method: %s)", n_imatrix, args.imatrix_method)
        for p in args.imatrix:
            logger.info("  - %s", p)

    # ── Step 1: Read model ────────────────────────────────────────────────
    logger.info("[1/4] Reading model...")
    model = read_model(args.model)
    logger.info("  Architecture: %s", model["architecture"])
    logger.info("  Tensors: %d", model["n_tensors"])
    logger.info("  Features: %s", json.dumps(model["features"], indent=2))

    total_elements = sum(v["n_elements"] for v in model.get("tensors", {}).values())

    # ── Resolve Architecture and Profile ──────────────────────────────────
    target_arg = args.profile or args.size or "7200"
    arch = detect_architecture_strategy(model)
    profile, profile_val, is_preset = resolve_profile(target_arg)
    target_mib = profile.calculate_budget_mib(total_elements, profile_val)
    wide_ladder = args.wide_ladder or profile.force_wide_ladder

    logger.info("Strategy: Architecture=[%s], Profile=[%s]", arch.name, profile.name)
    logger.info(
        "Target:   %.0f MiB (%.2f GB) [BPW: %.2f]",
        target_mib,
        target_mib / 1024,
        (target_mib * 8 * 1024 * 1024) / max(1, total_elements),
    )
    if wide_ladder:
        logger.info("  --wide-ladder: active (full ladder range enabled)")

    # ── Step 2: Merge / Read IMatrix ──────────────────────────────────────
    if n_imatrix > 1:
        logger.info("[2/4] Merging %d imatrix files via '%s'...", n_imatrix, args.imatrix_method)
        out_dir = cfg.get("imatrix_output_dir", "iMatrix")
        merged_path = merge_imatrix_files(
            paths=args.imatrix,
            method=args.imatrix_method,
            output_dir=out_dir,
        )
        effective_imatrix_path = str(merged_path)
        logger.info("  Merged iMatrix ready: %s", effective_imatrix_path)
    else:
        effective_imatrix_path = args.imatrix[0]

    logger.info("[2/4] Loading active imatrix: %s", effective_imatrix_path)
    imatrix = read_imatrix(effective_imatrix_path)

    # ── Step 3: Structural Ties & Importance Table ────────────────────────
    structural_tied = arch.get_structural_tied_groups(list(model.get("tensors", {}).keys()))
    if structural_tied:
        logger.info("[3/4] Structural tied groups (%d):", len(structural_tied))
        if args.verbose:
            for g in structural_tied:
                logger.info(
                    "    TIED (%d): %s  =  %s",
                    len(g),
                    g[0].replace(".weight", ""),
                    g[1].replace(".weight", ""),
                )
    else:
        logger.info("[3/4] Tensor evaluation: independent (no structural tied groups required)")

    imp_table = build_importance_table(imatrix, model)

    # ── Step 4: Classify tensors ─────────────────────────────────────────
    logger.info("[4/4] Classifying tensors (solver: %s)...", args.solver)

    if args.solver == "mckp":
        assignments, padded_ne_map = run_mckp_optimization(
            arch=arch,
            importance_table=imp_table,
            tied_groups=structural_tied,
            model=model,
            target_size_mib=target_mib,
            wide_ladder=wide_ladder,
            profile=profile,
        )
    else:
        assignments, padded_ne_map = run_greedy_optimization(
            arch=arch,
            importance_table=imp_table,
            tied_groups=structural_tied,
            model=model,
            target_size_mib=target_mib,
            wide_ladder=wide_ladder,
            profile=profile,
        )

    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}
    for tname, info in imp_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    _show_tier_summary(
        assignments,
        imp_table,
        ne_map,
        padded_ne_map,
        verbose=args.verbose,
    )

    stats = compute_stats(assignments, ne_map, padded_ne_map)
    stats["total_elements"] = sum(ne_map.values())
    flags = generate_flags(
        assignments=assignments,
        model=model,
        base_type=None,
        target_size_mib=target_mib,
        stats=stats,
    )
    flags["profile_tag"] = profile.name if is_preset else None
    flags["imatrix"] = effective_imatrix_path

    logger.info("Config (base=%s):", flags["base_type"])
    logger.info(format_flags(flags))

    if args.show_config:
        print(json.dumps(flags, indent=2))
        return

    # ── Dry-Run or Execution ──────────────────────────────────────────────
    if not args.run:
        logger.info("--- Dry Run ---")
        run_dry_run(
            flags=flags,
            model_path=args.model,
            quantizer_path=args.quantizer_path,
        )
        logger.info("Dry run only. Use --run to execute quantization.")
        return

    output_path = args.output
    if not output_path:
        out_dir = resolve_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        tag = profile.get_output_tag(target_mib)
        model_stem = os.path.splitext(os.path.basename(args.model))[0]
        output_path = os.path.join(out_dir, f"{model_stem}-{tag}.gguf")

    logger.info("--- Running Quantization ---")
    logger.info("Output: %s", output_path)
    run_quantization(
        flags=flags,
        model_path=args.model,
        output_path=output_path,
        quantizer_path=args.quantizer_path,
    )
    logger.info("Quantization complete: %s", output_path)


if __name__ == "__main__":
    main()
