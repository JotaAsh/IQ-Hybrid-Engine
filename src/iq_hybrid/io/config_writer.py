"""Config generator — compiles tensor assignments into llama-quantize flags.

Produces regex-based ``--tensor-type`` rules, determines the appropriate
base quantization type, and formats the final command-line configuration.
"""

import re
from collections.abc import Iterator
from typing import Any

from iq_hybrid.core.constants import QUANT_RANK
from iq_hybrid.core.types import TensorAssignments, TierName

# Mapping of effective BPW ranges to standard GGUF labels
BPW_LABEL_THRESHOLDS: list[tuple[float, TierName]] = [
    (1.85, "IQ1_S"),
    (2.20, "IQ2_XXS"),
    (2.45, "IQ2_XS"),
    (2.70, "IQ2_S"),
    (3.20, "IQ3_XXS"),
    (3.55, "IQ3_S"),
    (3.90, "Q3_K_M"),
    (4.40, "IQ4_XS"),
    (4.80, "IQ4_NL"),
    (5.20, "Q4_K_M"),
    (6.00, "Q5_K_M"),
    (7.20, "Q6_K"),
    (float("inf"), "Q8_0"),
]


def get_base_type(stats: dict[str, Any]) -> TierName:
    """Determine the base model label from its actual global BPW.

    Prevents the count or local mass of Q8_0 tensors from distorting
    the output filename label.
    """
    # 1. Get direct BPW if precomputed
    bpw = stats.get("bpw")

    # 2. Calculate BPW if total_mib and total_elements are available
    if bpw is None:
        total_mib = stats.get("total_size_mib", stats.get("total_mib", 0.0))
        total_elements = stats.get("total_elements", 0)

        if total_mib <= 0.0 and "by_tier_mib" in stats:
            total_mib = sum(stats["by_tier_mib"].values())

        if total_mib > 0.0 and total_elements > 0:
            bpw = (total_mib * 8 * 1024 * 1024) / total_elements

    # 3. Safety fallback if insufficient numerical data
    if bpw is None or bpw <= 0.0:
        return "Q4_K_M"

    # 4. Assign label for corresponding BPW threshold
    for threshold, label in BPW_LABEL_THRESHOLDS:
        if bpw < threshold:
            return label

    return "Q8_0"


def get_regex_priority(regex: str) -> int:
    """Score a regex pattern by specificity (higher = more specific = first match)."""
    score = 0

    if "nextn" in regex:
        score += 200
    if re.search(r"(blk|BLK)\\.3[0-2]\.", regex):
        score += 100
    if re.search(r"(blk|BLK)\\.0\.", regex):
        score += 90
    if re.search(r"(blk|BLK)\\.31\.", regex):
        score += 80
    # Layer-range group: more specific than bare \d+
    if r"(blk|BLK)\.(" in regex:
        score += 50
    elif r"(blk|BLK).\d" in regex:
        score += 30
    if regex.startswith(".*"):
        score -= 50
    if regex.endswith(r"\.weight"):
        score += 10

    return score


def _is_contiguous(lst: list[int], low: int, high: int) -> bool:
    """Return True if *lst* covers every integer from *low* to *high* inclusive."""
    if not lst:
        return False
    return len(lst) == (high - low + 1)


def _group_ranges(lst: list[int]) -> Iterator[tuple[int, int]]:
    """Yield (start, end) pairs for runs of consecutive integers in *lst*."""
    if not lst:
        return
    start = lst[0]
    end = lst[0]
    for i in range(1, len(lst)):
        if lst[i] == end + 1:
            end = lst[i]
        else:
            yield (start, end)
            start = end = lst[i]
    yield (start, end)


def _range_to_regex(start: int, end: int) -> str:
    """Convert an inclusive range [start, end] to a valid regex fragment."""
    if start == end:
        return str(start)
    if end <= 9:
        return f"[{start}-{end}]"
    # Enumerate all numbers as pipe alternatives
    alt = "|".join(str(i) for i in range(start, end + 1))
    return f"(?:{alt})"


def generate_flags(
    assignments: TensorAssignments,
    model: dict[str, Any],
    base_type: str | dict | None = None,
    target_size_mib: float | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate llama-quantize flags from tensor assignments.

    Returns a dict with keys: ``imatrix``, ``output_tensor_type``,
    ``token_embedding_type``, ``tensor_type_rules``, ``base_type``,
    ``target_size_mib``.
    """
    # Resolve base label by BPW
    if isinstance(base_type, dict):
        resolved_base_type = get_base_type(base_type)
    elif stats is not None:
        resolved_base_type = get_base_type(stats)
    elif isinstance(base_type, str):
        resolved_base_type = base_type
    elif target_size_mib and "tensors" in model:
        total_el = sum(v.get("n_elements", 0) for v in model["tensors"].values())
        calc_bpw = (target_size_mib * 8 * 1024 * 1024) / max(1, total_el)
        resolved_base_type = get_base_type({"bpw": calc_bpw})
    else:
        resolved_base_type = "Q4_K_M"

    output_type = assignments.get("output", assignments.get("output.weight", "IQ4_NL"))
    token_embd_type = assignments.get("token_embd", assignments.get("token_embd.weight", "IQ4_NL"))

    max_layer = model.get("features", {}).get("n_layers", 31)

    rules: list[tuple[str, int]] = []

    # Group blk tensors by (ttype, tier)
    type_tier_layers: dict[tuple[str, TierName], list[int]] = {}
    for tname, tier in assignments.items():
        parts = tname.split(".")
        if len(parts) >= 3 and parts[0] in ("blk", "BLK"):
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            ttype = parts[2]
            key = (ttype, tier)
            type_tier_layers.setdefault(key, []).append(layer)

    # Generate rules for blk tensor groups
    for (ttype, tier), layers in sorted(
        type_tier_layers.items(),
        key=lambda x: -QUANT_RANK.get(x[0][1], 0),
    ):
        layers = sorted(set(layers))

        if len(layers) >= 8 and _is_contiguous(layers, 0, max_layer):
            pattern = f"(blk|BLK)\\.\\d+\\.{ttype}={tier}"
        else:
            parts_list: list[str] = []
            for start, end in _group_ranges(layers):
                if start == end:
                    parts_list.append(str(start))
                else:
                    parts_list.append(_range_to_regex(start, end))
            desc = "|".join(parts_list)
            pattern = f"(blk|BLK)\\.({desc})\\.{ttype}={tier}"

        prio = (
            get_regex_priority(pattern)
            + (10 if tier == "Q8_0" else 5 if tier == "Q6_K" else 0)
            + (5 if len(layers) == 1 else 0)
            + (3 if "ffn_down" in ttype else 0)
        )

        rules.append((pattern, prio))

    # Generate rules for global tensors (non-blk)
    prefix = model.get("features", {}).get("prefix", "blk")
    for tname, tier in assignments.items():
        parts = tname.split(".")
        if len(parts) >= 2 and parts[0].lower() == prefix.lower():
            continue
        ttype = parts[0] if parts else tname
        if ttype in ("token_embd", "output"):
            continue
        # Build a catch-all pattern for global tensors
        pattern = f".*{re.escape(ttype)}.*={tier}"
        prio = get_regex_priority(pattern) + (5 if tier == "Q8_0" else 0)
        # Deduplicate
        if not any(p == pattern for p, _ in rules):
            rules.append((pattern, prio))

    rules.sort(key=lambda x: -x[1])

    return {
        "imatrix": None,
        "output_tensor_type": output_type,
        "token_embedding_type": token_embd_type,
        "tensor_type_rules": [f'--tensor-type "{r[0]}"' for r in rules],
        "base_type": resolved_base_type,
        "target_size_mib": target_size_mib,
    }


def format_flags(flags: dict[str, Any]) -> str:
    """Format flags dict into a human-readable multi-line string."""
    lines = [
        "  --output-tensor-type " + flags["output_tensor_type"],
        "  --token-embedding-type " + flags["token_embedding_type"],
    ]
    for rule in flags["tensor_type_rules"]:
        lines.append("  " + rule)
    return "\n".join(lines)
