"""Imatrix GGUF reader and importance table builder.

Parses activation importance matrices stored in GGUF format, detects tied
tensor groups, and constructs the unified importance table consumed by the
classifier.
"""

import logging
from typing import Any

import gguf
import numpy as np

from iq_hybrid.core.types import ImportanceTable

logger = logging.getLogger(__name__)


def read_imatrix(path: str) -> dict[str, Any]:
    """Parse an imatrix GGUF file and return per-tensor importance data.

    Returns a dict with keys: ``path``, ``tensors``, ``n_tensors``, ``meta``.
    """
    reader = gguf.GGUFReader(path)
    raw: dict[str, dict] = {}
    meta: dict[str, Any] = {}

    for key, field in reader.fields.items():
        try:
            data = field.data
            if isinstance(data, np.ndarray):
                data = data.tolist()
            elif isinstance(data, np.generic):
                data = data.item()
            meta[key] = data
        except Exception:
            logger.debug("Could not decode metadata field '%s', storing as string", key)
            meta[key] = str(field)

    for tensor in reader.tensors:
        name = tensor.name
        arr = np.array(tensor.data, dtype=np.float64)

        if name.endswith(".in_sum2"):
            base = name[:-8]
            raw.setdefault(base, {})["in_sum2"] = arr
        elif name.endswith(".counts"):
            base = name[:-7]
            raw.setdefault(base, {})["counts"] = float(np.mean(arr))

    result: dict[str, dict] = {}
    for base, data in raw.items():
        if "in_sum2" not in data:
            continue
        arr = data["in_sum2"]
        result[base] = {
            "importance_mean": float(np.mean(arr)),
            "importance_sum": float(np.sum(arr)),
            "importance_max": float(np.max(arr)),
            "importance_min": float(np.min(arr)),
            "n_elements": arr.size,
            "in_sum2_raw": arr,
        }

    return {
        "path": path,
        "tensors": result,
        "n_tensors": len(result),
        "meta": meta,
    }


def combine_imatrix(
    imatrix_list: list[dict[str, Any]],
    method: str = "max",
) -> dict[str, Any]:
    """Combine multiple imatrix dicts into one by aggregating importance.

    Args:
        imatrix_list: List of imatrix dicts from :func:`read_imatrix`.
        method: ``"max"`` (conservative, default) or ``"mean"``.

    Returns:
        A single combined imatrix dict.
    """
    if not imatrix_list:
        return {}
    if len(imatrix_list) == 1:
        return imatrix_list[0]

    # Collect all tensor names across all imatrix files
    all_names: set[str] = set()
    for im in imatrix_list:
        all_names.update(im["tensors"].keys())

    combined_tensors: dict[str, dict] = {}
    for name in all_names:
        means: list[float] = []
        in_sum2_raw = None
        n_elements = 0
        max_vals: list[float] = []
        min_vals: list[float] = []

        for im in imatrix_list:
            if name not in im["tensors"]:
                continue
            tensor = im["tensors"][name]
            means.append(tensor["importance_mean"])
            max_vals.append(tensor.get("importance_max", 0.0))
            min_vals.append(tensor.get("importance_min", float("inf")))
            if in_sum2_raw is None and "in_sum2_raw" in tensor:
                in_sum2_raw = tensor["in_sum2_raw"]
            n_elements = max(n_elements, tensor["n_elements"])

        if not means:
            continue

        imp_mean = sum(means) / len(means) if method == "mean" else max(means)

        combined_tensors[name] = {
            "importance_mean": imp_mean,
            "importance_sum": imp_mean * n_elements,
            "importance_max": max(max_vals) if max_vals else 0.0,
            "importance_min": min(min_vals) if min_vals else float("inf"),
            "n_elements": n_elements,
            "in_sum2_raw": in_sum2_raw,
        }

    # Merge metadata (first-seen wins)
    combined_meta: dict[str, Any] = {}
    for im in imatrix_list:
        for key, value in im["meta"].items():
            combined_meta.setdefault(key, value)

    return {
        "path": "+".join(im["path"] for im in imatrix_list),
        "tensors": combined_tensors,
        "n_tensors": len(combined_tensors),
        "meta": combined_meta,
    }


def detect_tied_groups(imatrix: dict[str, Any], atol: float = 1e-5) -> list[list[str]]:
    """Find tied tensor groups (identical importance arrays).

    Two tensors are considered tied if their ``in_sum2`` arrays are
    element-wise equal within *atol*.
    """
    names = sorted(imatrix["tensors"].keys())
    tied_groups: list[list[str]] = []
    visited: set[str] = set()

    for i, n1 in enumerate(names):
        if n1 in visited:
            continue
        group = [n1]
        arr1 = imatrix["tensors"][n1].get("in_sum2_raw")
        if arr1 is None:
            tied_groups.append(group)
            visited.add(n1)
            continue

        for j in range(i + 1, len(names)):
            n2 = names[j]
            if n2 in visited:
                continue
            arr2 = imatrix["tensors"][n2].get("in_sum2_raw")
            if arr2 is None:
                continue
            if arr1.shape == arr2.shape and np.allclose(arr1, arr2, atol=atol):
                group.append(n2)
                visited.add(n2)

        tied_groups.append(group)
        visited.add(n1)

    return tied_groups


def build_importance_table(
    imatrix: dict[str, Any],
    model: dict[str, Any],
) -> ImportanceTable:
    """Build unified importance table, merging imatrix with model tensor info.

    Returns a dict mapping tensor names to importance metadata including
    ``importance_mean``, ``importance_sum``, ``n_elements``, and ``type``.
    """
    table: ImportanceTable = {}
    for tname, info in imatrix["tensors"].items():
        ttype = _imatrix_type(tname)
        table[tname] = {
            "importance_mean": info["importance_mean"],
            "importance_sum": info["importance_sum"],
            "importance_max": info["importance_max"],
            "importance_min": info["importance_min"],
            "n_elements": info["n_elements"],
            "type": ttype,
        }

    # Also index by name without trailing dot (safety for legacy paths)
    for tname, info in list(table.items()):
        if tname.endswith("."):
            table[tname.rstrip(".")] = info

    return table


def _imatrix_type(name: str) -> str:
    """Extract the functional type from an imatrix tensor name."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return parts[2]
    return name
