"""Model reader for GGUF files (single or split shards).

Parses BF16 GGUF model files, detects the architecture, and returns a
structured model info dict with tensor metadata and feature flags.
"""

import logging
import re
from pathlib import Path
from typing import Any

import gguf
import numpy as np

from iq_hybrid.core.types import ModelInfo

logger = logging.getLogger(__name__)

SPLIT_PATTERN = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def detect_architecture(tensors: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    """Detect model architecture from tensor names and metadata.

    Returns one of: ``"qwen35"``, ``"bailingmoe3"``, ``"mellum2"``, ``"gemma4"``, ``"unknown"``.
    """
    names = list(tensors.keys())
    has_ssm = any("ssm_" in n for n in names)
    has_qkv = any("attn_qkv" in n for n in names)
    has_moe = any("exps" in n for n in names)
    has_shexp = any("shexp" in n for n in names)
    has_separate_qkv = any("attn_q.weight" in n for n in names)
    has_gemma_specific = any(
        t in n
        for t in ("layer_output_scale", "post_attention_norm", "post_ffw_norm")
        for n in names
    )

    is_bailing_meta = meta is not None and any(
        "bailing" in str(v).lower() for v in meta.values() if isinstance(v, str)
    )
    if (has_moe and (has_ssm or has_shexp)) or is_bailing_meta:
        return "bailingmoe3"
    if has_ssm and has_qkv:
        return "qwen35"
    if has_moe:
        return "mellum2"
    if has_separate_qkv or has_gemma_specific:
        return "gemma4"
    return "unknown"


def _detect_prefix(tensors: dict[str, Any]) -> str:
    """Detect whether tensors use ``blk`` or ``BLK`` prefix."""
    for name in tensors:
        if name.startswith("BLK."):
            return "BLK"
    return "blk"


def _estimate_layers(tensors: dict[str, Any]) -> int:
    """Estimate the number of transformer layers from block tensor indices."""
    max_layer = 0
    for name in tensors:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
            try:
                layer = int(parts[1])
                if layer > max_layer:
                    max_layer = layer
            except ValueError:
                pass
    return max_layer + 1  # layers are 0-indexed


def _find_split_paths(path: str) -> list[str]:
    """If *path* is a shard of a split GGUF (e.g. name-00001-of-00002.gguf),
    return the sorted list of all expected shard paths. Otherwise return [path].
    """
    match = SPLIT_PATTERN.search(path)
    if not match:
        return [path]

    total = int(match.group(2))
    width = len(match.group(1))  # typically 5
    prefix = path[: match.start()]

    return [f"{prefix}-{i:0{width}d}-of-{total:0{width}d}.gguf" for i in range(1, total + 1)]


def _read_single_shard(path: str) -> tuple[dict[str, dict], dict[str, Any]]:
    """Read tensors and metadata from a single GGUF shard."""
    reader = gguf.GGUFReader(path)
    tensors: dict[str, dict] = {}
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
            logger.debug("Could not decode field '%s' in shard '%s'", key, path)
            meta[key] = str(field)

    for tensor in reader.tensors:
        shape = list(tensor.shape)
        n_elements = int(np.prod(shape))
        tensors[tensor.name] = {
            "shape": shape,
            "n_elements": n_elements,
            "size_mib": n_elements * 2 / 1024 / 1024,
        }

    return tensors, meta


def read_model(path: str) -> ModelInfo:
    """Parse a BF16 GGUF model (single file or split shards).

    Returns a dict with keys: ``path``, ``architecture``, ``features``,
    ``tensors``, ``n_tensors``, ``meta``.

    Raises:
        FileNotFoundError: If the model file (or expected shards) cannot be found.
    """
    is_split = SPLIT_PATTERN.search(path) is not None
    shard_paths = _find_split_paths(path)

    tensors: dict[str, dict] = {}
    meta: dict[str, Any] = {}
    missing: list[str] = []

    for i, shard_path in enumerate(shard_paths):
        if not Path(shard_path).exists():
            missing.append(shard_path)
            continue
        shard_tensors, shard_meta = _read_single_shard(shard_path)
        tensors.update(shard_tensors)
        if i == 0:
            meta = shard_meta
        else:
            for key, value in shard_meta.items():
                meta.setdefault(key, value)

    if missing:
        if is_split:
            raise FileNotFoundError(
                "Missing shards required to read the full split GGUF model: " + ", ".join(missing)
            )
        else:
            raise FileNotFoundError(
                "Model file not found at the specified path: "
                + ", ".join(missing)
                + " (verify the path and filename are exact, including the extension)"
            )

    arch = detect_architecture(tensors, meta)
    prefix = _detect_prefix(tensors)
    n_layers = _estimate_layers(tensors)
    arch_features: dict[str, Any] = {
        "prefix": prefix,
        "n_layers": n_layers if n_layers > 0 else 32,
    }

    has_moe = any("exps" in n for n in tensors)
    if has_moe:
        arch_features["has_moe"] = True
        if arch_features.get("moe_intermediate_size", 0) == 0:
            for tname, tinfo in tensors.items():
                if any(x in tname for x in ("ffn_down_exps", "ffn_gate_exps", "ffn_up_exps")):
                    shape = tinfo.get("shape", [])
                    if len(shape) >= 2:
                        arch_features["moe_intermediate_size"] = (
                            shape[1] if len(shape) >= 3 else shape[0]
                        )
                        break
            if arch_features.get("moe_intermediate_size", 0) == 0 and arch == "mellum2":
                arch_features["moe_intermediate_size"] = 896

    has_nextn = any("nextn" in n for n in tensors)
    has_blk32 = n_layers <= 33 and any(
        n.startswith("blk.32.") or n.startswith("BLK.32.") for n in tensors
    )
    arch_features["has_mtp"] = has_nextn or has_blk32

    return {
        "path": path,
        "architecture": arch,
        "features": arch_features,
        "tensors": tensors,
        "n_tensors": len(tensors),
        "meta": meta,
    }
