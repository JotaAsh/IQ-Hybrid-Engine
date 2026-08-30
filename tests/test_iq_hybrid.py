"""Comprehensive test suite for IQ-Hybrid Engine."""

import json
import tempfile
from pathlib import Path

import gguf
import numpy as np
import pytest

from iq_hybrid.architectures.qwen_hybrid import QwenHybridArchitecture
from iq_hybrid.architectures.registry import detect_architecture_strategy
from iq_hybrid.core.constants import BITS_IN_MIB
from iq_hybrid.core.greedy_engine import run_greedy_optimization
from iq_hybrid.core.mckp_solver import run_mckp_optimization
from iq_hybrid.io.imatrix_merger import merge_imatrix_files
from iq_hybrid.profiles.registry import resolve_profile


def test_architecture_detection() -> None:
    """Test architecture identification from tensor names."""
    qwen_tensors = {"blk.0.ssm_alpha.weight": {}, "blk.0.attn_qkv.weight": {}}
    arch_qwen = detect_architecture_strategy({"tensors": qwen_tensors})
    assert arch_qwen.name == "qwen_hybrid"

    dense_tensors = {"blk.0.attn_q.weight": {}, "blk.0.ffn_down.weight": {}}
    arch_dense = detect_architecture_strategy({"tensors": dense_tensors})
    assert arch_dense.name == "dense"

    gemma_tensors = {"blk.0.layer_output_scale.weight": {}, "blk.0.attn_q.weight": {}}
    arch_gemma = detect_architecture_strategy({"tensors": gemma_tensors})
    assert arch_gemma.name in ("gemma_moe", "gemma4")


def test_dynamic_embedding_scaling() -> None:
    """Test vocab/output embedding scaling based on target BPW."""
    arch = QwenHybridArchitecture()
    model = {
        "tensors": {
            "token_embd.weight": {"n_elements": 100_000_000},
            "output.weight": {"n_elements": 100_000_000},
            "blk.0.attn_q.weight": {"n_elements": 800_000_000},
        }
    }
    total_el = sum(v["n_elements"] for v in model["tensors"].values())

    # High BPW (>= 7.5) -> Q8_0
    target_mib = (7.8 * total_el) / BITS_IN_MIB
    fixed = arch.get_fixed_tensors(model, {}, target_mib, wide_ladder=True)
    assert fixed["token_embd.weight"] == "Q8_0"
    assert fixed["output.weight"] == "Q8_0"

    # High-Mid BPW (>= 6.0) -> Q6_K
    target_mib = (6.5 * total_el) / BITS_IN_MIB
    fixed = arch.get_fixed_tensors(model, {}, target_mib, wide_ladder=True)
    assert fixed["token_embd.weight"] == "Q6_K"

    # Mid BPW (>= 5.4) -> Q5_K
    target_mib = (5.6 * total_el) / BITS_IN_MIB
    fixed = arch.get_fixed_tensors(model, {}, target_mib, wide_ladder=True)
    assert fixed["token_embd.weight"] == "Q5_K"

    # Standard BPW (>= 3.8) -> IQ4_NL
    target_mib = (4.4 * total_el) / BITS_IN_MIB
    fixed = arch.get_fixed_tensors(model, {}, target_mib, wide_ladder=True)
    assert fixed["token_embd.weight"] == "IQ4_NL"

    # Low BPW (< 3.8) in wide ladder -> Q3_K
    target_mib = (3.2 * total_el) / BITS_IN_MIB
    fixed = arch.get_fixed_tensors(model, {}, target_mib, wide_ladder=True)
    assert fixed["token_embd.weight"] == "Q3_K"


def test_mtp_layer_detection_and_q8_fix() -> None:
    """Test that MTP layers (e.g. blk.32 with nextn) are fixed to Q8_0 and norms to F16."""
    arch = QwenHybridArchitecture()
    model = {
        "features": {"n_layers": 33, "has_mtp": True},
        "tensors": {
            "blk.32.nextn.eh_proj.weight": {"n_elements": 10_000_000},
            "blk.32.nextn.enorm.weight": {"n_elements": 4096},
            "blk.32.attn_q.weight": {"n_elements": 20_000_000},
            "blk.32.ffn_down.weight": {"n_elements": 40_000_000},
            "blk.32.attn_norm.weight": {"n_elements": 4096},
            "blk.0.attn_qkv.weight": {"n_elements": 20_000_000},
        },
    }
    fixed = arch.get_fixed_tensors(model, {}, target_size_mib=100.0, wide_ladder=True)
    assert fixed["blk.32.nextn.eh_proj.weight"] == "Q8_0"
    assert fixed["blk.32.nextn.enorm.weight"] == "F16"
    assert fixed["blk.32.attn_q.weight"] == "Q8_0"
    assert fixed["blk.32.ffn_down.weight"] == "Q8_0"
    assert fixed["blk.32.attn_norm.weight"] == "F16"
    assert "blk.0.attn_qkv.weight" not in fixed  # Elastic tensor remains free to optimize


def test_independent_tensor_optimization() -> None:
    """Test that dense/hybrid model tensors optimize independently without artificial pinning."""
    arch = QwenHybridArchitecture()
    model = {
        "features": {"n_layers": 32, "has_ssm": True},
        "tensors": {
            "blk.28.attn_gate.weight": {"n_elements": 1_000_000},
            "blk.28.attn_qkv.weight": {"n_elements": 30_000_000},
        },
    }
    imp_table = {
        "blk.28.attn_gate.weight": {"importance_mean": 500_000.0, "n_elements": 1_000_000},
        "blk.28.attn_qkv.weight": {"importance_mean": 500_000.0, "n_elements": 30_000_000},
    }

    # Run Greedy Optimization
    res_greedy, _ = run_greedy_optimization(
        arch, imp_table, [], model, target_size_mib=50.0, wide_ladder=True
    )
    assert res_greedy["blk.28.attn_gate.weight"] == "Q8_0"
    assert res_greedy["blk.28.attn_qkv.weight"] == "Q8_0"

    # Run MCKP Optimization
    res_mckp, _ = run_mckp_optimization(
        arch, imp_table, [], model, target_size_mib=50.0, wide_ladder=True
    )
    assert res_mckp["blk.28.attn_gate.weight"] == "Q8_0"
    assert res_mckp["blk.28.attn_qkv.weight"] == "Q8_0"


def test_imatrix_merger_math_and_limits() -> None:
    """Test GGUF imatrix merging math (add, max, mean) and strict 3-file validation."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        f1 = tmp / "mat1.gguf"
        f2 = tmp / "mat2.gguf"

        # Create dummy imatrix GGUF files
        w1 = gguf.GGUFWriter(str(f1), "imatrix")
        w1.add_string("general.type", "imatrix")
        w1.add_array("imatrix.datasets", ["dataset1"])
        w1.add_tensor(
            "blk.0.attn_q.weight.in_sum2",
            np.array([2.0, 4.0], dtype=np.float32),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        w1.add_tensor(
            "blk.0.attn_q.weight.counts",
            np.array([10.0], dtype=np.float32),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        w1.write_header_to_file()
        w1.write_kv_data_to_file()
        w1.write_tensors_to_file()
        w1.close()

        w2 = gguf.GGUFWriter(str(f2), "imatrix")
        w2.add_string("general.type", "imatrix")
        w2.add_array("imatrix.datasets", ["dataset2"])
        w2.add_tensor(
            "blk.0.attn_q.weight.in_sum2",
            np.array([3.0, 6.0], dtype=np.float32),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        w2.add_tensor(
            "blk.0.attn_q.weight.counts",
            np.array([20.0], dtype=np.float32),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        w2.write_header_to_file()
        w2.write_kv_data_to_file()
        w2.write_tensors_to_file()
        w2.close()

        # 1. Test Additive Merge
        out_add = tmp / "merged_add.gguf"
        merge_imatrix_files([f1, f2], output_path=out_add, method="add")
        r_add = gguf.GGUFReader(str(out_add))
        t_sum2 = next(t for t in r_add.tensors if t.name.endswith(".in_sum2"))
        t_counts = next(t for t in r_add.tensors if t.name.endswith(".counts"))
        assert np.allclose(t_sum2.data, [5.0, 10.0])
        assert np.allclose(t_counts.data, [30.0])
        assert "imatrix.datasets" in r_add.fields
        del r_add

        # 2. Test File Limits (4 files must raise ValueError)
        with pytest.raises(ValueError):
            merge_imatrix_files([f1, f2, f1, f2])


def test_profile_resolution() -> None:
    """Test preset profiles resolution."""
    p_ql, val_ql, is_preset_ql = resolve_profile("quality_ladder")
    assert is_preset_ql is True
    assert p_ql.name == "quality_ladder"

    p_wl, val_wl, is_preset_wl = resolve_profile("wide_ladder")
    assert is_preset_wl is True
    assert p_wl.name == "wide_ladder"

    p_num, val_num, is_preset_num = resolve_profile("6310")
    assert is_preset_num is False
    assert val_num == 6310.0

    # Ensure removed presets raise ValueError
    with pytest.raises(ValueError):
        resolve_profile("Q3_E")
    with pytest.raises(ValueError):
        resolve_profile("Q4_E")
    with pytest.raises(ValueError):
        resolve_profile("Q5_E")


def test_quality_ladder_hard_floor() -> None:
    """Test that quality_ladder strictly prevents any tensor from taking IQ1, IQ2, or IQ3."""
    arch = QwenHybridArchitecture()
    profile, _, _ = resolve_profile("quality_ladder")

    model = {
        "features": {"n_layers": 32},
        "tensors": {
            "blk.0.ffn_down.weight": {"n_elements": 10_000_000},
            "blk.1.ffn_down.weight": {"n_elements": 10_000_000},
        },
    }
    imp_table = {
        "blk.0.ffn_down.weight": {"importance_mean": 1.0, "n_elements": 10_000_000},
        "blk.1.ffn_down.weight": {"importance_mean": 1.0, "n_elements": 10_000_000},
    }

    # Very small budget (e.g. 5 MiB)
    res, _ = run_greedy_optimization(
        arch=arch,
        importance_table=imp_table,
        tied_groups=[],
        model=model,
        target_size_mib=5.0,
        wide_ladder=True,
        profile=profile,
    )

    # Even under low budget, quality_ladder must stay at or above IQ4_XS
    for tier in res.values():
        assert tier not in ("IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")
        assert (
            tier.startswith("IQ4")
            or tier.startswith("Q5")
            or tier.startswith("Q6")
            or tier.startswith("Q8")
            or tier == "F16"
        )


def test_custom_json_profile_recipe() -> None:
    """Test loading and executing a custom JSON profile recipe."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        recipe_path = Path(tmpdir) / "custom_recipe.json"
        recipe_data = {
            "name": "MY_CUSTOM_Q4",
            "target_bpw": 4.85,
            "force_wide_ladder": True,
            "ladders": {
                "ffn_down": ["IQ4_XS", "Q5_K", "Q8_0"],
            },
            "bases": {
                "ffn_down": "IQ4_XS",
            },
        }
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(recipe_data, f)

        profile, val, is_preset = resolve_profile(str(recipe_path))
        assert is_preset is True
        assert profile.name == "MY_CUSTOM_Q4"
        assert val == 4.85

        arch = QwenHybridArchitecture()
        ladder = profile.get_tier_ladder(arch, "ffn_down", wide_ladder=True)
        assert ladder == ["IQ4_XS", "Q5_K", "Q8_0"]
