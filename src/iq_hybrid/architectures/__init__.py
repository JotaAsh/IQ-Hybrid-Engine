"""Architectures module for IQ-Hybrid Engine."""

from iq_hybrid.architectures.base import BaseArchitecture
from iq_hybrid.architectures.dense import DenseArchitecture
from iq_hybrid.architectures.gemma_moe import GemmaMoEArchitecture
from iq_hybrid.architectures.qwen_hybrid import QwenHybridArchitecture
from iq_hybrid.architectures.registry import detect_architecture_strategy

__all__ = [
    "BaseArchitecture",
    "DenseArchitecture",
    "GemmaMoEArchitecture",
    "QwenHybridArchitecture",
    "detect_architecture_strategy",
]
