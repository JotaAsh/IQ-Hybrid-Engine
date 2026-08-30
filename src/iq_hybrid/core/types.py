"""Central domain types and type aliases for IQ-Hybrid Engine."""

from typing import Any, Protocol, TypedDict

import numpy as np

# ── PEP 695 Type Aliases ────────────────────────────────────────────────────

type TierName = str
type TensorName = str
type TensorClass = str
type TensorAssignments = dict[TensorName, TierName]
type GroupRegistry = dict[int, tuple[list[TensorName], int, int]]


# ── TypedDict Structures ────────────────────────────────────────────────────


class ModelTensorInfo(TypedDict):
    shape: list[int]
    n_elements: int
    size_mib: float


class ModelInfo(TypedDict):
    path: str
    architecture: str
    features: dict[str, Any]
    tensors: dict[TensorName, ModelTensorInfo]
    n_tensors: int
    meta: dict[str, Any]


class TensorImportance(TypedDict, total=False):
    importance_mean: float
    importance_sum: float
    importance_max: float
    importance_min: float
    n_elements: int
    type: str
    in_sum2_raw: np.ndarray | None


type ImportanceTable = dict[TensorName, TensorImportance]


# ── Protocols ───────────────────────────────────────────────────────────────


class ArchitectureProtocol(Protocol):
    name: str

    def classify_tensor(self, name: TensorName) -> TensorClass: ...
    def get_tier_ladder(self, cls: TensorClass, wide_ladder: bool = False) -> list[TierName]: ...
    def get_ladder_base(self, cls: TensorClass, wide_ladder: bool = False) -> TierName: ...
    def get_ladder_ceiling(self, cls: TensorClass) -> TierName: ...
    def get_fixed_tensors(
        self, model: dict[str, Any], imp_table: ImportanceTable
    ) -> dict[TensorName, TierName]: ...
    def get_structural_tied_groups(
        self, tensor_names: list[TensorName]
    ) -> list[list[TensorName]]: ...
