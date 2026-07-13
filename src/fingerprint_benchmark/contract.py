"""Method-neutral pairwise benchmark v2 contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


BENCHMARK_CONTRACT_VERSION = "pairwise-benchmark-v2"
RESULT_SCHEMA_VERSION = "pairwise-result-v2"
TIMING_MODE_COLD_PAIR = "cold_pair"
WARMUP_POLICY = {
    "strategy": "first_manifest_pair",
    "pair_count": 1,
    "prepare_operations_per_pair": 2,
    "compare_operations_per_pair": 1,
}

HIGHER_IS_MORE_SIMILAR = "higher_is_more_similar"
LOWER_IS_MORE_SIMILAR = "lower_is_more_similar"
SCORE_DIRECTIONS = (HIGHER_IS_MORE_SIMILAR, LOWER_IS_MORE_SIMILAR)

OK = "ok"
PREPARE_A_FAILURE = "prepare_a_failure"
PREPARE_B_FAILURE = "prepare_b_failure"
COMPARISON_FAILURE = "comparison_failure"
RESULT_STATUSES = (OK, PREPARE_A_FAILURE, PREPARE_B_FAILURE, COMPARISON_FAILURE)


class MethodExecutionError(RuntimeError):
    """Expected method/runtime failure that should be recorded for one pair."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        method_internal_ms: float | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.method_internal_ms = method_internal_ms
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class PreparedRepresentation:
    """Opaque method representation plus explicit representation provenance."""

    method: str
    method_version: str
    representation_format: str
    representation_version: str
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareOutcome:
    """Result of one method preparation operation."""

    representation: PreparedRepresentation
    method_internal_ms: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompareOutcome:
    """Result of one method comparison operation."""

    raw_score: float
    method_internal_ms: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodMetadata:
    """Run-level method identity, score semantics, and provenance."""

    method: str
    method_version: str
    score_direction: str
    score_semantics: str
    implementation_provenance: dict[str, Any]
    config: dict[str, Any]
    runtime: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score_direction not in SCORE_DIRECTIONS:
            raise ValueError(
                f"Invalid score_direction {self.score_direction!r}; "
                f"expected one of {SCORE_DIRECTIONS}."
            )
        if not str(self.score_semantics).strip():
            raise ValueError("score_semantics must be non-empty.")


@dataclass(frozen=True)
class BenchmarkRunSpec:
    """Immutable identity binding a run to one manifest and implementation."""

    expected_dataset: str
    expected_protocol: str
    manifest_path: Path
    manifest_sha256: str
    method: str
    method_version: str
    benchmark_contract_version: str
    config_hash: str
    implementation_hash: str

    def as_dict(self) -> dict[str, str]:
        return {
            "expected_dataset": self.expected_dataset,
            "expected_protocol": self.expected_protocol,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "method": self.method,
            "method_version": self.method_version,
            "benchmark_contract_version": self.benchmark_contract_version,
            "config_hash": self.config_hash,
            "implementation_hash": self.implementation_hash,
        }


class MethodAdapter(Protocol):
    """Narrow pairwise adapter protocol.

    Implementations must not threshold, normalize, decide TA/FR, or cache
    representations between pairs when the runner is in cold-pair mode.
    """

    def metadata(self) -> MethodMetadata:
        ...

    def prepare(
        self,
        image_path: Path,
        image_metadata: Mapping[str, Any],
    ) -> PrepareOutcome:
        ...

    def compare(
        self,
        representation_a: PreparedRepresentation,
        representation_b: PreparedRepresentation,
    ) -> CompareOutcome:
        ...

    def close(self) -> None:
        ...
