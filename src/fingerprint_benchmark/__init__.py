"""Small pairwise benchmark layer for fingerprint research runs."""

from .contract import (
    BENCHMARK_CONTRACT_VERSION,
    COMPARISON_FAILURE,
    CompareOutcome,
    OK,
    PREPARE_A_FAILURE,
    PREPARE_B_FAILURE,
    PrepareOutcome,
    TIMING_MODE_COLD_PAIR,
    BenchmarkRunSpec,
    MethodExecutionError,
    MethodMetadata,
    PreparedRepresentation,
)

__all__ = [
    "BENCHMARK_CONTRACT_VERSION",
    "BenchmarkRunSpec",
    "COMPARISON_FAILURE",
    "CompareOutcome",
    "OK",
    "PREPARE_A_FAILURE",
    "PREPARE_B_FAILURE",
    "PrepareOutcome",
    "TIMING_MODE_COLD_PAIR",
    "MethodExecutionError",
    "MethodMetadata",
    "PreparedRepresentation",
]
