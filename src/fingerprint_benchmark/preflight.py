"""Manifest preflight for pairwise benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .contract import BenchmarkRunSpec
from .hashing import file_sha256
from .manifest import PairRecord, read_pair_manifest


class BenchmarkPreflightError(ValueError):
    """Raised before pair execution when a manifest/run identity is unsafe."""


_VALIDATOR_MODULES = {
    ("sd300b", "plain_self"): "fingerprint_data_discovery.sd300b_plain_self",
    ("sd300b", "roll_self"): "fingerprint_data_discovery.sd300b_roll_self",
    ("sd300b", "plain_roll"): "fingerprint_data_discovery.sd300b_plain_roll",
    ("sd300c", "plain_self"): "fingerprint_data_discovery.sd300c_plain_self",
    ("sd300c", "roll_self"): "fingerprint_data_discovery.sd300c_roll_self",
    ("sd300c", "plain_roll"): "fingerprint_data_discovery.sd300c_plain_roll",
}


def preflight_manifest(
    *,
    manifest_path: Path,
    expected_dataset: str,
    expected_protocol: str,
    run_spec: BenchmarkRunSpec,
    data_root: Path = DEFAULT_DATA_ROOT,
    dedicated_validator: Callable[[Path, Path], Any] | None = None,
) -> tuple[list[PairRecord], dict[str, Any]]:
    """Run the dedicated validator and bind every record to ``run_spec``."""

    validator = dedicated_validator or validator_for(expected_dataset, expected_protocol)
    try:
        validator_report = validator(manifest_path, data_root)
    except Exception as exc:
        raise BenchmarkPreflightError(
            f"Dedicated validator rejected {expected_dataset}/{expected_protocol} manifest: {exc}"
        ) from exc

    pairs = read_pair_manifest(manifest_path)
    if not pairs:
        raise BenchmarkPreflightError("Benchmark manifest is empty.")

    for index, pair in enumerate(pairs, start=1):
        if pair.dataset != expected_dataset or pair.protocol != expected_protocol:
            raise BenchmarkPreflightError(
                "Manifest identity mismatch at record "
                f"{index} ({pair.pair_id!r}): got {pair.dataset}/{pair.protocol}, "
                f"expected {expected_dataset}/{expected_protocol}."
            )

    actual_sha = file_sha256(manifest_path)
    expected = {
        "expected_dataset": expected_dataset,
        "expected_protocol": expected_protocol,
        "manifest_sha256": actual_sha,
    }
    actual = {
        "expected_dataset": run_spec.expected_dataset,
        "expected_protocol": run_spec.expected_protocol,
        "manifest_sha256": run_spec.manifest_sha256,
    }
    if actual != expected:
        raise BenchmarkPreflightError(
            f"Run specification does not match the validated manifest: expected {expected}, got {actual}."
        )
    if run_spec.manifest_path.resolve() != manifest_path.resolve():
        raise BenchmarkPreflightError(
            "Run specification manifest_path does not match the manifest supplied for execution."
        )

    return pairs, _report_dict(validator_report)


def validator_for(dataset: str, protocol: str) -> Callable[[Path, Path], Any]:
    module_name = _VALIDATOR_MODULES.get((dataset, protocol))
    if module_name is None:
        raise BenchmarkPreflightError(
            f"No dedicated manifest validator is registered for {dataset}/{protocol}."
        )
    module = import_module(module_name)
    return module.validate_manifest


def _report_dict(report: Any) -> dict[str, Any]:
    if is_dataclass(report):
        return asdict(report)
    if isinstance(report, dict):
        return dict(report)
    return {"result": str(report)}
