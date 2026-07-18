"""Final v3 freeze and protected joint-500 orchestration.

This module deliberately delegates scoring to the byte-identical v1
HarrisZPlusGeometricAdapter.  It changes only the publication identity,
preflight authorization binding, namespaces, validation policy, and reports.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import MethodMetadata
from ..hashing import canonical_json_bytes, file_sha256, stable_config_hash
from ..provenance import implementation_provenance
from . import pilot as legacy
from . import preflight as freeze_support
from .adapter import HarrisZPlusGeometricAdapter
from .config import HarrisZPlusConfig
from .preflight_v2 import EXPECTED_CANDIDATE_CONFIG_SHA256
from .preflight_v3 import (
    EXPECTED_CONTRACT_SHA256,
    METHOD_NAME,
    METHOD_VERSION,
    PASS_RELATIVE,
    PREFLIGHT_SCHEMA_VERSION,
    require_pilot_authorization,
)
from .provenance import implementation_source_hashes


DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
METHOD_RESULTS_RELATIVE = Path("results/harriszplus_rootsift_geometric_v3")
PILOT_RELATIVE = Path(
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v3"
)
PILOT_NAMESPACE = "harriszplus_rootsift_geometric_joint_500_v3"
PILOT_SCHEMA_VERSION = "harriszplus-joint-500-pilot-v3"
FREEZE_SCHEMA_VERSION = "harriszplus-config-freeze-v3"
AUTHORIZATION_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight.json"
)


class HarrisZPlusV3PublicationAdapter:
    """Versioned publication proxy over the unchanged score-producing adapter."""

    def __init__(self) -> None:
        self._inner = HarrisZPlusGeometricAdapter(
            HarrisZPlusConfig(backend="cuda", device="cuda:0")
        )
        self.config = self._inner.config

    def metadata(self) -> MethodMetadata:
        inner = self._inner.metadata()
        provenance = {
            **inner.implementation_provenance,
            "publication_wrapper": {
                "method_version": METHOD_VERSION,
                "score_producing_adapter_method_version": inner.method_version,
                "algorithm_sources_byte_exact": True,
                "candidate_algorithm_config_sha256": (
                    EXPECTED_CANDIDATE_CONFIG_SHA256
                ),
                "engineering_preflight_contract_sha256": (
                    EXPECTED_CONTRACT_SHA256
                ),
            },
        }
        return replace(
            inner,
            method_version=METHOD_VERSION,
            implementation_provenance=provenance,
            config={
                **inner.config,
                "method_version": METHOD_VERSION,
                "score_producing_adapter_method_version": inner.method_version,
                "candidate_algorithm_config_sha256": (
                    EXPECTED_CANDIDATE_CONFIG_SHA256
                ),
            },
        )

    def prepare(self, image_path: Path, image_metadata: Mapping[str, Any]) -> Any:
        return self._inner.prepare(image_path, image_metadata)

    def compare(self, representation_a: Any, representation_b: Any) -> Any:
        return self._inner.compare(representation_a, representation_b)

    def close(self) -> None:
        self._inner.close()


def v3_validation_policy() -> dict[str, Any]:
    """Return the exact final functional validation policy bound by the contract."""

    return {
        "schema_version": "harriszplus-functional-validation-policy-v3",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "response_atol": 5e-4,
        "response_rtol": 2e-4,
        "response_max_absolute_delta": 0.1,
        "minimum_real_pixel_coverage": 0.9999,
        "minimum_synthetic_pixel_coverage": 1.0,
        "candidate_count_allowed_delta": (
            "max(2, ceil(0.005 * max(cpu_count, cuda_count)))"
        ),
        "zero_versus_nonzero_candidate_count_fails": True,
        "spatial_tolerance_original_pixels": 0.5,
        "relative_scale_tolerance": 0.01,
        "minimum_bidirectional_matched_fraction": 0.995,
        "top_k": 3000,
        "minimum_top_k_bidirectional_overlap": 0.995,
        "maximum_downstream_raw_score_delta": 1,
        "minimum_exact_downstream_score_fraction": 0.95,
        "decision_threshold": 4,
        "decision_equality_required": True,
        "cuda_repeat_exact": True,
        "spearman_is_gate": False,
        "spearman_minimum_threshold": None,
        "auto_relaxation_allowed": False,
        "v4_relaxation_path_allowed": False,
        "frozen_before_500_results": True,
    }


def _new_adapter() -> HarrisZPlusV3PublicationAdapter:
    return HarrisZPlusV3PublicationAdapter()


def _v3_publication_markers(
    paths: Mapping[str, Path],
    *,
    exclude_before: bool,
) -> list[str]:
    """Allow frozen preflight evidence to predate the pilot integrity baseline."""

    before = (paths["pilot_root"] / "integrity/protected_before.json").resolve()
    markers: set[Path] = set()
    config_root = paths["method_root"] / "config"
    for root in (config_root, paths["pilot_root"]):
        if root.exists():
            markers.update(
                path.resolve() for path in root.rglob("*") if path.is_file()
            )
    if exclude_before:
        markers.discard(before)
    return [
        str(path)
        for path in sorted(markers, key=lambda item: str(item).lower())
    ]


def _configure_v3_runtime() -> None:
    """Bind the already-tested generic pilot helpers to the final v3 namespace."""

    legacy.METHOD_RESULTS_RELATIVE = METHOD_RESULTS_RELATIVE
    legacy.PILOT_RELATIVE = PILOT_RELATIVE
    legacy.PILOT_NAMESPACE = PILOT_NAMESPACE
    legacy.PILOT_SCHEMA_VERSION = PILOT_SCHEMA_VERSION
    legacy.SURVIVOR_SCHEMA_VERSION = (
        "harriszplus-per-dataset-self-survivors-v3"
    )
    legacy.METHOD_NAME = METHOD_NAME
    legacy.METHOD_VERSION = METHOD_VERSION
    legacy.PREFLIGHT_SCHEMA_VERSION = PREFLIGHT_SCHEMA_VERSION
    legacy._new_adapter = _new_adapter
    legacy._harriszplus_publication_markers = _v3_publication_markers

    freeze_support.METHOD_NAME = METHOD_NAME
    freeze_support.METHOD_VERSION = METHOD_VERSION
    freeze_support.PREFLIGHT_SCHEMA_VERSION = PREFLIGHT_SCHEMA_VERSION
    freeze_support.FREEZE_SCHEMA_VERSION = FREEZE_SCHEMA_VERSION
    freeze_support.fixed_validation_policy = v3_validation_policy


def _candidate_freeze_identity(project_root: Path) -> dict[str, str]:
    adapter = _new_adapter()
    try:
        metadata = adapter.metadata()
        runner_config = freeze_support._effective_runner_config(metadata)
        runtime_identity = freeze_support._canonical_runtime_identity(
            project_root, metadata.runtime
        )
        _, _, implementation_hash = implementation_provenance(
            adapter=adapter,
            method_metadata=metadata,
            startup_validation={},
            runner_source_path=(
                project_root / "src/fingerprint_benchmark/runner.py"
            ).resolve(),
        )
        return {
            "canonical_config_hash": stable_config_hash(runner_config),
            "implementation_hash": implementation_hash,
            "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
        }
    finally:
        adapter.close()


def _authorization_payload(project_root: Path) -> dict[str, Any]:
    pass_report = require_pilot_authorization(project_root=project_root)
    pass_path = project_root / PASS_RELATIVE
    if pass_report["sha256"] != file_sha256(pass_path):
        raise legacy.HarrisZPlusPilotError("The v3 pass artifact changed.")
    source_hashes = implementation_source_hashes(strict=True)[
        "required_score_producing_sources"
    ]
    if source_hashes != pass_report["algorithm_identity"][
        "expected_algorithm_source_sha256"
    ]:
        raise legacy.HarrisZPlusPilotError(
            "Score-producing sources changed after the v3 preflight."
        )
    peak = pass_report["performance_projection"].get("peak_vram_bytes")
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "authorized_after_frozen_v3_functional_pass",
        "passed": True,
        "pilot_500_authorized": True,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "engineering_preflight_v3": {
            "path": str(pass_path),
            "sha256": file_sha256(pass_path),
            "report_payload_sha256": pass_report["report_payload_sha256"],
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
        },
        "candidate_algorithm_config_sha256": (
            EXPECTED_CANDIDATE_CONFIG_SHA256
        ),
        "algorithm_source_sha256": source_hashes,
        "algorithm_sources_byte_exact_to_preflight": True,
        "candidate_freeze_identity": _candidate_freeze_identity(project_root),
        "ppi_coordinate_handling_all_passed": True,
        "device_binding": {
            "passed": True,
            "canonical_device": "cuda:0",
            "observed_device": pass_report["environment"]["device"],
        },
        "memory": {
            "passed": True,
            "v3_requirement": "no out-of-memory and no hidden CPU fallback",
            "out_of_memory_observed": False,
            "hidden_cpu_fallback_observed": False,
            "reported_peak_vram_bytes_information_only": peak,
            "capacity_fraction_is_correctness_gate": False,
        },
        "timing_synchronization": {
            "required_timing_fields_all_passed": True,
            "v3_timing_is_correctness_gate": False,
            "cuda_detector_uses_explicit_synchronization": True,
            "real_comparison_medians_reported": all(
                pass_report["performance_projection"]["datasets"][dataset][
                    "median_compare_ms"
                ]
                is not None
                for dataset in legacy.DATASETS
            ),
        },
        "validation_policy": v3_validation_policy(),
        "spearman_is_diagnostic_only": True,
        "no_v4_relaxation_path": True,
        "no_500_result_observed": True,
        "selection_sha256": legacy.EXPECTED_SELECTION_SHA256,
        "authorization_payload_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "preflight_sha256": file_sha256(pass_path),
                    "contract_sha256": EXPECTED_CONTRACT_SHA256,
                    "candidate_algorithm_config_sha256": (
                        EXPECTED_CANDIDATE_CONFIG_SHA256
                    ),
                    "selection_sha256": legacy.EXPECTED_SELECTION_SHA256,
                }
            )
        ).hexdigest(),
    }


def publish_pilot_authorization(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    _configure_v3_runtime()
    project_root = project_root.resolve()
    target = project_root / AUTHORIZATION_RELATIVE
    payload = _authorization_payload(project_root)
    legacy._publish_immutable_json(target, payload)
    if json.loads(target.read_text(encoding="utf-8")) != payload:
        raise legacy.HarrisZPlusPilotError(
            "Immutable v3 pilot authorization validation failed."
        )
    return {**payload, "path": str(target), "sha256": file_sha256(target)}


def _configure_reporting_v3() -> None:
    from . import reporting

    reporting.METHOD_NAME = METHOD_NAME
    reporting.METHOD_VERSION = METHOD_VERSION
    reporting.PILOT_NAMESPACE = PILOT_NAMESPACE
    reporting.SUPERVISOR_SCHEMA_VERSION = (
        "harriszplus-joint-500-supervisor-report-v3"
    )
    reporting.TECHNICAL_SCHEMA_VERSION = (
        "harriszplus-joint-500-technical-provenance-v3"
    )
    reporting.ARTIFACT_SCHEMA_VERSION = (
        "harriszplus-joint-500-artifact-manifest-v3"
    )


def run_complete_workflow(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    _configure_v3_runtime()
    authorization = publish_pilot_authorization(project_root=project_root)
    before = legacy.capture_integrity_before(project_root=project_root)
    token = before["validation"]
    manifests = legacy.prepare_base_manifests(
        project_root=project_root,
        data_root=data_root,
        integrity_before_validation=token,
    )
    freeze = legacy.freeze_after_preflight(
        project_root=project_root,
        integrity_before_validation=token,
    )
    runs = legacy.run_pilot(
        project_root=project_root,
        data_root=data_root,
        integrity_before_validation=token,
        progress_callback=lambda condition, current, total: print(
            f"{condition}: {current}/{total}", file=sys.stderr, flush=True
        ),
    )
    _configure_reporting_v3()
    final = legacy.finalize_outputs(project_root=project_root)
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "authorization": authorization,
        "integrity_before": before,
        "base_manifests": manifests,
        "freeze": freeze,
        "runs": runs,
        "final": final,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final HarrisZ+ v3 freeze and protected joint-500 workflow."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "command",
        choices=(
            "authorize",
            "integrity-before",
            "prepare",
            "freeze",
            "run",
            "finalize",
            "all",
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_v3_runtime()
    try:
        if args.command == "authorize":
            result = publish_pilot_authorization(project_root=args.project_root)
        elif args.command == "integrity-before":
            publish_pilot_authorization(project_root=args.project_root)
            result = legacy.capture_integrity_before(project_root=args.project_root)
        elif args.command == "prepare":
            result = legacy.prepare_base_manifests(
                project_root=args.project_root,
                data_root=args.data_root,
            )
        elif args.command == "freeze":
            result = legacy.freeze_after_preflight(project_root=args.project_root)
        elif args.command == "run":
            result = legacy.run_pilot(
                project_root=args.project_root,
                data_root=args.data_root,
                progress_callback=lambda condition, current, total: print(
                    f"{condition}: {current}/{total}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        elif args.command == "finalize":
            _configure_reporting_v3()
            result = legacy.finalize_outputs(project_root=args.project_root)
        elif args.command == "all":
            result = run_complete_workflow(
                project_root=args.project_root,
                data_root=args.data_root,
            )
        else:
            raise legacy.HarrisZPlusPilotError(
                f"Unsupported command: {args.command}"
            )
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
