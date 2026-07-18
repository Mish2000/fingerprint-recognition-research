"""Freeze and protected eight-run orchestration for PPI-aware HarrisZ+ v4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..hashing import canonical_json_bytes, file_sha256
from . import pilot as legacy
from . import preflight as freeze_support
from .adapter_v4 import HarrisZPlusPpiAwareGeometricAdapter
from .ppi_aware_v4 import (
    METHOD_NAME,
    METHOD_VERSION,
    PARENT_METHOD_VERSION,
    PpiAwareHarrisZPlusConfig,
)
from .preflight_v4 import (
    AUTHORIZATION_RELATIVE,
    DEFAULT_PROJECT_ROOT,
    FREEZE_SCHEMA_VERSION,
    METHOD_RESULTS_RELATIVE,
    PASS_RELATIVE,
    PILOT_RELATIVE,
    PREFLIGHT_SCHEMA_VERSION,
    require_pilot_authorization,
    v4_validation_policy,
)
from .v4_integrity import compare_inventories, protected_v1_v3_inventory


PILOT_NAMESPACE = (
    "harriszplus_rootsift_geometric_ppi_aware_joint_500_v4"
)
PILOT_SCHEMA_VERSION = "harriszplus-joint-500-pilot-v4"


def _new_adapter() -> HarrisZPlusPpiAwareGeometricAdapter:
    return HarrisZPlusPpiAwareGeometricAdapter(
        PpiAwareHarrisZPlusConfig().changed(
            backend="cuda", device="cuda:0"
        )
    )


def _v4_publication_markers(
    paths: Mapping[str, Path],
    *,
    exclude_before: bool,
) -> list[str]:
    """Permit frozen preflight evidence to predate pilot integrity-before."""

    before = (
        paths["pilot_root"] / "integrity/protected_before.json"
    ).resolve()
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


def configure_v4_runtime() -> None:
    """Bind the tested generic freeze/pilot machinery to the v4 namespace."""

    legacy.METHOD_RESULTS_RELATIVE = METHOD_RESULTS_RELATIVE
    legacy.PILOT_RELATIVE = PILOT_RELATIVE
    legacy.PILOT_NAMESPACE = PILOT_NAMESPACE
    legacy.PILOT_SCHEMA_VERSION = PILOT_SCHEMA_VERSION
    legacy.SURVIVOR_SCHEMA_VERSION = (
        "harriszplus-per-dataset-self-survivors-v4"
    )
    legacy.METHOD_NAME = METHOD_NAME
    legacy.METHOD_VERSION = METHOD_VERSION
    legacy.PREFLIGHT_SCHEMA_VERSION = PREFLIGHT_SCHEMA_VERSION
    legacy._new_adapter = _new_adapter
    legacy._harriszplus_publication_markers = _v4_publication_markers

    freeze_support.METHOD_NAME = METHOD_NAME
    freeze_support.METHOD_VERSION = METHOD_VERSION
    freeze_support.PREFLIGHT_SCHEMA_VERSION = PREFLIGHT_SCHEMA_VERSION
    freeze_support.FREEZE_SCHEMA_VERSION = FREEZE_SCHEMA_VERSION
    freeze_support.fixed_validation_policy = v4_validation_policy


def _authorization_payload(project_root: Path) -> dict[str, Any]:
    report = require_pilot_authorization(project_root=project_root)
    pass_path = project_root / PASS_RELATIVE
    adapter = _new_adapter()
    try:
        current = freeze_support._effective_runner_config(
            adapter.metadata()
        )
        current_config_hash = freeze_support.stable_config_hash(current)
        runtime_identity = freeze_support._canonical_runtime_identity(
            project_root, adapter.metadata().runtime
        )
        from ..provenance import implementation_provenance

        _, _, implementation_hash = implementation_provenance(
            adapter=adapter,
            method_metadata=adapter.metadata(),
            startup_validation={},
            runner_source_path=(
                project_root / "src/fingerprint_benchmark/runner.py"
            ).resolve(),
        )
    finally:
        adapter.close()
    current_identity = {
        "canonical_config_hash": current_config_hash,
        "implementation_hash": implementation_hash,
        "runtime_identity_hash": runtime_identity["runtime_identity_hash"],
    }
    if current_identity != report["candidate_freeze_identity"]:
        raise legacy.HarrisZPlusPilotError(
            "v4 config, implementation, or runtime changed after preflight."
        )
    physical = report["physical_scale_contract"]
    memory = report["performance_projection"]["vram"]
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "authorized_after_frozen_v4_ppi_aware_pass",
        "passed": True,
        "pilot_500_authorized": True,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "parent_method_version": PARENT_METHOD_VERSION,
        "engineering_preflight_v4": {
            "path": str(pass_path),
            "sha256": file_sha256(pass_path),
            "report_payload_sha256": report["report_payload_sha256"],
        },
        "physical_scale_contract": physical,
        "candidate_freeze_identity": current_identity,
        "ppi_coordinate_handling_all_passed": True,
        "device_binding": report["device_binding"],
        "memory": {
            **memory,
            "passed": memory["all_measurements_valid"],
            "allocated_and_reserved_summed": False,
        },
        "timing_synchronization": report["timing_synchronization"],
        "validation_policy": v4_validation_policy(),
        "no_parameter_tuning_performed": True,
        "no_500_result_observed": True,
        "selection_sha256": legacy.EXPECTED_SELECTION_SHA256,
        "authorization_payload_sha256": hashlib.sha256(
            canonical_json_bytes(
                {
                    "preflight_sha256": file_sha256(pass_path),
                    "physical_contract_sha256": physical["sha256"],
                    "candidate_freeze_identity": current_identity,
                    "selection_sha256": legacy.EXPECTED_SELECTION_SHA256,
                }
            )
        ).hexdigest(),
    }


def publish_pilot_authorization(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    configure_v4_runtime()
    project_root = project_root.resolve()
    target = project_root / AUTHORIZATION_RELATIVE
    payload = _authorization_payload(project_root)
    legacy._publish_immutable_json(target, payload)
    return {**payload, "path": str(target), "sha256": file_sha256(target)}


def _configure_reporting_v4() -> None:
    from . import reporting

    reporting.METHOD_NAME = METHOD_NAME
    reporting.METHOD_VERSION = METHOD_VERSION
    reporting.PILOT_NAMESPACE = PILOT_NAMESPACE
    reporting.SUPERVISOR_SCHEMA_VERSION = (
        "harriszplus-joint-500-supervisor-report-v4"
    )
    reporting.TECHNICAL_SCHEMA_VERSION = (
        "harriszplus-joint-500-technical-provenance-v4"
    )
    reporting.ARTIFACT_SCHEMA_VERSION = (
        "harriszplus-joint-500-artifact-manifest-v4"
    )


def finalize_v4(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    configure_v4_runtime()
    _configure_reporting_v4()
    root = project_root.resolve()
    pass_report = require_pilot_authorization(project_root=root)
    after = protected_v1_v3_inventory(root)
    comparison = compare_inventories(
        pass_report["integrity"]["before"], after
    )
    if not comparison["byte_identical"]:
        raise legacy.HarrisZPlusPilotError(
            "Protected v1-v3 assets changed during v4."
        )
    after_path = (
        root
        / METHOD_RESULTS_RELATIVE
        / "integrity/v1_v3_after.json"
    )
    comparison_path = (
        root
        / METHOD_RESULTS_RELATIVE
        / "integrity/v1_v3_comparison.json"
    )
    legacy._publish_immutable_json(after_path, after)
    legacy._publish_immutable_json(comparison_path, comparison)
    from .reporting_v4 import (
        build_v3_v4_scale_normalization_comparison,
    )
    from . import reporting

    paths = legacy.project_paths(root)
    authorization = json.loads(
        paths["preflight"].read_text(encoding="utf-8")
    )
    adapter = _new_adapter()
    try:
        freeze_validation = freeze_support.validate_frozen_configuration(
            config_directory=paths["config"],
            adapter=adapter,
            preflight_report=authorization,
        )
    finally:
        adapter.close()
    technical_comparison = build_v3_v4_scale_normalization_comparison(
        project_root=root
    )
    supervisor = reporting.build_supervisor_report(project_root=root)
    protected_integrity = legacy.capture_integrity_after(project_root=root)
    technical_provenance = reporting.build_technical_provenance(
        project_root=root,
        integrity_report=protected_integrity,
    )
    artifact_manifest = reporting.build_artifact_manifest(
        project_root=root
    )
    final = {
        "report": supervisor,
        "freeze_validation": freeze_validation,
        "integrity": protected_integrity,
        "technical_provenance": technical_provenance,
        "artifact_manifest": artifact_manifest,
        "v3_v4_technical_comparison": technical_comparison,
    }
    return {
        **final,
        "v1_v3_after": {
            **after,
            "path": str(after_path),
            "sha256": file_sha256(after_path),
        },
        "v1_v3_comparison": {
            **comparison,
            "path": str(comparison_path),
            "sha256": file_sha256(comparison_path),
        },
    }


def run_complete_workflow(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    configure_v4_runtime()
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
            f"{condition}: {current}/{total}",
            file=sys.stderr,
            flush=True,
        ),
    )
    final = finalize_v4(project_root=project_root)
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
        description="Freeze/run/finalize PPI-aware HarrisZ+ v4."
    )
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT_ROOT
    )
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
    configure_v4_runtime()
    try:
        if args.command == "authorize":
            result = publish_pilot_authorization(
                project_root=args.project_root
            )
        elif args.command == "integrity-before":
            publish_pilot_authorization(project_root=args.project_root)
            result = legacy.capture_integrity_before(
                project_root=args.project_root
            )
        elif args.command == "prepare":
            result = legacy.prepare_base_manifests(
                project_root=args.project_root,
                data_root=args.data_root,
            )
        elif args.command == "freeze":
            result = legacy.freeze_after_preflight(
                project_root=args.project_root
            )
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
            result = finalize_v4(project_root=args.project_root)
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


__all__ = [
    "PILOT_NAMESPACE",
    "PILOT_SCHEMA_VERSION",
    "configure_v4_runtime",
    "finalize_v4",
    "publish_pilot_authorization",
    "run_complete_workflow",
]
