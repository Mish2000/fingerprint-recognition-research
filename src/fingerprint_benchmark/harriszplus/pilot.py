"""Resumable, no-overwrite HarrisZ+ joint-500 pilot orchestration.

The workflow is intentionally narrow: it reuses the authoritative SourceAFIS
selection/manifests, performs the engineering preflight, freezes configuration,
then publishes exactly eight generic-runner bundles at short fixed paths.  Each
dataset derives its own survivor population before genuine and circular-shift
negative manifests are built.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import BENCHMARK_CONTRACT_VERSION, OK, BenchmarkRunSpec, MethodAdapter
from ..hashing import canonical_json_bytes, file_sha256, stable_hash
from ..manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from ..runner import (
    METADATA_FILENAME,
    RESULT_FILENAME,
    read_result_rows,
    run_benchmark_manifest,
    validate_result_bundle,
)
from .preflight import (
    DATASETS,
    EXPECTED_PPI,
    EXPECTED_SELECTION_COUNT,
    EXPECTED_SELECTION_SHA256,
    INTEGRITY_SCHEMA_VERSION,
    METHOD_NAME,
    METHOD_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    SELF_PROTOCOLS,
    HarrisZPlusPreflightError,
    SelectedIdentity,
    collect_narrow_protected_snapshot,
    compare_protected_snapshots,
    freeze_configuration,
    load_and_verify_selection,
    run_engineering_preflight,
    validate_authoritative_manifest,
    validate_frozen_configuration,
)


DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
METHOD_RESULTS_RELATIVE = Path("results/harriszplus_rootsift_geometric")
PILOT_RELATIVE = Path("results/pilots/harriszplus_rootsift_geometric_joint_500_v1")
SOURCE_PILOT_RELATIVE = Path("results/pilots/sourceafis_joint_500_v1")

PILOT_NAMESPACE = "harriszplus_rootsift_geometric_joint_500_v1"
PILOT_SCHEMA_VERSION = "harriszplus-joint-500-pilot-v1"
SURVIVOR_SCHEMA_VERSION = "harriszplus-per-dataset-self-survivors-v1"
THRESHOLD = 4
NEGATIVE_PROTOCOL = "plain_roll_next_subject_impostor"

RUN_CONDITIONS = (
    ("sd300b", "plain_self", "plain_self", "self"),
    ("sd300b", "roll_self", "roll_self", "self"),
    ("sd300b", "plain_roll_genuine", "plain_roll", "genuine"),
    ("sd300b", "plain_roll_negative", NEGATIVE_PROTOCOL, "negative"),
    ("sd300c", "plain_self", "plain_self", "self"),
    ("sd300c", "roll_self", "roll_self", "self"),
    ("sd300c", "plain_roll_genuine", "plain_roll", "genuine"),
    ("sd300c", "plain_roll_negative", NEGATIVE_PROTOCOL, "negative"),
)

SURVIVOR_COLUMNS = (
    "selection_index",
    "subject_id",
    "canonical_finger_position",
    "plain_self_pair_id",
    "plain_self_status",
    "plain_self_raw_score",
    "plain_self_accepted",
    "roll_self_pair_id",
    "roll_self_status",
    "roll_self_raw_score",
    "roll_self_accepted",
)
EXCLUDED_COLUMNS = (*SURVIVOR_COLUMNS, "reason_flags")
PAIRING_COLUMNS = (
    "negative_pair_id",
    "dataset",
    "protocol",
    "canonical_finger_position",
    "subject_id_a",
    "subject_id_b",
    "selection_index_a",
    "selection_index_b",
    "plain_group_index",
    "roll_group_index",
    "shift",
    "ppi",
    "raw_frgp_a",
    "raw_frgp_b",
    "path_a",
    "path_b",
    "source_plain_roll_pair_id_a",
    "source_plain_roll_pair_id_b",
)


class HarrisZPlusPilotError(ValueError):
    """Raised when the frozen pilot cannot be resumed safely."""


@dataclass(frozen=True)
class BundleView:
    dataset: str
    label: str
    protocol: str
    role: str
    manifest_path: Path
    bundle_path: Path
    pairs: list[PairRecord]
    rows: list[dict[str, str]]
    metadata: dict[str, Any]


def project_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "project_root": root,
        "method_root": root / METHOD_RESULTS_RELATIVE,
        "pilot_root": root / PILOT_RELATIVE,
        "source_pilot_root": root / SOURCE_PILOT_RELATIVE,
        "selection": root / SOURCE_PILOT_RELATIVE / "selected_identities.csv",
        "source_manifests": root / SOURCE_PILOT_RELATIVE / "manifests",
        "source_artifact_manifest": root / SOURCE_PILOT_RELATIVE / "artifact_manifest.json",
        "preflight": root / METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight.json",
        "config": root / METHOD_RESULTS_RELATIVE / "config",
    }


def capture_integrity_before(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    paths = project_paths(project_root)
    target = paths["pilot_root"] / "integrity/protected_before.json"
    if target.exists():
        validation = validate_integrity_before_gate(project_root=paths["project_root"])
        snapshot = _read_json(target)
        return {
            **snapshot,
            "path": str(target),
            "sha256": file_sha256(target),
            "validated_current": True,
            "validation": validation,
        }
    prior_publications = _harriszplus_publication_markers(paths, exclude_before=True)
    if prior_publications:
        raise HarrisZPlusPilotError(
            "Integrity-before must be captured before every HarrisZ+ preflight, config, manifest, "
            "or run publication; existing markers: " + ", ".join(prior_publications[:8])
        )
    snapshot = _collect_bound_protected_snapshot(paths["project_root"])
    _publish_immutable_json(target, snapshot)
    validation = _new_integrity_validation_token(paths, target, snapshot)
    return {
        **snapshot,
        "path": str(target),
        "sha256": file_sha256(target),
        "validated_current": True,
        "validation": validation,
    }


def capture_integrity_after(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    paths = project_paths(project_root)
    before_path = paths["pilot_root"] / "integrity/protected_before.json"
    before = _read_json(before_path)
    _validate_bound_snapshot(before)
    after = _collect_bound_protected_snapshot(paths["project_root"])
    report = compare_protected_snapshots(before, after)
    inventory_report = _compare_manifest_referenced_inventories(before, after)
    if inventory_report["status"] != "ok":
        raise HarrisZPlusPilotError(
            "Manifest-referenced dataset inventory changed during the pilot: "
            + json.dumps(inventory_report, sort_keys=True)
        )
    after_path = paths["pilot_root"] / "integrity/protected_after.json"
    report_path = paths["pilot_root"] / "integrity/protected_integrity.json"
    _publish_immutable_json(after_path, after)
    final_report = {
        **report,
        "manifest_referenced_dataset_inventory": inventory_report,
        "combined_snapshot_sha256_match": (
            before.get("combined_snapshot_sha256") == after.get("combined_snapshot_sha256")
        ),
        "before_snapshot_path": str(before_path),
        "before_snapshot_sha256": file_sha256(before_path),
        "after_snapshot_path": str(after_path),
        "after_snapshot_sha256": file_sha256(after_path),
    }
    _publish_immutable_json(report_path, final_report)
    return {**final_report, "path": str(report_path), "sha256": file_sha256(report_path)}


def validate_integrity_before_gate(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    """Require a current, internally bound baseline that predates all HarrisZ+ outputs."""

    paths = project_paths(project_root)
    before_path = paths["pilot_root"] / "integrity/protected_before.json"
    if not before_path.is_file():
        raise HarrisZPlusPilotError(
            "A valid integrity-before artifact is required before any HarrisZ+ pilot bundle."
        )
    before = _read_json(before_path)
    _validate_bound_snapshot(before)
    current = _collect_bound_protected_snapshot(paths["project_root"])
    protected_report = compare_protected_snapshots(before, current)
    inventory_report = _compare_manifest_referenced_inventories(before, current)
    if inventory_report["status"] != "ok":
        raise HarrisZPlusPilotError(
            "Integrity-before no longer matches the current manifest-referenced dataset inventory."
        )
    _validate_before_predates_publications(paths, before_path)
    return {
        "schema_version": "harriszplus-integrity-before-gate-v1",
        "validated": True,
        "before_path": str(before_path),
        "before_sha256": file_sha256(before_path),
        "combined_snapshot_sha256": before["combined_snapshot_sha256"],
        "protected_snapshot": protected_report,
        "manifest_referenced_dataset_inventory": inventory_report,
        "predates_all_harriszplus_publications": True,
    }


def _collect_bound_protected_snapshot(project_root: Path) -> dict[str, Any]:
    narrow = collect_narrow_protected_snapshot(project_root)
    dataset_inventory = _manifest_referenced_dataset_inventory(project_root)
    binding = {
        "narrow_tree_sha256": narrow["tree_sha256"],
        "manifest_referenced_dataset_inventory": dataset_inventory,
    }
    return {
        **narrow,
        "manifest_referenced_dataset_inventory": dataset_inventory,
        "combined_snapshot_sha256": stable_hash(binding),
        "combined_snapshot_contract": (
            "canonical SHA-256 over the narrow protected tree identity plus current all-protocol "
            "dataset stats and content hashes for every image referenced by the authoritative "
            "B/C plain-self and roll-self 500-row manifests"
        ),
    }


def _manifest_referenced_dataset_inventory(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    protocol_manifests = sorted(
        (project_root / "protocols").rglob("*.csv"),
        key=lambda path: path.as_posix().lower(),
    )
    if not protocol_manifests:
        raise HarrisZPlusPilotError("No protocol manifests exist for dataset integrity inventory.")
    protocol_records = [_artifact_record(path) for path in protocol_manifests]
    all_protocol_paths = sorted(
        {
            path.resolve()
            for manifest in protocol_manifests
            for pair in read_pair_manifest(manifest)
            for path in (pair.path_a, pair.path_b)
        },
        key=lambda path: str(path).lower(),
    )
    stat_digest = hashlib.sha256()
    stat_total_bytes = 0
    for path in all_protocol_paths:
        stat = path.stat()
        record = {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        stat_total_bytes += int(stat.st_size)
        stat_digest.update(canonical_json_bytes(record) + b"\n")

    paths = project_paths(project_root)
    selection = load_and_verify_selection(paths["selection"])
    source_artifacts = _source_artifact_records(paths["source_artifact_manifest"])
    authoritative_manifests: list[dict[str, Any]] = []
    authoritative_paths: set[Path] = set()
    for dataset in DATASETS:
        for protocol in SELF_PROTOCOLS:
            relative = f"manifests/{dataset}/{protocol}.csv"
            manifest_path = paths["source_pilot_root"] / relative
            artifact = source_artifacts.get(relative)
            if artifact is None:
                raise HarrisZPlusPilotError(
                    f"Authoritative dataset inventory manifest is unsealed: {relative}"
                )
            validate_authoritative_manifest(
                manifest_path,
                dataset=dataset,
                protocol=protocol,
                selection=selection,
                data_root=DEFAULT_DATA_ROOT,
                expected_sha256=artifact["sha256"],
                require_self=True,
            )
            authoritative_manifests.append(
                {
                    "path": str(manifest_path.resolve()),
                    "size": manifest_path.stat().st_size,
                    "sha256": file_sha256(manifest_path),
                    "dataset": dataset,
                    "protocol": protocol,
                    "row_count": EXPECTED_SELECTION_COUNT,
                }
            )
            for pair in read_pair_manifest(manifest_path):
                authoritative_paths.update((pair.path_a.resolve(), pair.path_b.resolve()))

    content_records: list[dict[str, Any]] = []
    for path in sorted(authoritative_paths, key=lambda item: str(item).lower()):
        stat = path.stat()
        content_records.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(path),
            }
        )
    authoritative_content_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in content_records)
    ).hexdigest()
    return {
        "schema_version": "harriszplus-manifest-referenced-dataset-inventory-v1",
        "full_dataset_tree_content_rescan_performed": False,
        "authoritative_pilot_images_content_scanned": True,
        "all_protocol_manifest_count": len(protocol_records),
        "all_protocol_manifests": protocol_records,
        "all_protocol_dataset_file_count": len(all_protocol_paths),
        "all_protocol_dataset_total_bytes": stat_total_bytes,
        "all_protocol_dataset_stat_inventory_sha256": stat_digest.hexdigest(),
        "authoritative_self_manifest_count": len(authoritative_manifests),
        "authoritative_self_manifests": authoritative_manifests,
        "authoritative_pilot_image_count": len(content_records),
        "authoritative_pilot_image_total_bytes": sum(
            record["size"] for record in content_records
        ),
        "authoritative_pilot_image_content_inventory_sha256": authoritative_content_sha,
        "authoritative_pilot_images": content_records,
        "policy": (
            "stat every image referenced by current repository protocol manifests and content-hash "
            "every unique image referenced by the four sealed SourceAFIS joint-500 B/C self "
            "manifests; do not scan unreferenced dataset files"
        ),
    }


def _validate_bound_snapshot(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("schema_version") != INTEGRITY_SCHEMA_VERSION:
        raise HarrisZPlusPilotError("Integrity-before snapshot schema mismatch.")
    inventory = snapshot.get("manifest_referenced_dataset_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("schema_version") != (
        "harriszplus-manifest-referenced-dataset-inventory-v1"
    ):
        raise HarrisZPlusPilotError("Integrity-before lacks the bound dataset inventory.")
    expected = stable_hash(
        {
            "narrow_tree_sha256": snapshot.get("tree_sha256"),
            "manifest_referenced_dataset_inventory": inventory,
        }
    )
    if snapshot.get("combined_snapshot_sha256") != expected:
        raise HarrisZPlusPilotError("Integrity-before combined snapshot binding is invalid.")


def _compare_manifest_referenced_inventories(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    first = before.get("manifest_referenced_dataset_inventory")
    second = after.get("manifest_referenced_dataset_inventory")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise HarrisZPlusPilotError("Dataset inventory is missing from protected snapshots.")
    keys = (
        "all_protocol_dataset_stat_inventory_sha256",
        "authoritative_pilot_image_content_inventory_sha256",
    )
    matches = {key: first.get(key) == second.get(key) for key in keys}
    return {
        "status": "ok" if all(matches.values()) else "changed",
        "matches": matches,
        "before": {key: first.get(key) for key in keys},
        "after": {key: second.get(key) for key in keys},
        "authoritative_pilot_image_count": first.get("authoritative_pilot_image_count"),
        "all_protocol_dataset_file_count": first.get("all_protocol_dataset_file_count"),
    }


def _harriszplus_publication_markers(
    paths: Mapping[str, Path], *, exclude_before: bool
) -> list[str]:
    before = (paths["pilot_root"] / "integrity/protected_before.json").resolve()
    markers: set[Path] = set()
    for root in (paths["method_root"], paths["pilot_root"]):
        if root.exists():
            markers.update(path.resolve() for path in root.rglob("*") if path.is_file())
    runs_root = paths["pilot_root"] / "runs"
    if runs_root.exists():
        markers.update(path.resolve() for path in runs_root.glob("*/*") if path.is_dir())
    if exclude_before:
        markers.discard(before)
    return [str(path) for path in sorted(markers, key=lambda item: str(item).lower())]


def _validate_before_predates_publications(
    paths: Mapping[str, Path], before_path: Path
) -> None:
    before_mtime_ns = before_path.stat().st_mtime_ns
    postdated = []
    for marker in _harriszplus_publication_markers(paths, exclude_before=True):
        path = Path(marker)
        if path.stat().st_mtime_ns < before_mtime_ns:
            postdated.append(marker)
    if postdated:
        raise HarrisZPlusPilotError(
            "Integrity-before was captured or modified after HarrisZ+ publications: "
            + ", ".join(postdated[:8])
        )


def _validate_integrity_gate_token(
    paths: Mapping[str, Path], validation: Mapping[str, Any]
) -> None:
    before_path = paths["pilot_root"] / "integrity/protected_before.json"
    if validation.get("validated") is not True or not before_path.is_file():
        raise HarrisZPlusPilotError("Current integrity-before validation token is required.")
    if validation.get("before_sha256") != file_sha256(before_path):
        raise HarrisZPlusPilotError("Integrity-before changed after validation.")
    before = _read_json(before_path)
    _validate_bound_snapshot(before)
    if validation.get("combined_snapshot_sha256") != before.get("combined_snapshot_sha256"):
        raise HarrisZPlusPilotError("Integrity-before validation token is stale.")
    _validate_before_predates_publications(paths, before_path)


def _new_integrity_validation_token(
    paths: Mapping[str, Path],
    before_path: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_bound_snapshot(snapshot)
    _validate_before_predates_publications(paths, before_path)
    inventory_report = _compare_manifest_referenced_inventories(snapshot, snapshot)
    return {
        "schema_version": "harriszplus-integrity-before-gate-v1",
        "validated": True,
        "before_path": str(before_path),
        "before_sha256": file_sha256(before_path),
        "combined_snapshot_sha256": snapshot["combined_snapshot_sha256"],
        "protected_snapshot": {
            "passed": True,
            "protected_artifacts_unchanged": True,
            "before_tree_sha256": snapshot["tree_sha256"],
            "after_tree_sha256": snapshot["tree_sha256"],
        },
        "manifest_referenced_dataset_inventory": inventory_report,
        "predates_all_harriszplus_publications": True,
    }


def _require_integrity_before_validation(
    paths: Mapping[str, Path],
    validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if validation is None:
        return validate_integrity_before_gate(project_root=paths["project_root"])
    _validate_integrity_gate_token(paths, validation)
    return dict(validation)


def prepare_base_manifests(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    integrity_before_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish byte-exact 500-row B/C self manifests from the recorded pilot."""

    paths = project_paths(project_root)
    integrity_before_validation = _require_integrity_before_validation(
        paths, integrity_before_validation
    )
    project_root = paths["project_root"]
    data_root = data_root.resolve()
    selection = load_and_verify_selection(paths["selection"])
    source_artifacts = _source_artifact_records(paths["source_artifact_manifest"])
    source_selection = source_artifacts.get("selected_identities.csv")
    if not source_selection or source_selection["sha256"] != EXPECTED_SELECTION_SHA256:
        raise HarrisZPlusPilotError(
            "SourceAFIS pilot artifact manifest does not bind the expected 500 selection."
        )

    manifest_records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for protocol in SELF_PROTOCOLS:
            relative = f"manifests/{dataset}/{protocol}.csv"
            source_path = paths["source_pilot_root"] / relative
            artifact = source_artifacts.get(relative)
            if artifact is None:
                raise HarrisZPlusPilotError(f"Authoritative artifact record is missing: {relative}")
            validation = validate_authoritative_manifest(
                source_path,
                dataset=dataset,
                protocol=protocol,
                selection=selection,
                data_root=data_root,
                expected_sha256=artifact["sha256"],
                require_self=True,
            )
            target = paths["pilot_root"] / relative
            _publish_immutable_bytes(target, source_path.read_bytes())
            if file_sha256(target) != artifact["sha256"]:
                raise HarrisZPlusPilotError(f"Copied manifest is not byte-exact: {target}")
            manifest_records.append(
                {
                    **validation,
                    "source_path": str(source_path),
                    "target_path": str(target),
                    "byte_exact_copy": True,
                }
            )

    identity_projection = [
        {
            "selection_index": item.selection_index,
            "subject_id": item.subject_id,
            "canonical_finger_position": item.canonical_finger_position,
            "source_identity_key": item.source_identity_key,
        }
        for item in selection
    ]
    reference = {
        "schema_version": "harriszplus-selected-identities-reference-v1",
        "source_path": str(paths["selection"]),
        "expected_sha256": EXPECTED_SELECTION_SHA256,
        "actual_sha256": file_sha256(paths["selection"]),
        "row_count": len(selection),
        "selection_index_contiguous": True,
        "identities_and_order_unchanged": True,
        "same_500_used_in_sd300b_and_sd300c": True,
        "canonical_positions_unchanged": True,
        "image_paths_preserved_from_authoritative_manifests": True,
        "identity_sequence_sha256": hashlib.sha256(
            canonical_json_bytes(identity_projection)
        ).hexdigest(),
        "source_artifact_manifest": {
            "path": str(paths["source_artifact_manifest"]),
            "sha256": file_sha256(paths["source_artifact_manifest"]),
        },
        "self_manifests": manifest_records,
    }
    reference_path = paths["pilot_root"] / "selected_identities_reference.json"
    _publish_immutable_json(reference_path, reference)
    provenance = {
        "schema_version": "harriszplus-base-manifest-provenance-v1",
        "selection_reference_path": str(reference_path),
        "selection_reference_sha256": file_sha256(reference_path),
        "dataset_tree_scanned": False,
        "authoritative_source": "sourceafis_joint_500_v1 recorded selection/manifests",
        "manifests": manifest_records,
    }
    provenance_path = paths["pilot_root"] / "manifests/base_manifest_provenance.json"
    _publish_immutable_json(provenance_path, provenance)
    return {**provenance, "path": str(provenance_path), "sha256": file_sha256(provenance_path)}


def execute_engineering_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    device: str | None = None,
    integrity_before_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run once or validate an existing immutable passing preflight artifact."""

    paths = project_paths(project_root)
    integrity_before_validation = _require_integrity_before_validation(
        paths, integrity_before_validation
    )
    if paths["preflight"].exists():
        report = _read_json(paths["preflight"])
        if (
            report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
            or report.get("passed") is not True
            or report.get("pilot_500_authorized") is not True
            or report.get("ppi_coordinate_handling_all_passed") is not True
            or not isinstance(report.get("device_binding"), Mapping)
            or report["device_binding"].get("passed") is not True
            or not isinstance(report.get("memory"), Mapping)
            or report["memory"].get("passed") is not True
            or not isinstance(report.get("timing_synchronization"), Mapping)
            or report["timing_synchronization"].get(
                "required_timing_fields_all_passed"
            )
            is not True
        ):
            raise HarrisZPlusPilotError(
                "Existing preflight artifact did not authorize the pilot with passing PPI handling."
            )
        return {**report, "path": str(paths["preflight"]), "sha256": file_sha256(paths["preflight"])}
    prepare_base_manifests(
        project_root=project_root,
        data_root=data_root,
        integrity_before_validation=integrity_before_validation,
    )
    report = run_engineering_preflight(
        project_root=paths["project_root"],
        data_root=data_root,
        selection_path=paths["selection"],
        source_manifest_root=paths["source_manifests"],
        device=device,
    )
    _publish_immutable_json(paths["preflight"], report)
    return {**report, "path": str(paths["preflight"]), "sha256": file_sha256(paths["preflight"])}


def freeze_after_preflight(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    integrity_before_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = project_paths(project_root)
    _require_integrity_before_validation(paths, integrity_before_validation)
    report = _read_json(paths["preflight"])
    adapter = _new_adapter()
    try:
        return freeze_configuration(
            project_root=paths["project_root"],
            adapter=adapter,
            preflight_report=report,
            config_directory=paths["config"],
        )
    finally:
        adapter.close()


def derive_survivors_and_pair_manifests(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    dataset: str,
) -> dict[str, Any]:
    """Intersect accepted self rows, then build genuine and shift-1 negative manifests."""

    if dataset not in DATASETS:
        raise HarrisZPlusPilotError(f"Unsupported dataset: {dataset}")
    paths = project_paths(project_root)
    selection = load_and_verify_selection(paths["selection"])
    selection_by_identity = {item.identity: item for item in selection}
    plain = load_bundle(project_root=project_root, dataset=dataset, label="plain_self")
    roll = load_bundle(project_root=project_root, dataset=dataset, label="roll_self")
    plain_by_identity = _result_by_identity(plain)
    roll_by_identity = _result_by_identity(roll)
    expected = [item.identity for item in selection]
    if list(plain_by_identity) != expected or list(roll_by_identity) != expected:
        raise HarrisZPlusPilotError(f"Self result identity/order mismatch for {dataset}.")

    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    survivor_identities: list[tuple[str, int]] = []
    reason_counts: Counter[str] = Counter()
    plain_accepted = 0
    roll_accepted = 0
    plain_failures = 0
    roll_failures = 0
    both_failed = 0
    both_nonaccepted = 0
    for item in selection:
        plain_pair, plain_row = plain_by_identity[item.identity]
        roll_pair, roll_row = roll_by_identity[item.identity]
        plain_decision = accepted_result_row(plain_row)
        roll_decision = accepted_result_row(roll_row)
        plain_accepted += int(plain_decision)
        roll_accepted += int(roll_decision)
        plain_failures += int(plain_row["status"] != OK)
        roll_failures += int(roll_row["status"] != OK)
        both_nonaccepted += int(not plain_decision and not roll_decision)
        reasons: list[str] = []
        if not plain_decision:
            reason = (
                "plain_self_failure" if plain_row["status"] != OK else "plain_self_score_below_4"
            )
            reasons.append(reason)
            reason_counts[reason] += 1
        if not roll_decision:
            reason = "roll_self_failure" if roll_row["status"] != OK else "roll_self_score_below_4"
            reasons.append(reason)
            reason_counts[reason] += 1
        if plain_row["status"] != OK and roll_row["status"] != OK:
            both_failed += 1
        record = {
            "selection_index": item.selection_index,
            "subject_id": item.subject_id,
            "canonical_finger_position": item.canonical_finger_position,
            "plain_self_pair_id": plain_pair.pair_id,
            "plain_self_status": plain_row["status"],
            "plain_self_raw_score": plain_row["raw_score"],
            "plain_self_accepted": _csv_bool(plain_decision),
            "roll_self_pair_id": roll_pair.pair_id,
            "roll_self_status": roll_row["status"],
            "roll_self_raw_score": roll_row["raw_score"],
            "roll_self_accepted": _csv_bool(roll_decision),
        }
        if not reasons:
            survivor_identities.append(item.identity)
            included_rows.append(record)
        else:
            excluded_rows.append({**record, "reason_flags": ";".join(reasons)})

    source_artifacts = _source_artifact_records(paths["source_artifact_manifest"])
    genuine_relative = f"manifests/{dataset}/plain_roll_genuine.csv"
    source_genuine = paths["source_pilot_root"] / genuine_relative
    source_record = source_artifacts.get(genuine_relative)
    if source_record is None:
        raise HarrisZPlusPilotError(f"Source genuine artifact is missing: {genuine_relative}")
    validate_authoritative_manifest(
        source_genuine,
        dataset=dataset,
        protocol="plain_roll",
        selection=selection,
        data_root=data_root,
        expected_sha256=source_record["sha256"],
        require_genuine=True,
    )
    source_genuine_by_identity = {
        (pair.subject_id, pair.canonical_finger_position): pair
        for pair in read_pair_manifest(source_genuine)
    }
    survivors = set(survivor_identities)
    genuine_pairs = [
        source_genuine_by_identity[item.identity]
        for item in selection
        if item.identity in survivors
    ]
    if [(pair.subject_id, pair.canonical_finger_position) for pair in genuine_pairs] != survivor_identities:
        raise HarrisZPlusPilotError(f"Genuine survivor order mismatch for {dataset}.")
    negative_pairs, pairing_rows = build_negative_pairs(
        dataset=dataset,
        genuine_pairs=genuine_pairs,
        selection_by_identity=selection_by_identity,
    )
    if len(negative_pairs) != len(genuine_pairs):
        raise HarrisZPlusPilotError("Negative pair count must equal survivor count.")

    survivor_root = paths["pilot_root"] / "survivors" / dataset
    manifest_root = paths["pilot_root"] / "manifests" / dataset
    included_path = survivor_root / "included_identities.csv"
    excluded_path = survivor_root / "excluded_identities.csv"
    summary_path = survivor_root / "summary.json"
    genuine_path = manifest_root / "plain_roll_genuine.csv"
    negative_path = manifest_root / "plain_roll_negative.csv"
    pairing_path = manifest_root / "pairing_map.csv"
    _publish_immutable_bytes(included_path, _csv_bytes(included_rows, SURVIVOR_COLUMNS))
    _publish_immutable_bytes(excluded_path, _csv_bytes(excluded_rows, EXCLUDED_COLUMNS))
    _publish_immutable_bytes(genuine_path, _pair_manifest_bytes(genuine_pairs))
    _publish_immutable_bytes(negative_path, _pair_manifest_bytes(negative_pairs))
    _publish_immutable_bytes(pairing_path, _csv_bytes(pairing_rows, PAIRING_COLUMNS))

    summary = {
        "schema_version": SURVIVOR_SCHEMA_VERSION,
        "dataset": dataset,
        "threshold": THRESHOLD,
        "acceptance_rule": "status == ok and geometric_inlier_count >= 4",
        "tie_score_4_accepted": True,
        "base_identity_count": len(selection),
        "plain_self_accepted_count": plain_accepted,
        "plain_self_rejected_count": sum(
            1
            for _, row in plain_by_identity.values()
            if row["status"] == OK and not accepted_result_row(row)
        ),
        "plain_self_failure_count": plain_failures,
        "plain_self_rejected_or_failure_count": len(selection) - plain_accepted,
        "roll_self_accepted_count": roll_accepted,
        "roll_self_rejected_count": sum(
            1
            for _, row in roll_by_identity.values()
            if row["status"] == OK and not accepted_result_row(row)
        ),
        "roll_self_failure_count": roll_failures,
        "roll_self_rejected_or_failure_count": len(selection) - roll_accepted,
        "both_self_failure_count": both_nonaccepted,
        "both_self_technical_failure_count": both_failed,
        "both_self_nonaccepted_count": both_nonaccepted,
        "survivor_count": len(survivor_identities),
        "excluded_count": len(excluded_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "survivor_definition": "accepted(plain_self) intersection accepted(roll_self), per dataset",
        "genuine_or_negative_result_used_for_filtering": False,
        "replacement_identity_count": 0,
        "genuine_pair_count": len(genuine_pairs),
        "negative_pair_count": len(negative_pairs),
        "negative_protocol": {
            "group_by_canonical_finger_position": True,
            "order_by_selection_index": True,
            "circular_shift": 1,
            "different_subjects": True,
            "each_plain_once": True,
            "each_roll_once": True,
            "genuine_contamination": False,
        },
        "artifacts": {
            "included_identities": _artifact_record(included_path),
            "excluded_identities": _artifact_record(excluded_path),
            "genuine_manifest": _artifact_record(genuine_path),
            "negative_manifest": _artifact_record(negative_path),
            "pairing_map": _artifact_record(pairing_path),
        },
    }
    _publish_immutable_json(summary_path, summary)
    validate_pilot_manifest(
        genuine_path,
        data_root,
        dataset=dataset,
        protocol="plain_roll",
        role="genuine",
        selection=selection,
        survivor_summary=summary,
    )
    validate_pilot_manifest(
        negative_path,
        data_root,
        dataset=dataset,
        protocol=NEGATIVE_PROTOCOL,
        role="negative",
        selection=selection,
        survivor_summary=summary,
        genuine_manifest_path=genuine_path,
    )
    return {**summary, "path": str(summary_path), "sha256": file_sha256(summary_path)}


def build_negative_pairs(
    *,
    dataset: str,
    genuine_pairs: Sequence[PairRecord],
    selection_by_identity: Mapping[tuple[str, int], SelectedIdentity],
) -> tuple[list[PairRecord], list[dict[str, Any]]]:
    """Build shift=1 within canonical finger, ordered only by selection_index."""

    grouped: dict[int, list[PairRecord]] = defaultdict(list)
    for pair in genuine_pairs:
        grouped[pair.canonical_finger_position].append(pair)
    negative: list[PairRecord] = []
    pairing: list[dict[str, Any]] = []
    for finger in sorted(grouped):
        group = sorted(
            grouped[finger],
            key=lambda pair: selection_by_identity[
                (pair.subject_id, pair.canonical_finger_position)
            ].selection_index,
        )
        if len(group) < 2:
            raise HarrisZPlusPilotError(
                f"Canonical finger {finger} in {dataset} has fewer than two survivors; "
                "a different-subject circular negative protocol is impossible."
            )
        for index, plain_pair in enumerate(group):
            roll_index = (index + 1) % len(group)
            roll_pair = group[roll_index]
            if plain_pair.subject_id == roll_pair.subject_id:
                raise HarrisZPlusPilotError("Negative circular shift produced the same subject.")
            identity_a = (plain_pair.subject_id, finger)
            identity_b = (roll_pair.subject_id, finger)
            selection_a = selection_by_identity[identity_a].selection_index
            selection_b = selection_by_identity[identity_b].selection_index
            pair_id = (
                f"{dataset}_plain_roll_next_subject_impostor_"
                f"{finger:02d}_{plain_pair.subject_id}_{roll_pair.subject_id}"
            )
            record = PairRecord(
                pair_id=pair_id,
                dataset=dataset,
                protocol=NEGATIVE_PROTOCOL,
                subject_id=plain_pair.subject_id,
                canonical_finger_position=finger,
                ppi=plain_pair.ppi,
                raw_frgp_a=plain_pair.raw_frgp_a,
                raw_frgp_b=roll_pair.raw_frgp_b,
                path_a=plain_pair.path_a,
                path_b=roll_pair.path_b,
            )
            negative.append(record)
            pairing.append(
                {
                    "negative_pair_id": pair_id,
                    "dataset": dataset,
                    "protocol": NEGATIVE_PROTOCOL,
                    "canonical_finger_position": finger,
                    "subject_id_a": plain_pair.subject_id,
                    "subject_id_b": roll_pair.subject_id,
                    "selection_index_a": selection_a,
                    "selection_index_b": selection_b,
                    "plain_group_index": index,
                    "roll_group_index": roll_index,
                    "shift": 1,
                    "ppi": plain_pair.ppi,
                    "raw_frgp_a": plain_pair.raw_frgp_a,
                    "raw_frgp_b": roll_pair.raw_frgp_b,
                    "path_a": str(plain_pair.path_a),
                    "path_b": str(roll_pair.path_b),
                    "source_plain_roll_pair_id_a": plain_pair.pair_id,
                    "source_plain_roll_pair_id_b": roll_pair.pair_id,
                }
            )
    if len({pair.path_a for pair in negative}) != len(negative):
        raise HarrisZPlusPilotError("Negative protocol does not use every PLAIN exactly once.")
    if len({pair.path_b for pair in negative}) != len(negative):
        raise HarrisZPlusPilotError("Negative protocol does not use every ROLL exactly once.")
    if any(pair.path_a == pair.path_b for pair in negative):
        raise HarrisZPlusPilotError("Negative protocol contains identical image paths.")
    return negative, pairing


def validate_pilot_manifest(
    manifest_path: Path,
    data_root: Path,
    *,
    dataset: str,
    protocol: str,
    role: str,
    selection: Sequence[SelectedIdentity],
    survivor_summary: Mapping[str, Any] | None = None,
    genuine_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Dedicated validator supplied to the generic runner for every pilot subset."""

    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    pairs = read_pair_manifest(manifest_path)
    expected_count = EXPECTED_SELECTION_COUNT if role == "self" else int(
        (survivor_summary or {}).get("survivor_count", -1)
    )
    if len(pairs) != expected_count:
        raise HarrisZPlusPilotError(
            f"{dataset}/{role} manifest row count mismatch: expected {expected_count}, got {len(pairs)}."
        )
    for pair in pairs:
        if pair.dataset != dataset or pair.protocol != protocol:
            raise HarrisZPlusPilotError(f"Manifest identity mismatch at {pair.pair_id}.")
        if pair.ppi != EXPECTED_PPI[dataset]:
            raise HarrisZPlusPilotError(f"Manifest PPI mismatch at {pair.pair_id}.")
        for path in (pair.path_a.resolve(), pair.path_b.resolve()):
            if not _is_relative_to(path, data_root) or not path.is_file():
                raise HarrisZPlusPilotError(f"Invalid data path in pilot manifest: {path}")

    selection_index = {item.identity: item.selection_index for item in selection}
    genuine_contamination = 0
    if role == "self":
        expected_identities = [item.identity for item in selection]
        actual = [(pair.subject_id, pair.canonical_finger_position) for pair in pairs]
        if actual != expected_identities:
            raise HarrisZPlusPilotError("Self manifest does not exactly follow selection order.")
        if any(pair.path_a != pair.path_b or pair.raw_frgp_a != pair.raw_frgp_b for pair in pairs):
            raise HarrisZPlusPilotError("Self manifest does not perform two prepares of one image path.")
    elif role == "genuine":
        actual_indexes = [
            selection_index[(pair.subject_id, pair.canonical_finger_position)] for pair in pairs
        ]
        if actual_indexes != sorted(actual_indexes):
            raise HarrisZPlusPilotError("Genuine manifest does not preserve selection_index order.")
        if any(pair.path_a == pair.path_b for pair in pairs):
            raise HarrisZPlusPilotError("Genuine PLAIN/ROLL manifest contains a self image pair.")
    elif role == "negative":
        if genuine_manifest_path is None:
            raise HarrisZPlusPilotError("Negative validation requires its survivor genuine manifest.")
        genuine = read_pair_manifest(genuine_manifest_path)
        expected_pairs, _ = build_negative_pairs(
            dataset=dataset,
            genuine_pairs=genuine,
            selection_by_identity={item.identity: item for item in selection},
        )
        if pairs != expected_pairs:
            raise HarrisZPlusPilotError("Negative manifest is not the exact frozen shift-1 construction.")
        roll_owner = {pair.path_b: pair.subject_id for pair in genuine}
        for pair in pairs:
            if roll_owner[pair.path_b] == pair.subject_id:
                genuine_contamination += 1
        if genuine_contamination:
            raise HarrisZPlusPilotError("Negative manifest contains genuine contamination.")
    else:
        raise HarrisZPlusPilotError(f"Unsupported manifest role: {role}")
    return {
        "validation_mode": f"exact_harriszplus_{role}_custom_validator_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset": dataset,
        "protocol": protocol,
        "role": role,
        "pair_count": len(pairs),
        "ppi": EXPECTED_PPI[dataset],
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "all_images_exist": True,
        "all_images_under_read_only_data_root": True,
        "dataset_tree_scanned": False,
        "genuine_contamination_count": genuine_contamination,
        "shift": 1 if role == "negative" else None,
        "cold_pair_required": True,
        "prepare_a_then_prepare_b_then_compare": True,
        "cross_pair_cache_allowed": False,
    }


def run_pilot(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    progress_callback: Callable[[str, int, int], None] | None = None,
    integrity_before_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume/publish the exact eight bundles; never repeat a valid published bundle."""

    paths = project_paths(project_root)
    integrity_before_validation = _require_integrity_before_validation(
        paths, integrity_before_validation
    )
    data_root = data_root.resolve()
    prepare_base_manifests(
        project_root=project_root,
        data_root=data_root,
        integrity_before_validation=integrity_before_validation,
    )
    preflight = _read_json(paths["preflight"])
    if (
        preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or preflight.get("pilot_500_authorized") is not True
        or preflight.get("ppi_coordinate_handling_all_passed") is not True
        or not isinstance(preflight.get("device_binding"), Mapping)
        or preflight["device_binding"].get("passed") is not True
        or not isinstance(preflight.get("memory"), Mapping)
        or preflight["memory"].get("passed") is not True
        or not isinstance(preflight.get("timing_synchronization"), Mapping)
        or preflight["timing_synchronization"].get(
            "required_timing_fields_all_passed"
        )
        is not True
    ):
        raise HarrisZPlusPilotError(
            "Passing engineering preflight, including PPI coordinate handling, is required "
            "before any pilot run."
        )
    selection = load_and_verify_selection(paths["selection"])
    completed: list[dict[str, Any]] = []

    # Dataset-local ordering is intentional: two self bundles gate that
    # dataset's survivor-derived genuine and negative bundles.
    for dataset in DATASETS:
        for label in SELF_PROTOCOLS:
            completed.append(
                _run_condition(
                    project_root=paths["project_root"],
                    data_root=data_root,
                    dataset=dataset,
                    label=label,
                    protocol=label,
                    role="self",
                    selection=selection,
                    preflight=preflight,
                    integrity_before_validation=integrity_before_validation,
                    progress_callback=progress_callback,
                )
            )
        survivor = derive_survivors_and_pair_manifests(
            project_root=paths["project_root"],
            data_root=data_root,
            dataset=dataset,
        )
        for label, protocol, role in (
            ("plain_roll_genuine", "plain_roll", "genuine"),
            ("plain_roll_negative", NEGATIVE_PROTOCOL, "negative"),
        ):
            completed.append(
                _run_condition(
                    project_root=paths["project_root"],
                    data_root=data_root,
                    dataset=dataset,
                    label=label,
                    protocol=protocol,
                    role=role,
                    selection=selection,
                    preflight=preflight,
                    integrity_before_validation=integrity_before_validation,
                    survivor_summary=survivor,
                    progress_callback=progress_callback,
                )
            )

    if [(item["dataset"], item["label"]) for item in completed] != [
        (dataset, label) for dataset, label, _, _ in RUN_CONDITIONS
    ]:
        raise HarrisZPlusPilotError("Eight-run execution order changed.")
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "namespace": PILOT_NAMESPACE,
        "completed_valid_bundle_count": len(completed),
        "full_repeat_count": 0,
        "bundles": completed,
    }


def _run_condition(
    *,
    project_root: Path,
    data_root: Path,
    dataset: str,
    label: str,
    protocol: str,
    role: str,
    selection: Sequence[SelectedIdentity],
    preflight: Mapping[str, Any],
    integrity_before_validation: Mapping[str, Any] | None = None,
    survivor_summary: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    paths = project_paths(project_root)
    if integrity_before_validation is None:
        integrity_before_validation = validate_integrity_before_gate(
            project_root=paths["project_root"]
        )
    _validate_integrity_gate_token(paths, integrity_before_validation)
    manifest_name = (
        f"{label}.csv" if role == "self" else f"plain_roll_{role}.csv"
    )
    manifest_path = paths["pilot_root"] / "manifests" / dataset / manifest_name
    bundle_path = paths["pilot_root"] / "runs" / dataset / label
    genuine_path = paths["pilot_root"] / "manifests" / dataset / "plain_roll_genuine.csv"

    validator = lambda path, root: validate_pilot_manifest(
        path,
        root,
        dataset=dataset,
        protocol=protocol,
        role=role,
        selection=selection,
        survivor_summary=survivor_summary,
        genuine_manifest_path=genuine_path if role == "negative" else None,
    )
    validator(manifest_path, data_root)
    adapter = _new_adapter()
    try:
        freeze = validate_frozen_configuration(
            config_directory=paths["config"],
            adapter=adapter,
            preflight_report=preflight,
        )
        metadata = adapter.metadata()
        cache_flags = (
            metadata.config.get("representation_cache"),
            metadata.config.get("cross_pair_cache"),
        )
        if cache_flags != (False, False):
            raise HarrisZPlusPilotError(
                "Adapter must explicitly declare both representation_cache=false and "
                "cross_pair_cache=false."
            )
        startup_validation = {
            "engineering_preflight_sha256": file_sha256(paths["preflight"]),
            "integrity_before_sha256": integrity_before_validation["before_sha256"],
            "integrity_before_combined_snapshot_sha256": integrity_before_validation[
                "combined_snapshot_sha256"
            ],
            "config_freeze_manifest_sha256": file_sha256(
                paths["config"] / "freeze_manifest.json"
            ),
            "canonical_config_hash": freeze["canonical_config_hash"],
            "implementation_hash": freeze["implementation_hash"],
            "runtime_identity_hash": freeze["runtime_identity_hash"],
            "decision_rule_hash": freeze["decision_rule_hash"],
            "generic_runner": True,
            "timing_mode": "cold_pair",
            "prepare_operations_per_pair": 2,
            "cross_pair_cache": False,
        }
        result = run_benchmark_manifest(
            manifest_path=manifest_path,
            adapter=adapter,
            expected_dataset=dataset,
            expected_protocol=protocol,
            results_root=paths["pilot_root"] / "unused-generic-root",
            startup_validation=startup_validation,
            data_root=data_root,
            dedicated_validator=validator,
            skip_existing=True,
            bundle_directory=bundle_path,
            progress_callback=(
                (lambda current, total: progress_callback(f"{dataset}/{label}", current, total))
                if progress_callback is not None
                else None
            ),
        )
    finally:
        adapter.close()
    view = load_bundle(project_root=project_root, dataset=dataset, label=label)
    return {
        "dataset": dataset,
        "label": label,
        "protocol": protocol,
        "role": role,
        "bundle_path": str(bundle_path),
        "row_count": len(view.rows),
        "pairs_sha256": file_sha256(bundle_path / RESULT_FILENAME),
        "run_metadata_sha256": file_sha256(bundle_path / METADATA_FILENAME),
        "score_payload_sha256": result["result"]["score_payload_sha256"],
        "config_hash": result["config_hash"],
        "implementation_hash": result["implementation_hash"],
        "valid_published_bundle_reused_or_created": True,
    }


def _expected_bundle_startup_validation(paths: Mapping[str, Path]) -> dict[str, Any]:
    preflight_path = paths["preflight"]
    before_path = paths["pilot_root"] / "integrity/protected_before.json"
    freeze_path = paths["config"] / "freeze_manifest.json"
    before = _read_json(before_path)
    _validate_bound_snapshot(before)
    freeze = _read_json(freeze_path)
    required_freeze_fields = (
        "canonical_config_hash",
        "implementation_hash",
        "runtime_identity_hash",
        "decision_rule_hash",
    )
    if any(not isinstance(freeze.get(field), str) for field in required_freeze_fields):
        raise HarrisZPlusPilotError("Current freeze lacks required bundle identity hashes.")
    return {
        "engineering_preflight_sha256": file_sha256(preflight_path),
        "integrity_before_sha256": file_sha256(before_path),
        "integrity_before_combined_snapshot_sha256": before[
            "combined_snapshot_sha256"
        ],
        "config_freeze_manifest_sha256": file_sha256(freeze_path),
        "canonical_config_hash": freeze["canonical_config_hash"],
        "implementation_hash": freeze["implementation_hash"],
        "runtime_identity_hash": freeze["runtime_identity_hash"],
        "decision_rule_hash": freeze["decision_rule_hash"],
        "generic_runner": True,
        "timing_mode": "cold_pair",
        "prepare_operations_per_pair": 2,
        "cross_pair_cache": False,
    }


def _validate_bundle_startup_validation(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if (
        metadata.get("config_hash") != expected.get("canonical_config_hash")
        or metadata.get("implementation_hash") != expected.get("implementation_hash")
        or metadata.get("timing_mode") != "cold_pair"
        or metadata.get("startup_validation") != expected
    ):
        raise HarrisZPlusPilotError(
            "Published bundle startup validation does not match the current preflight, "
            "integrity-before baseline, frozen config/runtime/decision, and cold-pair policy."
        )


def load_bundle(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    dataset: str,
    label: str,
) -> BundleView:
    condition = next(
        (item for item in RUN_CONDITIONS if item[0] == dataset and item[1] == label),
        None,
    )
    if condition is None:
        raise HarrisZPlusPilotError(f"Unsupported bundle condition: {dataset}/{label}")
    _, _, protocol, role = condition
    paths = project_paths(project_root)
    manifest_name = f"{label}.csv" if role == "self" else f"plain_roll_{role}.csv"
    manifest_path = paths["pilot_root"] / "manifests" / dataset / manifest_name
    bundle_path = paths["pilot_root"] / "runs" / dataset / label
    metadata = _read_json(bundle_path / METADATA_FILENAME)
    raw_spec = metadata.get("run_spec")
    if not isinstance(raw_spec, dict):
        raise HarrisZPlusPilotError(f"Bundle lacks run_spec: {bundle_path}")
    spec_values = dict(raw_spec)
    spec_values["manifest_path"] = Path(spec_values["manifest_path"])
    spec = BenchmarkRunSpec(**spec_values)
    if (
        spec.expected_dataset != dataset
        or spec.expected_protocol != protocol
        or spec.method != METHOD_NAME
        or spec.method_version != METHOD_VERSION
        or spec.benchmark_contract_version != BENCHMARK_CONTRACT_VERSION
        or spec.manifest_path.resolve() != manifest_path.resolve()
    ):
        raise HarrisZPlusPilotError(f"Bundle run identity mismatch: {bundle_path}")
    expected_startup = _expected_bundle_startup_validation(paths)
    _validate_bundle_startup_validation(metadata, expected_startup)
    if (
        spec.config_hash != expected_startup["canonical_config_hash"]
        or spec.implementation_hash != expected_startup["implementation_hash"]
    ):
        raise HarrisZPlusPilotError(
            "Bundle run specification does not match the current frozen identities."
        )
    pairs = read_pair_manifest(manifest_path)
    validate_result_bundle(
        bundle_path,
        manifest_records=pairs,
        run_spec=spec,
        score_direction=metadata["score_direction"],
        score_semantics=metadata["score_semantics"],
    )
    rows = read_result_rows(bundle_path / RESULT_FILENAME)
    return BundleView(
        dataset=dataset,
        label=label,
        protocol=protocol,
        role=role,
        manifest_path=manifest_path,
        bundle_path=bundle_path,
        pairs=pairs,
        rows=rows,
        metadata=metadata,
    )


def accepted_result_row(row: Mapping[str, str]) -> bool:
    """Frozen threshold-4 pilot decision; adapter output remains unthresholded."""

    if row.get("status") != OK:
        return False
    score = float(row.get("raw_score", "nan"))
    if not math.isfinite(score) or score < 0 or not score.is_integer():
        raise HarrisZPlusPilotError(f"HarrisZ+ result score is not a non-negative integer: {score}")
    return int(score) >= THRESHOLD


def finalize_outputs(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    """Build report, after-integrity, technical provenance, and final artifact manifest."""

    from .reporting import (
        build_artifact_manifest,
        build_supervisor_report,
        build_technical_provenance,
    )

    paths = project_paths(project_root)
    preflight = _read_json(paths["preflight"])
    adapter = _new_adapter()
    try:
        freeze_validation = validate_frozen_configuration(
            config_directory=paths["config"],
            adapter=adapter,
            preflight_report=preflight,
        )
    finally:
        adapter.close()
    report = build_supervisor_report(project_root=project_root)
    integrity = capture_integrity_after(project_root=project_root)
    technical = build_technical_provenance(
        project_root=project_root,
        integrity_report=integrity,
    )
    artifacts = build_artifact_manifest(project_root=project_root)
    return {
        "report": report,
        "freeze_validation": freeze_validation,
        "integrity": integrity,
        "technical_provenance": technical,
        "artifact_manifest": artifacts,
    }


def run_complete_workflow(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    device: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Execute all gates in order.  This is the only full-pilot convenience entrypoint."""

    before = capture_integrity_before(project_root=project_root)
    integrity_before_validation = before["validation"]
    manifests = prepare_base_manifests(
        project_root=project_root,
        data_root=data_root,
        integrity_before_validation=integrity_before_validation,
    )
    preflight = execute_engineering_preflight(
        project_root=project_root,
        data_root=data_root,
        device=device,
        integrity_before_validation=integrity_before_validation,
    )
    freeze = freeze_after_preflight(
        project_root=project_root,
        integrity_before_validation=integrity_before_validation,
    )
    runs = run_pilot(
        project_root=project_root,
        data_root=data_root,
        progress_callback=progress_callback,
        integrity_before_validation=integrity_before_validation,
    )
    final = finalize_outputs(project_root=project_root)
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "integrity_before": before,
        "base_manifests": manifests,
        "preflight": preflight,
        "freeze": freeze,
        "runs": runs,
        "final": final,
    }


def _new_adapter() -> MethodAdapter:
    from .adapter import HarrisZPlusGeometricAdapter
    from .config import HarrisZPlusConfig

    return HarrisZPlusGeometricAdapter(HarrisZPlusConfig(device="cuda:0"))


def _result_by_identity(
    bundle: BundleView,
) -> dict[tuple[str, int], tuple[PairRecord, dict[str, str]]]:
    result: dict[tuple[str, int], tuple[PairRecord, dict[str, str]]] = {}
    for pair, row in zip(bundle.pairs, bundle.rows, strict=True):
        identity = pair.subject_id, pair.canonical_finger_position
        if identity in result:
            raise HarrisZPlusPilotError(f"Duplicate result identity: {identity}")
        if row["subject_id"] != identity[0] or int(row["canonical_finger_position"]) != identity[1]:
            raise HarrisZPlusPilotError(f"Result identity mismatch at {pair.pair_id}.")
        result[identity] = (pair, row)
    return result


def _source_artifact_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    files = payload.get("files")
    if not isinstance(files, list):
        raise HarrisZPlusPilotError(f"Artifact manifest lacks file list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise HarrisZPlusPilotError(f"Invalid artifact record in {path}")
        result[record["path"]] = record
    return result


def _pair_manifest_bytes(pairs: Iterable[PairRecord]) -> bytes:
    rows = []
    for pair in pairs:
        rows.append(
            {
                "pair_id": pair.pair_id,
                "dataset": pair.dataset,
                "protocol": pair.protocol,
                "subject_id": pair.subject_id,
                "canonical_finger_position": pair.canonical_finger_position,
                "ppi": pair.ppi,
                "raw_frgp_a": pair.raw_frgp_a,
                "raw_frgp_b": pair.raw_frgp_b,
                "path_a": str(pair.path_a),
                "path_b": str(pair.path_b),
            }
        )
    return _csv_bytes(rows, MANIFEST_COLUMNS)


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue().encode("utf-8")


def _publish_immutable_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_immutable_bytes(path, data)


def _publish_immutable_bytes(path: Path, data: bytes) -> None:
    """Create a file exactly once; byte-identical reuse is validation, never overwrite."""

    path = path.resolve()
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise HarrisZPlusPilotError(f"Immutable output already exists with different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Windows defaults low-level descriptors to text mode unless O_BINARY is
    # requested explicitly.  Without it, os.write translates LF to CRLF and a
    # byte-exact manifest copy is silently changed during publication.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags, 0o644)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarrisZPlusPilotError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HarrisZPlusPilotError(f"JSON artifact must contain an object: {path}")
    return value


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HarrisZ+ RootSIFT geometric engineering preflight and joint-500 pilot."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("integrity-before")
    subparsers.add_parser("prepare")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--device")
    subparsers.add_parser("freeze")
    subparsers.add_parser("run")
    subparsers.add_parser("report")
    subparsers.add_parser("integrity-after")
    subparsers.add_parser("finalize")
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "integrity-before":
            result = capture_integrity_before(project_root=args.project_root)
        elif args.command == "prepare":
            result = prepare_base_manifests(
                project_root=args.project_root,
                data_root=args.data_root,
            )
        elif args.command == "preflight":
            result = execute_engineering_preflight(
                project_root=args.project_root,
                data_root=args.data_root,
                device=args.device,
            )
        elif args.command == "freeze":
            result = freeze_after_preflight(project_root=args.project_root)
        elif args.command == "run":
            result = run_pilot(
                project_root=args.project_root,
                data_root=args.data_root,
                progress_callback=lambda condition, current, total: print(
                    f"{condition}: {current}/{total}", file=sys.stderr, flush=True
                ),
            )
        elif args.command == "report":
            from .reporting import build_supervisor_report

            result = build_supervisor_report(project_root=args.project_root)
        elif args.command == "integrity-after":
            result = capture_integrity_after(project_root=args.project_root)
        elif args.command == "finalize":
            result = finalize_outputs(project_root=args.project_root)
        elif args.command == "all":
            result = run_complete_workflow(
                project_root=args.project_root,
                data_root=args.data_root,
                device=args.device,
                progress_callback=lambda condition, current, total: print(
                    f"{condition}: {current}/{total}", file=sys.stderr, flush=True
                ),
            )
        else:
            raise HarrisZPlusPilotError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
