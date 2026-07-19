"""Neutral joint-500 screening protocol, execution, and reporting support."""

from __future__ import annotations

import base64
import csv
from dataclasses import asdict, is_dataclass
from functools import partial
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from fingerprint_data_discovery.canonical_fingers import canonical_finger_position
from fingerprint_data_discovery.nist_sd300 import (
    DATASETS as DATASET_SPECS,
    DEFAULT_DATA_ROOT,
    validate_image_path,
)

from .hashing import file_sha256, stable_hash
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest
from .preflight import validator_for


PROTOCOL_NAME = "detector_only_joint_500_v1"
PROTOCOL_VERSION = "detector-only-joint-500-v1"
SEED = "detector-only-joint-500-v1"
DATASETS = ("sd300b", "sd300c")
BASE_PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
PAIR_KINDS = ("plain_self", "roll_self", "plain_roll_genuine", "plain_roll_impostor")
FINGER_POSITIONS = tuple(range(1, 11))
PER_POSITION = 50
COHORT_SIZE = 500
PROTOCOL_DIRECTORY = Path("protocols") / PROTOCOL_NAME
SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION = "sourceafis-final-minutiae-preflight-v2"
SOURCEAFIS_ENCODED_RAW_DIFFERENCE_REASON = "decoder_pixel_semantics_differ"
SOURCEAFIS_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "protocol_name",
        "protocol_sha256",
        "detector_version",
        "sidecar_jar_sha256",
        "image_count",
        "identities_per_dataset",
        "impressions",
        "items",
        "biometric_bytes_persisted",
    }
)
SOURCEAFIS_PREFLIGHT_ITEM_FIELDS = frozenset(
    {
        "logical_image_id",
        "dataset",
        "subject_id",
        "canonical_finger_position",
        "impression",
        "ppi",
        "encoded_template_sha256",
        "canonical_raw_template_sha256",
        "final_minutiae_template_sha256",
        "raw_template_final_minutiae_equal",
        "encoded_raw_template_equal",
        "encoded_raw_equivalence_required",
        "encoded_raw_pixel_ingestion_equivalence",
        "encoded_raw_difference_reason",
        "repeated_encoded_template_equal",
        "repeated_raw_template_equal",
        "repeated_final_minutiae_equal",
        "minutia_count",
    }
)
SELECTED_COLUMNS = [
    "protocol_version",
    "subject_id",
    "canonical_finger_position",
    "selection_rank",
    "selection_hash",
]
PAIRING_COLUMNS = [
    "logical_pair_id",
    "canonical_finger_position",
    "probe_subject_id",
    "candidate_subject_id",
    "selection_rank_probe",
    "selection_rank_candidate",
]


class Joint500ProtocolError(ValueError):
    """Raised when joint-500 selection or artifacts violate the frozen protocol."""


def build_protocol_artifacts(
    *,
    repository_root: Path = Path("."),
    check: bool = False,
) -> dict[str, Any]:
    """Build or byte-check the deterministic protocol artifact set."""

    root = repository_root.resolve()
    target = root / PROTOCOL_DIRECTORY
    artifacts, report = protocol_artifact_bytes(root)
    if check:
        mismatches = [
            relative
            for relative, expected in artifacts.items()
            if not (target / relative).is_file() or (target / relative).read_bytes() != expected
        ]
        extras = _extra_artifacts(target, set(artifacts))
        if mismatches or extras:
            raise Joint500ProtocolError(
                f"Protocol --check failed; mismatches={mismatches}, extras={extras}."
            )
        return {**report, "mode": "check", "byte_exact": True}

    target.mkdir(parents=True, exist_ok=True)
    for relative, content in artifacts.items():
        output = target / relative
        if output.exists():
            if output.read_bytes() != content:
                raise Joint500ProtocolError(
                    f"Refusing to overwrite different protocol artifact: {output}"
                )
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
    extras = _extra_artifacts(target, set(artifacts))
    if extras:
        raise Joint500ProtocolError(f"Unexpected files exist in protocol directory: {extras}.")
    return {**report, "mode": "build", "byte_exact": True}


def protocol_artifact_bytes(repository_root: Path) -> tuple[dict[Path, bytes], dict[str, Any]]:
    """Construct every protocol artifact in memory from the six base manifests."""

    base, base_hashes = _read_base_manifests(repository_root)
    eligible = _eligible_identities(base)
    selected = _select_identities(eligible)
    pairing = _impostor_pairing(selected)

    artifacts: dict[Path, bytes] = {}
    artifacts[Path("selected_identities.csv")] = _csv_bytes(selected, SELECTED_COLUMNS)
    artifacts[Path("impostor_pairing.csv")] = _csv_bytes(pairing, PAIRING_COLUMNS)
    for dataset in DATASETS:
        for pair_kind in PAIR_KINDS:
            rows = _pair_manifest_rows(
                dataset=dataset,
                pair_kind=pair_kind,
                selected=selected,
                pairing=pairing,
                base=base,
            )
            artifacts[Path(dataset) / f"{pair_kind}.csv"] = _csv_bytes(rows, MANIFEST_COLUMNS)

    artifact_hashes = {
        relative.as_posix(): _bytes_sha256(content)
        for relative, content in sorted(artifacts.items(), key=lambda item: item[0].as_posix())
    }
    per_position_counts = {
        str(position): sum(
            1 for row in selected if int(row["canonical_finger_position"]) == position
        )
        for position in FINGER_POSITIONS
    }
    metadata_core = {
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "selection_algorithm": "sha256_rank_by_position_then_first_unselected_subject_v1",
        "selection_hash_input": (
            "protocol_version + newline + seed + newline + canonical_finger_position + newline + subject_id"
        ),
        "impostor_pairing_algorithm": "within_position_selection_rank_circular_shift_1_v1",
        "cohort_size": COHORT_SIZE,
        "per_position_counts": per_position_counts,
        "unique_subject_count": len({row["subject_id"] for row in selected}),
        "base_manifest_sha256": base_hashes,
        "artifact_sha256": artifact_hashes,
        "artifact_hash_scope": "all generated artifacts except protocol_metadata.json to avoid recursive self-hash",
        "timestamp_policy": "omitted_from_artifacts_and_protocol_hash_for_byte_exact_rebuild",
        "method_references": [],
        "score_references": [],
        "result_references": [],
        "self_failure_policy": "record_failure_without_cohort_filtering_or_replacement",
    }
    metadata = {**metadata_core, "protocol_sha256": stable_hash(metadata_core)}
    artifacts[Path("protocol_metadata.json")] = _json_bytes(metadata)
    report = {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": metadata["protocol_sha256"],
        "cohort_size": len(selected),
        "per_position_counts": per_position_counts,
        "unique_subject_count": len({row["subject_id"] for row in selected}),
        "base_manifest_sha256": base_hashes,
        "artifact_sha256": {
            **artifact_hashes,
            "protocol_metadata.json": _bytes_sha256(artifacts[Path("protocol_metadata.json")]),
        },
    }
    return artifacts, report


def validate_protocol_artifacts(
    *,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    """Rebuild in memory and prove every cohort/pairing/artifact invariant."""

    root = repository_root.resolve()
    target = root / PROTOCOL_DIRECTORY
    expected, build_report = protocol_artifact_bytes(root)
    if not target.is_dir():
        raise Joint500ProtocolError(f"Protocol directory does not exist: {target}")
    extras = _extra_artifacts(target, set(expected))
    if extras:
        raise Joint500ProtocolError(f"Unexpected protocol artifacts: {extras}.")
    for relative, content in expected.items():
        path = target / relative
        if not path.is_file():
            raise Joint500ProtocolError(f"Missing protocol artifact: {relative.as_posix()}")
        if path.read_bytes() != content:
            raise Joint500ProtocolError(f"Protocol artifact mismatch: {relative.as_posix()}")

    selected = _read_csv_dicts(target / "selected_identities.csv", SELECTED_COLUMNS)
    pairing = _read_csv_dicts(target / "impostor_pairing.csv", PAIRING_COLUMNS)
    if len(selected) != COHORT_SIZE:
        raise Joint500ProtocolError("Selected cohort must contain exactly 500 identities.")
    subjects = [row["subject_id"] for row in selected]
    if len(set(subjects)) != COHORT_SIZE:
        raise Joint500ProtocolError("Selected cohort must contain 500 unique subjects.")
    for position in FINGER_POSITIONS:
        position_rows = [
            row for row in selected if int(row["canonical_finger_position"]) == position
        ]
        if len(position_rows) != PER_POSITION:
            raise Joint500ProtocolError(f"Finger position {position} does not contain exactly 50 identities.")
        if [int(row["selection_rank"]) for row in position_rows] != list(range(1, PER_POSITION + 1)):
            raise Joint500ProtocolError(f"Finger position {position} has invalid selection ranks.")
    _validate_pairing_rows(pairing, selected)
    _validate_pair_manifests(target, selected, pairing)

    metadata = json.loads((target / "protocol_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("method_references") or metadata.get("score_references") or metadata.get("result_references"):
        raise Joint500ProtocolError("Protocol metadata must not reference methods, scores, or results.")
    if metadata.get("protocol_sha256") != build_report["protocol_sha256"]:
        raise Joint500ProtocolError("Protocol hash mismatch.")
    return {
        **build_report,
        "status": "ok",
        "validated_manifest_count": len(DATASETS) * len(PAIR_KINDS),
        "same_identities_b_c": True,
        "eligible_against_all_six_base_manifests": True,
        "impostor_bijection": True,
        "same_impostor_logic_b_c": True,
        "self_filtering": False,
        "method_score_result_dependency": False,
        "byte_exact_rebuild": True,
    }


def validate_joint_dataset_preflight(
    *,
    repository_root: Path = Path("."),
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Validate all base and derived manifests against one explicit dataset root."""

    root = repository_root.resolve()
    resolved_data_root = Path(data_root).resolve()
    if not resolved_data_root.is_dir():
        raise Joint500ProtocolError(f"Joint-500 data root does not exist: {resolved_data_root}")
    protocol_report = validate_protocol_artifacts(repository_root=root)
    metadata_path = root / PROTOCOL_DIRECTORY / "protocol_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_base_hashes = metadata.get("base_manifest_sha256")
    if not isinstance(expected_base_hashes, dict):
        raise Joint500ProtocolError("Protocol metadata base_manifest_sha256 must be an object.")

    validator_reports: dict[str, Any] = {}
    for dataset in DATASETS:
        for protocol in BASE_PROTOCOLS:
            relative = Path("protocols") / dataset / f"{protocol}.csv"
            path = root / relative
            try:
                result = validator_for(dataset, protocol)(path, resolved_data_root)
            except Exception as exc:
                raise Joint500ProtocolError(
                    f"Base manifest validation failed for {dataset}/{protocol}: {exc}"
                ) from exc
            actual_sha = file_sha256(path)
            if expected_base_hashes.get(relative.as_posix()) != actual_sha:
                raise Joint500ProtocolError(
                    f"Base manifest SHA-256 mismatch for {relative.as_posix()}."
                )
            validator_reports[relative.as_posix()] = _validator_report(result)

    base, base_hashes = _read_base_manifests(root)
    if base_hashes != expected_base_hashes:
        raise Joint500ProtocolError("Validated base manifest hashes do not match protocol metadata.")
    validated_path_count = _validate_derived_dataset_paths(
        repository_root=root,
        data_root=resolved_data_root,
        base=base,
    )
    derived_manifest_hashes = {
        protocol_manifest_path(dataset, pair_kind).as_posix(): file_sha256(
            root / protocol_manifest_path(dataset, pair_kind)
        )
        for dataset in DATASETS
        for pair_kind in PAIR_KINDS
    }
    return {
        **protocol_report,
        "dataset_preflight_status": "ok",
        "data_root": str(resolved_data_root),
        "base_manifest_validator_results": validator_reports,
        "derived_manifest_sha256": derived_manifest_hashes,
        "derived_dataset_path_count": validated_path_count,
    }


def validate_joint_manifest(
    manifest_path: Path,
    data_root: Path,
    *,
    dataset_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Dedicated runner validator for one of the eight derived manifests."""

    resolved = manifest_path.resolve()
    repository_root = _repository_root_from_manifest(resolved)
    report = (
        validate_joint_dataset_preflight(
            repository_root=repository_root,
            data_root=data_root,
        )
        if dataset_preflight is None
        else _validate_dataset_preflight_binding(dataset_preflight, data_root)
    )
    protocol_root = (repository_root / PROTOCOL_DIRECTORY).resolve()
    try:
        relative = resolved.relative_to(protocol_root)
    except ValueError as exc:
        raise Joint500ProtocolError("Manifest is outside the joint-500 protocol directory.") from exc
    expected = {Path(dataset) / f"{kind}.csv" for dataset in DATASETS for kind in PAIR_KINDS}
    if relative not in expected:
        raise Joint500ProtocolError(f"Not a joint-500 pair manifest: {relative.as_posix()}")
    relative_to_repository = (PROTOCOL_DIRECTORY / relative).as_posix()
    expected_sha = (report.get("derived_manifest_sha256") or {}).get(relative_to_repository)
    if expected_sha is None or file_sha256(resolved) != expected_sha:
        raise Joint500ProtocolError(
            f"Derived manifest changed after dataset preflight: {relative_to_repository}"
        )
    return {**report, "validated_manifest": relative.as_posix()}


def protocol_manifest_path(dataset: str, pair_kind: str) -> Path:
    if dataset not in DATASETS or pair_kind not in PAIR_KINDS:
        raise Joint500ProtocolError(f"Unsupported joint-500 manifest identity: {dataset}/{pair_kind}.")
    return PROTOCOL_DIRECTORY / dataset / f"{pair_kind}.csv"


def row_protocol(pair_kind: str) -> str:
    if pair_kind not in PAIR_KINDS:
        raise Joint500ProtocolError(f"Unsupported pair kind: {pair_kind}.")
    return f"{PROTOCOL_NAME}_{pair_kind}"


def preflight_artifact_path(results_root: Path) -> Path:
    return results_root / PROTOCOL_NAME / "preflight_sourceafis.json"


def run_sourceafis_preflight(
    *,
    client: Any,
    jar_sha256: str,
    results_root: Path = Path("results"),
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    """Prove canonical raw-template/final-minutiae parity on 20 cohort images."""

    import cv2

    from .detectors.sourceafis_final_minutiae import DETECTOR_VERSION

    validation = validate_protocol_artifacts(repository_root=repository_root)
    selected_path = repository_root.resolve() / PROTOCOL_DIRECTORY / "selected_identities.csv"
    selected = _read_csv_dicts(selected_path, SELECTED_COLUMNS)[:5]
    if len(selected) != 5:
        raise Joint500ProtocolError("SourceAFIS preflight requires five selected identities.")
    items: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manifests = {
            kind: {
                (record.subject_id, record.canonical_finger_position): record
                for record in read_pair_manifest(
                    repository_root.resolve() / protocol_manifest_path(dataset, kind)
                )
            }
            for kind in ("plain_self", "roll_self")
        }
        for identity in selected:
            key = (identity["subject_id"], int(identity["canonical_finger_position"]))
            for source_kind, side in (("plain_self", "plain"), ("roll_self", "roll")):
                record = manifests[source_kind][key]
                image_path = record.path_a
                try:
                    encoded = image_path.read_bytes()
                except OSError as exc:
                    raise Joint500ProtocolError(f"Cannot read preflight image {image_path}: {exc}") from exc
                encoded_first = client.extract_template(encoded, float(record.ppi))
                encoded_second = client.extract_template(encoded, float(record.ppi))
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.ndim != 2 or image.dtype.name != "uint8":
                    raise Joint500ProtocolError(f"OpenCV cannot read uint8 grayscale preflight image: {image_path}")
                pixels = image.tobytes(order="C")
                raw_first = client.extract_template_raw(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                raw_second = client.extract_template_raw(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                final_first = client.extract_final_minutiae(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                final_second = client.extract_final_minutiae(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                encoded_first_sha256 = hashlib.sha256(
                    base64.b64decode(encoded_first.template_base64.encode("ascii"), validate=True)
                ).hexdigest()
                encoded_second_sha256 = hashlib.sha256(
                    base64.b64decode(encoded_second.template_base64.encode("ascii"), validate=True)
                ).hexdigest()
                repeated_encoded_template_equal = encoded_first_sha256 == encoded_second_sha256
                repeated_raw_template_equal = (
                    raw_first.template_sha256 == raw_second.template_sha256
                    and raw_first.template_base64 == raw_second.template_base64
                )
                repeated_final_minutiae_equal = (
                    _deterministic_minutia_payload(final_first)
                    == _deterministic_minutia_payload(final_second)
                )
                raw_template_final_minutiae_equal = (
                    raw_first.template_sha256 == final_first.template_sha256
                    and raw_second.template_sha256 == final_second.template_sha256
                )
                if not raw_template_final_minutiae_equal:
                    raise Joint500ProtocolError(
                        f"SourceAFIS raw-template/final-minutiae parity mismatch for {dataset}/{key}/{side}."
                    )
                if not repeated_encoded_template_equal:
                    raise Joint500ProtocolError(
                        f"Repeated SourceAFIS encoded extraction mismatch for {dataset}/{key}/{side}."
                    )
                if not repeated_raw_template_equal:
                    raise Joint500ProtocolError(
                        f"Repeated SourceAFIS raw-template extraction mismatch for {dataset}/{key}/{side}."
                    )
                if not repeated_final_minutiae_equal:
                    raise Joint500ProtocolError(
                        f"Repeated SourceAFIS final-minutiae extraction mismatch for {dataset}/{key}/{side}."
                    )
                items.append(
                    {
                        "logical_image_id": f"{dataset}:{key[0]}:{key[1]:02d}:{side}",
                        "dataset": dataset,
                        "subject_id": key[0],
                        "canonical_finger_position": key[1],
                        "impression": side,
                        "ppi": record.ppi,
                        "encoded_template_sha256": encoded_first_sha256,
                        "canonical_raw_template_sha256": raw_first.template_sha256,
                        "final_minutiae_template_sha256": final_first.template_sha256,
                        "raw_template_final_minutiae_equal": raw_template_final_minutiae_equal,
                        "encoded_raw_template_equal": encoded_first_sha256 == raw_first.template_sha256,
                        "encoded_raw_equivalence_required": False,
                        "encoded_raw_pixel_ingestion_equivalence": False,
                        "encoded_raw_difference_reason": SOURCEAFIS_ENCODED_RAW_DIFFERENCE_REASON,
                        "repeated_encoded_template_equal": repeated_encoded_template_equal,
                        "repeated_raw_template_equal": repeated_raw_template_equal,
                        "repeated_final_minutiae_equal": repeated_final_minutiae_equal,
                        "minutia_count": final_first.minutia_count,
                    }
                )
    if len(items) != 20:
        raise Joint500ProtocolError(f"SourceAFIS preflight must contain 20 images, got {len(items)}.")
    artifact = {
        "schema_version": SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION,
        "status": "ok",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": validation["protocol_sha256"],
        "detector_version": DETECTOR_VERSION,
        "sidecar_jar_sha256": _validate_sha256(jar_sha256, "sidecar JAR"),
        "image_count": len(items),
        "identities_per_dataset": 5,
        "impressions": ["plain", "roll"],
        "items": items,
        "biometric_bytes_persisted": False,
    }
    output = preflight_artifact_path(results_root)
    content = _json_bytes(artifact)
    if output.exists() and output.read_bytes() != content:
        raise Joint500ProtocolError(
            f"Refusing to overwrite a different SourceAFIS preflight artifact: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(content)
    return validate_sourceafis_preflight(
        results_root=results_root,
        protocol_sha256=validation["protocol_sha256"],
        jar_sha256=jar_sha256,
    )


def validate_sourceafis_preflight(
    *,
    results_root: Path,
    protocol_sha256: str,
    jar_sha256: str,
) -> dict[str, Any]:
    from .detectors.sourceafis_final_minutiae import DETECTOR_VERSION

    path = preflight_artifact_path(results_root).resolve()
    if not path.is_file():
        raise Joint500ProtocolError(
            f"Matching SourceAFIS preflight is required before joint-500 run: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Joint500ProtocolError(f"Cannot read SourceAFIS preflight artifact: {exc}") from exc
    _validate_sourceafis_preflight_payload(
        payload,
        protocol_sha256=protocol_sha256,
        jar_sha256=jar_sha256,
        detector_version=DETECTOR_VERSION,
    )
    return {
        **payload,
        "preflight_path": str(path),
        "preflight_sha256": file_sha256(path),
    }


def _validate_sourceafis_preflight_payload(
    payload: Any,
    *,
    protocol_sha256: str,
    jar_sha256: str,
    detector_version: str,
) -> None:
    if not isinstance(payload, dict) or set(payload) != SOURCEAFIS_PREFLIGHT_FIELDS:
        actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise Joint500ProtocolError(
            f"SourceAFIS preflight top-level schema mismatch: {actual}."
        )
    expected_scalars = {
        "schema_version": SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION,
        "status": "ok",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": _validate_sha256(protocol_sha256, "protocol"),
        "detector_version": detector_version,
        "sidecar_jar_sha256": _validate_sha256(jar_sha256, "sidecar JAR"),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight field {key!r} mismatch: "
                f"expected {expected!r}, got {payload.get(key)!r}."
            )
    if type(payload.get("image_count")) is not int or payload["image_count"] != 20:
        raise Joint500ProtocolError("SourceAFIS preflight image_count must be integer 20.")
    if type(payload.get("identities_per_dataset")) is not int or payload["identities_per_dataset"] != 5:
        raise Joint500ProtocolError(
            "SourceAFIS preflight identities_per_dataset must be integer 5."
        )
    if payload.get("impressions") != ["plain", "roll"]:
        raise Joint500ProtocolError(
            "SourceAFIS preflight impressions must be exactly ['plain', 'roll']."
        )
    if payload.get("biometric_bytes_persisted") is not False:
        raise Joint500ProtocolError("SourceAFIS preflight must not persist biometric bytes.")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 20:
        raise Joint500ProtocolError("SourceAFIS preflight items must contain exactly 20 entries.")

    logical_ids: set[str] = set()
    identities: dict[str, dict[tuple[str, int], set[str]]] = {
        dataset: {} for dataset in DATASETS
    }
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != SOURCEAFIS_PREFLIGHT_ITEM_FIELDS:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} schema mismatch."
            )
        dataset = item.get("dataset")
        if dataset not in DATASETS:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} has invalid dataset {dataset!r}."
            )
        subject_id = item.get("subject_id")
        if not isinstance(subject_id, str) or len(subject_id) != 8 or not subject_id.isdigit():
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} has invalid subject_id."
            )
        position = item.get("canonical_finger_position")
        if type(position) is not int or position not in FINGER_POSITIONS:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} has invalid canonical finger position."
            )
        impression = item.get("impression")
        if impression not in {"plain", "roll"}:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} has invalid impression."
            )
        expected_logical_id = f"{dataset}:{subject_id}:{position:02d}:{impression}"
        if item.get("logical_image_id") != expected_logical_id:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} logical_image_id mismatch."
            )
        if expected_logical_id in logical_ids:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight logical_image_id is duplicated: {expected_logical_id}."
            )
        logical_ids.add(expected_logical_id)
        ppi = item.get("ppi")
        if type(ppi) is not int or ppi != DATASET_SPECS[dataset].ppi:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} PPI does not match {dataset}."
            )
        for field_name in (
            "encoded_template_sha256",
            "canonical_raw_template_sha256",
            "final_minutiae_template_sha256",
        ):
            _validate_sha256(item.get(field_name), f"preflight item {index} {field_name}")
        minutia_count = item.get("minutia_count")
        if type(minutia_count) is not int or minutia_count < 0:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} minutia_count must be a nonnegative integer."
            )
        required_true = (
            "raw_template_final_minutiae_equal",
            "repeated_encoded_template_equal",
            "repeated_raw_template_equal",
            "repeated_final_minutiae_equal",
        )
        if any(item.get(field_name) is not True for field_name in required_true):
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} did not pass parity and repeatability."
            )
        if type(item.get("encoded_raw_template_equal")) is not bool:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} encoded_raw_template_equal must be boolean."
            )
        expected_encoded_raw_equal = (
            item["encoded_template_sha256"] == item["canonical_raw_template_sha256"]
        )
        if item["encoded_raw_template_equal"] is not expected_encoded_raw_equal:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} encoded/raw equality does not match its hashes."
            )
        if item.get("encoded_raw_equivalence_required") is not False:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} must classify encoded/raw template equality as diagnostic."
            )
        if item.get("encoded_raw_pixel_ingestion_equivalence") is not False:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} must record non-equivalent decoder pixel ingestion."
            )
        if item.get("encoded_raw_difference_reason") != SOURCEAFIS_ENCODED_RAW_DIFFERENCE_REASON:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} lacks the required decoder root-cause classification."
            )
        if item["canonical_raw_template_sha256"] != item["final_minutiae_template_sha256"]:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight item {index} raw template hashes do not match."
            )
        identity = (subject_id, position)
        identities[dataset].setdefault(identity, set()).add(impression)

    expected_impressions = {"plain", "roll"}
    for dataset, dataset_identities in identities.items():
        if len(dataset_identities) != 5:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight must contain exactly five identities for {dataset}."
            )
        if any(impressions != expected_impressions for impressions in dataset_identities.values()):
            raise Joint500ProtocolError(
                f"SourceAFIS preflight must contain Plain and Roll for every {dataset} identity."
            )
    if set(identities["sd300b"]) != set(identities["sd300c"]):
        raise Joint500ProtocolError(
            "SourceAFIS preflight must use the same five logical identities in SD300b and SD300c."
        )


def run_joint500(
    *,
    method: str,
    dataset: str | None = None,
    pair_kind: str | None = None,
    results_root: Path = Path("results"),
    data_root: Path = DEFAULT_DATA_ROOT,
    repository_root: Path = Path("."),
    sidecar_jar: Path | None = None,
    service_url: str = "http://127.0.0.1:8765",
    timeout_seconds: float = 120.0,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run selected bundles; SourceAFIS receives one fresh JVM per bundle."""

    from .detectors.opencv_gftt_harris import (
        METHOD_NAME as HARRIS_METHOD,
        OpenCVGFTTHarrisRootSIFTGeometricAdapter,
    )
    from .detectors.sourceafis_final_minutiae import (
        METHOD_NAME as SOURCEAFIS_METHOD,
        SourceAfisFinalMinutiaeRootSIFTGeometricAdapter,
    )
    from .runner import run_benchmark_manifest
    from .sourceafis_client import SourceAfisSidecarClient, validate_health
    from .sourceafis_sidecar import ManagedSourceAfisSidecar

    if method not in {HARRIS_METHOD, SOURCEAFIS_METHOD}:
        raise Joint500ProtocolError(f"Unsupported joint-500 method: {method}.")
    validation = validate_joint_dataset_preflight(
        repository_root=repository_root,
        data_root=data_root,
    )
    dedicated_validator = partial(validate_joint_manifest, dataset_preflight=validation)
    datasets = (dataset,) if dataset is not None else DATASETS
    kinds = (pair_kind,) if pair_kind is not None else PAIR_KINDS
    if any(item not in DATASETS for item in datasets) or any(item not in PAIR_KINDS for item in kinds):
        raise Joint500ProtocolError("Invalid dataset or pair-kind filter.")
    if method == SOURCEAFIS_METHOD:
        if sidecar_jar is None or not sidecar_jar.is_file():
            raise Joint500ProtocolError("SourceAFIS joint-500 run requires an existing sidecar JAR.")
        sourceafis_preflight = validate_sourceafis_preflight(
            results_root=results_root,
            protocol_sha256=validation["protocol_sha256"],
            jar_sha256=file_sha256(sidecar_jar),
        )
    else:
        sourceafis_preflight = None
    reports: list[dict[str, Any]] = []
    for active_dataset in datasets:
        for active_kind in kinds:
            manifest = repository_root.resolve() / protocol_manifest_path(active_dataset, active_kind)
            bundle = _bundle_directory(results_root, active_dataset, active_kind, method)
            if method == HARRIS_METHOD:
                adapter = OpenCVGFTTHarrisRootSIFTGeometricAdapter()
                reports.append(
                    run_benchmark_manifest(
                        manifest_path=manifest,
                        adapter=adapter,
                        expected_dataset=active_dataset,
                        expected_protocol=row_protocol(active_kind),
                        results_root=results_root,
                        startup_validation={},
                        data_root=data_root,
                        dedicated_validator=dedicated_validator,
                        skip_existing=skip_existing,
                        bundle_directory=bundle,
                    )
                )
                continue
            with ManagedSourceAfisSidecar(
                sidecar_jar,
                service_url,
                timeout_seconds=timeout_seconds,
            ) as sidecar:
                client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
                try:
                    health = client.health()
                    validate_health(health)
                    adapter = SourceAfisFinalMinutiaeRootSIFTGeometricAdapter(client, health=health)
                    reports.append(
                        run_benchmark_manifest(
                            manifest_path=manifest,
                            adapter=adapter,
                            expected_dataset=active_dataset,
                            expected_protocol=row_protocol(active_kind),
                            results_root=results_root,
                            startup_validation=_sidecar_startup_dict(
                                sidecar.startup,
                                health.raw,
                                sourceafis_preflight=sourceafis_preflight,
                            ),
                            data_root=data_root,
                            dedicated_validator=dedicated_validator,
                            skip_existing=skip_existing,
                            bundle_directory=bundle,
                        )
                    )
                finally:
                    client.close()
    return reports


def report_joint500(
    *,
    results_root: Path = Path("results"),
    output_directory: Path | None = None,
    repository_root: Path = Path("."),
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Create a screening-only report from validated benchmark-v2 bundles."""

    protocol_results = (results_root / PROTOCOL_NAME).resolve()
    repository_root = repository_root.resolve()
    protocol_validation = validate_protocol_artifacts(repository_root=repository_root)
    bundles = _validated_joint500_bundles(
        protocol_results=protocol_results,
        repository_root=repository_root,
        protocol_sha256=protocol_validation["protocol_sha256"],
    )
    if not bundles:
        raise Joint500ProtocolError(f"No joint-500 result bundles found under {protocol_results}.")
    expected_matrix = {
        (method, dataset, pair_kind)
        for method in _joint_method_versions()
        for dataset in DATASETS
        for pair_kind in PAIR_KINDS
    }
    actual_matrix = set(bundles)
    missing = sorted(expected_matrix - actual_matrix)
    if missing and not allow_partial:
        raise Joint500ProtocolError(
            f"Complete joint-500 report requires 16 validated bundles; missing={missing}."
        )
    complete_protocol_matrix = actual_matrix == expected_matrix

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    method_identities: dict[str, tuple[str, str, str, str, str]] = {}
    for key, bundle in sorted(bundles.items()):
        method, dataset, pair_kind = key
        metadata = bundle["metadata"]
        identity = (
            metadata["method_version"],
            metadata["config_hash"],
            metadata["implementation_hash"],
            metadata["score_direction"],
            metadata["score_semantics"],
        )
        previous = method_identities.setdefault(method, identity)
        if identity != previous:
            raise Joint500ProtocolError(
                f"Mixed method version/config/implementation detected for {method}."
            )
        bundle_rows = bundle["rows"]
        grouped[(method, dataset, pair_kind)] = bundle_rows
    summaries = [
        _screening_group_summary(method, dataset, pair_kind, group)
        for (method, dataset, pair_kind), group in sorted(grouped.items())
    ]
    screening = []
    methods = sorted({key[0] for key in grouped})
    for method in methods:
        for dataset in DATASETS:
            genuine = grouped.get((method, dataset, "plain_roll_genuine"))
            impostor = grouped.get((method, dataset, "plain_roll_impostor"))
            if genuine is None or impostor is None:
                continue
            screening.append(_screening_roc(method, dataset, genuine, impostor))
    paired = _paired_bc_analysis(grouped)
    report = {
        "schema_version": "detector-only-joint-500-screening-report-v2",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": protocol_validation["protocol_sha256"],
        "screening_only": True,
        "complete_protocol_matrix": complete_protocol_matrix,
        "validated_bundle_count": len(bundles),
        "required_bundle_count": 16,
        "allow_partial": bool(allow_partial),
        "impostor_count": 500,
        "far_resolution": 0.002,
        "threshold_calibration": "none",
        "reported_far_operating_points": [0.01],
        "summaries": summaries,
        "genuine_impostor_screening": screening,
        "paired_sd300b_sd300c": paired,
        "interpretation_limits": [
            "Raw score magnitude must not be compared between methods.",
            "Detector point count is part of detector output and may differ between SourceAFIS minutiae and Harris.",
            "Self pairs are engineering diagnostics and never filter the cohort.",
            "Five hundred impostors do not support a FAR 0.1% conclusion.",
            "This is a development/screening cohort, not held-out evaluation.",
            "A negative result tests minutia locations with common RootSIFT downstream, not minutiae-native matching.",
            "SD300b and SD300c are paired views and are not pooled as independent samples.",
            "ROC, AUC, EER, and TAR-at-FAR metrics are conditional on successful comparisons.",
            "Operational metrics use a fail-closed policy over all 500 requested rows.",
        ],
    }
    output = output_directory or (protocol_results / "report")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(_json_bytes(report))
    (output / "summary.csv").write_bytes(_summary_csv_bytes(summaries))
    (output / "report.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    return {
        "status": "ok",
        "output_directory": str(output.resolve()),
        "bundle_count": len(bundles),
        "validated_bundle_count": len(bundles),
        "complete_protocol_matrix": complete_protocol_matrix,
        "summary_count": len(summaries),
        "screening_comparison_count": len(screening),
    }


def _validated_joint500_bundles(
    *,
    protocol_results: Path,
    repository_root: Path,
    protocol_sha256: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    method_versions = _joint_method_versions()
    candidate_directories = {
        path.parent.resolve()
        for filename in ("pairs.csv", "run_metadata.json")
        for path in protocol_results.rglob(filename)
    }
    bundles: dict[tuple[str, str, str], dict[str, Any]] = {}
    for directory in sorted(candidate_directories, key=lambda path: path.as_posix()):
        try:
            relative = directory.relative_to(protocol_results)
        except ValueError as exc:
            raise Joint500ProtocolError(f"Bundle is outside result root: {directory}") from exc
        if len(relative.parts) != 3:
            raise Joint500ProtocolError(
                f"Bare or misplaced joint-500 bundle artifact: {directory}"
            )
        dataset, pair_kind, method = relative.parts
        if dataset not in DATASETS:
            raise Joint500ProtocolError(f"Unknown joint-500 bundle dataset: {dataset!r}.")
        if pair_kind not in PAIR_KINDS:
            raise Joint500ProtocolError(f"Unknown joint-500 bundle pair kind: {pair_kind!r}.")
        if method not in method_versions:
            raise Joint500ProtocolError(f"Unknown joint-500 bundle method: {method!r}.")
        key = (method, dataset, pair_kind)
        if key in bundles:
            raise Joint500ProtocolError(f"Duplicate joint-500 bundle identity: {key}.")
        bundles[key] = _validate_joint500_bundle(
            bundle_directory=directory,
            repository_root=repository_root,
            dataset=dataset,
            pair_kind=pair_kind,
            method=method,
            expected_method_version=method_versions[method],
            results_root=protocol_results.parent,
            protocol_sha256=protocol_sha256,
        )
    return bundles


def _validate_joint500_bundle(
    *,
    bundle_directory: Path,
    repository_root: Path,
    dataset: str,
    pair_kind: str,
    method: str,
    expected_method_version: str,
    results_root: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    from .contract import BENCHMARK_CONTRACT_VERSION, HIGHER_IS_MORE_SIMILAR, BenchmarkRunSpec
    from .runner import RESULT_FILENAME, validate_result_bundle

    result_path = bundle_directory / "pairs.csv"
    metadata_path = bundle_directory / "run_metadata.json"
    if not result_path.is_file() or not metadata_path.is_file():
        raise Joint500ProtocolError(
            f"Partial bundle must contain pairs.csv and run_metadata.json: {bundle_directory}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Joint500ProtocolError(f"Cannot read bundle metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise Joint500ProtocolError(f"Bundle metadata must be an object: {metadata_path}")
    if metadata.get("method") != method or metadata.get("method_version") != expected_method_version:
        raise Joint500ProtocolError(
            f"Bundle method identity mismatch for {bundle_directory}."
        )
    config_hash = _validate_sha256(metadata.get("config_hash"), "bundle config")
    implementation_hash = _validate_sha256(
        metadata.get("implementation_hash"), "bundle implementation"
    )
    score_direction = metadata.get("score_direction")
    score_semantics = metadata.get("score_semantics")
    if score_direction != HIGHER_IS_MORE_SIMILAR or not isinstance(score_semantics, str) or not score_semantics:
        raise Joint500ProtocolError(f"Bundle score identity is invalid: {bundle_directory}")
    manifest_path = (repository_root / protocol_manifest_path(dataset, pair_kind)).resolve()
    manifest_records = read_pair_manifest(manifest_path)
    if len(manifest_records) != COHORT_SIZE:
        raise Joint500ProtocolError(
            f"Joint-500 report manifest must contain 500 rows: {manifest_path}"
        )
    run_spec = BenchmarkRunSpec(
        expected_dataset=dataset,
        expected_protocol=row_protocol(pair_kind),
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        method=method,
        method_version=expected_method_version,
        benchmark_contract_version=BENCHMARK_CONTRACT_VERSION,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
    )
    try:
        validated_metadata = validate_result_bundle(
            bundle_directory,
            manifest_records=manifest_records,
            run_spec=run_spec,
            score_direction=score_direction,
            score_semantics=score_semantics,
        )
    except Exception as exc:
        raise Joint500ProtocolError(
            f"Benchmark-v2 bundle validation failed for {bundle_directory}: {exc}"
        ) from exc
    if method == _sourceafis_method_name():
        _validate_sourceafis_bundle_preflight(
            metadata=validated_metadata,
            results_root=results_root,
            protocol_sha256=protocol_sha256,
        )
    with (bundle_directory / RESULT_FILENAME).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {"metadata": validated_metadata, "rows": rows}


def _validate_sourceafis_bundle_preflight(
    *,
    metadata: Mapping[str, Any],
    results_root: Path,
    protocol_sha256: str,
) -> None:
    startup = metadata.get("startup_validation")
    if not isinstance(startup, dict):
        raise Joint500ProtocolError("SourceAFIS bundle startup_validation must be an object.")
    summary = startup.get("sourceafis_preflight")
    expected_fields = {
        "path",
        "sha256",
        "schema_version",
        "protocol_sha256",
        "detector_version",
        "sidecar_jar_sha256",
        "image_count",
    }
    if not isinstance(summary, dict) or set(summary) != expected_fields:
        raise Joint500ProtocolError("SourceAFIS bundle preflight metadata schema mismatch.")
    jar_sha256 = summary["sidecar_jar_sha256"]
    implementation_components = metadata.get("implementation_hash_components")
    if (
        startup.get("jar_sha256") != jar_sha256
        or not isinstance(implementation_components, dict)
        or implementation_components.get("sidecar_jar_sha256") != jar_sha256
    ):
        raise Joint500ProtocolError(
            "SourceAFIS bundle preflight JAR SHA does not match startup/implementation metadata."
        )
    expected_path = preflight_artifact_path(results_root).resolve()
    if Path(str(summary["path"])).resolve() != expected_path:
        raise Joint500ProtocolError("SourceAFIS bundle preflight path mismatch.")
    validated = validate_sourceafis_preflight(
        results_root=results_root,
        protocol_sha256=protocol_sha256,
        jar_sha256=jar_sha256,
    )
    expected = {
        "path": validated["preflight_path"],
        "sha256": validated["preflight_sha256"],
        "schema_version": validated["schema_version"],
        "protocol_sha256": validated["protocol_sha256"],
        "detector_version": validated["detector_version"],
        "sidecar_jar_sha256": validated["sidecar_jar_sha256"],
        "image_count": validated["image_count"],
    }
    if summary != expected:
        raise Joint500ProtocolError("SourceAFIS bundle preflight metadata is stale or tampered.")


def _joint_method_versions() -> dict[str, str]:
    from .detectors.opencv_gftt_harris import (
        METHOD_NAME as HARRIS_METHOD,
        METHOD_VERSION as HARRIS_VERSION,
    )
    from .detectors.sourceafis_final_minutiae import (
        METHOD_NAME as SOURCEAFIS_METHOD,
        METHOD_VERSION as SOURCEAFIS_VERSION,
    )

    return {
        HARRIS_METHOD: HARRIS_VERSION,
        SOURCEAFIS_METHOD: SOURCEAFIS_VERSION,
    }


def _sourceafis_method_name() -> str:
    from .detectors.sourceafis_final_minutiae import METHOD_NAME

    return METHOD_NAME


def _read_base_manifests(
    repository_root: Path,
) -> tuple[dict[tuple[str, str], list[PairRecord]], dict[str, str]]:
    base: dict[tuple[str, str], list[PairRecord]] = {}
    hashes: dict[str, str] = {}
    for dataset in DATASETS:
        for protocol in BASE_PROTOCOLS:
            relative = Path("protocols") / dataset / f"{protocol}.csv"
            path = repository_root / relative
            rows = read_pair_manifest(path)
            base[(dataset, protocol)] = rows
            hashes[relative.as_posix()] = file_sha256(path)
    return base, hashes


def _validate_derived_dataset_paths(
    *,
    repository_root: Path,
    data_root: Path,
    base: Mapping[tuple[str, str], Sequence[PairRecord]],
) -> int:
    target = repository_root / PROTOCOL_DIRECTORY
    selected = _read_csv_dicts(target / "selected_identities.csv", SELECTED_COLUMNS)
    pairing = _read_csv_dicts(target / "impostor_pairing.csv", PAIRING_COLUMNS)
    validated_path_count = 0
    for dataset in DATASETS:
        for pair_kind in PAIR_KINDS:
            actual = read_pair_manifest(target / dataset / f"{pair_kind}.csv")
            expected = _pair_manifest_rows(
                dataset=dataset,
                pair_kind=pair_kind,
                selected=selected,
                pairing=pairing,
                base=base,
            )
            if len(actual) != len(expected):
                raise Joint500ProtocolError(
                    f"Derived manifest row count differs from base derivation: {dataset}/{pair_kind}."
                )
            impressions = (
                ("plain", "plain")
                if pair_kind == "plain_self"
                else ("roll", "roll")
                if pair_kind == "roll_self"
                else ("plain", "roll")
            )
            for row_number, (record, expected_row) in enumerate(
                zip(actual, expected, strict=True),
                start=2,
            ):
                if _pair_record_row(record) != expected_row:
                    raise Joint500ProtocolError(
                        f"Derived row does not match its base-manifest derivation at "
                        f"{dataset}/{pair_kind} line {row_number}."
                    )
                for side, path, impression, raw_frgp in (
                    ("a", record.path_a, impressions[0], record.raw_frgp_a),
                    ("b", record.path_b, impressions[1], record.raw_frgp_b),
                ):
                    _validate_derived_image_reference(
                        path=path,
                        data_root=data_root,
                        dataset=dataset,
                        impression=impression,
                        expected_ppi=record.ppi,
                        expected_frgp=raw_frgp,
                        expected_position=record.canonical_finger_position,
                        label=f"{dataset}/{pair_kind} line {row_number} side {side}",
                    )
                    validated_path_count += 1
    return validated_path_count


def _validate_derived_image_reference(
    *,
    path: Path,
    data_root: Path,
    dataset: str,
    impression: str,
    expected_ppi: int,
    expected_frgp: int,
    expected_position: int,
    label: str,
) -> None:
    if not path.is_file():
        raise Joint500ProtocolError(f"Derived dataset file is missing for {label}: {path}")
    spec = DATASET_SPECS[dataset]
    resolved = path.resolve()
    expected_root = spec.impression_dir(data_root, impression).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise Joint500ProtocolError(
            f"Derived dataset path is outside the expected {dataset}/{impression} root for {label}: {path}"
        ) from exc
    try:
        source = validate_image_path(resolved, spec, impression)
    except Exception as exc:
        raise Joint500ProtocolError(f"Derived dataset path schema is invalid for {label}: {exc}") from exc
    if source.ppi != expected_ppi or source.frgp != expected_frgp:
        raise Joint500ProtocolError(
            f"Derived dataset PPI/FRGP metadata mismatch for {label}: "
            f"expected {expected_ppi}/{expected_frgp}, got {source.ppi}/{source.frgp}."
        )
    mapped = canonical_finger_position(source.impression_type, source.frgp)
    if mapped != expected_position:
        raise Joint500ProtocolError(
            f"Derived dataset canonical position mismatch for {label}: "
            f"expected {expected_position}, got {mapped}."
        )


def _pair_record_row(record: PairRecord) -> dict[str, str]:
    return {
        "pair_id": record.pair_id,
        "dataset": record.dataset,
        "protocol": record.protocol,
        "subject_id": record.subject_id,
        "canonical_finger_position": str(record.canonical_finger_position),
        "ppi": str(record.ppi),
        "raw_frgp_a": str(record.raw_frgp_a),
        "raw_frgp_b": str(record.raw_frgp_b),
        "path_a": str(record.path_a),
        "path_b": str(record.path_b),
    }


def _validate_dataset_preflight_binding(
    dataset_preflight: Mapping[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    report = dict(dataset_preflight)
    if report.get("status") != "ok" or report.get("dataset_preflight_status") != "ok":
        raise Joint500ProtocolError("Shared joint-500 dataset preflight is not successful.")
    expected_root = Path(data_root).resolve()
    if Path(str(report.get("data_root"))).resolve() != expected_root:
        raise Joint500ProtocolError(
            f"Joint-500 data root differs from shared preflight: {expected_root}"
        )
    hashes = report.get("derived_manifest_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        protocol_manifest_path(dataset, pair_kind).as_posix()
        for dataset in DATASETS
        for pair_kind in PAIR_KINDS
    }:
        raise Joint500ProtocolError("Shared dataset preflight has invalid derived manifest hashes.")
    return report


def _validator_report(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


def _eligible_identities(
    base: Mapping[tuple[str, str], Sequence[PairRecord]],
) -> dict[int, list[str]]:
    indices: dict[tuple[str, str], dict[tuple[str, int], list[PairRecord]]] = {}
    for key, records in base.items():
        current: dict[tuple[str, int], list[PairRecord]] = {}
        for record in records:
            identity = (record.subject_id, record.canonical_finger_position)
            current.setdefault(identity, []).append(record)
        indices[key] = current
    all_keys = set.intersection(*(set(index) for index in indices.values()))
    eligible: dict[int, list[str]] = {position: [] for position in FINGER_POSITIONS}
    for subject_id, position in sorted(all_keys):
        if position not in eligible:
            continue
        if any(len(indices[key][(subject_id, position)]) != 1 for key in indices):
            continue
        records = {
            key: indices[key][(subject_id, position)][0]
            for key in indices
        }
        if _consistent_identity(records):
            eligible[position].append(subject_id)
    return eligible


def _consistent_identity(records: Mapping[tuple[str, str], PairRecord]) -> bool:
    for dataset in DATASETS:
        plain = records[(dataset, "plain_self")]
        roll = records[(dataset, "roll_self")]
        genuine = records[(dataset, "plain_roll")]
        if plain.path_a != plain.path_b or plain.raw_frgp_a != plain.raw_frgp_b:
            return False
        if roll.path_a != roll.path_b or roll.raw_frgp_a != roll.raw_frgp_b:
            return False
        if genuine.path_a != plain.path_a or genuine.raw_frgp_a != plain.raw_frgp_a:
            return False
        if genuine.path_b != roll.path_a or genuine.raw_frgp_b != roll.raw_frgp_a:
            return False
        if not (plain.ppi == roll.ppi == genuine.ppi):
            return False
    b = records[("sd300b", "plain_roll")]
    c = records[("sd300c", "plain_roll")]
    return b.raw_frgp_a == c.raw_frgp_a and b.raw_frgp_b == c.raw_frgp_b


def _select_identities(eligible: Mapping[int, Sequence[str]]) -> list[dict[str, str]]:
    selected_subjects: set[str] = set()
    selected: list[dict[str, str]] = []
    for position in FINGER_POSITIONS:
        ranked = sorted(
            (
                (_selection_hash(position, subject_id), subject_id)
                for subject_id in eligible.get(position, ())
            ),
            key=lambda item: (item[0], item[1]),
        )
        chosen = [item for item in ranked if item[1] not in selected_subjects][:PER_POSITION]
        if len(chosen) != PER_POSITION:
            raise Joint500ProtocolError(
                f"Cannot select 50 unique subjects for finger position {position}; found {len(chosen)}."
            )
        for rank, (selection_hash, subject_id) in enumerate(chosen, start=1):
            selected_subjects.add(subject_id)
            selected.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "subject_id": subject_id,
                    "canonical_finger_position": str(position),
                    "selection_rank": str(rank),
                    "selection_hash": selection_hash,
                }
            )
    if len(selected) != COHORT_SIZE or len(selected_subjects) != COHORT_SIZE:
        raise Joint500ProtocolError("Frozen selection did not produce 500 identities and subjects.")
    return selected


def _selection_hash(position: int, subject_id: str) -> str:
    payload = f"{PROTOCOL_VERSION}\n{SEED}\n{position}\n{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _impostor_pairing(selected: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for position in FINGER_POSITIONS:
        rows = sorted(
            (row for row in selected if int(row["canonical_finger_position"]) == position),
            key=lambda row: int(row["selection_rank"]),
        )
        for index, probe in enumerate(rows):
            candidate = rows[(index + 1) % len(rows)]
            probe_subject = probe["subject_id"]
            candidate_subject = candidate["subject_id"]
            result.append(
                {
                    "logical_pair_id": (
                        f"{PROTOCOL_NAME}_impostor_{position:02d}_{probe_subject}_to_{candidate_subject}"
                    ),
                    "canonical_finger_position": str(position),
                    "probe_subject_id": probe_subject,
                    "candidate_subject_id": candidate_subject,
                    "selection_rank_probe": probe["selection_rank"],
                    "selection_rank_candidate": candidate["selection_rank"],
                }
            )
    return result


def _pair_manifest_rows(
    *,
    dataset: str,
    pair_kind: str,
    selected: Sequence[Mapping[str, str]],
    pairing: Sequence[Mapping[str, str]],
    base: Mapping[tuple[str, str], Sequence[PairRecord]],
) -> list[dict[str, str]]:
    protocol_lookup = {
        protocol: {
            (record.subject_id, record.canonical_finger_position): record
            for record in base[(dataset, protocol)]
        }
        for protocol in BASE_PROTOCOLS
    }
    rows: list[dict[str, str]] = []
    if pair_kind != "plain_roll_impostor":
        source_protocol = "plain_roll" if pair_kind == "plain_roll_genuine" else pair_kind
        for identity in selected:
            key = (identity["subject_id"], int(identity["canonical_finger_position"]))
            source = protocol_lookup[source_protocol][key]
            rows.append(_derived_source_row(source, dataset, pair_kind))
        return rows
    for pair in pairing:
        position = int(pair["canonical_finger_position"])
        probe = protocol_lookup["plain_self"][(pair["probe_subject_id"], position)]
        candidate = protocol_lookup["roll_self"][(pair["candidate_subject_id"], position)]
        rows.append(
            {
                "pair_id": f"{dataset}_{pair['logical_pair_id']}",
                "dataset": dataset,
                "protocol": row_protocol(pair_kind),
                "subject_id": pair["probe_subject_id"],
                "canonical_finger_position": str(position),
                "ppi": str(probe.ppi),
                "raw_frgp_a": str(probe.raw_frgp_a),
                "raw_frgp_b": str(candidate.raw_frgp_a),
                "path_a": str(probe.path_a),
                "path_b": str(candidate.path_a),
            }
        )
    return rows


def _derived_source_row(source: PairRecord, dataset: str, pair_kind: str) -> dict[str, str]:
    return {
        "pair_id": (
            f"{dataset}_{PROTOCOL_NAME}_{pair_kind}_{source.subject_id}_"
            f"{source.canonical_finger_position:02d}"
        ),
        "dataset": dataset,
        "protocol": row_protocol(pair_kind),
        "subject_id": source.subject_id,
        "canonical_finger_position": str(source.canonical_finger_position),
        "ppi": str(source.ppi),
        "raw_frgp_a": str(source.raw_frgp_a),
        "raw_frgp_b": str(source.raw_frgp_b),
        "path_a": str(source.path_a),
        "path_b": str(source.path_b),
    }


def _validate_pairing_rows(
    pairing: Sequence[Mapping[str, str]],
    selected: Sequence[Mapping[str, str]],
) -> None:
    if len(pairing) != COHORT_SIZE:
        raise Joint500ProtocolError("Impostor pairing must contain exactly 500 pairs.")
    if len({row["logical_pair_id"] for row in pairing}) != COHORT_SIZE:
        raise Joint500ProtocolError("Impostor logical pair IDs must be unique.")
    selected_keys = {
        (row["subject_id"], int(row["canonical_finger_position"]))
        for row in selected
    }
    for position in FINGER_POSITIONS:
        rows = [row for row in pairing if int(row["canonical_finger_position"]) == position]
        probes = [row["probe_subject_id"] for row in rows]
        candidates = [row["candidate_subject_id"] for row in rows]
        if len(rows) != PER_POSITION or len(set(probes)) != PER_POSITION or len(set(candidates)) != PER_POSITION:
            raise Joint500ProtocolError(f"Impostor pairing for position {position} is not bijective.")
        if any(probe == candidate for probe, candidate in zip(probes, candidates, strict=True)):
            raise Joint500ProtocolError(f"Impostor pairing for position {position} contains a self pair.")
        if any((subject, position) not in selected_keys for subject in probes + candidates):
            raise Joint500ProtocolError(f"Impostor pairing for position {position} leaves the selected cohort.")


def _validate_pair_manifests(
    target: Path,
    selected: Sequence[Mapping[str, str]],
    pairing: Sequence[Mapping[str, str]],
) -> None:
    expected_subject_positions = {
        (row["subject_id"], int(row["canonical_finger_position"]))
        for row in selected
    }
    logical_by_dataset: dict[tuple[str, str], set[str]] = {}
    for dataset in DATASETS:
        for pair_kind in PAIR_KINDS:
            path = target / dataset / f"{pair_kind}.csv"
            records = read_pair_manifest(path)
            if len(records) != COHORT_SIZE:
                raise Joint500ProtocolError(f"{dataset}/{pair_kind} must contain exactly 500 rows.")
            if any(record.protocol != row_protocol(pair_kind) for record in records):
                raise Joint500ProtocolError(f"{dataset}/{pair_kind} has an invalid row protocol.")
            if pair_kind != "plain_roll_impostor":
                actual = {(record.subject_id, record.canonical_finger_position) for record in records}
                if actual != expected_subject_positions:
                    raise Joint500ProtocolError(f"{dataset}/{pair_kind} does not contain the selected identities.")
            logical = {record.pair_id[len(dataset) + 1 :] for record in records}
            logical_by_dataset[(dataset, pair_kind)] = logical
    for pair_kind in PAIR_KINDS:
        if logical_by_dataset[("sd300b", pair_kind)] != logical_by_dataset[("sd300c", pair_kind)]:
            raise Joint500ProtocolError(f"B/C logical rows differ for pair kind {pair_kind}.")
    expected_impostors = {row["logical_pair_id"] for row in pairing}
    if logical_by_dataset[("sd300b", "plain_roll_impostor")] != expected_impostors:
        raise Joint500ProtocolError("Impostor manifests do not match impostor_pairing.csv.")


def _csv_bytes(rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_csv_dicts(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise Joint500ProtocolError(f"CSV schema mismatch for {path}.")
        rows = list(reader)
    if any(None in row or any(row.get(column) in (None, "") for column in columns) for row in rows):
        raise Joint500ProtocolError(f"CSV contains missing or extra values: {path}.")
    return rows


def _extra_artifacts(target: Path, expected: set[Path]) -> list[str]:
    if not target.exists():
        return []
    actual = {
        path.relative_to(target)
        for path in target.rglob("*")
        if path.is_file()
    }
    return sorted(path.as_posix() for path in actual - expected)


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _repository_root_from_manifest(manifest: Path) -> Path:
    for candidate in (manifest, *manifest.parents):
        if (candidate / ".git").exists():
            return candidate
    raise Joint500ProtocolError(f"Cannot locate repository root from manifest: {manifest}")


def _deterministic_minutia_payload(extracted: Any) -> tuple[Any, ...]:
    return (
        extracted.sourceafis_version,
        extracted.template_version,
        extracted.effective_dpi,
        extracted.native_width,
        extracted.native_height,
        extracted.scaled_width,
        extracted.scaled_height,
        extracted.coordinate_space,
        extracted.selection_stage,
        extracted.selection_semantics,
        extracted.source_order_semantics,
        extracted.template_sha256,
        extracted.minutiae,
    )


def _validate_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise Joint500ProtocolError(f"{label} SHA-256 must be 64 lowercase hexadecimal characters.")
    return value


def _bundle_directory(results_root: Path, dataset: str, pair_kind: str, method: str) -> Path:
    return (results_root / PROTOCOL_NAME / dataset / pair_kind / method).resolve()


def _sidecar_startup_dict(
    startup: Any,
    health: Mapping[str, Any],
    *,
    sourceafis_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if startup is None:
        raise Joint500ProtocolError("Managed SourceAFIS startup metadata is unavailable.")
    result = {
        "managed_by_runner": startup.managed_by_runner,
        "service_url": startup.service_url,
        "startup_ms": startup.startup_ms,
        "validation_result": startup.validation_result,
        "command": startup.command,
        "jar_path": startup.jar_path,
        "jar_sha256": startup.jar_sha256,
        "java_executable": startup.java_executable,
        "health": dict(health),
    }
    if sourceafis_preflight is not None:
        result["sourceafis_preflight"] = {
            "path": sourceafis_preflight["preflight_path"],
            "sha256": sourceafis_preflight["preflight_sha256"],
            "schema_version": sourceafis_preflight["schema_version"],
            "protocol_sha256": sourceafis_preflight["protocol_sha256"],
            "detector_version": sourceafis_preflight["detector_version"],
            "sidecar_jar_sha256": sourceafis_preflight["sidecar_jar_sha256"],
            "image_count": sourceafis_preflight["image_count"],
        }
    return result


def _pair_kind_from_protocol(protocol: str) -> str:
    prefix = f"{PROTOCOL_NAME}_"
    if not protocol.startswith(prefix) or protocol[len(prefix) :] not in PAIR_KINDS:
        raise Joint500ProtocolError(f"Unexpected result protocol: {protocol!r}.")
    return protocol[len(prefix) :]


def _screening_group_summary(
    method: str,
    dataset: str,
    pair_kind: str,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    failure_codes: dict[str, int] = {}
    for row in rows:
        if row["error_code"]:
            failure_codes[row["error_code"]] = failure_codes.get(row["error_code"], 0) + 1
    prepare_diagnostics = [
        diagnostics
        for row in ok
        for diagnostics in (
            _diagnostics(row.get("prepare_a_diagnostics", "")),
            _diagnostics(row.get("prepare_b_diagnostics", "")),
        )
        if diagnostics
    ]
    compare_diagnostics = [
        diagnostics
        for row in ok
        if (diagnostics := _diagnostics(row.get("compare_diagnostics", "")))
    ]
    return {
        "method": method,
        "dataset": dataset,
        "pair_kind": pair_kind,
        "requested_rows": len(rows),
        "successful_rows": len(ok),
        "prepare_failures": sum(
            row["status"] in {"prepare_a_failure", "prepare_b_failure"} for row in rows
        ),
        "comparison_failures": sum(row["status"] == "comparison_failure" for row in rows),
        "failure_codes": dict(sorted(failure_codes.items())),
        "detector_point_count": _quantiles(
            _numbers(diagnostics.get("detector_point_count") for diagnostics in prepare_diagnostics)
        ),
        "descriptor_count": _quantiles(
            _numbers(diagnostics.get("representation_descriptor_count") for diagnostics in prepare_diagnostics)
        ),
        "mutual_matches": _quantiles(
            _numbers(diagnostics.get("mutual_match_count") for diagnostics in compare_diagnostics)
        ),
        "geometric_inliers": _quantiles(
            _numbers(diagnostics.get("geometric_inlier_count") for diagnostics in compare_diagnostics)
        ),
        "inlier_ratio": _quantiles(
            _numbers(diagnostics.get("inlier_ratio") for diagnostics in compare_diagnostics)
        ),
        "inliers_over_minimum_keypoint_count": _quantiles(
            _numbers(
                (diagnostics.get("score_components") or {}).get("inliers_over_min_keypoints")
                for diagnostics in compare_diagnostics
                if isinstance(diagnostics.get("score_components"), dict)
            )
        ),
        "detector_time_ms": _quantiles(
            _numbers(diagnostics.get("detector_time_ms") for diagnostics in prepare_diagnostics)
        ),
        "preparation_time_ms": _quantiles(
            _numbers(
                value
                for row in rows
                for value in (row.get("prepare_a_ms"), row.get("prepare_b_ms"))
            )
        ),
        "comparison_time_ms": _quantiles(_numbers(row.get("compare_ms") for row in rows)),
        "raw_score_quantiles": _quantiles(_numbers(row.get("raw_score") for row in ok)),
    }


def _screening_roc(
    method: str,
    dataset: str,
    genuine_rows: Sequence[Mapping[str, str]],
    impostor_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    genuine = _numbers(row.get("raw_score") for row in genuine_rows if row["status"] == "ok")
    impostor = _numbers(row.get("raw_score") for row in impostor_rows if row["status"] == "ok")
    if not genuine or not impostor:
        raise Joint500ProtocolError(f"ROC requires successful genuine and impostor rows for {method}/{dataset}.")
    thresholds = [
        math.nextafter(max(genuine + impostor), math.inf),
        *sorted(set(genuine + impostor), reverse=True),
    ]
    roc = [
        {
            "threshold": threshold,
            "far": sum(score >= threshold for score in impostor) / len(impostor),
            "tar": sum(score >= threshold for score in genuine) / len(genuine),
        }
        for threshold in thresholds
    ]
    auc = (
        sum(
            1.0 if g > i else 0.5 if g == i else 0.0
            for g in genuine
            for i in impostor
        )
        / (len(genuine) * len(impostor))
    )
    eer_point = min(roc, key=lambda point: abs(point["far"] - (1.0 - point["tar"])))
    candidates = [point for point in roc if point["far"] <= 0.01]
    far_one = max(candidates, key=lambda point: (point["tar"], point["far"], -point["threshold"]))
    genuine_failure_count = len(genuine_rows) - len(genuine)
    impostor_failure_count = len(impostor_rows) - len(impostor)
    selected_threshold = far_one["threshold"]
    return {
        "method": method,
        "dataset": dataset,
        "conditional_on_success": True,
        "genuine_requested": len(genuine_rows),
        "genuine_successful": len(genuine),
        "genuine_failure_count": genuine_failure_count,
        "genuine_failure_rate": genuine_failure_count / len(genuine_rows),
        "impostor_requested": len(impostor_rows),
        "impostor_successful": len(impostor),
        "impostor_failure_count": impostor_failure_count,
        "impostor_failure_rate": impostor_failure_count / len(impostor_rows),
        "conditional_auc": auc,
        "conditional_screening_eer": (eer_point["far"] + (1.0 - eer_point["tar"])) / 2.0,
        "conditional_eer_threshold": eer_point["threshold"],
        "conditional_tar_at_far_1_percent": far_one["tar"],
        "conditional_actual_far_at_1_percent": far_one["far"],
        "conditional_far_1_percent_threshold": selected_threshold,
        "conditional_roc": roc,
        "selected_threshold": selected_threshold,
        "selected_threshold_source": "conditional_far_1_percent_report_only",
        "operational_tar_at_selected_threshold": (
            sum(score >= selected_threshold for score in genuine) / len(genuine_rows)
        ),
        "operational_far_at_selected_threshold": (
            sum(score >= selected_threshold for score in impostor) / len(impostor_rows)
        ),
        "failure_policy": "fail_closed",
        "threshold_calibration": "none_report_only_empirical",
    }


def _paired_bc_analysis(
    grouped: Mapping[tuple[str, str, str], Sequence[Mapping[str, str]]],
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    methods = sorted({method for method, _, _ in grouped})
    for method in methods:
        for pair_kind in PAIR_KINDS:
            b_rows = grouped.get((method, "sd300b", pair_kind))
            c_rows = grouped.get((method, "sd300c", pair_kind))
            if b_rows is None or c_rows is None:
                continue
            b = {_logical_result_key(row): row for row in b_rows}
            c = {_logical_result_key(row): row for row in c_rows}
            if set(b) != set(c):
                raise Joint500ProtocolError(f"B/C result logical keys differ for {method}/{pair_kind}.")
            score_b: list[float] = []
            score_c: list[float] = []
            deltas: dict[str, list[float]] = {
                "score": [],
                "point_count": [],
                "descriptor_count": [],
                "mutual_matches": [],
                "geometric_inliers": [],
            }
            transitions: dict[str, int] = {}
            for key in sorted(b):
                row_b, row_c = b[key], c[key]
                transition = f"{row_b['status']}->{row_c['status']}"
                transitions[transition] = transitions.get(transition, 0) + 1
                if row_b["status"] != "ok" or row_c["status"] != "ok":
                    continue
                sb, sc = float(row_b["raw_score"]), float(row_c["raw_score"])
                score_b.append(sb)
                score_c.append(sc)
                deltas["score"].append(sc - sb)
                metrics_b = _row_metrics(row_b)
                metrics_c = _row_metrics(row_c)
                for metric in ("point_count", "descriptor_count", "mutual_matches", "geometric_inliers"):
                    if metrics_b[metric] is not None and metrics_c[metric] is not None:
                        deltas[metric].append(float(metrics_c[metric]) - float(metrics_b[metric]))
            analyses.append(
                {
                    "method": method,
                    "pair_kind": pair_kind,
                    "logical_pair_count": len(b),
                    "paired_success_count": len(score_b),
                    "success_failure_transitions": dict(sorted(transitions.items())),
                    "score_delta_c_minus_b": _delta_summary(deltas["score"]),
                    "point_count_delta_c_minus_b": _delta_summary(deltas["point_count"]),
                    "descriptor_count_delta_c_minus_b": _delta_summary(deltas["descriptor_count"]),
                    "mutual_match_delta_c_minus_b": _delta_summary(deltas["mutual_matches"]),
                    "inlier_delta_c_minus_b": _delta_summary(deltas["geometric_inliers"]),
                    "score_correlation": _pearson(score_b, score_c),
                    "join_semantics": "logical_pair_id_not_row_position",
                }
            )
    return analyses


def _logical_result_key(row: Mapping[str, str]) -> str:
    dataset = row["dataset"]
    prefix = f"{dataset}_"
    if not row["pair_id"].startswith(prefix):
        raise Joint500ProtocolError(f"Result pair_id does not start with dataset: {row['pair_id']!r}.")
    return row["pair_id"][len(prefix) :]


def _row_metrics(row: Mapping[str, str]) -> dict[str, float | None]:
    prepare = [
        _diagnostics(row.get("prepare_a_diagnostics", "")),
        _diagnostics(row.get("prepare_b_diagnostics", "")),
    ]
    compare = _diagnostics(row.get("compare_diagnostics", ""))
    return {
        "point_count": _mean_optional(
            _numbers(item.get("detector_point_count") for item in prepare if item)
        ),
        "descriptor_count": _mean_optional(
            _numbers(item.get("representation_descriptor_count") for item in prepare if item)
        ),
        "mutual_matches": _number_or_none(compare.get("mutual_match_count")),
        "geometric_inliers": _number_or_none(compare.get("geometric_inlier_count")),
    }


def _diagnostics(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Joint500ProtocolError("Result diagnostics contain invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise Joint500ProtocolError("Result diagnostics must encode a JSON object.")
    return parsed


def _numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value in (None, "") or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            result.append(parsed)
    return result


def _number_or_none(value: Any) -> float | None:
    numbers = _numbers([value])
    return numbers[0] if numbers else None


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "p25": None, "median": None, "p75": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.5),
        "p75": _percentile(ordered, 0.75),
        "maximum": ordered[-1],
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _mean_optional(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _delta_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "median_absolute_delta": None, "mean": None}
    return {
        "count": len(values),
        "median": statistics.median(values),
        "median_absolute_delta": statistics.median(abs(value) for value in values),
        "mean": statistics.fmean(values),
    }


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = statistics.fmean(first)
    mean_second = statistics.fmean(second)
    numerator = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second, strict=True))
    denominator = math.sqrt(
        sum((a - mean_first) ** 2 for a in first)
        * sum((b - mean_second) ** 2 for b in second)
    )
    return numerator / denominator if denominator > 0 else None


def _summary_csv_bytes(summaries: Sequence[Mapping[str, Any]]) -> bytes:
    fields = [
        "method",
        "dataset",
        "pair_kind",
        "requested_rows",
        "successful_rows",
        "prepare_failures",
        "comparison_failures",
        "failure_codes",
        "detector_point_count",
        "descriptor_count",
        "mutual_matches",
        "geometric_inliers",
        "inlier_ratio",
        "inliers_over_minimum_keypoint_count",
        "detector_time_ms",
        "preparation_time_ms",
        "comparison_time_ms",
        "raw_score_quantiles",
    ]
    flattened = [
        {
            field: (
                json.dumps(summary[field], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                if isinstance(summary[field], dict)
                else summary[field]
            )
            for field in fields
        }
        for summary in summaries
    ]
    return _csv_bytes(flattened, fields)


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Detector-only joint-500 screening report",
        "",
        "This report is screening-only. Threshold calibration is not performed.",
        "",
        f"- Complete protocol matrix: {str(report['complete_protocol_matrix']).lower()}",
        f"- Validated benchmark-v2 bundles: {report['validated_bundle_count']} / {report['required_bundle_count']}",
        f"- Bundled summary groups: {len(report['summaries'])}",
        f"- Genuine/impostor comparisons: {len(report['genuine_impostor_screening'])}",
        "- Impostor count per complete comparison: 500 (FAR resolution 0.002)",
        "- Reported operating point: FAR 1% only",
        "",
        "## Conditional and operational metrics",
        "",
        "ROC, conditional AUC, conditional screening EER, and conditional TAR at FAR 1% use only successful comparisons.",
        "Genuine and impostor failure counts/rates are reported separately.",
        "Operational TAR/FAR use all 500 requested rows with fail-closed handling: genuine failures are non-matches and impostor failures are non-accepts.",
        "",
        "## Interpretation limits",
        "",
    ]
    lines.extend(f"- {item}" for item in report["interpretation_limits"])
    lines.extend(
        [
            "",
            "## B/C pairing",
            "",
            "SD300b and SD300c rows are joined by logical identity/pair keys. They are not pooled as independent samples.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "COHORT_SIZE",
    "DATASETS",
    "PAIR_KINDS",
    "PROTOCOL_DIRECTORY",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "SEED",
    "SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION",
    "Joint500ProtocolError",
    "build_protocol_artifacts",
    "preflight_artifact_path",
    "protocol_artifact_bytes",
    "protocol_manifest_path",
    "row_protocol",
    "report_joint500",
    "run_joint500",
    "run_sourceafis_preflight",
    "validate_joint_dataset_preflight",
    "validate_joint_manifest",
    "validate_protocol_artifacts",
    "validate_sourceafis_preflight",
]
