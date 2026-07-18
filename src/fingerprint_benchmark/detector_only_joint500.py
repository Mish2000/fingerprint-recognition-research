"""Neutral joint-500 screening protocol, execution, and reporting support."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .hashing import file_sha256, stable_hash
from .manifest import MANIFEST_COLUMNS, PairRecord, read_pair_manifest


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


def validate_joint_manifest(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    """Dedicated runner validator for one of the eight derived manifests."""

    del data_root
    resolved = manifest_path.resolve()
    repository_root = _repository_root_from_manifest(resolved)
    report = validate_protocol_artifacts(repository_root=repository_root)
    protocol_root = (repository_root / PROTOCOL_DIRECTORY).resolve()
    try:
        relative = resolved.relative_to(protocol_root)
    except ValueError as exc:
        raise Joint500ProtocolError("Manifest is outside the joint-500 protocol directory.") from exc
    expected = {Path(dataset) / f"{kind}.csv" for dataset in DATASETS for kind in PAIR_KINDS}
    if relative not in expected:
        raise Joint500ProtocolError(f"Not a joint-500 pair manifest: {relative.as_posix()}")
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
    """Prove encoded-image/raw-grayscale template parity on 20 cohort images."""

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
                template = client.extract_template(encoded, float(record.ppi))
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None or image.ndim != 2 or image.dtype.name != "uint8":
                    raise Joint500ProtocolError(f"OpenCV cannot read uint8 grayscale preflight image: {image_path}")
                pixels = image.tobytes(order="C")
                first = client.extract_final_minutiae(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                second = client.extract_final_minutiae(
                    pixels,
                    int(image.shape[1]),
                    int(image.shape[0]),
                    float(record.ppi),
                )
                template_sha256 = hashlib.sha256(
                    base64.b64decode(template.template_base64.encode("ascii"), validate=True)
                ).hexdigest()
                if first.template_sha256 != template_sha256:
                    raise Joint500ProtocolError(
                        f"SourceAFIS encoded/raw template parity mismatch for {dataset}/{key}/{side}."
                    )
                if _deterministic_minutia_payload(first) != _deterministic_minutia_payload(second):
                    raise Joint500ProtocolError(
                        f"Repeated SourceAFIS raw extraction mismatch for {dataset}/{key}/{side}."
                    )
                items.append(
                    {
                        "logical_image_id": f"{dataset}:{key[0]}:{key[1]:02d}:{side}",
                        "dataset": dataset,
                        "subject_id": key[0],
                        "canonical_finger_position": key[1],
                        "impression": side,
                        "ppi": record.ppi,
                        "template_sha256": template_sha256,
                        "minutia_count": first.minutia_count,
                        "parity": True,
                        "repeated_raw_payload_equal": True,
                    }
                )
    if len(items) != 20:
        raise Joint500ProtocolError(f"SourceAFIS preflight must contain 20 images, got {len(items)}.")
    artifact = {
        "schema_version": "sourceafis-final-minutiae-preflight-v1",
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
    return {**artifact, "path": str(output.resolve()), "sha256": _bytes_sha256(content)}


def validate_sourceafis_preflight(
    *,
    results_root: Path,
    protocol_sha256: str,
    jar_sha256: str,
) -> dict[str, Any]:
    from .detectors.sourceafis_final_minutiae import DETECTOR_VERSION

    path = preflight_artifact_path(results_root)
    if not path.is_file():
        raise Joint500ProtocolError(
            f"Matching SourceAFIS preflight is required before joint-500 run: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "ok",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": protocol_sha256,
        "detector_version": DETECTOR_VERSION,
        "sidecar_jar_sha256": _validate_sha256(jar_sha256, "sidecar JAR"),
        "image_count": 20,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise Joint500ProtocolError(
                f"SourceAFIS preflight field {key!r} mismatch: expected {value!r}, got {payload.get(key)!r}."
            )
    if payload.get("biometric_bytes_persisted") is not False:
        raise Joint500ProtocolError("SourceAFIS preflight must not persist biometric bytes.")
    return payload


def run_joint500(
    *,
    method: str,
    dataset: str | None = None,
    pair_kind: str | None = None,
    results_root: Path = Path("results"),
    data_root: Path = Path("C:/fingerprint-datasets/NIST"),
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
    validation = validate_protocol_artifacts(repository_root=repository_root)
    datasets = (dataset,) if dataset is not None else DATASETS
    kinds = (pair_kind,) if pair_kind is not None else PAIR_KINDS
    if any(item not in DATASETS for item in datasets) or any(item not in PAIR_KINDS for item in kinds):
        raise Joint500ProtocolError("Invalid dataset or pair-kind filter.")
    if method == SOURCEAFIS_METHOD:
        if sidecar_jar is None or not sidecar_jar.is_file():
            raise Joint500ProtocolError("SourceAFIS joint-500 run requires an existing sidecar JAR.")
        validate_sourceafis_preflight(
            results_root=results_root,
            protocol_sha256=validation["protocol_sha256"],
            jar_sha256=file_sha256(sidecar_jar),
        )
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
                        dedicated_validator=validate_joint_manifest,
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
                            startup_validation=_sidecar_startup_dict(sidecar.startup, health.raw),
                            data_root=data_root,
                            dedicated_validator=validate_joint_manifest,
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
) -> dict[str, Any]:
    """Create a screening-only report from already persisted bundles."""

    protocol_results = results_root / PROTOCOL_NAME
    result_paths = sorted(protocol_results.glob("*/*/*/pairs.csv"))
    if not result_paths:
        raise Joint500ProtocolError(f"No joint-500 result bundles found under {protocol_results}.")
    rows: list[dict[str, str]] = []
    for path in result_paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        pair_kind = _pair_kind_from_protocol(row["protocol"])
        grouped.setdefault((row["method"], row["dataset"], pair_kind), []).append(row)
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
        "schema_version": "detector-only-joint-500-screening-report-v1",
        "protocol_name": PROTOCOL_NAME,
        "screening_only": True,
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
        "bundle_count": len(result_paths),
        "summary_count": len(summaries),
        "screening_comparison_count": len(screening),
    }


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


def _sidecar_startup_dict(startup: Any, health: Mapping[str, Any]) -> dict[str, Any]:
    if startup is None:
        raise Joint500ProtocolError("Managed SourceAFIS startup metadata is unavailable.")
    return {
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
    return {
        "method": method,
        "dataset": dataset,
        "genuine_requested": len(genuine_rows),
        "genuine_successful": len(genuine),
        "impostor_requested": len(impostor_rows),
        "impostor_successful": len(impostor),
        "auc": auc,
        "screening_eer": (eer_point["far"] + (1.0 - eer_point["tar"])) / 2.0,
        "eer_threshold": eer_point["threshold"],
        "tar_at_far_1_percent": far_one["tar"],
        "actual_achieved_far": far_one["far"],
        "far_1_percent_threshold": far_one["threshold"],
        "roc": roc,
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
        f"- Bundled summary groups: {len(report['summaries'])}",
        f"- Genuine/impostor comparisons: {len(report['genuine_impostor_screening'])}",
        "- Impostor count per complete comparison: 500 (FAR resolution 0.002)",
        "- Reported operating point: FAR 1% only",
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
    "Joint500ProtocolError",
    "build_protocol_artifacts",
    "preflight_artifact_path",
    "protocol_artifact_bytes",
    "protocol_manifest_path",
    "row_protocol",
    "report_joint500",
    "run_joint500",
    "run_sourceafis_preflight",
    "validate_joint_manifest",
    "validate_protocol_artifacts",
    "validate_sourceafis_preflight",
]
