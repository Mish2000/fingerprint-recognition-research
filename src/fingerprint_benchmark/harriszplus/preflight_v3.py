"""Final functional CPU/CUDA engineering preflight for HarrisZ+ candidate v3.

Only validation, fixture, provenance, and reporting behavior lives here.  The
score-producing HarrisZ+ sources are imported unchanged.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from ..contract import (
    COMPARISON_FAILURE,
    OK,
    PREPARE_A_FAILURE,
    PREPARE_B_FAILURE,
    MethodExecutionError,
)
from ..hashing import canonical_json_bytes, file_sha256, stable_config_hash, stable_hash
from ..manifest import PairRecord
from .adapter import HarrisZPlusGeometricAdapter
from .config import HarrisZPlusConfig
from .cuda_detector import detect_harriszplus_cuda
from .preflight import (
    DATASETS,
    MINIMUM_RESPONSE_PIXEL_COVERAGE,
    SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE,
    _canonical_validation_keypoint_index_order,
    _configure_and_describe_cuda,
    _effective_runner_config,
    _keypoint_record,
    _keypoints,
    _spearman_rank_correlation,
    _without_timing_fields,
    detector_result_sha256,
    synthetic_suite,
)
from .preflight_v2 import (
    EXPECTED_CANDIDATE_CONFIG_SHA256,
    EXPECTED_V1_ALGORITHM_SOURCE_SHA256,
    SPATIAL_TOLERANCE_ORIGINAL_PX,
    RELATIVE_SCALE_TOLERANCE,
    MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION,
    _cuda_repeat_comparison,
    _detector_absolute_conditions,
    _directional_keypoint_matches,
    _response_map_comparison,
    _semantic_pair_comparison,
    compare_candidate_counts_v2,
    compare_final_keypoints_v2,
    count_equivalence,
    representation_sha256,
)
from .provenance import implementation_source_hashes
from .reference_cpu import detect_harriszplus_cpu


METHOD_NAME = "harriszplus_rootsift_geometric"
METHOD_VERSION = "harriszplus-rootsift-geometric-v3"
PREFLIGHT_CONTRACT = "engineering-preflight-v3"
PREFLIGHT_SCHEMA_VERSION = "harriszplus-engineering-preflight-v3"
DEFAULT_PROJECT_ROOT = Path(r"C:\fingerprint-recognition-research")
METHOD_RESULTS_RELATIVE = Path("results/harriszplus_rootsift_geometric_v3")
PILOT_RELATIVE = Path(
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v3"
)
CONTRACT_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_contract_v3.json"
)
PASS_RELATIVE = METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_pass.json"
FAILURE_RELATIVE = (
    METHOD_RESULTS_RELATIVE / "preflight/engineering_preflight_failure.json"
)
FIXTURE_ROOT_RELATIVE = METHOD_RESULTS_RELATIVE / "fixtures"
IDENTITIES_RELATIVE = FIXTURE_ROOT_RELATIVE / "engineering_identities_v3.csv"
PAIRS_RELATIVE = FIXTURE_ROOT_RELATIVE / "engineering_pairs_v3.csv"
FIXTURE_PROVENANCE_RELATIVE = (
    FIXTURE_ROOT_RELATIVE / "engineering_fixture_provenance_v3.json"
)
SHARED_ROOT_RELATIVE = Path("results/shared_accuracy/sourceafis_sift_v1")
PILOT_SELECTION_RELATIVE = Path(
    "results/pilots/sourceafis_joint_500_v1/selected_identities.csv"
)

EXPECTED_CONTRACT_SHA256 = (
    "a0278413a2e6eb2c5308642b90de86afe8bfcb00d783aa1b092d18ca510d68d3"
)
EXPECTED_SPLIT_SHA256 = (
    "19412c0edccf5ae4ab8e2246cd911a3aaf3d6e96ce060d9278762c05cae03bc0"
)
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = {
    "sd300b": "e5bf4de12f720e9d6b893bbcfa6816bb73d8a81288dd703362a453a2d263d75d",
    "sd300c": "801b3e610e59d9068ff8901b989adce0cbd20d0c2b246022c0589f9f19828172",
}
EXPECTED_PILOT_SELECTION_SHA256 = (
    "942363780986aab4b28df97ab67421ac8322ead5c9fd5131446f90eb8cdca7e9"
)
EXPECTED_PARENT_SHA256 = {
    "results/harriszplus_rootsift_geometric/preflight/engineering_preflight_failure.json": (
        "9b822a9a2bc0e67e8b0bf3d9658b55d84865bb6c64f3f82a5900debebbb8cd42"
    ),
    "results/harriszplus_rootsift_geometric/preflight/README.md": (
        "ea19481a2d73f0a9e63244b01d01668cccd240f2b639c4d0e67071d20d158fef"
    ),
    "docs/harriszplus_preflight_v1_failure_analysis.md": (
        "a17ae4eb55bb4664801d0da23412a61c609eb8b946deb3f97349a95073aeef24"
    ),
    "results/harriszplus_rootsift_geometric_v2/preflight/engineering_preflight_contract_v2.json": (
        "3dfa83653375763eedc9b6df40168d1d7b686d727613af54390f53c16540db9a"
    ),
    "results/harriszplus_rootsift_geometric_v2/preflight/engineering_preflight_failure.json": (
        "db18ba8747de4a436fcff78e259ca2d56aa0f639c4f9febe3bee0d2f52d953ce"
    ),
    "docs/harriszplus_preflight_v2_failure_analysis.md": (
        "13dd150d0c2cb2229737bb88b74d70afd12854cb291ba8c60758b9a767c80622"
    ),
}

THRESHOLD = 4
MAX_KEYPOINTS = 3000
TOP_RESPONSE_COUNT = 500
MINIMUM_EXACT_SCORE_FRACTION = 0.95
MAXIMUM_RAW_SCORE_DELTA = 1
PAIR_CLASSES = ("plain_self", "roll_self", "genuine", "negative")
VALID_STATUSES = (OK, PREPARE_A_FAILURE, PREPARE_B_FAILURE, COMPARISON_FAILURE)
NO_V4_RELAXATION_PATH = True

IDENTITY_COLUMNS = (
    "selection_index",
    "subject_id",
    "canonical_finger_position",
    "identity_key",
    "sd300b_plain_path",
    "sd300b_roll_path",
    "sd300b_plain_raw_frgp",
    "sd300b_roll_raw_frgp",
    "sd300c_plain_path",
    "sd300c_roll_path",
    "sd300c_plain_raw_frgp",
    "sd300c_roll_raw_frgp",
)
PAIR_COLUMNS = (
    "pair_id",
    "dataset",
    "pair_class",
    "subject_id_a",
    "subject_id_b",
    "canonical_finger_position",
    "ppi",
    "raw_frgp_a",
    "raw_frgp_b",
    "path_a",
    "path_b",
    "anchor_selection_index",
    "negative_shift",
)


class HarrisZPlusPreflightV3Error(ValueError):
    """Raised when v3 fixture or preflight evidence violates the frozen contract."""


@dataclass(frozen=True)
class EngineeringIdentity:
    selection_index: int
    subject_id: str
    canonical_finger_position: int
    identity_key: str
    sd300b_plain_path: Path
    sd300b_roll_path: Path
    sd300b_plain_raw_frgp: int
    sd300b_roll_raw_frgp: int
    sd300c_plain_path: Path
    sd300c_roll_path: Path
    sd300c_plain_raw_frgp: int
    sd300c_roll_raw_frgp: int


@dataclass(frozen=True)
class EngineeringPair:
    pair_id: str
    dataset: str
    pair_class: str
    subject_id_a: str
    subject_id_b: str
    canonical_finger_position: int
    ppi: int
    raw_frgp_a: int
    raw_frgp_b: int
    path_a: Path
    path_b: Path
    anchor_selection_index: int
    negative_shift: int

    def as_pair_record(self) -> PairRecord:
        return PairRecord(
            pair_id=self.pair_id,
            dataset=self.dataset,
            protocol=f"engineering_v3_{self.pair_class}",
            subject_id=self.subject_id_a,
            canonical_finger_position=self.canonical_finger_position,
            ppi=self.ppi,
            raw_frgp_a=self.raw_frgp_a,
            raw_frgp_b=self.raw_frgp_b,
            path_a=self.path_a,
            path_b=self.path_b,
        )


def _contract_path(project_root: Path) -> Path:
    return project_root.resolve() / CONTRACT_RELATIVE


def _validate_frozen_contract(project_root: Path) -> dict[str, Any]:
    path = _contract_path(project_root)
    actual = file_sha256(path)
    if actual != EXPECTED_CONTRACT_SHA256:
        raise HarrisZPlusPreflightV3Error(
            f"Frozen v3 contract changed: expected {EXPECTED_CONTRACT_SHA256}, got {actual}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "frozen"
        or payload.get("immutable") is not True
        or payload.get("frozen_before_fixture_materialization") is not True
        or payload.get("spearman_diagnostics", {}).get("is_gate") is not False
        or payload.get("failure_policy", {}).get("v4_relaxation_path_allowed")
        is not False
    ):
        raise HarrisZPlusPreflightV3Error("The v3 contract is not validly frozen.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _development_rows(project_root: Path) -> dict[str, dict[tuple[str, int], dict[str, str]]]:
    shared_root = project_root / SHARED_ROOT_RELATIVE
    split_path = shared_root / "shared_split_reference.json"
    if file_sha256(split_path) != EXPECTED_SPLIT_SHA256:
        raise HarrisZPlusPreflightV3Error("Shared development split changed.")
    output: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for dataset in DATASETS:
        path = shared_root / f"manifests/{dataset}_development_genuine.csv"
        if file_sha256(path) != EXPECTED_DEVELOPMENT_MANIFEST_SHA256[dataset]:
            raise HarrisZPlusPreflightV3Error(
                f"Development genuine manifest changed for {dataset}."
            )
        index: dict[tuple[str, int], dict[str, str]] = {}
        for row in _read_csv(path):
            key = (row["subject_id_a"], int(row["canonical_finger_position"]))
            if row["subject_id_a"] != row["subject_id_b"] or key in index:
                raise HarrisZPlusPreflightV3Error(
                    f"Invalid development genuine identity for {dataset}: {key}."
                )
            index[key] = row
        output[dataset] = index
    return output


def _pilot_identity_keys(project_root: Path) -> set[tuple[str, int]]:
    path = project_root / PILOT_SELECTION_RELATIVE
    if file_sha256(path) != EXPECTED_PILOT_SELECTION_SHA256:
        raise HarrisZPlusPreflightV3Error("Protected 500 selection changed.")
    return {
        (row["subject_id"], int(row["canonical_finger_position"]))
        for row in _read_csv(path)
    }


def _eligible_identities(
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[tuple[str, int], dict[str, str]]]]:
    development = _development_rows(project_root)
    pilot_keys = _pilot_identity_keys(project_root)
    common = sorted(
        (set(development["sd300b"]) & set(development["sd300c"])) - pilot_keys,
        key=lambda item: (item[0], item[1]),
    )
    eligible: list[dict[str, Any]] = []
    for subject_id, finger in common:
        b = development["sd300b"][(subject_id, finger)]
        c = development["sd300c"][(subject_id, finger)]
        paths = tuple(Path(value) for value in (b["path_a"], b["path_b"], c["path_a"], c["path_b"]))
        if not all(path.is_file() for path in paths):
            raise HarrisZPlusPreflightV3Error(
                f"Eligible engineering identity has a missing image: {subject_id}|{finger}."
            )
        eligible.append(
            {
                "subject_id": subject_id,
                "canonical_finger_position": finger,
                "identity_key": f"{subject_id}|{finger}",
                "sd300b": b,
                "sd300c": c,
            }
        )
    if len(eligible) < 10:
        raise HarrisZPlusPreflightV3Error("Fewer than ten external development identities exist.")
    return eligible, development


def _identity_rows(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors, start=1):
        b = anchor["sd300b"]
        c = anchor["sd300c"]
        rows.append(
            {
                "selection_index": index,
                "subject_id": anchor["subject_id"],
                "canonical_finger_position": anchor["canonical_finger_position"],
                "identity_key": anchor["identity_key"],
                "sd300b_plain_path": b["path_a"],
                "sd300b_roll_path": b["path_b"],
                "sd300b_plain_raw_frgp": b["raw_frgp_a"],
                "sd300b_roll_raw_frgp": b["raw_frgp_b"],
                "sd300c_plain_path": c["path_a"],
                "sd300c_roll_path": c["path_b"],
                "sd300c_plain_raw_frgp": c["raw_frgp_a"],
                "sd300c_roll_raw_frgp": c["raw_frgp_b"],
            }
        )
    return rows


def _next_subject_partner(
    anchor: Mapping[str, Any],
    eligible: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    finger = int(anchor["canonical_finger_position"])
    group = [
        row
        for row in eligible
        if int(row["canonical_finger_position"]) == finger
    ]
    anchor_index = next(
        index for index, row in enumerate(group) if row["identity_key"] == anchor["identity_key"]
    )
    for shift in range(1, len(group) + 1):
        candidate = group[(anchor_index + shift) % len(group)]
        if candidate["subject_id"] != anchor["subject_id"]:
            return candidate
    raise HarrisZPlusPreflightV3Error(
        f"No different-subject negative partner exists for {anchor['identity_key']}."
    )


def _pair_rows(
    anchors: Sequence[Mapping[str, Any]],
    eligible: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        ppi = 1000 if dataset == "sd300b" else 2000
        for pair_class, selected in (
            ("plain_self", anchors[:5]),
            ("roll_self", anchors[5:]),
            ("genuine", anchors),
            ("negative", anchors),
        ):
            for anchor in selected:
                selection_index = anchors.index(anchor) + 1
                source = anchor[dataset]
                if pair_class == "plain_self":
                    subject_b = anchor["subject_id"]
                    raw_a = raw_b = source["raw_frgp_a"]
                    path_a = path_b = source["path_a"]
                    shift = 0
                elif pair_class == "roll_self":
                    subject_b = anchor["subject_id"]
                    raw_a = raw_b = source["raw_frgp_b"]
                    path_a = path_b = source["path_b"]
                    shift = 0
                elif pair_class == "genuine":
                    subject_b = anchor["subject_id"]
                    raw_a = source["raw_frgp_a"]
                    raw_b = source["raw_frgp_b"]
                    path_a = source["path_a"]
                    path_b = source["path_b"]
                    shift = 0
                else:
                    partner = _next_subject_partner(anchor, eligible)
                    partner_source = partner[dataset]
                    subject_b = partner["subject_id"]
                    raw_a = source["raw_frgp_a"]
                    raw_b = partner_source["raw_frgp_b"]
                    path_a = source["path_a"]
                    path_b = partner_source["path_b"]
                    shift = 1
                finger = int(anchor["canonical_finger_position"])
                rows.append(
                    {
                        "pair_id": (
                            f"engineering_v3_{dataset}_{pair_class}_"
                            f"{selection_index:02d}_{anchor['subject_id']}_{finger:02d}"
                        ),
                        "dataset": dataset,
                        "pair_class": pair_class,
                        "subject_id_a": anchor["subject_id"],
                        "subject_id_b": subject_b,
                        "canonical_finger_position": finger,
                        "ppi": ppi,
                        "raw_frgp_a": raw_a,
                        "raw_frgp_b": raw_b,
                        "path_a": path_a,
                        "path_b": path_b,
                        "anchor_selection_index": selection_index,
                        "negative_shift": shift,
                    }
                )
    return rows


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in columns})
    return stream.getvalue().encode("utf-8")


def _publish_exclusive_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def prepare_engineering_fixtures(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    """Materialize the score-independent v3 identities and 60 pairs once."""

    project_root = project_root.resolve()
    _validate_frozen_contract(project_root)
    paths = {
        "identities": project_root / IDENTITIES_RELATIVE,
        "pairs": project_root / PAIRS_RELATIVE,
        "provenance": project_root / FIXTURE_PROVENANCE_RELATIVE,
    }
    existing = [path.exists() for path in paths.values()]
    eligible, _ = _eligible_identities(project_root)
    anchors = eligible[:10]
    identity_rows = _identity_rows(anchors)
    pair_rows = _pair_rows(anchors, eligible)
    identities_bytes = _csv_bytes(identity_rows, IDENTITY_COLUMNS)
    pairs_bytes = _csv_bytes(pair_rows, PAIR_COLUMNS)
    identities_sha = hashlib.sha256(identities_bytes).hexdigest()
    pairs_sha = hashlib.sha256(pairs_bytes).hexdigest()
    provenance = {
        "schema_version": "harriszplus-engineering-fixture-v3",
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "selection_rule": (
            "first 10 eligible development fingerprint identities sorted by "
            "subject_id then canonical_finger_position"
        ),
        "eligible_identity_count": len(eligible),
        "pilot_identity_exclusion_count": 500,
        "identity_count": len(identity_rows),
        "pair_count": len(pair_rows),
        "pair_count_by_dataset_and_class": {
            dataset: {
                pair_class: sum(
                    row["dataset"] == dataset and row["pair_class"] == pair_class
                    for row in pair_rows
                )
                for pair_class in PAIR_CLASSES
            }
            for dataset in DATASETS
        },
        "negative_pairing": (
            "next different subject in the global eligible same-canonical-finger "
            "development pool, circular shift=1"
        ),
        "identities_sha256": identities_sha,
        "pairs_sha256": pairs_sha,
        "source_sha256": {
            "shared_split": EXPECTED_SPLIT_SHA256,
            "development_genuine": EXPECTED_DEVELOPMENT_MANIFEST_SHA256,
            "pilot_selection_exclusion": EXPECTED_PILOT_SELECTION_SHA256,
        },
        "selection_used_harriszplus_scores": False,
        "fixture_frozen_before_harriszplus_score": True,
    }
    expected = {
        "identities": identities_bytes,
        "pairs": pairs_bytes,
        "provenance": _json_bytes(provenance),
    }
    if any(existing):
        if not all(existing):
            raise HarrisZPlusPreflightV3Error("Partial immutable v3 fixture exists.")
        for key, path in paths.items():
            if path.read_bytes() != expected[key]:
                raise HarrisZPlusPreflightV3Error(
                    f"Existing v3 fixture changed: {path}."
                )
    else:
        for key in ("identities", "pairs", "provenance"):
            _publish_exclusive_bytes(paths[key], expected[key])
    return {
        **provenance,
        "paths": {key: str(path) for key, path in paths.items()},
        "artifact_sha256": {
            key: file_sha256(path) for key, path in paths.items()
        },
    }


def load_engineering_identities(
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> list[EngineeringIdentity]:
    path = project_root.resolve() / IDENTITIES_RELATIVE
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != IDENTITY_COLUMNS:
            raise HarrisZPlusPreflightV3Error("Engineering identity fixture schema changed.")
        rows = [
            EngineeringIdentity(
                selection_index=int(row["selection_index"]),
                subject_id=row["subject_id"],
                canonical_finger_position=int(row["canonical_finger_position"]),
                identity_key=row["identity_key"],
                sd300b_plain_path=Path(row["sd300b_plain_path"]),
                sd300b_roll_path=Path(row["sd300b_roll_path"]),
                sd300b_plain_raw_frgp=int(row["sd300b_plain_raw_frgp"]),
                sd300b_roll_raw_frgp=int(row["sd300b_roll_raw_frgp"]),
                sd300c_plain_path=Path(row["sd300c_plain_path"]),
                sd300c_roll_path=Path(row["sd300c_roll_path"]),
                sd300c_plain_raw_frgp=int(row["sd300c_plain_raw_frgp"]),
                sd300c_roll_raw_frgp=int(row["sd300c_roll_raw_frgp"]),
            )
            for row in reader
        ]
    if [row.selection_index for row in rows] != list(range(1, 11)):
        raise HarrisZPlusPreflightV3Error("Engineering selection order changed.")
    return rows


def load_engineering_pairs(
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> list[EngineeringPair]:
    path = project_root.resolve() / PAIRS_RELATIVE
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            raise HarrisZPlusPreflightV3Error("Engineering pair fixture schema changed.")
        rows = [
            EngineeringPair(
                pair_id=row["pair_id"],
                dataset=row["dataset"],
                pair_class=row["pair_class"],
                subject_id_a=row["subject_id_a"],
                subject_id_b=row["subject_id_b"],
                canonical_finger_position=int(row["canonical_finger_position"]),
                ppi=int(row["ppi"]),
                raw_frgp_a=int(row["raw_frgp_a"]),
                raw_frgp_b=int(row["raw_frgp_b"]),
                path_a=Path(row["path_a"]),
                path_b=Path(row["path_b"]),
                anchor_selection_index=int(row["anchor_selection_index"]),
                negative_shift=int(row["negative_shift"]),
            )
            for row in reader
        ]
    if len(rows) != 60 or len({row.pair_id for row in rows}) != 60:
        raise HarrisZPlusPreflightV3Error("Engineering pair fixture is not exact 60.")
    _validate_pair_fixture(rows)
    return rows


def _validate_pair_fixture(rows: Sequence[EngineeringPair]) -> None:
    for dataset in DATASETS:
        for pair_class, expected in (
            ("plain_self", 5),
            ("roll_self", 5),
            ("genuine", 10),
            ("negative", 10),
        ):
            selected = [
                row
                for row in rows
                if row.dataset == dataset and row.pair_class == pair_class
            ]
            if len(selected) != expected:
                raise HarrisZPlusPreflightV3Error(
                    f"Wrong fixture count for {dataset}/{pair_class}."
                )
            for row in selected:
                if not row.path_a.is_file() or not row.path_b.is_file():
                    raise HarrisZPlusPreflightV3Error("Fixture image is missing.")
                if pair_class in ("plain_self", "roll_self") and (
                    row.path_a != row.path_b
                    or row.subject_id_a != row.subject_id_b
                ):
                    raise HarrisZPlusPreflightV3Error("Invalid self fixture.")
                if pair_class == "genuine" and row.subject_id_a != row.subject_id_b:
                    raise HarrisZPlusPreflightV3Error("Invalid genuine fixture.")
                if pair_class == "negative" and (
                    row.subject_id_a == row.subject_id_b
                    or row.negative_shift != 1
                ):
                    raise HarrisZPlusPreflightV3Error("Invalid negative fixture.")


def _algorithm_identity(project_root: Path) -> dict[str, Any]:
    actual_sources = implementation_source_hashes(strict=True)[
        "required_score_producing_sources"
    ]
    adapter = HarrisZPlusGeometricAdapter(
        HarrisZPlusConfig(backend="cuda", device="cuda:0")
    )
    try:
        config_hash = stable_config_hash(_effective_runner_config(adapter.metadata()))
    finally:
        adapter.close()
    parent_actual = {
        path: file_sha256(project_root / path) for path in EXPECTED_PARENT_SHA256
    }
    return {
        "algorithm_sources_byte_exact_to_v2": (
            actual_sources == EXPECTED_V1_ALGORITHM_SOURCE_SHA256
        ),
        "algorithm_source_sha256": actual_sources,
        "expected_algorithm_source_sha256": EXPECTED_V1_ALGORITHM_SOURCE_SHA256,
        "candidate_config_sha256": config_hash,
        "candidate_config_unchanged": (
            config_hash == EXPECTED_CANDIDATE_CONFIG_SHA256
        ),
        "parent_artifact_sha256": parent_actual,
        "parent_artifacts_unchanged": parent_actual == EXPECTED_PARENT_SHA256,
        "algorithm_changed": not (
            actual_sources == EXPECTED_V1_ALGORITHM_SOURCE_SHA256
            and config_hash == EXPECTED_CANDIDATE_CONFIG_SHA256
        ),
        "validation_contract_only_changed": True,
    }


def _spearman_diagnostics(
    cpu_points: Sequence[Mapping[str, Any]],
    cuda_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = _directional_keypoint_matches(cpu_points, cuda_points)
    cpu_order = _canonical_validation_keypoint_index_order(cpu_points)
    cuda_order = _canonical_validation_keypoint_index_order(cuda_points)
    cpu_rank = {raw: rank for rank, raw in enumerate(cpu_order)}
    cuda_rank = {raw: rank for rank, raw in enumerate(cuda_order)}
    rank_rows = [
        (
            cpu_index,
            cuda_index,
            cpu_rank[cpu_index],
            cuda_rank[cuda_index],
        )
        for cpu_index, cuda_index, _, _ in matches
    ]
    global_rho = _spearman_rank_correlation(
        [(row[2], row[3]) for row in rank_rows]
    )
    scales = sorted(
        {
            int(cpu_points[row[0]]["scale_index"])
            for row in rank_rows
        }
    )
    per_scale: dict[str, Any] = {}
    for scale in scales:
        scale_rows = [
            (row[2], row[3])
            for row in rank_rows
            if int(cpu_points[row[0]]["scale_index"]) == scale
        ]
        per_scale[str(scale)] = {
            "matched_count": len(scale_rows),
            "spearman": _spearman_rank_correlation(scale_rows),
        }
    top = [
        (row[2], row[3])
        for row in rank_rows
        if row[2] < min(TOP_RESPONSE_COUNT, len(cpu_points))
    ]
    return {
        "is_gate": False,
        "minimum_pass_threshold": None,
        "global_spearman": global_rho,
        "per_scale": per_scale,
        "top_response_count": len(top),
        "top_response_spearman": _spearman_rank_correlation(top),
    }


def _synthetic_comparison(cpu: Any, cuda: Any) -> dict[str, Any]:
    responses = _response_map_comparison(
        cpu,
        cuda,
        minimum_pixel_coverage=SYNTHETIC_MINIMUM_RESPONSE_PIXEL_COVERAGE,
    )
    candidates = compare_candidate_counts_v2(cpu, cuda)
    final = compare_final_keypoints_v2(cpu, cuda)
    points_cpu = [_keypoint_record(point) for point in _keypoints(cpu)]
    points_cuda = [_keypoint_record(point) for point in _keypoints(cuda)]
    spearman = _spearman_diagnostics(points_cpu, points_cuda)
    semantic_final_pass = bool(
        final["count_gate_passed"]
        and final["bidirectional_matching_passed"]
    )
    return {
        "response_maps": responses,
        "intermediate_candidate_counts": candidates,
        "final_keypoint_equivalence": final,
        "spearman_diagnostics": spearman,
        "spearman_is_gate": False,
        "passed": bool(
            responses["passed"] and candidates["passed"] and semantic_final_pass
        ),
    }


def _prepare_record(outcome: Any, *, expected_backend: str) -> dict[str, Any]:
    representation = outcome.representation
    payload = representation.payload
    descriptors = np.asarray(payload.descriptors)
    points = np.asarray(payload.points)
    sizes = np.asarray(payload.sizes)
    finite = bool(
        np.isfinite(descriptors).all()
        and np.isfinite(points).all()
        and np.isfinite(sizes).all()
    )
    coordinates = bool(
        points.ndim == 2
        and points.shape[1] == 2
        and np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] < int(payload.width))
        and np.all(points[:, 1] >= 0.0)
        and np.all(points[:, 1] < int(payload.height))
    )
    backend = outcome.diagnostics.get("detector_backend")
    return {
        "status": OK,
        "failure_stage": None,
        "outcome": outcome,
        "representation_sha256": representation_sha256(representation),
        "deterministic_diagnostics_sha256": stable_hash(
            _without_timing_fields(outcome.diagnostics)
        ),
        "descriptor_count": int(descriptors.shape[0]),
        "descriptor_available": int(descriptors.shape[0]) >= 2,
        "keypoint_count": int(points.shape[0]),
        "descriptors_finite": finite,
        "coordinates_within_image_bounds": coordinates,
        "positive_scales": bool(np.all(sizes > 0.0)),
        "keypoint_cap_ok": int(points.shape[0]) <= MAX_KEYPOINTS,
        "detector_backend": backend,
        "hidden_cpu_fallback_absent": backend == expected_backend,
        "candidates_after_duplicate_removal": int(
            outcome.diagnostics.get("candidates_after_duplicate_removal", -1)
        ),
        "prepare_total_ms": _finite_number(
            outcome.diagnostics.get("prepare_total_ms")
        ),
        "detector_gpu_wall_ms": _finite_number(
            outcome.diagnostics.get("detector_gpu_wall_ms")
        ),
        "descriptor_cpu_ms": _finite_number(
            outcome.diagnostics.get("descriptor_cpu_ms")
        ),
        "peak_vram_allocated": _finite_number(
            outcome.diagnostics.get("peak_vram_allocated")
        ),
        "peak_vram_reserved": _finite_number(
            outcome.diagnostics.get("peak_vram_reserved")
        ),
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _safe_prepare(
    adapter: HarrisZPlusGeometricAdapter,
    *,
    path: Path,
    metadata: Mapping[str, Any],
    expected_backend: str,
) -> dict[str, Any]:
    try:
        outcome = adapter.prepare(path, metadata)
    except MethodExecutionError as exc:
        return {
            "status": PREPARE_A_FAILURE,
            "failure_stage": "prepare",
            "error_code": exc.error_code,
            "error_message": exc.message,
            "outcome": None,
            "descriptor_count": 0,
            "descriptor_available": False,
            "representation_sha256": None,
            "deterministic_diagnostics_sha256": stable_hash(
                _without_timing_fields(exc.diagnostics)
            ),
        }
    return _prepare_record(outcome, expected_backend=expected_backend)


def _representation_points(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcome = record.get("outcome")
    if outcome is None:
        return []
    payload = outcome.representation.payload
    source_indices = payload.metadata.get(
        "harriszplus_source_indices", list(range(payload.keypoint_count))
    )
    return [
        {
            "x": float(payload.points[index, 0]),
            "y": float(payload.points[index, 1]),
            "response": float(payload.responses[index]),
            "scale_index": int(payload.class_ids[index]),
            "sigma": float(payload.sizes[index]),
            "size": float(payload.sizes[index]),
            "source_index": int(source_indices[index]),
        }
        for index in range(payload.keypoint_count)
    ]


def top_k_equivalence(
    cpu_record: Mapping[str, Any],
    cuda_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen cap-active top-3000 set overlap gate."""

    cpu_pre_cap = int(cpu_record.get("candidates_after_duplicate_removal", -1))
    cuda_pre_cap = int(cuda_record.get("candidates_after_duplicate_removal", -1))
    active = cpu_pre_cap > MAX_KEYPOINTS and cuda_pre_cap > MAX_KEYPOINTS
    if not active:
        return {
            "applicable": False,
            "reason": "both_backends_not_above_3000_before_cap",
            "cpu_candidates_before_cap": cpu_pre_cap,
            "cuda_candidates_before_cap": cuda_pre_cap,
            "passed": True,
        }
    cpu_points = _representation_points(cpu_record)
    cuda_points = _representation_points(cuda_record)
    forward = _directional_keypoint_matches(cpu_points, cuda_points)
    reverse = _directional_keypoint_matches(cuda_points, cpu_points)
    cpu_fraction = len(forward) / len(cpu_points) if cpu_points else 0.0
    cuda_fraction = len(reverse) / len(cuda_points) if cuda_points else 0.0
    overlap = min(cpu_fraction, cuda_fraction)
    cpu_matched = {row[0] for row in forward}
    cuda_matched = {row[0] for row in reverse}
    cpu_only = set(range(len(cpu_points))) - cpu_matched
    cuda_only = set(range(len(cuda_points))) - cuda_matched
    cpu_unique_source_indices = {
        int(point["source_index"]) for point in cpu_points
    }
    cuda_unique_source_indices = {
        int(point["source_index"]) for point in cuda_points
    }
    cutoff_start = math.floor(0.9 * MAX_KEYPOINTS)
    counts_exact = len(cpu_points) == len(cuda_points) == MAX_KEYPOINTS
    entries_unique = bool(
        len(cpu_unique_source_indices) == len(cpu_points)
        and len(cuda_unique_source_indices) == len(cuda_points)
    )
    return {
        "applicable": True,
        "cpu_candidates_before_cap": cpu_pre_cap,
        "cuda_candidates_before_cap": cuda_pre_cap,
        "cpu_top_k_count": len(cpu_points),
        "cuda_top_k_count": len(cuda_points),
        "counts_exact_3000_3000": counts_exact,
        "cpu_unique_source_index_count": len(cpu_unique_source_indices),
        "cuda_unique_source_index_count": len(cuda_unique_source_indices),
        "top_k_entries_unique": entries_unique,
        "cpu_to_cuda_matched_count": len(forward),
        "cuda_to_cpu_matched_count": len(reverse),
        "bidirectional_overlap": overlap,
        "minimum_bidirectional_overlap": MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION,
        "cpu_only_count": len(cpu_only),
        "cuda_only_count": len(cuda_only),
        "cpu_only_in_cutoff_region_count": sum(
            index >= cutoff_start for index in cpu_only
        ),
        "cuda_only_in_cutoff_region_count": sum(
            index >= cutoff_start for index in cuda_only
        ),
        "cutoff_region_start_rank_zero_based": cutoff_start,
        "differences_concentrated_around_cutoff": bool(
            (not cpu_only or all(index >= cutoff_start for index in cpu_only))
            and (not cuda_only or all(index >= cutoff_start for index in cuda_only))
        ),
        "passed": bool(
            counts_exact
            and entries_unique
            and overlap >= MINIMUM_BIDIRECTIONAL_MATCHED_FRACTION
        ),
    }


def _real_representation_comparison(
    cpu: Mapping[str, Any],
    cuda: Mapping[str, Any],
) -> dict[str, Any]:
    status_equal = cpu["status"] == cuda["status"]
    descriptor_availability_equal = (
        cpu["descriptor_available"] == cuda["descriptor_available"]
    )
    if cpu.get("outcome") is None or cuda.get("outcome") is None:
        return {
            "status_equal": status_equal,
            "descriptor_availability_equal": descriptor_availability_equal,
            "passed": False,
            "reason": "prepare_failure",
        }
    cpu_points = _representation_points(cpu)
    cuda_points = _representation_points(cuda)
    final = compare_final_keypoints_v2(
        SimpleNamespace(keypoints=cpu_points),
        SimpleNamespace(keypoints=cuda_points),
    )
    spearman = _spearman_diagnostics(cpu_points, cuda_points)
    descriptor_count = count_equivalence(
        int(cpu["descriptor_count"]), int(cuda["descriptor_count"])
    )
    top_k = top_k_equivalence(cpu, cuda)
    absolute = all(
        bool(row.get(field))
        for row in (cpu, cuda)
        for field in (
            "descriptors_finite",
            "coordinates_within_image_bounds",
            "positive_scales",
            "keypoint_cap_ok",
            "hidden_cpu_fallback_absent",
        )
    )
    semantic_final = bool(
        final["count_gate_passed"] and final["bidirectional_matching_passed"]
    )
    return {
        "status_equal": status_equal,
        "descriptor_availability_equal": descriptor_availability_equal,
        "descriptor_count_equivalence": descriptor_count,
        "final_keypoint_equivalence": final,
        "top_k_equivalence": top_k,
        "spearman_diagnostics": spearman,
        "spearman_is_gate": False,
        "absolute_conditions_passed": absolute,
        "passed": bool(
            status_equal
            and descriptor_availability_equal
            and descriptor_count["passed"]
            and semantic_final
            and top_k["passed"]
            and absolute
        ),
    }


def _prepared_for_pair(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "descriptor_count": int(record.get("descriptor_count", 0)),
        "representation_sha256": record.get("representation_sha256"),
    }


def _pair_failure(status: str, stage: str, record: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "status": status,
        "failure_stage": stage,
        "error_code": record.get("error_code"),
        "error_message": record.get("error_message"),
        "raw_score": None,
        "decision_threshold_4": False,
        "prepare_a": None,
        "prepare_b": None,
    }
    payload["payload_sha256"] = stable_hash(payload)
    return payload


def _compare_prepared_pair(
    adapter: HarrisZPlusGeometricAdapter,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    if first.get("outcome") is None:
        return _pair_failure(PREPARE_A_FAILURE, "prepare_a", first)
    if second.get("outcome") is None:
        return _pair_failure(PREPARE_B_FAILURE, "prepare_b", second)
    try:
        comparison = adapter.compare(
            first["outcome"].representation,
            second["outcome"].representation,
        )
    except MethodExecutionError as exc:
        payload = {
            "status": COMPARISON_FAILURE,
            "failure_stage": "compare",
            "error_code": exc.error_code,
            "error_message": exc.message,
            "raw_score": None,
            "decision_threshold_4": False,
            "prepare_a": _prepared_for_pair(first),
            "prepare_b": _prepared_for_pair(second),
        }
        payload["payload_sha256"] = stable_hash(payload)
        return payload
    score = int(comparison.raw_score)
    payload = {
        "status": OK,
        "failure_stage": None,
        "raw_score": score,
        "decision_threshold_4": score >= THRESHOLD,
        "prepare_a": _prepared_for_pair(first),
        "prepare_b": _prepared_for_pair(second),
        "compare_deterministic_diagnostics_sha256": stable_hash(
            _without_timing_fields(comparison.diagnostics)
        ),
        "compare_total_ms": _finite_number(
            comparison.diagnostics.get("compare_total_ms")
        ),
    }
    payload["payload_sha256"] = stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "compare_total_ms"
        }
    )
    return payload


def exact_score_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 1.0
    return sum(
        row["cpu_cuda_semantic_equivalence"]["raw_score_exact"] for row in rows
    ) / len(rows)


def aggregate_decision_equivalence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    passed = True
    for dataset in DATASETS:
        for pair_class in PAIR_CLASSES:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["pair_class"] == pair_class
            ]
            cpu_accepted = sum(row["cpu"]["decision_threshold_4"] for row in selected)
            cuda_accepted = sum(
                row["cuda_first"]["decision_threshold_4"] for row in selected
            )
            result = {
                "pair_count": len(selected),
                "cpu_accepted": cpu_accepted,
                "cpu_rejected": len(selected) - cpu_accepted,
                "cuda_accepted": cuda_accepted,
                "cuda_rejected": len(selected) - cuda_accepted,
                "passed": cpu_accepted == cuda_accepted,
            }
            groups[f"{dataset}/{pair_class}"] = result
            passed = passed and result["passed"]
    return {"groups": groups, "passed": bool(passed)}


def _timing_report(
    real_records: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        records = [
            row["cuda_first"]
            for row in real_records
            if row["dataset"] == dataset and row["cuda_first"].get("outcome") is not None
        ]
        pairs = [
            row["cuda_first"]
            for row in pair_rows
            if row["dataset"] == dataset and row["cuda_first"]["status"] == OK
        ]
        prepares = [
            float(row["prepare_total_ms"])
            for row in records
            if row.get("prepare_total_ms") is not None
        ]
        detectors = [
            float(row["detector_gpu_wall_ms"])
            for row in records
            if row.get("detector_gpu_wall_ms") is not None
        ]
        descriptors = [
            float(row["descriptor_cpu_ms"])
            for row in records
            if row.get("descriptor_cpu_ms") is not None
        ]
        compares = [
            float(row["compare_total_ms"])
            for row in pairs
            if row.get("compare_total_ms") is not None
        ]
        median_prepare = statistics.median(prepares) if prepares else None
        median_compare = statistics.median(compares) if compares else None
        projected = (
            500.0 * (2.0 * median_prepare + median_compare)
            if median_prepare is not None and median_compare is not None
            else None
        )
        by_dataset[dataset] = {
            "median_prepare_ms": median_prepare,
            "median_detector_ms": statistics.median(detectors)
            if detectors
            else None,
            "median_descriptor_ms": statistics.median(descriptors)
            if descriptors
            else None,
            "median_compare_ms": median_compare,
            "projected_500_pair_run_ms": projected,
        }
    peaks = [
        float(record[key])
        for row in real_records
        for record in (row["cuda_first"], row["cuda_repeat"])
        for key in ("peak_vram_allocated", "peak_vram_reserved")
        if record.get(key) is not None
    ]
    b = by_dataset["sd300b"]["projected_500_pair_run_ms"]
    c = by_dataset["sd300c"]["projected_500_pair_run_ms"]
    total = 4.0 * float(b) + 4.0 * float(c) if b is not None and c is not None else None
    return {
        "information_only_not_correctness_gate": True,
        "datasets": by_dataset,
        "peak_vram_bytes": max(peaks, default=None),
        "projected_all_eight_runs_ms": total,
        "projected_all_eight_runs_hours": (
            total / 3_600_000.0 if total is not None else None
        ),
    }


def _real_image_metadata(
    pairs: Sequence[EngineeringPair],
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        record = pair.as_pair_record()
        for side, path, raw in (
            ("a", pair.path_a, pair.raw_frgp_a),
            ("b", pair.path_b, pair.raw_frgp_b),
        ):
            key = str(path.resolve()).lower()
            candidate = {
                "path": path,
                "dataset": pair.dataset,
                "ppi": pair.ppi,
                "raw_frgp": raw,
                "subject_id": pair.subject_id_a if side == "a" else pair.subject_id_b,
                "canonical_finger_position": pair.canonical_finger_position,
                "metadata": (
                    record.image_metadata_a()
                    if side == "a"
                    else record.image_metadata_b()
                ),
            }
            existing = by_path.get(key)
            if existing is not None and (
                existing["dataset"] != candidate["dataset"]
                or existing["ppi"] != candidate["ppi"]
                or existing["raw_frgp"] != candidate["raw_frgp"]
            ):
                raise HarrisZPlusPreflightV3Error(
                    f"Inconsistent metadata for engineering image {path}."
                )
            by_path[key] = candidate
    return sorted(
        by_path.values(),
        key=lambda row: (row["dataset"], str(row["path"]).lower()),
    )


def run_engineering_preflight_v3(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    """Execute every frozen v3 gate without reading any 500-pilot result."""

    project_root = project_root.resolve()
    data_root = data_root.resolve()
    contract = _validate_frozen_contract(project_root)
    fixture = prepare_engineering_fixtures(project_root=project_root)
    identities = load_engineering_identities(project_root)
    pairs = load_engineering_pairs(project_root)
    identity = _algorithm_identity(project_root)
    if identity["algorithm_changed"] or not identity["parent_artifacts_unchanged"]:
        raise HarrisZPlusPreflightV3Error(
            "Algorithm or protected v1/v2 artifacts changed before v3 preflight."
        )
    torch, environment = _configure_and_describe_cuda("cuda:0")
    cpu_config = HarrisZPlusConfig(backend="reference_cpu", device=None)
    cuda_config = HarrisZPlusConfig(backend="cuda", device="cuda:0")

    synthetic_rows: list[dict[str, Any]] = []
    for name, source in synthetic_suite().items():
        source_u8 = np.ascontiguousarray(
            np.clip(np.rint(source), 0.0, 255.0), dtype=np.uint8
        )
        image = source_u8.astype(np.float32)
        doubled = cv2.resize(
            source_u8,
            (source_u8.shape[1] * 2, source_u8.shape[0] * 2),
            interpolation=cv2.INTER_LANCZOS4,
        ).astype(np.float32, copy=False)
        cpu = detect_harriszplus_cpu(
            image,
            cpu_config,
            doubled_image=doubled,
            return_response_maps=True,
        )
        cuda_first = detect_harriszplus_cuda(
            image,
            cuda_config,
            device="cuda:0",
            doubled_image=doubled,
            return_response_maps=True,
        )
        torch.cuda.synchronize("cuda:0")
        cuda_repeat = detect_harriszplus_cuda(
            image,
            cuda_config,
            device="cuda:0",
            doubled_image=doubled,
            return_response_maps=True,
        )
        torch.cuda.synchronize("cuda:0")
        comparison = _synthetic_comparison(cpu, cuda_first)
        cpu_absolute = _detector_absolute_conditions(
            cpu,
            expected_backend="reference_cpu",
            image_shape=image.shape,
        )
        cuda_absolute = _detector_absolute_conditions(
            cuda_first,
            expected_backend="cuda",
            image_shape=image.shape,
        )
        first_hash = detector_result_sha256(cuda_first)
        repeat_hash = detector_result_sha256(cuda_repeat)
        repeat_exact = first_hash == repeat_hash
        flat_zero = (
            len(_keypoints(cpu)) == len(_keypoints(cuda_first)) == 0
            if name == "flat"
            else None
        )
        synthetic_rows.append(
            {
                "case_id": f"synthetic:{name}",
                "cpu_keypoint_count": len(_keypoints(cpu)),
                "cuda_keypoint_count": len(_keypoints(cuda_first)),
                "cpu_cuda": comparison,
                "cpu_absolute_conditions": cpu_absolute,
                "cuda_absolute_conditions": cuda_absolute,
                "cuda_first_sha256": first_hash,
                "cuda_repeat_sha256": repeat_hash,
                "cuda_repeat_exact": repeat_exact,
                "flat_zero_keypoints": flat_zero,
                "passed": bool(
                    comparison["passed"]
                    and cpu_absolute["passed"]
                    and cuda_absolute["passed"]
                    and repeat_exact
                    and flat_zero is not False
                ),
            }
        )

    real_images = _real_image_metadata(pairs)
    cpu_adapter = HarrisZPlusGeometricAdapter(cpu_config)
    cuda_adapter = HarrisZPlusGeometricAdapter(cuda_config)
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    real_rows: list[dict[str, Any]] = []
    try:
        for image in real_images:
            metadata = image["metadata"]
            path = image["path"]
            cpu = _safe_prepare(
                cpu_adapter,
                path=path,
                metadata=metadata,
                expected_backend="reference_cpu",
            )
            cuda_first = _safe_prepare(
                cuda_adapter,
                path=path,
                metadata=metadata,
                expected_backend="cuda",
            )
            cuda_repeat = _safe_prepare(
                cuda_adapter,
                path=path,
                metadata=metadata,
                expected_backend="cuda",
            )
            comparison = _real_representation_comparison(cpu, cuda_first)
            repeat_exact = bool(
                cuda_first.get("status") == cuda_repeat.get("status")
                and cuda_first.get("failure_stage") == cuda_repeat.get("failure_stage")
                and cuda_first.get("representation_sha256")
                == cuda_repeat.get("representation_sha256")
                and cuda_first.get("deterministic_diagnostics_sha256")
                == cuda_repeat.get("deterministic_diagnostics_sha256")
            )
            key = str(path.resolve()).lower()
            prepared[("cpu", key)] = cpu
            prepared[("cuda_first", key)] = cuda_first
            prepared[("cuda_repeat", key)] = cuda_repeat
            real_rows.append(
                {
                    "path": str(path),
                    "dataset": image["dataset"],
                    "subject_id": image["subject_id"],
                    "canonical_finger_position": image[
                        "canonical_finger_position"
                    ],
                    "cpu": _json_record(cpu),
                    "cuda_first": _json_record(cuda_first),
                    "cuda_repeat": _json_record(cuda_repeat),
                    "cpu_cuda": comparison,
                    "cuda_repeat_exact": repeat_exact,
                    "passed": bool(comparison["passed"] and repeat_exact),
                }
            )

        pair_rows: list[dict[str, Any]] = []
        for pair in pairs:
            a_key = str(pair.path_a.resolve()).lower()
            b_key = str(pair.path_b.resolve()).lower()
            cpu_result = _compare_prepared_pair(
                cpu_adapter,
                prepared[("cpu", a_key)],
                prepared[("cpu", b_key)],
            )
            cuda_first_result = _compare_prepared_pair(
                cuda_adapter,
                prepared[("cuda_first", a_key)],
                prepared[("cuda_first", b_key)],
            )
            cuda_repeat_result = _compare_prepared_pair(
                cuda_adapter,
                prepared[("cuda_repeat", a_key)],
                prepared[("cuda_repeat", b_key)],
            )
            semantic = _semantic_pair_comparison(cpu_result, cuda_first_result)
            repeat = _cuda_repeat_comparison(
                cuda_first_result, cuda_repeat_result
            )
            pair_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "pair_class": pair.pair_class,
                    "subject_id_a": pair.subject_id_a,
                    "subject_id_b": pair.subject_id_b,
                    "canonical_finger_position": pair.canonical_finger_position,
                    "cpu": cpu_result,
                    "cuda_first": cuda_first_result,
                    "cuda_repeat": cuda_repeat_result,
                    "cpu_cuda_semantic_equivalence": semantic,
                    "cuda_repeat_exact": repeat,
                    "passed": bool(semantic["passed"] and repeat["passed"]),
                }
            )
    finally:
        cpu_adapter.close()
        cuda_adapter.close()

    exact_rate = exact_score_rate(pair_rows)
    maximum_delta = max(
        (
            row["cpu_cuda_semantic_equivalence"]["raw_score_absolute_delta"]
            for row in pair_rows
            if row["cpu_cuda_semantic_equivalence"][
                "raw_score_absolute_delta"
            ]
            is not None
        ),
        default=0,
    )
    aggregate = aggregate_decision_equivalence(pair_rows)
    downstream = {
        "pair_count": len(pair_rows),
        "status_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["status_equal"]
            for row in pair_rows
        ),
        "failure_stage_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["failure_stage_equal"]
            for row in pair_rows
        ),
        "decision_all_equal": all(
            row["cpu_cuda_semantic_equivalence"]["decision_threshold_4_equal"]
            for row in pair_rows
        ),
        "one_backend_failure_absent": all(
            (row["cpu"]["status"] == OK) == (row["cuda_first"]["status"] == OK)
            for row in pair_rows
        ),
        "exact_score_count": sum(
            row["cpu_cuda_semantic_equivalence"]["raw_score_exact"]
            for row in pair_rows
        ),
        "exact_score_rate": exact_rate,
        "minimum_exact_score_rate": MINIMUM_EXACT_SCORE_FRACTION,
        "maximum_raw_score_absolute_delta": maximum_delta,
        "maximum_allowed_raw_score_absolute_delta": MAXIMUM_RAW_SCORE_DELTA,
        "aggregate_decision_equivalence": aggregate,
    }
    downstream["passed"] = bool(
        downstream["pair_count"] == 60
        and downstream["status_all_equal"]
        and downstream["failure_stage_all_equal"]
        and downstream["decision_all_equal"]
        and downstream["one_backend_failure_absent"]
        and exact_rate >= MINIMUM_EXACT_SCORE_FRACTION
        and maximum_delta <= MAXIMUM_RAW_SCORE_DELTA
        and aggregate["passed"]
        and all(row["passed"] for row in pair_rows)
    )
    reproducibility = {
        "synthetic_all_exact": all(
            row["cuda_repeat_exact"] for row in synthetic_rows
        ),
        "real_representations_all_exact": all(
            row["cuda_repeat_exact"] for row in real_rows
        ),
        "real_comparisons_all_exact": all(
            row["cuda_repeat_exact"]["passed"] for row in pair_rows
        ),
        "synthetic_representations_sha256": stable_hash(
            [
                {
                    "case_id": row["case_id"],
                    "first": row["cuda_first_sha256"],
                    "repeat": row["cuda_repeat_sha256"],
                }
                for row in synthetic_rows
            ]
        ),
        "real_representations_sha256": stable_hash(
            [
                {
                    "path": row["path"],
                    "first": row["cuda_first"]["representation_sha256"],
                    "repeat": row["cuda_repeat"]["representation_sha256"],
                }
                for row in real_rows
            ]
        ),
        "real_comparisons_sha256": stable_hash(
            [
                {
                    "pair_id": row["pair_id"],
                    "first": row["cuda_first"]["payload_sha256"],
                    "repeat": row["cuda_repeat"]["payload_sha256"],
                }
                for row in pair_rows
            ]
        ),
    }
    reproducibility["passed"] = all(
        reproducibility[key]
        for key in (
            "synthetic_all_exact",
            "real_representations_all_exact",
            "real_comparisons_all_exact",
        )
    )
    top_k_applicable = [
        row["cpu_cuda"]["top_k_equivalence"]
        for row in real_rows
        if row["cpu_cuda"].get("top_k_equivalence", {}).get("applicable")
    ]
    correctness_passed = bool(
        len(identities) == 10
        and len(pairs) == 60
        and all(row["passed"] for row in synthetic_rows)
        and all(row["passed"] for row in real_rows)
        and all(row["passed"] for row in top_k_applicable)
        and downstream["passed"]
        and reproducibility["passed"]
    )
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "method_name": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "preflight_contract": PREFLIGHT_CONTRACT,
        "passed": correctness_passed,
        "pilot_500_authorized": correctness_passed,
        "contract": {
            "path": str(_contract_path(project_root)),
            "sha256": EXPECTED_CONTRACT_SHA256,
            "frozen_before_preflight": True,
        },
        "fixture": fixture,
        "engineering_identities": [
            {
                "selection_index": row.selection_index,
                "subject_id": row.subject_id,
                "canonical_finger_position": row.canonical_finger_position,
                "identity_key": row.identity_key,
            }
            for row in identities
        ],
        "algorithm_identity": identity,
        "algorithm_changed": False,
        "validation_contract_changed": True,
        "synthetic_results": synthetic_rows,
        "real_image_count_after_deduplication": len(real_rows),
        "real_image_results": real_rows,
        "top_k_applicable_image_count": len(top_k_applicable),
        "real_pair_results": pair_rows,
        "downstream_semantic_validation": downstream,
        "cuda_reproducibility": reproducibility,
        "performance_projection": _timing_report(real_rows, pair_rows),
        "environment": environment,
        "spearman_is_diagnostic_only": True,
        "spearman_can_cause_failure": False,
        "no_v4_relaxation_path": NO_V4_RELAXATION_PATH,
        "no_parameter_tuning_performed": True,
        "no_tolerance_changed_after_result": True,
        "no_500_result_observed": True,
        "sourceafis_or_sift_rerun": False,
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "outcome"
    }


def publish_engineering_preflight_v3(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    failure_path = project_root / FAILURE_RELATIVE
    existing = [path for path in (pass_path, failure_path) if path.exists()]
    if existing:
        if len(existing) != 1:
            raise HarrisZPlusPreflightV3Error(
                "Both pass and failure v3 artifacts exist."
            )
        report = json.loads(existing[0].read_text(encoding="utf-8"))
        return {
            **report,
            "path": str(existing[0]),
            "sha256": file_sha256(existing[0]),
            "reused_immutable_artifact": True,
        }
    report = run_engineering_preflight_v3(
        project_root=project_root,
        data_root=data_root,
    )
    target = pass_path if report["passed"] else failure_path
    _publish_exclusive_bytes(target, _json_bytes(report))
    return {
        **report,
        "path": str(target),
        "sha256": file_sha256(target),
        "reused_immutable_artifact": False,
    }


def require_pilot_authorization(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    pass_path = project_root / PASS_RELATIVE
    failure_path = project_root / FAILURE_RELATIVE
    if failure_path.exists():
        raise HarrisZPlusPreflightV3Error(
            "The frozen v3 preflight failed; a 500 run is forbidden."
        )
    if not pass_path.is_file():
        raise HarrisZPlusPreflightV3Error(
            "No v3 engineering_preflight_pass.json exists; a 500 run is forbidden."
        )
    report = json.loads(pass_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or report.get("passed") is not True
        or report.get("pilot_500_authorized") is not True
        or report.get("contract", {}).get("sha256")
        != EXPECTED_CONTRACT_SHA256
        or report.get("no_500_result_observed") is not True
    ):
        raise HarrisZPlusPreflightV3Error(
            "The v3 pass artifact does not authorize the 500 protocol."
        )
    return {**report, "path": str(pass_path), "sha256": file_sha256(pass_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or run frozen HarrisZ+ engineering preflight v3."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "command",
        choices=("prepare-fixtures", "preflight"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare-fixtures":
            result = prepare_engineering_fixtures(project_root=args.project_root)
            summary = {
                "identity_count": result["identity_count"],
                "pair_count": result["pair_count"],
                "artifact_sha256": result["artifact_sha256"],
            }
            code = 0
        else:
            result = publish_engineering_preflight_v3(
                project_root=args.project_root,
                data_root=args.data_root,
            )
            summary = {
                "passed": result["passed"],
                "pilot_500_authorized": result["pilot_500_authorized"],
                "path": result["path"],
                "sha256": result["sha256"],
            }
            code = 0 if result["passed"] else 1
    except (HarrisZPlusPreflightV3Error, ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
