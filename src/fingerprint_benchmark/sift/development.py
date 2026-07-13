"""Leakage-safe subject split, shared pilot, ablations, and config selection."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterable

import numpy as np

from fingerprint_benchmark.contract import MethodExecutionError
from fingerprint_benchmark.hashing import canonical_json_bytes, file_sha256, stable_config_hash, stable_hash
from fingerprint_benchmark.io import write_csv_atomic, write_json_atomic
from fingerprint_benchmark.manifest import PairRecord, read_pair_manifest

from .adapter import SiftGeometricAdapter
from .config import SCORE_MODES, SiftGeometricConfig


SPLIT_VERSION = "sift-subject-split-v1"
SPLIT_HASH_FUNCTION = "sha256"
DEVELOPMENT_PERCENTAGE = 20
PRIMARY_TARGET_FAR = 0.01
REPORT_TARGET_FARS = (0.001, 0.005, 0.01)
PILOT_SCHEMA_VERSION = "sift-pilot-pairs-v1"


@dataclass(frozen=True)
class PilotPair:
    pair_id: str
    dataset: str
    protocol: str
    label: int
    subject_id_a: str
    subject_id_b: str
    canonical_finger_position_a: int
    canonical_finger_position_b: int
    ppi: int
    path_a: str
    path_b: str
    image_sha256_a: str
    image_sha256_b: str
    split_assignment: str
    pair_generation_reason: str

    def metadata(self, side: str) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "side": side,
            "dataset": self.dataset,
            "protocol": self.protocol,
            "subject_id": self.subject_id_a if side == "a" else self.subject_id_b,
            "canonical_finger_position": (
                self.canonical_finger_position_a if side == "a" else self.canonical_finger_position_b
            ),
            "ppi": self.ppi,
            "path": self.path_a if side == "a" else self.path_b,
        }


@dataclass(frozen=True)
class AblationSpec:
    candidate_id: str
    parent_candidate_id: str | None
    changed_field: str
    config: SiftGeometricConfig
    selection_eligible: bool = True


def build_subject_split(repo_root: Path) -> dict[str, Any]:
    manifests = _manifest_map(repo_root)
    by_dataset: dict[str, set[str]] = {}
    for (dataset, _), path in manifests.items():
        by_dataset.setdefault(dataset, set()).update(pair.subject_id for pair in read_pair_manifest(path))
    if by_dataset.get("sd300b") != by_dataset.get("sd300c"):
        raise ValueError("SD300b and SD300c subject sets are not identical.")
    subjects = sorted(by_dataset["sd300b"])
    development = [subject for subject in subjects if subject_assignment(subject) == "development"]
    evaluation = [subject for subject in subjects if subject_assignment(subject) == "evaluation"]
    payload = {
        "split_version": SPLIT_VERSION,
        "split_rule": (
            f"development iff first 64 bits of sha256('{SPLIT_VERSION}:' + subject_id) modulo 100 "
            f"is less than {DEVELOPMENT_PERCENTAGE}; otherwise evaluation"
        ),
        "hash_function": SPLIT_HASH_FUNCTION,
        "development_percentage": DEVELOPMENT_PERCENTAGE,
        "dataset_subject_alignment": True,
        "development_subjects": development,
        "evaluation_subjects": evaluation,
        "development_count": len(development),
        "evaluation_count": len(evaluation),
    }
    payload["subject_lists_sha256"] = stable_hash(
        {"development": development, "evaluation": evaluation}
    )
    return payload


def subject_assignment(subject_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_VERSION}:{subject_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "development" if bucket < DEVELOPMENT_PERCENTAGE else "evaluation"


def write_subject_split(repo_root: Path, output_path: Path) -> dict[str, Any]:
    payload = build_subject_split(repo_root)
    write_json_atomic(payload, output_path)
    payload["file_sha256"] = file_sha256(output_path)
    return payload


def build_pilot_pairs(
    repo_root: Path,
    split_payload: dict[str, Any],
    *,
    genuine_per_condition: int,
    impostors_per_dataset: int,
) -> list[PilotPair]:
    development = set(split_payload["development_subjects"])
    evaluation = set(split_payload["evaluation_subjects"])
    manifests = _manifest_map(repo_root)
    pairs: list[PilotPair] = []
    for (dataset, protocol), manifest_path in sorted(manifests.items()):
        records = [pair for pair in read_pair_manifest(manifest_path) if pair.subject_id in development]
        selected = sorted(records, key=lambda pair: _selection_key(pair.pair_id))[:genuine_per_condition]
        for pair in selected:
            pairs.append(_genuine_pilot_pair(pair))
    for dataset in ("sd300b", "sd300c"):
        source = [
            pair
            for pair in read_pair_manifest(manifests[(dataset, "plain_roll")])
            if pair.subject_id in development
        ]
        ordered = sorted(source, key=lambda pair: _selection_key(f"impostor:{pair.pair_id}"))
        used: set[tuple[str, str]] = set()
        for offset in range(1, len(ordered)):
            for index, left in enumerate(ordered):
                right = ordered[(index + offset) % len(ordered)]
                identity_a = (left.subject_id, left.canonical_finger_position)
                identity_b = (right.subject_id, right.canonical_finger_position)
                if identity_a == identity_b or left.path_a.resolve() == right.path_b.resolve():
                    continue
                key = (str(left.path_a).lower(), str(right.path_b).lower())
                if key in used:
                    continue
                used.add(key)
                pairs.append(_impostor_pilot_pair(dataset, len(used), left, right))
                if len(used) >= impostors_per_dataset:
                    break
            if len(used) >= impostors_per_dataset:
                break
        if len(used) != impostors_per_dataset:
            raise ValueError(f"Could only generate {len(used)} impostors for {dataset}.")
    if any(pair.subject_id_a in evaluation or pair.subject_id_b in evaluation for pair in pairs):
        raise AssertionError("Evaluation subject leaked into pilot generation.")
    if any(
        pair.label == 0
        and pair.subject_id_a == pair.subject_id_b
        and pair.canonical_finger_position_a == pair.canonical_finger_position_b
        for pair in pairs
    ):
        raise AssertionError("Pilot contains a same-identity impostor.")
    return pairs


def write_pilot_pairs(pairs: list[PilotPair], output_path: Path) -> None:
    rows = [asdict(pair) for pair in pairs]
    write_csv_atomic(rows, output_path, list(rows[0]))


def read_pilot_pairs(path: Path) -> list[PilotPair]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                PilotPair(
                    **{
                        **raw,
                        "label": int(raw["label"]),
                        "canonical_finger_position_a": int(raw["canonical_finger_position_a"]),
                        "canonical_finger_position_b": int(raw["canonical_finger_position_b"]),
                        "ppi": int(raw["ppi"]),
                    }
                )
            )
    return rows


def ablation_specs() -> tuple[AblationSpec, ...]:
    reference = SiftGeometricConfig(
        image_policy="reference_reproduction",
        mask_mode="none",
        descriptor_mode="standard",
        matching_mode="one_way",
        geometry_model="affine_full_2d",
        score_mode="inliers_times_inlier_ratio_times_log1p_matches",
        normalize_coordinates_by_ppi=False,
    )
    base = SiftGeometricConfig(
        image_policy="native",
        mask_mode="none",
        descriptor_mode="standard",
        matching_mode="one_way",
        geometry_model="affine_full_2d",
        score_mode="inliers_times_inlier_ratio_times_log1p_matches",
    )
    partial = base.changed(geometry_model="affine_partial_2d")
    root = partial.changed(descriptor_mode="rootsift")
    union = root.changed(matching_mode="bidirectional_union")
    mutual = union.changed(matching_mode="mutual")
    masked = mutual.changed(mask_mode="valid_region")
    return (
        AblationSpec(
            "reference_reproduction",
            None,
            "internal development reference; never eligible as the final public method",
            reference,
            selection_eligible=False,
        ),
        AblationSpec("native_base", None, "native image policy versus internal reference", base),
        AblationSpec("partial_affine", "native_base", "geometry_model", partial),
        AblationSpec("rootsift", "partial_affine", "descriptor_mode", root),
        AblationSpec("bidirectional_union", "rootsift", "matching_mode", union),
        AblationSpec("mutual_consistency", "bidirectional_union", "matching_mode", mutual),
        AblationSpec("valid_region", "mutual_consistency", "mask_mode", masked),
    )


def run_pilot_candidate(
    pairs: list[PilotPair],
    spec: AblationSpec,
    *,
    progress: callable | None = None,
) -> list[dict[str, Any]]:
    adapter = SiftGeometricAdapter(spec.config)
    rows: list[dict[str, Any]] = []
    try:
        for index, pair in enumerate(pairs, start=1):
            started = perf_counter_ns()
            base = {
                **asdict(pair),
                "candidate_id": spec.candidate_id,
                "parent_candidate_id": spec.parent_candidate_id or "",
                "changed_field": spec.changed_field,
                "config_hash": stable_config_hash(spec.config.as_dict()),
            }
            try:
                prepared_a = adapter.prepare(Path(pair.path_a), pair.metadata("a"))
                prepared_b = adapter.prepare(Path(pair.path_b), pair.metadata("b"))
                compared = adapter.compare(prepared_a.representation, prepared_b.representation)
                diagnostics = compared.diagnostics
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "error_code": "",
                        "raw_score": float(compared.raw_score),
                        "score_components_json": json.dumps(
                            diagnostics["score_components"], sort_keys=True, separators=(",", ":")
                        ),
                        "geometry_success": bool(diagnostics["geometry_success"]),
                        "geometry_failure_reason": diagnostics["geometry_failure_reason"] or "",
                        "keypoint_count_a": int(diagnostics["keypoint_count_a"]),
                        "keypoint_count_b": int(diagnostics["keypoint_count_b"]),
                        "matches": int(diagnostics["matches_submitted_to_geometry"]),
                        "inliers": int(diagnostics["geometric_inlier_count"]),
                        "prepare_a_ms": float(prepared_a.method_internal_ms or 0.0),
                        "prepare_b_ms": float(prepared_b.method_internal_ms or 0.0),
                        "compare_ms": float(compared.method_internal_ms or 0.0),
                        "total_ms": _elapsed(started),
                    }
                )
            except MethodExecutionError as exc:
                rows.append(
                    {
                        **base,
                        "status": "failure",
                        "error_code": exc.error_code,
                        "raw_score": "",
                        "score_components_json": "{}",
                        "geometry_success": False,
                        "geometry_failure_reason": "operation_failure",
                        "keypoint_count_a": "",
                        "keypoint_count_b": "",
                        "matches": "",
                        "inliers": "",
                        "prepare_a_ms": "",
                        "prepare_b_ms": "",
                        "compare_ms": "",
                        "total_ms": _elapsed(started),
                    }
                )
            if progress is not None and (index == 1 or index % 20 == 0 or index == len(pairs)):
                progress(index, len(pairs))
    finally:
        adapter.close()
    return rows


def write_pilot_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    write_csv_atomic(rows, output_path, list(rows[0]))


def select_final_config(
    all_rows: list[dict[str, Any]],
    specs: tuple[AblationSpec, ...],
) -> tuple[SiftGeometricConfig, dict[str, Any], list[dict[str, Any]]]:
    spec_by_id = {spec.candidate_id: spec for spec in specs}
    expanded: list[dict[str, Any]] = []
    for candidate_id in spec_by_id:
        candidate_rows = [row for row in all_rows if row["candidate_id"] == candidate_id]
        for score_mode in SCORE_MODES:
            scored = []
            for row in candidate_rows:
                copy = dict(row)
                components = json.loads(str(row.get("score_components_json") or "{}"))
                copy["selection_score"] = (
                    float(components.get(score_mode, 0.0)) if row["status"] == "ok" else float("nan")
                )
                scored.append(copy)
            expanded.append(_candidate_metric(candidate_id, score_mode, scored))
    eligible = [
        row
        for row in expanded
        if spec_by_id[row["candidate_id"]].selection_eligible
        and row["preparation_failure_rate"] <= 0.05
        and row["macro_impostor_acceptance_at_primary"] <= row["primary_far_guardrail"]
    ]
    if not eligible:
        raise RuntimeError("No pilot candidate passed preparation and impostor safety guardrails.")
    eligible.sort(
        key=lambda row: (
            -row["macro_plain_roll_tar_at_primary"],
            -row["macro_self_acceptance_at_primary"],
            row["geometry_failure_rate"],
            row["zero_score_rate"],
            row["median_total_ms"],
            row["candidate_id"],
            row["score_mode"],
        )
    )
    selected = eligible[0]
    config = spec_by_id[selected["candidate_id"]].config.changed(score_mode=selected["score_mode"])
    thresholds = _thresholds_for_selected(all_rows, selected["candidate_id"], selected["score_mode"])
    decision_rule = {
        "decision_rule_schema": "sift-decision-rule-v1",
        "selection_split": "development",
        "primary_target_far": PRIMARY_TARGET_FAR,
        "acceptance_operator": "raw_score >= dataset_threshold",
        "score_mode": selected["score_mode"],
        "thresholds_by_dataset": thresholds,
    }
    decision_rule["decision_rule_hash"] = stable_hash(decision_rule)
    for row in expanded:
        row["selected"] = (
            row["candidate_id"] == selected["candidate_id"] and row["score_mode"] == selected["score_mode"]
        )
        row["conclusion"] = "retain" if row["selected"] else "reject"
        if row["selected"]:
            row["decision_rationale"] = "best development-only ordered objective after safety guardrails"
        elif row["preparation_failure_rate"] > 0.05:
            row["decision_rationale"] = "rejected: preparation failure rate exceeded five percent"
        elif row["macro_impostor_acceptance_at_primary"] > row["primary_far_guardrail"]:
            row["decision_rationale"] = "rejected: development impostor acceptance exceeded primary FAR guardrail"
        else:
            row["decision_rationale"] = "rejected: lower ordered development objective than selected configuration"
    return config, decision_rule, expanded


def paired_reference_comparison(
    all_rows: list[dict[str, Any]],
    *,
    selected_candidate_id: str,
    score_mode: str,
    selected_thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    reference_id = "reference_reproduction"
    reference_rows = {
        str(row["pair_id"]): row for row in all_rows if row["candidate_id"] == reference_id
    }
    selected_rows = {
        str(row["pair_id"]): row
        for row in all_rows
        if row["candidate_id"] == selected_candidate_id
    }
    if set(reference_rows) != set(selected_rows):
        raise ValueError("Internal reference and selected pilot pair IDs do not align exactly.")
    reference_thresholds: dict[str, float] = {}
    for dataset in ("sd300b", "sd300c"):
        negatives = []
        for row in reference_rows.values():
            if row["dataset"] == dataset and int(row["label"]) == 0 and row["status"] == "ok":
                negatives.append(float(json.loads(str(row["score_components_json"]))[score_mode]))
        reference_thresholds[dataset] = threshold_for_far(negatives, PRIMARY_TARGET_FAR)[0]
    output = []
    for pair_id in sorted(selected_rows):
        selected = selected_rows[pair_id]
        reference = reference_rows[pair_id]
        selected_components = json.loads(str(selected.get("score_components_json") or "{}"))
        reference_components = json.loads(str(reference.get("score_components_json") or "{}"))
        selected_score = float(selected_components.get(score_mode, 0.0))
        reference_score = float(reference_components.get(score_mode, 0.0))
        dataset = str(selected["dataset"])
        selected_accept = selected_score >= float(selected_thresholds[dataset]["primary_threshold"])
        reference_accept = reference_score >= reference_thresholds[dataset]
        output.append(
            {
                "pair_id": pair_id,
                "dataset": dataset,
                "protocol": selected["protocol"],
                "label": selected["label"],
                "score_mode": score_mode,
                "reference_score": reference_score,
                "selected_score": selected_score,
                "score_delta_selected_minus_reference": selected_score - reference_score,
                "reference_accepted": reference_accept,
                "selected_accepted": selected_accept,
                "decision_agreement": reference_accept == selected_accept,
                "reference_accept_selected_reject": reference_accept and not selected_accept,
                "reference_reject_selected_accept": (not reference_accept) and selected_accept,
                "reference_zero_selected_positive": reference_score == 0.0 and selected_score > 0.0,
                "reference_positive_selected_zero": reference_score > 0.0 and selected_score == 0.0,
                "reference_geometry_success": _as_bool(reference.get("geometry_success", False)),
                "selected_geometry_success": _as_bool(selected.get("geometry_success", False)),
                "latency_delta_ms": float(selected["total_ms"]) - float(reference["total_ms"]),
                "keypoint_delta_a": _optional_delta(selected, reference, "keypoint_count_a"),
                "keypoint_delta_b": _optional_delta(selected, reference, "keypoint_count_b"),
                "match_count_delta": _optional_delta(selected, reference, "matches"),
                "inlier_count_delta": _optional_delta(selected, reference, "inliers"),
            }
        )
    return output


def _candidate_metric(candidate_id: str, score_mode: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds: dict[str, float] = {}
    actual_fars: dict[str, float] = {}
    for dataset in ("sd300b", "sd300c"):
        negatives = [
            row["selection_score"]
            for row in rows
            if row["dataset"] == dataset and int(row["label"]) == 0 and math.isfinite(row["selection_score"])
        ]
        threshold, actual_far = threshold_for_far(negatives, PRIMARY_TARGET_FAR)
        thresholds[dataset] = threshold
        actual_fars[dataset] = actual_far
    def accepted(row: dict[str, Any]) -> bool:
        return math.isfinite(row["selection_score"]) and row["selection_score"] >= thresholds[row["dataset"]]
    per_dataset_plain_roll = []
    per_dataset_self = []
    target_metrics: dict[float, dict[str, list[float]]] = {
        target: {"tar": [], "far": []} for target in REPORT_TARGET_FARS
    }
    dataset_auc: list[float] = []
    dataset_eer: list[float] = []
    for dataset in ("sd300b", "sd300c"):
        genuine = [row for row in rows if row["dataset"] == dataset and int(row["label"]) == 1]
        plain_roll = [row for row in genuine if row["protocol"] == "plain_roll"]
        self_rows = [row for row in genuine if row["protocol"] in {"plain_self", "roll_self"}]
        per_dataset_plain_roll.append(_rate(plain_roll, accepted))
        per_dataset_self.append(_rate(self_rows, accepted))
        negatives = [
            row["selection_score"]
            for row in rows
            if row["dataset"] == dataset
            and int(row["label"]) == 0
            and math.isfinite(row["selection_score"])
        ]
        for target in REPORT_TARGET_FARS:
            threshold, actual = threshold_for_far(negatives, target)
            target_metrics[target]["far"].append(actual)
            target_metrics[target]["tar"].append(
                _rate(
                    plain_roll,
                    lambda row, threshold=threshold: math.isfinite(row["selection_score"])
                    and row["selection_score"] >= threshold,
                )
            )
        labeled = [
            (int(row["label"]), float(row["selection_score"]))
            for row in rows
            if row["dataset"] == dataset and math.isfinite(row["selection_score"])
        ]
        auc, eer = auc_eer(labeled)
        dataset_auc.append(auc)
        dataset_eer.append(eer)
    successes = [row for row in rows if row["status"] == "ok"]
    zero_rate = _rate(successes, lambda row: float(row["selection_score"]) == 0.0)
    geometry_failure = _rate(successes, lambda row: not _as_bool(row["geometry_success"]))
    preparation_failure = _rate(rows, lambda row: row["status"] != "ok")
    total_values = [float(row["total_ms"]) for row in rows]
    metric = {
        "candidate_id": candidate_id,
        "selection_eligible": candidate_id != "reference_reproduction",
        "score_mode": score_mode,
        "primary_target_far": PRIMARY_TARGET_FAR,
        "thresholds_by_dataset_json": json.dumps(thresholds, sort_keys=True),
        "macro_impostor_acceptance_at_primary": float(np.mean(list(actual_fars.values()))),
        "primary_far_guardrail": PRIMARY_TARGET_FAR + 1e-12,
        "macro_plain_roll_tar_at_primary": float(np.mean(per_dataset_plain_roll)),
        "macro_self_acceptance_at_primary": float(np.mean(per_dataset_self)),
        "zero_score_rate": zero_rate,
        "geometry_failure_rate": geometry_failure,
        "preparation_failure_rate": preparation_failure,
        "median_total_ms": float(np.median(total_values)) if total_values else float("nan"),
        "macro_roc_auc": float(np.mean(dataset_auc)),
        "macro_eer": float(np.mean(dataset_eer)),
    }
    for target in REPORT_TARGET_FARS:
        suffix = str(target).replace(".", "_")
        metric[f"macro_plain_roll_tar_at_far_{suffix}"] = float(np.mean(target_metrics[target]["tar"]))
        metric[f"macro_impostor_far_at_far_{suffix}"] = float(np.mean(target_metrics[target]["far"]))
    return metric


def _thresholds_for_selected(
    rows: list[dict[str, Any]], candidate_id: str, score_mode: str
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for dataset in ("sd300b", "sd300c"):
        scores = []
        for row in rows:
            if row["candidate_id"] != candidate_id or row["dataset"] != dataset or int(row["label"]) != 0:
                continue
            components = json.loads(str(row.get("score_components_json") or "{}"))
            if row["status"] == "ok":
                scores.append(float(components[score_mode]))
        dataset_rules: dict[str, float] = {}
        for target in REPORT_TARGET_FARS:
            threshold, actual = threshold_for_far(scores, target)
            dataset_rules[f"far_{target:g}"] = threshold
            dataset_rules[f"actual_far_{target:g}"] = actual
        dataset_rules["primary_threshold"] = dataset_rules[f"far_{PRIMARY_TARGET_FAR:g}"]
        output[dataset] = dataset_rules
    return output


def threshold_for_far(scores: Iterable[float], target_far: float) -> tuple[float, float]:
    values = np.asarray([float(value) for value in scores if math.isfinite(float(value))], dtype=float)
    if values.size == 0:
        raise ValueError("Cannot calibrate a threshold without finite impostor scores.")
    for threshold in sorted(np.unique(values)):
        actual = float(np.mean(values >= threshold))
        if actual <= target_far + 1e-15:
            return float(threshold), actual
    return math.nextafter(float(np.max(values)), math.inf), 0.0


def auc_eer(labeled_scores: Iterable[tuple[int, float]]) -> tuple[float, float]:
    pairs = [(int(label), float(score)) for label, score in labeled_scores if math.isfinite(float(score))]
    positives = np.asarray([score for label, score in pairs if label == 1], dtype=float)
    negatives = np.asarray([score for label, score in pairs if label == 0], dtype=float)
    if positives.size == 0 or negatives.size == 0:
        return float("nan"), float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    auc = float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)
    thresholds = np.unique(np.concatenate([positives, negatives]))
    thresholds = np.concatenate(
        ([math.nextafter(float(np.max(thresholds)), math.inf)], thresholds[::-1])
    )
    fars = np.asarray([np.mean(negatives >= threshold) for threshold in thresholds], dtype=float)
    frrs = np.asarray([np.mean(positives < threshold) for threshold in thresholds], dtype=float)
    index = int(np.argmin(np.abs(fars - frrs)))
    return auc, float((fars[index] + frrs[index]) / 2.0)


def _rate(rows: list[dict[str, Any]], predicate: callable) -> float:
    return float(sum(bool(predicate(row)) for row in rows) / len(rows)) if rows else 0.0


def _optional_delta(
    selected: dict[str, Any], reference: dict[str, Any], key: str
) -> float | str:
    if selected.get(key) in (None, "") or reference.get(key) in (None, ""):
        return ""
    return float(selected[key]) - float(reference[key])


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _genuine_pilot_pair(pair: PairRecord) -> PilotPair:
    return PilotPair(
        pair_id=f"pilot_genuine_{pair.pair_id}",
        dataset=pair.dataset,
        protocol=pair.protocol,
        label=1,
        subject_id_a=pair.subject_id,
        subject_id_b=pair.subject_id,
        canonical_finger_position_a=pair.canonical_finger_position,
        canonical_finger_position_b=pair.canonical_finger_position,
        ppi=pair.ppi,
        path_a=str(pair.path_a),
        path_b=str(pair.path_b),
        image_sha256_a=file_sha256(pair.path_a),
        image_sha256_b=file_sha256(pair.path_b),
        split_assignment="development",
        pair_generation_reason=f"deterministic genuine sample from {pair.protocol}",
    )


def _impostor_pilot_pair(dataset: str, index: int, left: PairRecord, right: PairRecord) -> PilotPair:
    return PilotPair(
        pair_id=f"pilot_impostor_{dataset}_{index:05d}",
        dataset=dataset,
        protocol="impostor_sanity",
        label=0,
        subject_id_a=left.subject_id,
        subject_id_b=right.subject_id,
        canonical_finger_position_a=left.canonical_finger_position,
        canonical_finger_position_b=right.canonical_finger_position,
        ppi=left.ppi,
        path_a=str(left.path_a),
        path_b=str(right.path_b),
        image_sha256_a=file_sha256(left.path_a),
        image_sha256_b=file_sha256(right.path_b),
        split_assignment="development",
        pair_generation_reason="deterministic different-anatomical-identity plain-to-roll impostor",
    )


def _manifest_map(repo_root: Path) -> dict[tuple[str, str], Path]:
    return {
        (dataset, protocol): repo_root / "protocols" / dataset / f"{protocol}.csv"
        for dataset in ("sd300b", "sd300c")
        for protocol in ("plain_self", "roll_self", "plain_roll")
    }


def _selection_key(value: str) -> str:
    return hashlib.sha256(f"{PILOT_SCHEMA_VERSION}:{value}".encode("utf-8")).hexdigest()


def _elapsed(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0
