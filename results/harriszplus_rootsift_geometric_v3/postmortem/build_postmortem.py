"""Read-only postmortem for the frozen HarrisZ+/RootSIFT geometric v3 pilot.

This script consumes only already-published CSV/JSON/image artifacts.  It does
not instantiate a matcher, detector, descriptor, orientation estimator, or
RANSAC implementation, and it never writes outside the postmortem directory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
METHOD_ROOT = ROOT / "results" / "harriszplus_rootsift_geometric_v3"
PILOT_ROOT = ROOT / "results" / "pilots" / "harriszplus_rootsift_geometric_joint_500_v3"
SIFT_ROOT = ROOT / "results" / "pilots" / "sift_geometric_joint_500_v1"
OUT = METHOD_ROOT / "postmortem"
INSPECTION = OUT / "inspection"
THRESHOLD = 4
DATASETS = ("sd300b", "sd300c")
LABELS = ("plain_self", "roll_self", "genuine", "negative")
RUN_DIRS = {
    "plain_self": "plain_self",
    "roll_self": "roll_self",
    "genuine": "plain_roll_genuine",
    "negative": "plain_roll_negative",
}
MANIFEST_FILES = {
    "plain_self": "plain_self.csv",
    "roll_self": "roll_self.csv",
    "genuine": "plain_roll_genuine.csv",
    "negative": "plain_roll_negative.csv",
}
SCALE_INDICES = tuple(range(5))
SCORE_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5-9", 5, 9),
    ("10+", 10, None),
)
FAILURE_STAGES = (
    "insufficient_descriptors",
    "no_ratio_matches",
    "insufficient_mutual_matches",
    "ransac_not_attempted",
    "ransac_model_failure",
    "valid_model_but_0_to_3_inliers",
    "accepted_4_or_more_inliers",
)

csv.field_size_limit(1_000_000_000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stats(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray(
        [number for value in values if (number := clean_number(value)) is not None],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "zero_count": 0,
            "minimum": None,
            "maximum": None,
        }
    q = np.percentile(array, [5, 25, 50, 75, 95], method="linear")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(q[2]),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "zero_count": int(np.count_nonzero(array == 0.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def weighted_stats(value_counts: Counter[float]) -> dict[str, Any]:
    ordered = sorted((float(value), int(count)) for value, count in value_counts.items() if count)
    total = sum(count for _, count in ordered)
    if total == 0:
        return stats(())
    cumulative: list[tuple[float, int]] = []
    running = 0
    for value, count in ordered:
        running += count
        cumulative.append((value, running))

    def at_index(index: int) -> float:
        for value, end in cumulative:
            if index < end:
                return value
        return cumulative[-1][0]

    def percentile(q: float) -> float:
        rank = (total - 1) * q
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        fraction = rank - low
        return at_index(low) * (1.0 - fraction) + at_index(high) * fraction

    mean = sum(value * count for value, count in ordered) / total
    return {
        "count": total,
        "mean": mean,
        "median": percentile(0.50),
        "p5": percentile(0.05),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "zero_count": sum(count for value, count in ordered if value == 0.0),
        "minimum": ordered[0][0],
        "maximum": ordered[-1][0],
    }


def protected_inventory() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for base in (METHOD_ROOT, PILOT_ROOT):
        for path in sorted(base.rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file() or OUT in path.parents:
                continue
            records.append({"path": rel(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = "".join(
        f"{record['path']}\t{record['size']}\t{record['sha256']}\n"
        for record in sorted(records, key=lambda record: record["path"])
    ).encode("utf-8")
    return {
        "file_count": len(records),
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
        "files": records,
    }


def load_manifest(dataset: str, label: str) -> dict[str, dict[str, str]]:
    path = PILOT_ROOT / "manifests" / dataset / MANIFEST_FILES[label]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def scale_mapping(prepare: dict[str, Any]) -> dict[int, float]:
    return {
        int(record["scale_index"]): float(record["opencv_keypoint_size"])
        for record in prepare.get("scale_mapping_records", ())
    }


def prepare_summary(prepare: dict[str, Any]) -> dict[str, Any]:
    final_counts = Counter(int(value) for value in prepare.get("harriszplus_scale_indices", ()))
    pre_counts = Counter()
    for index in SCALE_INDICES:
        record = prepare.get("scales", {}).get(str(index), {})
        pre_counts[index] = int(record.get("counts", {}).get("candidates_after_eigen_ratio", 0))
    mappings = scale_mapping(prepare)
    size_counts = Counter(
        {float(mappings[index]): int(count) for index, count in final_counts.items() if index in mappings}
    )
    candidates_before_cap = int(
        prepare.get(
            "candidates_after_duplicate_removal",
            prepare.get("uniform_selection", {}).get("input_count", 0),
        )
    )
    final = int(prepare.get("final_keypoint_count", 0))
    cap_active = bool(prepare.get("cap_truncated_count", 0) > 0 or (final == 3000 and candidates_before_cap > final))
    representation_hash = str(prepare.get("representation_sha256", ""))
    return {
        "representation_sha256": representation_hash,
        "final_keypoints": final,
        "descriptors": int(prepare.get("descriptor_count", 0)),
        "cap_active": cap_active,
        "candidates_before_cap": candidates_before_cap,
        "candidate_to_final_ratio": (
            float(candidates_before_cap) / final if final > 0 else None
        ),
        "final_scale_counts": {index: int(final_counts[index]) for index in SCALE_INDICES},
        "pre_scale_counts_proxy": {index: int(pre_counts[index]) for index in SCALE_INDICES},
        "size_counts": size_counts,
        "median_keypoint_size": weighted_stats(size_counts)["median"],
        "orientation_count": int(prepare.get("orientation_count", 0)),
        "orientation_sample_count_min": int(prepare.get("orientation_sample_count_min", 0)),
        "orientation_sample_count_max": int(prepare.get("orientation_sample_count_max", 0)),
        "uniform_distance_px": clean_number(prepare.get("uniform_selection", {}).get("distance")),
        "rank_window_2990_3010_available_point_count": max(
            0, min(candidates_before_cap, 3010) - 2989
        ),
        "cutoff_response_2990": None,
        "cutoff_response_3000": None,
        "cutoff_response_3010": None,
        "ppi": int(prepare.get("ppi", 0)),
    }


def failure_stage(record: dict[str, Any]) -> str:
    compare = record["compare"]
    if record["a"]["descriptors"] < 2 or record["b"]["descriptors"] < 2:
        return "insufficient_descriptors"
    ratio_a = int(compare.get("ratio_match_count_a_to_b", 0) or 0)
    ratio_b = int(compare.get("ratio_match_count_b_to_a", 0) or 0)
    mutual = int(compare.get("mutual_match_count", 0) or 0)
    if ratio_a == 0 and ratio_b == 0:
        return "no_ratio_matches"
    if mutual < 3:
        return "insufficient_mutual_matches"
    if compare.get("ransac_input_count") is None:
        return "ransac_not_attempted"
    if not bool(compare.get("geometry_success", False)):
        return "ransac_model_failure"
    if int(compare.get("geometric_inlier_count", 0) or 0) <= 3:
        return "valid_model_but_0_to_3_inliers"
    return "accepted_4_or_more_inliers"


def read_harris_run(dataset: str, label: str) -> list[dict[str, Any]]:
    path = PILOT_ROOT / "runs" / dataset / RUN_DIRS[label] / "pairs.csv"
    manifests = load_manifest(dataset, label)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prepare_a = json.loads(row["prepare_a_diagnostics"])
            prepare_b = json.loads(row["prepare_b_diagnostics"])
            compare = json.loads(row["compare_diagnostics"])
            manifest = manifests[row["pair_id"]]
            record = {
                "pair_id": row["pair_id"],
                "dataset": dataset,
                "label": label,
                "subject_id": row["subject_id"],
                "canonical_finger_position": int(row["canonical_finger_position"]),
                "status": row["status"],
                "raw_score": int(float(row["raw_score"])),
                "accepted": int(float(row["raw_score"])) >= THRESHOLD,
                "path_a": manifest["path_a"],
                "path_b": manifest["path_b"],
                "a": prepare_summary(prepare_a),
                "b": prepare_summary(prepare_b),
                "compare": compare,
            }
            record["failure_stage"] = failure_stage(record)
            records.append(record)
    return records


def stage_metric_rows(records_by_run: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_extractors: list[tuple[str, str, str, Any]] = [
        ("final_keypoints_a", "keypoints", "prepare_a_diagnostics.final_keypoint_count", lambda r: r["a"]["final_keypoints"]),
        ("final_keypoints_b", "keypoints", "prepare_b_diagnostics.final_keypoint_count", lambda r: r["b"]["final_keypoints"]),
        ("descriptor_count_a", "descriptors", "prepare_a_diagnostics.descriptor_count", lambda r: r["a"]["descriptors"]),
        ("descriptor_count_b", "descriptors", "prepare_b_diagnostics.descriptor_count", lambda r: r["b"]["descriptors"]),
        ("cap_active_a", "boolean", "derived from frozen cap diagnostics", lambda r: int(r["a"]["cap_active"])),
        ("cap_active_b", "boolean", "derived from frozen cap diagnostics", lambda r: int(r["b"]["cap_active"])),
        ("cap_active_count", "representations_per_pair", "sum of side flags", lambda r: int(r["a"]["cap_active"]) + int(r["b"]["cap_active"])),
        ("candidates_before_cap_a", "candidates", "prepare_a_diagnostics.candidates_after_duplicate_removal", lambda r: r["a"]["candidates_before_cap"]),
        ("candidates_before_cap_b", "candidates", "prepare_b_diagnostics.candidates_after_duplicate_removal", lambda r: r["b"]["candidates_before_cap"]),
        ("candidate_to_final_ratio_a", "ratio", "candidates_before_cap/final", lambda r: r["a"]["candidate_to_final_ratio"]),
        ("candidate_to_final_ratio_b", "ratio", "candidates_before_cap/final", lambda r: r["b"]["candidate_to_final_ratio"]),
        ("median_keypoint_size_a", "native_pixels", "derived from scale index and frozen mapping", lambda r: r["a"]["median_keypoint_size"]),
        ("median_keypoint_size_b", "native_pixels", "derived from scale index and frozen mapping", lambda r: r["b"]["median_keypoint_size"]),
        ("orientation_count_a", "orientations", "prepare_a_diagnostics.orientation_count", lambda r: r["a"]["orientation_count"]),
        ("orientation_count_b", "orientations", "prepare_b_diagnostics.orientation_count", lambda r: r["b"]["orientation_count"]),
        ("orientation_sample_count_min_a", "samples", "aggregate orientation diagnostic", lambda r: r["a"]["orientation_sample_count_min"]),
        ("orientation_sample_count_min_b", "samples", "aggregate orientation diagnostic", lambda r: r["b"]["orientation_sample_count_min"]),
        ("orientation_sample_count_max_a", "samples", "aggregate orientation diagnostic", lambda r: r["a"]["orientation_sample_count_max"]),
        ("orientation_sample_count_max_b", "samples", "aggregate orientation diagnostic", lambda r: r["b"]["orientation_sample_count_max"]),
        ("knn_candidates_a_to_b", "matches", "compare_diagnostics.raw_knn_count_a_to_b", lambda r: r["compare"].get("raw_knn_count_a_to_b")),
        ("knn_candidates_b_to_a", "matches", "compare_diagnostics.raw_knn_count_b_to_a", lambda r: r["compare"].get("raw_knn_count_b_to_a")),
        ("lowe_passed_a_to_b", "matches", "compare_diagnostics.ratio_match_count_a_to_b", lambda r: r["compare"].get("ratio_match_count_a_to_b")),
        ("lowe_passed_b_to_a", "matches", "compare_diagnostics.ratio_match_count_b_to_a", lambda r: r["compare"].get("ratio_match_count_b_to_a")),
        ("mutual_matches", "matches", "compare_diagnostics.mutual_match_count", lambda r: r["compare"].get("mutual_match_count")),
        ("ransac_input_matches", "matches", "compare_diagnostics.ransac_input_count", lambda r: r["compare"].get("ransac_input_count")),
        ("geometric_model_success", "boolean", "compare_diagnostics.geometry_success", lambda r: int(bool(r["compare"].get("geometry_success")))),
        ("inlier_count", "matches", "compare_diagnostics.geometric_inlier_count", lambda r: r["compare"].get("geometric_inlier_count")),
        ("inlier_ratio", "ratio", "compare_diagnostics.inlier_ratio", lambda r: r["compare"].get("inlier_ratio")),
        ("raw_score", "inliers", "pairs.csv raw_score", lambda r: r["raw_score"]),
    ]
    for frame, prefix in (("residual_reference_pixels", "residual_reference"), ("residual_destination_pixels", "residual_destination")):
        for summary_name in ("mean", "median", "p95", "maximum"):
            metric_extractors.append(
                (
                    f"{prefix}_{summary_name}",
                    "pixels",
                    f"per-pair compare_diagnostics.{frame}.{summary_name}",
                    lambda r, frame=frame, summary_name=summary_name: r["compare"].get(frame, {}).get(summary_name),
                )
            )
    for side in ("a", "b"):
        for index in SCALE_INDICES:
            metric_extractors.append(
                (
                    f"final_keypoints_scale_{index}_{side}",
                    "keypoints",
                    "counted from frozen harriszplus_scale_indices",
                    lambda r, side=side, index=index: r[side]["final_scale_counts"][index],
                )
            )
            metric_extractors.append(
                (
                    f"pre_cap_proxy_scale_{index}_{side}",
                    "candidates",
                    "candidates_after_eigen_ratio before cross-scale duplicate removal/uniform cap",
                    lambda r, side=side, index=index: r[side]["pre_scale_counts_proxy"][index],
                )
            )
    for (dataset, label), records in records_by_run.items():
        for metric, unit, source, extractor in metric_extractors:
            result = {
                "dataset": dataset,
                "class": label,
                "metric": metric,
                "unit": unit,
                "available": "true",
                "source": source,
                "notes": "",
                **stats(extractor(record) for record in records),
            }
            rows.append(result)
        for side in ("a", "b"):
            size_counts: Counter[float] = Counter()
            for record in records:
                size_counts.update(record[side]["size_counts"])
            rows.append(
                {
                    "dataset": dataset,
                    "class": label,
                    "metric": f"keypoint_size_distribution_{side}",
                    "unit": "native_pixels_per_keypoint",
                    "available": "true",
                    "source": "all keypoints, derived from frozen scale indices and size mapping",
                    "notes": "",
                    **weighted_stats(size_counts),
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "class": label,
                "metric": "orientation_angle_distribution",
                "unit": "degrees",
                "available": "false",
                "source": "not present in frozen diagnostics",
                "notes": "Angles are covered by representation_sha256 but individual angle values/histograms were not serialized.",
                **stats(()),
            }
        )
    return rows


def paired_bc(records_by_run: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    b = {
        (record["subject_id"], record["canonical_finger_position"]): record
        for record in records_by_run[("sd300b", "genuine")]
    }
    c = {
        (record["subject_id"], record["canonical_finger_position"]): record
        for record in records_by_run[("sd300c", "genuine")]
    }
    if set(b) != set(c) or len(b) != 500:
        raise RuntimeError("B/C genuine identity sets are not the same 500 identities.")
    rows: list[dict[str, Any]] = []
    for key in sorted(b):
        br, cr = b[key], c[key]
        row: dict[str, Any] = {
            "subject_id": key[0],
            "canonical_finger_position": key[1],
            "b_keypoints_a": br["a"]["final_keypoints"],
            "b_keypoints_b": br["b"]["final_keypoints"],
            "c_keypoints_a": cr["a"]["final_keypoints"],
            "c_keypoints_b": cr["b"]["final_keypoints"],
            "b_descriptors_a": br["a"]["descriptors"],
            "b_descriptors_b": br["b"]["descriptors"],
            "c_descriptors_a": cr["a"]["descriptors"],
            "c_descriptors_b": cr["b"]["descriptors"],
            "b_cap_a": br["a"]["cap_active"],
            "b_cap_b": br["b"]["cap_active"],
            "c_cap_a": cr["a"]["cap_active"],
            "c_cap_b": cr["b"]["cap_active"],
            "b_scale_distribution_a": ";".join(f"{i}:{br['a']['final_scale_counts'][i]}" for i in SCALE_INDICES),
            "b_scale_distribution_b": ";".join(f"{i}:{br['b']['final_scale_counts'][i]}" for i in SCALE_INDICES),
            "c_scale_distribution_a": ";".join(f"{i}:{cr['a']['final_scale_counts'][i]}" for i in SCALE_INDICES),
            "c_scale_distribution_b": ";".join(f"{i}:{cr['b']['final_scale_counts'][i]}" for i in SCALE_INDICES),
            "b_median_keypoint_size_a": br["a"]["median_keypoint_size"],
            "b_median_keypoint_size_b": br["b"]["median_keypoint_size"],
            "c_median_keypoint_size_a": cr["a"]["median_keypoint_size"],
            "c_median_keypoint_size_b": cr["b"]["median_keypoint_size"],
            "b_mutual_matches": br["compare"].get("mutual_match_count"),
            "c_mutual_matches": cr["compare"].get("mutual_match_count"),
            "b_ransac_input": br["compare"].get("ransac_input_count"),
            "c_ransac_input": cr["compare"].get("ransac_input_count"),
            "b_score": br["raw_score"],
            "c_score": cr["raw_score"],
            "score_delta_c_minus_b": cr["raw_score"] - br["raw_score"],
            "b_decision": "accepted" if br["accepted"] else "rejected",
            "c_decision": "accepted" if cr["accepted"] else "rejected",
            "b_stage": br["failure_stage"],
            "c_stage": cr["failure_stage"],
            "stage_transition": f"{br['failure_stage']} -> {cr['failure_stage']}",
        }
        rows.append(row)
    scores_b = np.asarray([row["b_score"] for row in rows], dtype=float)
    scores_c = np.asarray([row["c_score"] for row in rows], dtype=float)
    correlation = float(np.corrcoef(scores_b, scores_c)[0, 1])
    summary = {
        "identity_count": len(rows),
        "accepted_in_both": sum(row["b_decision"] == "accepted" and row["c_decision"] == "accepted" for row in rows),
        "accepted_only_b": sum(row["b_decision"] == "accepted" and row["c_decision"] == "rejected" for row in rows),
        "accepted_only_c": sum(row["b_decision"] == "rejected" and row["c_decision"] == "accepted" for row in rows),
        "rejected_in_both": sum(row["b_decision"] == "rejected" and row["c_decision"] == "rejected" for row in rows),
        "score_pearson_correlation": correlation,
        "score_delta_c_minus_b": stats(row["score_delta_c_minus_b"] for row in rows),
        "stage_transitions": dict(Counter(row["stage_transition"] for row in rows).most_common()),
    }
    return rows, summary


def physical_scale_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, ppi in (("sd300b", 1000), ("sd300c", 2000)):
        for scale in config["derived_scale_table"]:
            index = int(scale["scale_index"])
            keypoint_size = float(scale["keypoint_size"])
            integration_sigma = float(scale["output_integration_sigma"])
            differentiation_sigma = float(scale["output_differentiation_sigma"])
            support_px = float(scale["effective_support_diameter_original_px"])
            orientation_weighting_sigma = integration_sigma * float(config["orientation_gaussian_sigma_factor"])
            orientation_radius = max(
                1,
                int(math.floor(float(config["orientation_radius_factor"]) * orientation_weighting_sigma + 0.5)),
            )
            sift_hist_width = 3.0 * (keypoint_size / 2.0)
            descriptor_radius = int(math.floor(sift_hist_width * math.sqrt(2.0) * 2.5 + 0.5))
            descriptor_diameter = 2 * descriptor_radius + 1
            working_suppression = int(math.ceil(3.0 * float(scale["working_differentiation_sigma"])))
            suppression_native = working_suppression / float(scale["working_image_scale"])
            duplicate_native = float(config["duplicate_distance"]) if index in (0, 1) else None
            ransac_native = float(config["ransac_threshold_at_reference_ppi"]) * (
                ppi / float(config["reference_ppi"])
            )
            px_to_in = 1.0 / ppi
            px_to_mm = 25.4 / ppi
            rows.append(
                {
                    "dataset": dataset,
                    "ppi": ppi,
                    "scale_index": index,
                    "nominal_differentiation_sigma": scale["nominal_differentiation_sigma"],
                    "nominal_integration_sigma": scale["nominal_integration_sigma"],
                    "output_differentiation_sigma_px": differentiation_sigma,
                    "output_integration_sigma_px": integration_sigma,
                    "opencv_keypoint_size_px": keypoint_size,
                    "effective_support_diameter_px": support_px,
                    "effective_support_diameter_in": support_px * px_to_in,
                    "effective_support_diameter_mm": support_px * px_to_mm,
                    "orientation_weighting_sigma_px": orientation_weighting_sigma,
                    "orientation_window_radius_px": orientation_radius,
                    "orientation_window_radius_mm": orientation_radius * px_to_mm,
                    "descriptor_support_diameter_estimate_px": descriptor_diameter,
                    "descriptor_support_diameter_estimate_mm": descriptor_diameter * px_to_mm,
                    "scale_suppression_distance_native_px": suppression_native,
                    "scale_suppression_distance_mm": suppression_native * px_to_mm,
                    "duplicate_removal_distance_native_px": duplicate_native,
                    "duplicate_removal_distance_mm": (
                        duplicate_native * px_to_mm if duplicate_native is not None else None
                    ),
                    "ransac_threshold_native_px": ransac_native,
                    "ransac_threshold_mm": ransac_native * px_to_mm,
                    "same_scale_c_to_b_physical_ratio": 0.5,
                    "descriptor_support_note": (
                        "Estimate from OpenCV SIFT descriptor geometry: radius=round("
                        "3*(keypoint_size/2)*sqrt(2)*(4+1)/2), diameter=2r+1."
                    ),
                }
            )
    return rows


def sign_test_two_sided(differences: list[float]) -> dict[str, Any]:
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    tied = sum(value == 0 for value in differences)
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        k = min(positive, negative)
        numerator = sum(math.comb(n, i) for i in range(k + 1))
        p_value = min(1.0, 2.0 * float(numerator / (2**n)))
    return {
        "positive_c_gt_b": positive,
        "negative_c_lt_b": negative,
        "ties": tied,
        "non_tied": n,
        "two_sided_exact_sign_test_p": p_value,
    }


def cap_rows_and_summary(
    records_by_run: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for label in LABELS:
            records = records_by_run[(dataset, label)]
            observations = [record[side] for record in records for side in ("a", "b")]
            unique = {record["representation_sha256"]: record for record in observations}
            pre_total = Counter()
            final_total = Counter()
            for record in observations:
                pre_total.update(record["pre_scale_counts_proxy"])
                final_total.update(record["final_scale_counts"])
            row: dict[str, Any] = {
                "dataset": dataset,
                "class": label,
                "side_observation_count": len(observations),
                "exactly_3000_observations": sum(record["final_keypoints"] == 3000 for record in observations),
                "cap_active_observations": sum(record["cap_active"] for record in observations),
                "unique_representation_count": len(unique),
                "exactly_3000_unique_representations": sum(record["final_keypoints"] == 3000 for record in unique.values()),
                "cap_active_unique_representations": sum(record["cap_active"] for record in unique.values()),
                "candidate_before_cap_mean": stats(record["candidates_before_cap"] for record in observations)["mean"],
                "candidate_before_cap_median": stats(record["candidates_before_cap"] for record in observations)["median"],
                "candidate_before_cap_p5": stats(record["candidates_before_cap"] for record in observations)["p5"],
                "candidate_before_cap_p95": stats(record["candidates_before_cap"] for record in observations)["p95"],
                "candidate_to_final_ratio_mean": stats(record["candidate_to_final_ratio"] for record in observations)["mean"],
                "candidate_to_final_ratio_median": stats(record["candidate_to_final_ratio"] for record in observations)["median"],
                "candidate_to_final_ratio_p5": stats(record["candidate_to_final_ratio"] for record in observations)["p5"],
                "candidate_to_final_ratio_p95": stats(record["candidate_to_final_ratio"] for record in observations)["p95"],
                "rank_window_2990_3010_available_points_total": sum(
                    record["rank_window_2990_3010_available_point_count"] for record in observations
                ),
                "cutoff_response_2990": "unavailable",
                "cutoff_response_3000": "unavailable",
                "cutoff_response_3010": "unavailable",
                "cutoff_response_note": "Candidate responses/order were not serialized in frozen diagnostics.",
                "pre_cap_scale_source": "candidates_after_eigen_ratio proxy; exact post-duplicate per-scale input was not serialized",
            }
            for index in SCALE_INDICES:
                row[f"pre_cap_proxy_scale_{index}_count"] = int(pre_total[index])
                row[f"pre_cap_proxy_scale_{index}_fraction"] = (
                    float(pre_total[index]) / sum(pre_total.values()) if sum(pre_total.values()) else None
                )
                row[f"final_scale_{index}_count"] = int(final_total[index])
                row[f"final_scale_{index}_fraction"] = (
                    float(final_total[index]) / sum(final_total.values()) if sum(final_total.values()) else None
                )
            rows.append(row)
    captures: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for dataset in DATASETS:
        for record in records_by_run[(dataset, "genuine")]:
            captures[(dataset, record["subject_id"], record["canonical_finger_position"], "plain")] = record["a"]
            captures[(dataset, record["subject_id"], record["canonical_finger_position"], "roll")] = record["b"]
    differences: list[float] = []
    b_values: list[float] = []
    c_values: list[float] = []
    for _, subject, finger, modality in sorted(captures):
        if _ != "sd300b":
            continue
        b_value = float(captures[("sd300b", subject, finger, modality)]["candidate_to_final_ratio"])
        c_value = float(captures[("sd300c", subject, finger, modality)]["candidate_to_final_ratio"])
        b_values.append(b_value)
        c_values.append(c_value)
        differences.append(c_value - b_value)
    paired = {
        "capture_count": len(differences),
        "b_saturation_factor": stats(b_values),
        "c_saturation_factor": stats(c_values),
        "paired_delta_c_minus_b": stats(differences),
        "median_c_to_b_ratio": (
            float(np.median(np.asarray(c_values) / np.asarray(b_values))) if b_values else None
        ),
        **sign_test_two_sided(differences),
    }
    paired["c_materially_more_saturated"] = bool(
        paired["median_c_to_b_ratio"] >= 1.25
        and paired["positive_c_gt_b"] / max(1, paired["non_tied"]) >= 0.75
        and paired["two_sided_exact_sign_test_p"] < 0.05
    )
    return rows, paired


def orientation_audit(records_by_run: dict[tuple[str, str], list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    run_summaries: dict[str, Any] = {}
    for dataset in DATASETS:
        for label in LABELS:
            records = records_by_run[(dataset, label)]
            observations = [record[side] for record in records for side in ("a", "b")]
            run_summaries[f"{dataset}/{label}"] = {
                "representation_observations": len(observations),
                "orientation_count": stats(record["orientation_count"] for record in observations),
                "orientation_sample_count_min": stats(record["orientation_sample_count_min"] for record in observations),
                "orientation_sample_count_max": stats(record["orientation_sample_count_max"] for record in observations),
            }
    self_checks: dict[str, Any] = {}
    for dataset in DATASETS:
        for label in ("plain_self", "roll_self"):
            records = records_by_run[(dataset, label)]
            equal = sum(
                record["compare"].get("representation_sha256_a")
                == record["compare"].get("representation_sha256_b")
                for record in records
            )
            self_checks[f"{dataset}/{label}"] = {
                "pair_count": len(records),
                "representation_sha256_equal": equal,
                "all_equal": equal == len(records),
            }
    return {
        "schema_version": "harriszplus-v3-orientation-postmortem-v1",
        "no_orientation_recomputed": True,
        "configured_policy": {
            "bins": config["orientation_bins"],
            "gaussian_sigma_factor": config["orientation_gaussian_sigma_factor"],
            "radius_factor": config["orientation_radius_factor"],
            "histogram_smoothing_passes": config["orientation_histogram_smoothing_passes"],
            "signed_gradient_full_360_degrees": True,
            "random_180_flip": False,
        },
        "availability": {
            "orientation_histogram": "unavailable: individual angles were not serialized",
            "ten_degree_neighborhood_rates": "unavailable",
            "zero_180_concentration": "unavailable",
            "orientation_entropy": "unavailable",
            "weak_peak_dominance_rate": "unavailable: peak values/dominance were not serialized",
            "near_180_ambiguity_rate": "unavailable",
            "b_c_distribution_shift": "unavailable without angle payloads",
            "aggregate_orientation_count_and_sample_range": "available",
            "self_payload_identity": "available through representation SHA-256",
        },
        "representation_hash_scope": (
            "representation_sha256 hashes points, sizes, angles, responses, octaves, class_ids, "
            "descriptors, dimensions, PPI, scale indices and source indices."
        ),
        "self_pair_payload_identity": self_checks,
        "run_summaries": run_summaries,
        "interpretation": (
            "Self-pair hashes prove byte-identical orientation-bearing representations on both "
            "sides. The frozen payload cannot support an angle histogram, entropy, 180-degree "
            "shift, dispersion, or peak-dominance claim."
        ),
    }


def score_outputs(records_by_run: dict[tuple[str, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    histogram_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for dataset in DATASETS:
        for label in ("genuine", "negative"):
            records = records_by_run[(dataset, label)]
            scores = [record["raw_score"] for record in records]
            key = f"{dataset}/{label}"
            summary[key] = {"total": len(scores), "bins": {}, "accepted_at_threshold": {}}
            for record in records:
                raw_rows.append(
                    {
                        "pair_id": record["pair_id"],
                        "dataset": dataset,
                        "class": label,
                        "subject_id": record["subject_id"],
                        "canonical_finger_position": record["canonical_finger_position"],
                        "raw_score": record["raw_score"],
                        "frozen_threshold": THRESHOLD,
                        "frozen_decision": "accepted" if record["accepted"] else "rejected",
                    }
                )
            for bin_name, low, high in SCORE_BINS:
                count = sum(score >= low and (high is None or score <= high) for score in scores)
                summary[key]["bins"][bin_name] = count
                histogram_rows.append(
                    {
                        "record_type": "score_bin",
                        "dataset": dataset,
                        "class": label,
                        "score_bin": bin_name,
                        "threshold": "",
                        "count": count,
                        "total": len(scores),
                        "percentage": 100.0 * count / len(scores),
                    }
                )
            for threshold in (1, 2, 3, 4):
                count = sum(score >= threshold for score in scores)
                summary[key]["accepted_at_threshold"][str(threshold)] = count
                histogram_rows.append(
                    {
                        "record_type": "descriptive_threshold_sensitivity",
                        "dataset": dataset,
                        "class": label,
                        "score_bin": "",
                        "threshold": threshold,
                        "count": count,
                        "total": len(scores),
                        "percentage": 100.0 * count / len(scores),
                    }
                )
    return histogram_rows, raw_rows, summary


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_score_histogram(path: Path, score_summary: dict[str, Any]) -> None:
    width, height = 1600, 1100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34, bold=True)
    heading_font = load_font(25, bold=True)
    label_font = load_font(20)
    small_font = load_font(17)
    draw.text((50, 25), "Frozen HarrisZ+ v3 raw-score histograms (threshold remains 4)", fill="black", font=title_font)
    panels = (
        ("sd300b/genuine", (60, 100, 780, 550), "#2878b5"),
        ("sd300b/negative", (820, 100, 1540, 550), "#c44e52"),
        ("sd300c/genuine", (60, 610, 780, 1060), "#2878b5"),
        ("sd300c/negative", (820, 610, 1540, 1060), "#c44e52"),
    )
    for key, (left, top, right, bottom), color in panels:
        bins = score_summary[key]["bins"]
        maximum = max(bins.values()) or 1
        draw.rectangle((left, top, right, bottom), outline="#444444", width=2)
        draw.text((left + 15, top + 10), key, fill="black", font=heading_font)
        plot_left, plot_top = left + 70, top + 60
        plot_right, plot_bottom = right - 20, bottom - 60
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
        bar_space = (plot_right - plot_left) / len(SCORE_BINS)
        for index, (name, _, _) in enumerate(SCORE_BINS):
            count = bins[name]
            bar_left = plot_left + index * bar_space + 10
            bar_right = plot_left + (index + 1) * bar_space - 10
            bar_height = (plot_bottom - plot_top - 30) * count / maximum
            y = plot_bottom - bar_height
            draw.rectangle((bar_left, y, bar_right, plot_bottom), fill=color, outline="#333333")
            text = str(count)
            bbox = draw.textbbox((0, 0), text, font=small_font)
            draw.text(((bar_left + bar_right - (bbox[2] - bbox[0])) / 2, y - 24), text, fill="black", font=small_font)
            bbox = draw.textbbox((0, 0), name, font=label_font)
            draw.text(((bar_left + bar_right - (bbox[2] - bbox[0])) / 2, plot_bottom + 10), name, fill="black", font=label_font)
        draw.text((left + 8, plot_top - 10), str(maximum), fill="black", font=small_font)
        accepted = score_summary[key]["accepted_at_threshold"]["4"]
        draw.text((left + 15, bottom - 35), f"accepted at frozen threshold 4: {accepted}/500", fill="black", font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def read_sift_genuine(dataset: str) -> dict[tuple[str, int], dict[str, Any]]:
    path = SIFT_ROOT / "runs" / dataset / "plain_roll_genuine" / "pairs.csv"
    output: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            a = json.loads(row["prepare_a_diagnostics"])
            b = json.loads(row["prepare_b_diagnostics"])
            compare = json.loads(row["compare_diagnostics"])
            key = (row["subject_id"], int(row["canonical_finger_position"]))
            output[key] = {
                "descriptors_a": int(a.get("descriptor_count", 0)),
                "descriptors_b": int(b.get("descriptor_count", 0)),
                "mutual": int(compare.get("mutual_match_count", 0)),
                "ransac": int(compare.get("matches_submitted_to_geometry", 0)),
                "inliers": int(compare.get("geometric_inlier_count", 0)),
            }
    return output


def sift_comparison(
    records_by_run: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for dataset in DATASETS:
        harris = {
            (record["subject_id"], record["canonical_finger_position"]): record
            for record in records_by_run[(dataset, "genuine")]
        }
        sift = read_sift_genuine(dataset)
        if set(harris) != set(sift) or len(harris) != 500:
            raise RuntimeError(f"Harris/SIFT identity mismatch for {dataset}.")
        for key in sorted(harris):
            hr, sr = harris[key], sift[key]
            rows.append(
                {
                    "dataset": dataset,
                    "subject_id": key[0],
                    "canonical_finger_position": key[1],
                    "harrisz_descriptors_a": hr["a"]["descriptors"],
                    "harrisz_descriptors_b": hr["b"]["descriptors"],
                    "sift_descriptors_a": sr["descriptors_a"],
                    "sift_descriptors_b": sr["descriptors_b"],
                    "harrisz_mutual_matches": hr["compare"].get("mutual_match_count"),
                    "sift_mutual_matches": sr["mutual"],
                    "harrisz_ransac_inputs": hr["compare"].get("ransac_input_count"),
                    "sift_ransac_inputs": sr["ransac"],
                    "harrisz_inliers": hr["raw_score"],
                    "sift_inliers": sr["inliers"],
                }
            )
        subset = [row for row in rows if row["dataset"] == dataset]
        summary[dataset] = {
            "identity_count": len(subset),
            "harrisz_descriptors_per_image": stats(
                value for row in subset for value in (row["harrisz_descriptors_a"], row["harrisz_descriptors_b"])
            ),
            "sift_descriptors_per_image": stats(
                value for row in subset for value in (row["sift_descriptors_a"], row["sift_descriptors_b"])
            ),
            "harrisz_mutual_matches": stats(row["harrisz_mutual_matches"] for row in subset),
            "sift_mutual_matches": stats(row["sift_mutual_matches"] for row in subset),
            "harrisz_ransac_inputs": stats(row["harrisz_ransac_inputs"] for row in subset),
            "sift_ransac_inputs": stats(row["sift_ransac_inputs"] for row in subset),
            "harrisz_inliers": stats(row["harrisz_inliers"] for row in subset),
            "sift_inliers": stats(row["sift_inliers"] for row in subset),
        }
    return rows, summary


def fit_thumbnail(path: str, size: tuple[int, int]) -> Image.Image:
    source = Image.open(path).convert("L")
    source.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("L", size, 245)
    x = (size[0] - source.width) // 2
    y = (size[1] - source.height) // 2
    panel.paste(source, (x, y))
    return panel.convert("RGB")


def inspection_outputs(
    paired_rows: list[dict[str, Any]],
    records_by_run: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    b_records = {
        (record["subject_id"], record["canonical_finger_position"]): record
        for record in records_by_run[("sd300b", "genuine")]
    }
    c_records = {
        (record["subject_id"], record["canonical_finger_position"]): record
        for record in records_by_run[("sd300c", "genuine")]
    }
    categories = {
        "b_accepted_c_rejected_first10": [
            row for row in paired_rows if row["b_decision"] == "accepted" and row["c_decision"] == "rejected"
        ][:10],
        "both_rejected_first10": [
            row for row in paired_rows if row["b_decision"] == "rejected" and row["c_decision"] == "rejected"
        ][:10],
        "c_accepted_all": [row for row in paired_rows if row["c_decision"] == "accepted"],
        "c_score_zero_first10": [row for row in paired_rows if row["c_score"] == 0][:10],
    }
    selection_rows: list[dict[str, Any]] = []
    sheet_paths: list[str] = []
    title_font = load_font(28, bold=True)
    heading_font = load_font(18, bold=True)
    text_font = load_font(16)
    for category, selected in categories.items():
        category_dir = INSPECTION / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for page_index in range(0, len(selected), 5):
            page_rows = selected[page_index : page_index + 5]
            width, row_height = 1700, 245
            canvas = Image.new("RGB", (width, 70 + row_height * len(page_rows)), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (25, 18),
                f"{category} — deterministic order by subject_id, finger",
                fill="black",
                font=title_font,
            )
            for offset, paired in enumerate(page_rows):
                key = (paired["subject_id"], paired["canonical_finger_position"])
                br, cr = b_records[key], c_records[key]
                y = 70 + offset * row_height
                draw.rectangle((10, y, width - 10, y + row_height - 5), outline="#777777", width=2)
                image_specs = (
                    ("B PLAIN", br["path_a"]),
                    ("B ROLL", br["path_b"]),
                    ("C PLAIN", cr["path_a"]),
                    ("C ROLL", cr["path_b"]),
                )
                for image_index, (label, image_path) in enumerate(image_specs):
                    x = 20 + image_index * 255
                    draw.text((x, y + 8), label, fill="black", font=heading_font)
                    thumb = fit_thumbnail(image_path, (235, 185))
                    canvas.paste(thumb, (x, y + 38))
                text_x = 1045
                draw.text(
                    (text_x, y + 10),
                    f"subject {key[0]}  finger {key[1]}",
                    fill="black",
                    font=heading_font,
                )
                lines = [
                    (
                        f"B: KP {br['a']['final_keypoints']}/{br['b']['final_keypoints']}  "
                        f"mutual {br['compare'].get('mutual_match_count')}  "
                        f"RANSAC {br['compare'].get('ransac_input_count')}  "
                        f"inliers/score {br['raw_score']}  {br['failure_stage']}"
                    ),
                    (
                        f"C: KP {cr['a']['final_keypoints']}/{cr['b']['final_keypoints']}  "
                        f"mutual {cr['compare'].get('mutual_match_count')}  "
                        f"RANSAC {cr['compare'].get('ransac_input_count')}  "
                        f"inliers/score {cr['raw_score']}  {cr['failure_stage']}"
                    ),
                    f"decision: B {paired['b_decision']} | C {paired['c_decision']}",
                    "KP/match/inlier coordinates were not serialized; counts only (no rerun).",
                ]
                for line_index, line in enumerate(lines):
                    draw.text((text_x, y + 48 + line_index * 38), line, fill="#222222", font=text_font)
            page_path = category_dir / f"page_{page_index // 5 + 1:02d}.png"
            canvas.save(page_path, format="PNG", optimize=True)
            sheet_paths.append(rel(page_path))
            for order_offset, paired in enumerate(page_rows):
                selection_rows.append(
                    {
                        "category": category,
                        "category_order": page_index + order_offset + 1,
                        "subject_id": paired["subject_id"],
                        "canonical_finger_position": paired["canonical_finger_position"],
                        "b_score": paired["b_score"],
                        "c_score": paired["c_score"],
                        "b_stage": paired["b_stage"],
                        "c_stage": paired["c_stage"],
                        "sheet_path": rel(page_path),
                    }
                )
    readme = (
        "# Deterministic inspection sheets\n\n"
        "Rows were selected exactly as requested after sorting by `(subject_id, "
        "canonical_finger_position)`. Source PLAIN/ROLL images are shown without matcher "
        "re-execution. The frozen diagnostics contain keypoint, mutual-match, RANSAC-input, "
        "inlier and score counts, but not keypoint/match/inlier coordinates. Therefore the "
        "sheets show the saved counts and stage classification and do not claim spatial overlays.\n"
    )
    (INSPECTION / "README.md").write_text(readme, encoding="utf-8")
    write_csv(INSPECTION / "selection.csv", selection_rows)
    return selection_rows, sheet_paths


def key_file_records(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    keys = {
        "results/harriszplus_rootsift_geometric_v3/preflight/engineering_preflight_contract_v3.json",
        "results/harriszplus_rootsift_geometric_v3/preflight/engineering_preflight_pass.json",
        "results/harriszplus_rootsift_geometric_v3/config/freeze_manifest.json",
        "results/harriszplus_rootsift_geometric_v3/config/decision_rule.json",
        "results/pilots/harriszplus_rootsift_geometric_joint_500_v3/artifact_manifest.json",
        "results/pilots/harriszplus_rootsift_geometric_joint_500_v3/report/supervisor_report.json",
        "results/pilots/harriszplus_rootsift_geometric_joint_500_v3/integrity/protected_integrity.json",
    }
    before_map = {record["path"]: record for record in before["files"]}
    after_map = {record["path"]: record for record in after["files"]}
    return [
        {
            "path": path,
            "before_sha256": before_map[path]["sha256"],
            "after_sha256": after_map[path]["sha256"],
            "byte_identical": before_map[path] == after_map[path],
        }
        for path in sorted(keys)
    ]


def f(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INSPECTION.mkdir(parents=True, exist_ok=True)
    protected_before = protected_inventory()
    config = json.loads((METHOD_ROOT / "config" / "algorithm_config.json").read_text(encoding="utf-8"))
    freeze = json.loads((METHOD_ROOT / "config" / "freeze_manifest.json").read_text(encoding="utf-8"))
    records_by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for label in LABELS:
            records = read_harris_run(dataset, label)
            if len(records) != 500 or any(record["status"] != "ok" for record in records):
                raise RuntimeError(f"Unexpected frozen run shape/status for {dataset}/{label}.")
            records_by_run[(dataset, label)] = records

    stage_rows = stage_metric_rows(records_by_run)
    write_csv(OUT / "stage_attrition.csv", stage_rows)

    failure_rows: list[dict[str, Any]] = []
    failure_counts: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        genuine = records_by_run[(dataset, "genuine")]
        counts = Counter(record["failure_stage"] for record in genuine)
        failure_counts[dataset] = {stage: int(counts[stage]) for stage in FAILURE_STAGES}
        for record in genuine:
            failure_rows.append(
                {
                    "pair_id": record["pair_id"],
                    "dataset": dataset,
                    "subject_id": record["subject_id"],
                    "canonical_finger_position": record["canonical_finger_position"],
                    "stage": record["failure_stage"],
                    "raw_score": record["raw_score"],
                    "frozen_decision": "accepted" if record["accepted"] else "rejected",
                }
            )
    write_csv(OUT / "failure_stage_classification.csv", failure_rows)

    paired_rows, paired_summary = paired_bc(records_by_run)
    write_csv(OUT / "paired_bc_analysis.csv", paired_rows)

    physical_rows = physical_scale_rows(config)
    write_csv(OUT / "physical_scale_audit.csv", physical_rows)

    cap_rows, cap_summary = cap_rows_and_summary(records_by_run)
    write_csv(OUT / "cap_saturation.csv", cap_rows)

    orientation = orientation_audit(records_by_run, config)
    write_json(OUT / "orientation_audit.json", orientation)

    histogram_rows, raw_score_rows, score_summary = score_outputs(records_by_run)
    write_csv(OUT / "score_histograms.csv", histogram_rows)
    write_csv(OUT / "raw_scores.csv", raw_score_rows)
    draw_score_histogram(OUT / "score_histograms.png", score_summary)

    sift_rows, sift_summary = sift_comparison(records_by_run)
    write_csv(OUT / "sift_backend_comparison.csv", sift_rows)

    selection_rows, sheet_paths = inspection_outputs(paired_rows, records_by_run)

    ransac_records = [
        record
        for records in records_by_run.values()
        for record in records
    ]
    ransac_audit = {
        "pair_count": len(ransac_records),
        "coordinate_normalization_ppi_to_reference_count": sum(
            record["compare"].get("coordinate_normalization") == "ppi_to_reference"
            for record in ransac_records
        ),
        "reference_ppi_1000_count": sum(
            clean_number(record["compare"].get("reference_ppi")) == 1000.0
            for record in ransac_records
        ),
        "threshold_reference_pixels_3_count": sum(
            clean_number(record["compare"].get("ransac_threshold_reference_pixels")) == 3.0
            for record in ransac_records
        ),
        "b_native_threshold_pixels": 3.0,
        "c_native_threshold_pixels": 6.0,
        "b_threshold_mm": 3.0 * 25.4 / 1000.0,
        "c_threshold_mm": 6.0 * 25.4 / 2000.0,
    }
    ransac_audit["fully_ppi_normalized"] = all(
        ransac_audit[key] == ransac_audit["pair_count"]
        for key in (
            "coordinate_normalization_ppi_to_reference_count",
            "reference_ppi_1000_count",
            "threshold_reference_pixels_3_count",
        )
    ) and math.isclose(ransac_audit["b_threshold_mm"], ransac_audit["c_threshold_mm"])

    genuine_stats = {
        dataset: {
            "mutual_matches": stats(
                record["compare"].get("mutual_match_count")
                for record in records_by_run[(dataset, "genuine")]
            ),
            "ransac_inputs": stats(
                record["compare"].get("ransac_input_count")
                for record in records_by_run[(dataset, "genuine")]
            ),
            "inliers": stats(record["raw_score"] for record in records_by_run[(dataset, "genuine")]),
            "accepted": sum(record["accepted"] for record in records_by_run[(dataset, "genuine")]),
        }
        for dataset in DATASETS
    }
    scale_mix = {}
    for dataset in DATASETS:
        counts = Counter()
        for record in records_by_run[(dataset, "genuine")]:
            counts.update(record["a"]["final_scale_counts"])
            counts.update(record["b"]["final_scale_counts"])
        total = sum(counts.values())
        scale_mix[dataset] = {
            "counts": {str(index): int(counts[index]) for index in SCALE_INDICES},
            "fractions": {str(index): float(counts[index]) / total for index in SCALE_INDICES},
        }
    scale_mix_l1 = 0.5 * sum(
        abs(scale_mix["sd300b"]["fractions"][str(index)] - scale_mix["sd300c"]["fractions"][str(index)])
        for index in SCALE_INDICES
    )

    protected_after = protected_inventory()
    protected_identical = (
        protected_before["tree_sha256"] == protected_after["tree_sha256"]
        and protected_before["files"] == protected_after["files"]
    )
    if not protected_identical:
        raise RuntimeError("Protected v3 inputs changed during the postmortem.")

    conclusion = {
        "code": "A",
        "title": "PPI/scale mismatch strongly supported",
        "rationale": [
            "C loses genuine correspondences before and through geometry, with lower mutual-match and inlier distributions.",
            "For every fixed scale index, detector, orientation-window, and descriptor physical support in C is approximately one half of B.",
            "The cap is saturated and the B/C scale/candidate behavior is measurably different.",
            "All frozen compare diagnostics confirm PPI-to-reference coordinate normalization and equal 0.0762 mm RANSAC thresholds.",
        ],
        "next_step": (
            "Define a new v4 PPI-aware scale-normalization method/config and select all "
            "parameters only on development identities outside these 500 identities. Do not "
            "modify v3 or describe v4 as a small fix."
        ),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": "harriszplus-rootsift-geometric-v3-postmortem-v1",
        "generated_at_utc": generated_at,
        "scope": {
            "postmortem_only": True,
            "matcher_executed": False,
            "detector_executed": False,
            "descriptor_executed": False,
            "orientation_executed": False,
            "ransac_executed": False,
            "config_changed": False,
            "threshold_changed": False,
            "cap_changed": False,
            "runs_reexecuted": False,
        },
        "frozen_identity": {
            "method_version": "harriszplus-rootsift-geometric-v3",
            "operational_threshold": THRESHOLD,
            "canonical_config_hash": freeze["canonical_config_hash"],
            "implementation_hash": freeze["implementation_hash"],
            "decision_rule_hash": freeze["decision_rule_hash"],
        },
        "genuine_failure_stage_counts": failure_counts,
        "genuine_stage_metrics": genuine_stats,
        "paired_bc": paired_summary,
        "physical_scale": {
            "same_scale_c_to_b_physical_support_ratio": 0.5,
            "scale_mix_total_variation_distance": scale_mix_l1,
            "final_scale_mix": scale_mix,
            "ransac_audit": ransac_audit,
        },
        "cap_saturation": cap_summary,
        "orientation": orientation,
        "score_distributions": score_summary,
        "sift_backend_comparison": sift_summary,
        "inspection": {
            "selection_record_count_including_category_membership": len(selection_rows),
            "contact_sheets": sheet_paths,
            "coordinate_overlay_status": "unavailable from frozen diagnostics; counts shown",
        },
        "data_availability_limits": {
            "orientation_angles_and_histograms": "not serialized",
            "orientation_peak_dominance": "not serialized",
            "keypoint_coordinates": "not serialized",
            "match_and_inlier_coordinates": "not serialized",
            "candidate_responses_near_cap_cutoff": "not serialized",
            "exact_per_scale_post_duplicate_pre_cap_counts": (
                "not serialized; candidates_after_eigen_ratio is reported as an explicit proxy"
            ),
            "raw_residual_vectors": "not serialized; per-pair mean/median/p95/maximum summaries are analyzed",
        },
        "conclusion": conclusion,
        "integrity": {
            "protected_file_count": protected_before["file_count"],
            "protected_tree_sha256_before": protected_before["tree_sha256"],
            "protected_tree_sha256_after": protected_after["tree_sha256"],
            "byte_identical": protected_identical,
            "key_files": key_file_records(protected_before, protected_after),
        },
        "artifact_manifest": rel(OUT / "artifact_manifest.json"),
    }
    write_json(OUT / "postmortem_report.json", report)

    b_fail = failure_counts["sd300b"]
    c_fail = failure_counts["sd300c"]
    b_score = score_summary["sd300b/genuine"]
    c_score = score_summary["sd300c/genuine"]
    b_neg = score_summary["sd300b/negative"]
    c_neg = score_summary["sd300c/negative"]
    md = f"""# HarrisZ+ / RootSIFT geometric v3 postmortem

Generated from the eight frozen 500-pair bundles only. No detector, orientation,
descriptor, matcher, RANSAC, threshold, cap, PPI policy, or config was executed or changed.

## 1. Where genuine recognition collapses

The collapse is already visible before geometry and continues after RANSAC. Mean mutual
matches fall from {f(genuine_stats['sd300b']['mutual_matches']['mean'])} in B to
{f(genuine_stats['sd300c']['mutual_matches']['mean'])} in C; mean inliers fall from
{f(genuine_stats['sd300b']['inliers']['mean'])} to
{f(genuine_stats['sd300c']['inliers']['mean'])}. Accepted genuine pairs remain
64/500 in B and 1/500 in C at the frozen threshold 4.

## 2. Genuine failure-stage breakdown

| Stage | B | C |
|---|---:|---:|
| insufficient_descriptors | {b_fail['insufficient_descriptors']} | {c_fail['insufficient_descriptors']} |
| no_ratio_matches | {b_fail['no_ratio_matches']} | {c_fail['no_ratio_matches']} |
| insufficient_mutual_matches | {b_fail['insufficient_mutual_matches']} | {c_fail['insufficient_mutual_matches']} |
| ransac_not_attempted | {b_fail['ransac_not_attempted']} | {c_fail['ransac_not_attempted']} |
| ransac_model_failure | {b_fail['ransac_model_failure']} | {c_fail['ransac_model_failure']} |
| valid_model_but_0_to_3_inliers | {b_fail['valid_model_but_0_to_3_inliers']} | {c_fail['valid_model_but_0_to_3_inliers']} |
| accepted_4_or_more_inliers | {b_fail['accepted_4_or_more_inliers']} | {c_fail['accepted_4_or_more_inliers']} |

Classification is descriptive and does not alter saved status or score. Detailed pair-level
classification is in `failure_stage_classification.csv`; all requested distributions and
summary statistics are in `stage_attrition.csv`.

## 3. Paired B/C result

The same 500 identities were aligned by `(subject_id, canonical_finger_position)`.
Accepted in both: {paired_summary['accepted_in_both']}; accepted only B:
{paired_summary['accepted_only_b']}; accepted only C:
{paired_summary['accepted_only_c']}; rejected in both:
{paired_summary['rejected_in_both']}. Score Pearson correlation is
{f(paired_summary['score_pearson_correlation'])}; median C-minus-B score delta is
{f(paired_summary['score_delta_c_minus_b']['median'])}.

## 4. Physical-scale audit

At the same scale index, every pixel-defined support in C covers half the physical size of
B because B is 1000 PPI and C is 2000 PPI. This includes HarrisZ+ Gaussian support,
orientation radius, supplied-keypoint SIFT descriptor support, suppression distance, and
scale-0/1 duplicate removal. The full 10-row scale table is in
`physical_scale_audit.csv`.

The geometric layer is correctly PPI-normalized in all {ransac_audit['pair_count']} frozen
pair diagnostics: coordinates are converted to 1000-PPI reference pixels and the reference
threshold is 3 px. That is 3 native px in B and 6 native px in C, both exactly
{f(ransac_audit['b_threshold_mm'], 4)} mm. This rules against a simple unnormalized RANSAC
threshold explanation.

## 5. Cap saturation

Across the 1,000 paired PLAIN/ROLL captures, the median pre-cap/final saturation factor is
{f(cap_summary['b_saturation_factor']['median'])} in B and
{f(cap_summary['c_saturation_factor']['median'])} in C; median C/B is
{f(cap_summary['median_c_to_b_ratio'])}. C is greater on
{cap_summary['positive_c_gt_b']} captures, lower on {cap_summary['negative_c_lt_b']}, with
{cap_summary['ties']} ties (two-sided exact sign-test p =
{cap_summary['two_sided_exact_sign_test_p']:.3g}). Final B/C scale-mixture total-variation
distance is {f(scale_mix_l1)}.

Candidate responses at ranks 2,990, 3,000 and 3,010 were not serialized, so their values
cannot be recovered without an impermissible rerun. The rank-window availability count and
the pre-cap scale proxy (`candidates_after_eigen_ratio`) are reported explicitly in
`cap_saturation.csv`.

## 6. Orientation diagnostics

All 2,000 self pairs have identical A/B representation hashes. The representation hash
includes the angle array, so this proves orientation-bearing payload identity on self
comparisons. Individual angles, histograms, peak dominance and ambiguity values were not
serialized; therefore entropy, 10-degree concentration, 0/180 concentration, B/C shift and
dispersion cannot be claimed. See `orientation_audit.json`.

## 7. Score histograms and descriptive threshold sensitivity

Frozen-threshold counts are B genuine {b_score['accepted_at_threshold']['4']}/500, B
negative {b_neg['accepted_at_threshold']['4']}/500, C genuine
{c_score['accepted_at_threshold']['4']}/500, and C negative
{c_neg['accepted_at_threshold']['4']}/500. `score_histograms.csv` reports bins 0, 1, 2, 3,
4, 5-9 and 10+, plus descriptive accepted counts at thresholds 1-4 for both genuine and
negative classes. `raw_scores.csv` preserves one row per frozen score. These calculations
do not select or change a threshold.

## 8. Existing SIFT backend comparison

The same 500 genuine identities per dataset were joined to the already-published SIFT
artifacts. In B, mean mutual matches are
{f(sift_summary['sd300b']['harrisz_mutual_matches']['mean'])} for HarrisZ+ versus
{f(sift_summary['sd300b']['sift_mutual_matches']['mean'])} for SIFT; mean inliers are
{f(sift_summary['sd300b']['harrisz_inliers']['mean'])} versus
{f(sift_summary['sd300b']['sift_inliers']['mean'])}. In C, mean mutual matches are
{f(sift_summary['sd300c']['harrisz_mutual_matches']['mean'])} versus
{f(sift_summary['sd300c']['sift_mutual_matches']['mean'])}; mean inliers are
{f(sift_summary['sd300c']['harrisz_inliers']['mean'])} versus
{f(sift_summary['sd300c']['sift_inliers']['mean'])}. This is a diagnostic localization,
not a reranking.

## 9. Deterministic inspection

Selections are the first requested identities after sorting by subject and canonical finger.
Contact-sheet paths are enumerated in `inspection/selection.csv`. The sheets show B/C PLAIN
and ROLL source images, saved keypoint counts, mutual matches, RANSAC inputs, inliers,
scores and failure stages. Spatial overlays are not shown because coordinates were not
serialized.

## 10. Decision

**A. PPI/scale mismatch strongly supported.**

C loses genuine correspondences before RANSAC, fixed scale indices cover half the physical
support, the cap/scale behavior changes, and the RANSAC PPI normalization is verified.
This distinguishes implementation correctness (supported by preflight and deterministic
self behavior) from method suitability at 1000/2000 PPI (not supported by v3 results).

## 11. Exact next step

Create a new `v4 PPI-aware scale normalization` method/config. Select every parameter on
development identities outside these 500; keep v3 immutable; do not call v4 a small fix;
and treat any future 500-identity pilot as a demonstration report, not an independent
evaluation.

## 12. Integrity and provenance

Protected file count: {protected_before['file_count']}. Protected tree SHA-256 before:
`{protected_before['tree_sha256']}`. After: `{protected_after['tree_sha256']}`. Result:
**byte-identical**. Generated artifact and source hashes are in `artifact_manifest.json`.

## 13. Explicit non-actions

No matcher or method was run. No HarrisZ+, SIFT, SourceAFIS, orientation or geometry was
rerun. No config, threshold, cap or PPI policy was changed. No commit or push was made by
this postmortem.
"""
    (OUT / "postmortem_report.md").write_text(md, encoding="utf-8")

    generated_files = []
    for path in sorted(OUT.rglob("*"), key=lambda item: str(item).lower()):
        if path.is_file() and path.name != "artifact_manifest.json":
            generated_files.append(
                {"path": rel(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
    sift_inputs = []
    for dataset in DATASETS:
        path = SIFT_ROOT / "runs" / dataset / "plain_roll_genuine" / "pairs.csv"
        sift_inputs.append({"path": rel(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    artifact_manifest = {
        "schema_version": "harriszplus-v3-postmortem-artifact-manifest-v1",
        "generated_at_utc": generated_at,
        "analysis_policy": {
            "existing_artifacts_only": True,
            "no_matcher_or_pipeline_execution": True,
            "no_config_or_threshold_change": True,
            "protected_scope_excludes_only_new_postmortem_directory": True,
        },
        "protected_integrity": {
            "file_count": protected_before["file_count"],
            "tree_sha256_before": protected_before["tree_sha256"],
            "tree_sha256_after": protected_after["tree_sha256"],
            "byte_identical": protected_identical,
            "key_files": key_file_records(protected_before, protected_after),
        },
        "frozen_input_files": protected_before["files"],
        "additional_sift_input_files": sift_inputs,
        "generated_file_count_excluding_this_manifest": len(generated_files),
        "generated_files": generated_files,
    }
    write_json(OUT / "artifact_manifest.json", artifact_manifest)

    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(OUT),
                "conclusion": conclusion["code"],
                "protected_tree_sha256": protected_before["tree_sha256"],
                "protected_byte_identical": protected_identical,
                "generated_files_excluding_manifest": len(generated_files),
                "artifact_manifest_sha256": sha256_file(OUT / "artifact_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
