"""Command-line orchestration for development, parity, and the six primary runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

from fingerprint_benchmark.hashing import file_sha256
from fingerprint_benchmark.io import write_csv_atomic, write_json_atomic
from fingerprint_benchmark.runner import run_benchmark_manifest

from .adapter import SiftGeometricAdapter
from .config import SiftGeometricConfig
from .development import (
    ablation_specs,
    build_pilot_pairs,
    read_pilot_pairs,
    paired_reference_comparison,
    run_pilot_candidate,
    select_final_config,
    write_pilot_pairs,
    write_pilot_results,
    write_subject_split,
)
from .integrity import protected_input_inventory
from .reporting import build_final_reports
from .reference import reference_findings


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = DEFAULT_REPO_ROOT / "results"
DEVELOPMENT_DIRNAME = "development"
FINAL_CONFIG_FILENAME = "sift_geometric_config.json"


def prepare_development(
    repo_root: Path,
    results_root: Path,
    *,
    genuine_per_condition: int,
    impostors_per_dataset: int,
) -> dict[str, Any]:
    development = _development_dir(results_root)
    development.mkdir(parents=True, exist_ok=True)
    split_path = development / "subject_split.json"
    split_payload = write_subject_split(repo_root, split_path)
    pilot = build_pilot_pairs(
        repo_root,
        split_payload,
        genuine_per_condition=genuine_per_condition,
        impostors_per_dataset=impostors_per_dataset,
    )
    pilot_path = development / "pilot_pairs.csv"
    write_pilot_pairs(pilot, pilot_path)
    inventory_path = development / "protected_inputs_before.json"
    inventory = protected_input_inventory(repo_root)
    write_json_atomic(inventory, inventory_path)
    return {
        "subject_split": str(split_path),
        "subject_split_sha256": file_sha256(split_path),
        "pilot_pairs": str(pilot_path),
        "pilot_pairs_sha256": file_sha256(pilot_path),
        "pilot_pair_count": len(pilot),
        "protected_inputs": str(inventory_path),
        "protected_inputs_sha256": file_sha256(inventory_path),
    }


def run_internal_reference_parity(
    repo_root: Path,
    old_repo_root: Path,
    results_root: Path,
    *,
    pair_count: int,
) -> dict[str, Any]:
    development = _development_dir(results_root)
    pilot = [pair for pair in read_pilot_pairs(development / "pilot_pairs.csv") if pair.label == 1][:pair_count]
    if not pilot:
        raise ValueError("No genuine pilot pairs are available for parity.")
    old_path = str(old_repo_root.resolve())
    sys.path.insert(0, old_path)
    try:
        from src.fpbench.matchers.matching_baseline import (  # type: ignore
            SIFTConfig as ReferenceSiftConfig,
            match_sift as reference_match,
            ransac_inliers_for_model as reference_geometry,
            sift_extract as reference_extract,
        )
        from src.fpbench.preprocess.preprocess import (  # type: ignore
            PreprocessConfig as ReferencePreprocessConfig,
            load_gray as reference_load_gray,
            preprocess_image as reference_preprocess,
        )

        config = SiftGeometricConfig(
            image_policy="reference_reproduction",
            mask_mode="none",
            descriptor_mode="standard",
            matching_mode="one_way",
            geometry_model="affine_full_2d",
            score_mode="inliers_times_inlier_ratio_times_log1p_matches",
            normalize_coordinates_by_ppi=False,
        )
        adapter = SiftGeometricAdapter(config)
        rows: list[dict[str, Any]] = []
        for pair in pilot:
            reference_preprocess_config = ReferencePreprocessConfig(target_size=768, blur_ksize=0)
            reference_sift_config = ReferenceSiftConfig(nfeatures=3000)
            reference_image_a = reference_preprocess(reference_load_gray(pair.path_a), reference_preprocess_config)
            reference_image_b = reference_preprocess(reference_load_gray(pair.path_b), reference_preprocess_config)
            old_keypoints_a, old_descriptors_a = reference_extract(
                reference_image_a, None, reference_sift_config
            )
            old_keypoints_b, old_descriptors_b = reference_extract(
                reference_image_b, None, reference_sift_config
            )
            old_good = reference_match(old_descriptors_a, old_descriptors_b, ratio=0.75)
            old_inliers, _ = reference_geometry(
                old_keypoints_a,
                old_keypoints_b,
                old_good,
                ransac_model="affine_full_2d",
                ransac_thresh=3.0,
            )
            old_matches = len(old_good)
            old_ratio = float(old_inliers / old_matches) if old_matches else 0.0
            old_score = (
                float(old_inliers) * old_ratio * __import__("math").log1p(float(old_matches))
                if old_inliers > 0 and old_matches > 0
                else 0.0
            )
            new_a = adapter.prepare(Path(pair.path_a), pair.metadata("a"))
            new_b = adapter.prepare(Path(pair.path_b), pair.metadata("b"))
            new_result = adapter.compare(new_a.representation, new_b.representation)
            diag = new_result.diagnostics
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "reference_score": float(old_score),
                    "reproduction_score": float(new_result.raw_score),
                    "reference_keypoints_a": len(old_keypoints_a),
                    "reproduction_keypoints_a": diag["keypoint_count_a"],
                    "reference_keypoints_b": len(old_keypoints_b),
                    "reproduction_keypoints_b": diag["keypoint_count_b"],
                    "reference_ratio_matches": int(old_matches),
                    "reproduction_ratio_matches": diag["ratio_match_count_a_to_b"],
                    "reference_inliers": int(old_inliers),
                    "reproduction_inliers": diag["geometric_inlier_count"],
                    "score_equal": float(old_score) == float(new_result.raw_score),
                    "counts_equal": (
                        len(old_keypoints_a) == diag["keypoint_count_a"]
                        and len(old_keypoints_b) == diag["keypoint_count_b"]
                        and int(old_matches) == diag["ratio_match_count_a_to_b"]
                        and int(old_inliers) == diag["geometric_inlier_count"]
                    ),
                }
            )
    finally:
        if sys.path and sys.path[0] == old_path:
            sys.path.pop(0)
    findings = reference_findings(old_repo_root)
    write_json_atomic(findings, development / "reference_findings.json")
    report = {
        "parity_schema": "sift-reference-parity-v1",
        "reference_repository": str(old_repo_root.resolve()),
        "reference_files": findings["source_files"],
        "reference_findings_sha256": findings["findings_sha256"],
        "pair_count": len(rows),
        "exact_score_matches": sum(bool(row["score_equal"]) for row in rows),
        "exact_count_matches": sum(bool(row["counts_equal"]) for row in rows),
        "status": "pass" if all(row["score_equal"] and row["counts_equal"] for row in rows) else "deviation",
        "rows": rows,
    }
    write_json_atomic(report, development / "reference_parity_report.json")
    return report


def run_pilot(results_root: Path) -> dict[str, Any]:
    development = _development_dir(results_root)
    if (development / FINAL_CONFIG_FILENAME).exists():
        raise FileExistsError(
            "The final SIFT config is already frozen. Remove development artifacts explicitly before a new pilot."
        )
    pairs = read_pilot_pairs(development / "pilot_pairs.csv")
    specs = ablation_specs()
    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[pilot] candidate={spec.candidate_id}", flush=True)
        rows = run_pilot_candidate(
            pairs,
            spec,
            progress=lambda current, total, candidate=spec.candidate_id: print(
                f"[pilot] {candidate} {current}/{total}", flush=True
            ),
        )
        candidate_dir = development / "ablations" / spec.candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(spec.config.as_dict(), candidate_dir / "config.json")
        write_pilot_results(rows, candidate_dir / "pilot_results.csv")
        all_rows.extend(rows)
    return finalize_pilot(results_root, all_rows=all_rows)


def run_reference_pilot(results_root: Path) -> dict[str, Any]:
    development = _development_dir(results_root)
    pairs = read_pilot_pairs(development / "pilot_pairs.csv")
    spec = next(item for item in ablation_specs() if not item.selection_eligible)
    print(f"[pilot] candidate={spec.candidate_id}", flush=True)
    rows = run_pilot_candidate(
        pairs,
        spec,
        progress=lambda current, total: print(
            f"[pilot] {spec.candidate_id} {current}/{total}", flush=True
        ),
    )
    candidate_dir = development / "ablations" / spec.candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(spec.config.as_dict(), candidate_dir / "config.json")
    write_pilot_results(rows, candidate_dir / "pilot_results.csv")
    return {
        "candidate_id": spec.candidate_id,
        "selection_eligible": False,
        "pair_count": len(rows),
        "results": str(candidate_dir / "pilot_results.csv"),
    }


def finalize_pilot(
    results_root: Path,
    *,
    all_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    development = _development_dir(results_root)
    specs = ablation_specs()
    if all_rows is None:
        all_rows = []
        for spec in specs:
            path = development / "ablations" / spec.candidate_id / "pilot_results.csv"
            with path.open("r", newline="", encoding="utf-8") as handle:
                all_rows.extend(list(csv.DictReader(handle)))
    final_config, decision_rule, metrics = select_final_config(all_rows, specs)
    config_path = development / FINAL_CONFIG_FILENAME
    decision_path = development / "decision_rule.json"
    metrics_path = development / "ablation_report.csv"
    write_json_atomic(final_config.as_dict(), config_path)
    write_json_atomic(decision_rule, decision_path)
    write_csv_atomic(metrics, metrics_path, list(metrics[0]))
    selection = next(row for row in metrics if row["selected"])
    if any(row["candidate_id"] == "reference_reproduction" for row in all_rows):
        paired = paired_reference_comparison(
            all_rows,
            selected_candidate_id=str(selection["candidate_id"]),
            score_mode=str(selection["score_mode"]),
            selected_thresholds=decision_rule["thresholds_by_dataset"],
        )
        paired_path = development / "paired_reference_comparison.csv"
        write_csv_atomic(paired, paired_path, list(paired[0]))
    else:
        paired_path = None
    report = {
        "pilot_report_schema": "sift-pilot-report-v1",
        "pilot_pairs_sha256": file_sha256(development / "pilot_pairs.csv"),
        "candidate_count": len(specs),
        "score_candidate_count": 4,
        "selected": selection,
        "final_config_path": str(config_path),
        "final_config_sha256": file_sha256(config_path),
        "decision_rule_path": str(decision_path),
        "decision_rule_sha256": file_sha256(decision_path),
        "selection_policy": (
            "development only; safe primary-FAR candidates ranked by macro plain-roll TAR, self acceptance, "
            "geometry failure, zero-score rate, latency, then stable identifiers"
        ),
        "paired_reference_comparison": None if paired_path is None else str(paired_path),
    }
    write_json_atomic(report, development / "pilot_report.json")
    return report


def run_primary_bundles(
    repo_root: Path,
    results_root: Path,
    *,
    data_root: Path,
    skip_existing: bool,
) -> list[dict[str, Any]]:
    safety = run_safety_gate(results_root)
    if safety["status"] != "pass":
        raise RuntimeError(f"SIFT safety gate did not pass: {safety}")
    config_path = _development_dir(results_root) / FINAL_CONFIG_FILENAME
    decision_path = _development_dir(results_root) / "decision_rule.json"
    config_sha256 = file_sha256(config_path)
    decision_sha256 = file_sha256(decision_path)
    config = SiftGeometricConfig.from_json(config_path)
    metadata_rows: list[dict[str, Any]] = []
    for dataset in ("sd300b", "sd300c"):
        for protocol in ("plain_self", "roll_self", "plain_roll"):
            adapter = SiftGeometricAdapter(config)
            print(f"[primary] {dataset}/{protocol}", flush=True)
            try:
                metadata = run_benchmark_manifest(
                    manifest_path=repo_root / "protocols" / dataset / f"{protocol}.csv",
                    adapter=adapter,
                    expected_dataset=dataset,
                    expected_protocol=protocol,
                    results_root=results_root,
                    data_root=data_root,
                    skip_existing=skip_existing,
                    progress_callback=lambda current, total, d=dataset, p=protocol: print(
                        f"[primary] {d}/{p} {current}/{total}", flush=True
                    ),
                )
            finally:
                adapter.close()
            metadata_rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "bundle": str(Path(metadata["result"]["path"]).parent),
                    "config_hash": metadata["run_spec"]["config_hash"],
                    "implementation_hash": metadata["run_spec"]["implementation_hash"],
                    "result_sha256": metadata["result"]["sha256"],
                    "score_payload_sha256": metadata["result"]["score_payload_sha256"],
                }
            )
            if file_sha256(config_path) != config_sha256 or file_sha256(decision_path) != decision_sha256:
                raise RuntimeError("Frozen SIFT config or decision rule changed during primary evaluation.")
    return metadata_rows


def run_safety_gate(results_root: Path, *, repeated_pair_count: int = 4) -> dict[str, Any]:
    development = _development_dir(results_root)
    config_path = development / FINAL_CONFIG_FILENAME
    config_sha_before = file_sha256(config_path)
    config = SiftGeometricConfig.from_json(config_path)
    parity = json.loads((development / "reference_parity_report.json").read_text(encoding="utf-8"))
    with (development / "ablation_report.csv").open("r", newline="", encoding="utf-8") as handle:
        selected_rows = [row for row in csv.DictReader(handle) if str(row.get("selected", "")).lower() == "true"]
    if len(selected_rows) != 1:
        raise ValueError(f"Expected one selected pilot row, found {len(selected_rows)}.")
    selected = selected_rows[0]
    pairs = read_pilot_pairs(development / "pilot_pairs.csv")[:repeated_pair_count]
    repeated = []
    adapter = SiftGeometricAdapter(config)
    try:
        for pair in pairs:
            observations = []
            for _ in range(2):
                prepared_a = adapter.prepare(Path(pair.path_a), pair.metadata("a"))
                prepared_b = adapter.prepare(Path(pair.path_b), pair.metadata("b"))
                compared = adapter.compare(prepared_a.representation, prepared_b.representation)
                observations.append(
                    {
                        "raw_score": float(compared.raw_score),
                        "keypoint_count_a": compared.diagnostics["keypoint_count_a"],
                        "keypoint_count_b": compared.diagnostics["keypoint_count_b"],
                        "ratio_match_count_a_to_b": compared.diagnostics["ratio_match_count_a_to_b"],
                        "matches_submitted_to_geometry": compared.diagnostics[
                            "matches_submitted_to_geometry"
                        ],
                        "geometric_inlier_count": compared.diagnostics["geometric_inlier_count"],
                    }
                )
            repeated.append(
                {
                    "pair_id": pair.pair_id,
                    "first": observations[0],
                    "second": observations[1],
                    "exactly_equal": observations[0] == observations[1],
                }
            )
        metadata = adapter.metadata()
    finally:
        adapter.close()
    checks = {
        "reference_parity_passed": parity.get("status") == "pass",
        "opencv_version_pinned": metadata.implementation_provenance.get("opencv_version") == "4.12.0",
        "opencv_distribution_exact": metadata.implementation_provenance.get("opencv_distribution")
        == {"name": "opencv-python", "version": "4.12.0.88", "status": "ok"},
        "pilot_preparation_failure_rate_acceptable": float(selected["preparation_failure_rate"]) <= 0.05,
        "pilot_self_acceptance_not_broadly_failed": float(selected["macro_self_acceptance_at_primary"]) >= 0.80,
        "pilot_impostor_acceptance_within_guardrail": float(
            selected["macro_impostor_acceptance_at_primary"]
        ) <= float(selected["primary_far_guardrail"]),
        "pilot_zero_score_rate_not_extreme": float(selected["zero_score_rate"]) < 0.95,
        "pilot_geometry_failure_rate_not_extreme": float(selected["geometry_failure_rate"]) < 0.95,
        "repeated_subset_exact": all(row["exactly_equal"] for row in repeated),
        "repeated_scores_finite": all(
            math.isfinite(float(row["first"]["raw_score"]))
            and math.isfinite(float(row["second"]["raw_score"]))
            for row in repeated
        ),
        "config_unchanged_during_gate": file_sha256(config_path) == config_sha_before,
    }
    report = {
        "safety_gate_schema": "sift-safety-gate-v1",
        "status": "pass" if all(checks.values()) else "stop",
        "checks": checks,
        "selected_pilot_metrics": selected,
        "repeated_subset": repeated,
        "config_sha256": config_sha_before,
    }
    write_json_atomic(report, development / "safety_gate.json")
    return report


def _development_dir(results_root: Path) -> Path:
    return results_root.resolve() / "sift_geometric" / DEVELOPMENT_DIRNAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe SIFT geometric study.")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "parity",
            "pilot",
            "pilot-reference",
            "finalize",
            "safety",
            "run",
            "report",
            "all",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\fingerprint-datasets"))
    parser.add_argument("--reference-repo", type=Path, default=Path(r"C:\fingerprint-research"))
    parser.add_argument("--genuine-per-condition", type=int, default=12)
    parser.add_argument("--impostors-per-dataset", type=int, default=200)
    parser.add_argument("--parity-pairs", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    results_root = args.results_root.resolve()
    if args.command in {"prepare", "all"}:
        print(
            json.dumps(
                prepare_development(
                    repo_root,
                    results_root,
                    genuine_per_condition=args.genuine_per_condition,
                    impostors_per_dataset=args.impostors_per_dataset,
                ),
                indent=2,
            )
        )
    if args.command in {"parity", "all"}:
        print(
            json.dumps(
                run_internal_reference_parity(
                    repo_root,
                    args.reference_repo.resolve(),
                    results_root,
                    pair_count=args.parity_pairs,
                ),
                indent=2,
            )
        )
    if args.command in {"pilot", "all"}:
        print(json.dumps(run_pilot(results_root), indent=2))
    if args.command == "pilot-reference":
        print(json.dumps(run_reference_pilot(results_root), indent=2))
    if args.command == "finalize":
        print(json.dumps(finalize_pilot(results_root), indent=2))
    if args.command in {"safety", "all"}:
        print(json.dumps(run_safety_gate(results_root), indent=2))
    if args.command in {"run", "all"}:
        print(
            json.dumps(
                run_primary_bundles(
                    repo_root,
                    results_root,
                    data_root=args.data_root.resolve(),
                    skip_existing=args.skip_existing,
                ),
                indent=2,
            )
        )
    if args.command in {"report", "all"}:
        print(json.dumps(build_final_reports(repo_root, results_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
