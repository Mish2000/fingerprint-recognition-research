"""Read-only provenance and extracted facts for the historical SIFT reference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fingerprint_benchmark.hashing import file_sha256, stable_hash


REFERENCE_FILES = (
    "src/fpbench/matchers/matching_baseline.py",
    "src/fpbench/matchers/dedicated_matcher.py",
    "src/fpbench/preprocess/preprocess.py",
    "pipelines/benchmark/eval_classic.py",
    "pipelines/benchmark/method_profiles.py",
    "pipelines/benchmark/evaluate.py",
    "docs/benchmark_methods.md",
    "scripts/diagnostics/run_sift_plain_roll_experiments.py",
    "scripts/diagnostics/run_sift_plain_roll_v2_grid3_validation.py",
    "scripts/diagnostics/run_sift_plain_roll_v2_hypothesis_tests.py",
    "scripts/diagnostics/sift_score_variant_sweep.py",
    "scripts/diagnostics/official_sift_plain_roll_v2_comparison.py",
    "tests/test_eval_classic_parity.py",
    "tests/test_eval_classic_wiring.py",
    "tests/test_sift_plain_roll_v2_external_validation.py",
    "tests/test_sift_plain_roll_v2_grid3_validation.py",
    "tests/test_sift_plain_roll_v2_hypothesis_tests.py",
)
REFERENCE_ARTIFACT_DIRECTORY = (
    "artifacts/reports/benchmark/plain_roll_final_baselines_v2_anatomical_full_pairs"
)


def reference_findings(reference_root: Path) -> dict[str, Any]:
    source_hashes = {
        relative: file_sha256(reference_root / Path(relative)) for relative in REFERENCE_FILES
    }
    artifact_root = reference_root / Path(REFERENCE_ARTIFACT_DIRECTORY)
    artifact_hashes = {
        str(path.relative_to(artifact_root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file()
    }
    payload = {
        "reference_findings_schema": "sift-reference-findings-v1",
        "reference_repository": str(reference_root.resolve()),
        "usage_policy": (
            "read-only implementation reference and development baseline; historical metrics are not selection inputs for new evaluation"
        ),
        "source_files": source_hashes,
        "source_inventory_sha256": stable_hash(source_hashes),
        "artifact_directory": str(artifact_root.resolve()),
        "artifact_file_count": len(artifact_hashes),
        "artifact_files": artifact_hashes,
        "artifact_inventory_sha256": stable_hash(artifact_hashes),
        "opencv_sift_constructor": {
            "nfeatures": 3000,
            "nOctaveLayers": 3,
            "contrastThreshold": 0.04,
            "edgeThreshold": 10.0,
            "sigma": 1.6,
            "note": "nfeatures is an OpenCV target/configured cap and observed output can differ by a keypoint",
        },
        "preprocessing": {
            "load": "cv2.imread(path, cv2.IMREAD_GRAYSCALE)",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "resize_policy": "scale long edge to 768 with INTER_AREA and center in zero-valued 768 square",
            "blur": "disabled",
            "post_resize_normalization": "cv2.normalize to 0..255 uint8",
            "mask": "none",
        },
        "matching": {
            "descriptor_type": "standard OpenCV SIFT float32",
            "matcher": "BFMatcher",
            "norm": "NORM_L2",
            "cross_check": False,
            "direction": "A_to_B",
            "knn_k": 2,
            "lowe_ratio": 0.75,
            "lowe_operator": "strict_less_than",
        },
        "geometry": {
            "model": "estimateAffine2D full affine",
            "method": "RANSAC",
            "minimum_ratio_matches": 3,
            "reprojection_threshold_pixels": 3.0,
            "confidence": 0.99,
            "maximum_iterations": 2000,
            "refine_iterations": 10,
            "rng_seed": 0,
            "coordinate_policy": "resized-image pixels; not PPI-normalized",
        },
        "score": {
            "formula": "inliers * (inliers / matches) * log1p(matches)",
            "components": ["inlier_count", "inlier_ratio", "ratio_match_count"],
            "direction": "higher_is_more_similar",
        },
        "failure_to_score": {
            "unreadable_image": 0.0,
            "missing_descriptors": 0.0,
            "fewer_than_eight_descriptors_on_either_side": 0.0,
            "fewer_than_three_ratio_matches": 0.0,
            "failed_or_missing_affine_inlier_mask": 0.0,
            "zero_inliers": 0.0,
            "status_policy": "old score files did not distinguish these preparation and geometry failures from valid zero scores",
        },
        "historical_observations_not_used_for_selection": {
            "composite_score": "raised low-FAR plain-to-roll TAR compared with inliers over minimum keypoints",
            "timing": "in-process feature caching mixed cache hits into reported timings",
            "resolution": "fixed pixel RANSAC threshold was not physically comparable across 1000 and 2000 PPI",
            "ablation": "resize, enhancement, geometry, and score were often changed together",
            "score_sweep": "exploratory same-set sweeps were not independent evaluation evidence",
            "crop_probes": "rescued genuine pairs but also raised some impostor scores and required validation-only guardrails",
            "low_far_test_tar_at_one_percent": {
                "sd300b": 0.4859392575928009,
                "sd300c": 0.4544431946006749,
            },
            "test_far_at_one_percent": {
                "sd300b": 0.007874015748031496,
                "sd300c": 0.008998875140607425,
            },
        },
    }
    payload["findings_sha256"] = stable_hash(payload)
    return payload
