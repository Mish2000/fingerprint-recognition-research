"""Technical v3/v4 scale-normalization ablation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

from ..hashing import file_sha256
from .preflight_v4 import METHOD_RESULTS_RELATIVE, PILOT_RELATIVE


V3_PILOT_RELATIVE = Path(
    "results/pilots/harriszplus_rootsift_geometric_joint_500_v3"
)
CONDITIONS = (
    ("sd300b", "plain_self"),
    ("sd300b", "roll_self"),
    ("sd300b", "plain_roll_genuine"),
    ("sd300b", "plain_roll_negative"),
    ("sd300c", "plain_self"),
    ("sd300c", "roll_self"),
    ("sd300c", "plain_roll_genuine"),
    ("sd300c", "plain_roll_negative"),
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_cell(row: Mapping[str, str], field: str) -> dict[str, Any]:
    value = row.get(field, "")
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _numbers(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            output.append(number)
    return output


def _median(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    return statistics.median(numbers) if numbers else None


def _bundle_summary(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    ok = [row for row in rows if row.get("status") == "ok"]
    prepare = [
        _json_cell(row, field)
        for row in ok
        for field in ("prepare_a_diagnostics", "prepare_b_diagnostics")
    ]
    compare = [_json_cell(row, "compare_diagnostics") for row in ok]
    scale_counts = {
        str(index): {
            "median_a_plus_b_candidates_after_eigen_ratio": _median(
                record.get("scales", {})
                .get(str(index), {})
                .get("counts", {})
                .get("candidates_after_eigen_ratio")
                for record in prepare
            ),
            "median_final_representation_fraction": _median(
                (
                    record.get("harriszplus_scale_indices", []).count(index)
                    / max(
                        1,
                        len(record.get("harriszplus_scale_indices", [])),
                    )
                )
                for record in prepare
            ),
        }
        for index in range(5)
    }
    sizes = [
        item.get("opencv_keypoint_size")
        for record in prepare
        for item in record.get("scale_mapping_records", [])
        if isinstance(item, Mapping)
    ]
    physical_total = max(
        _numbers(
            record.get("vram_physical_total_bytes")
            for record in prepare
        ),
        default=None,
    )
    return {
        "row_count": len(rows),
        "ok_count": len(ok),
        "failure_count": len(rows) - len(ok),
        "accepted_at_threshold_4": sum(
            float(row["raw_score"]) >= 4.0 for row in ok
        ),
        "raw_score_median": _median(row.get("raw_score") for row in ok),
        "raw_score_mean": (
            statistics.fmean(_numbers(row.get("raw_score") for row in ok))
            if ok
            else None
        ),
        "candidate_count_median": _median(
            record.get("candidates_after_duplicate_removal")
            for record in prepare
        ),
        "border_safe_candidate_count_median": _median(
            record.get("candidates_after_border_exclusion")
            for record in prepare
        ),
        "final_keypoint_count_median": _median(
            record.get("final_keypoint_count") for record in prepare
        ),
        "saturation_factor_median": _median(
            float(record.get("final_keypoint_count", 0)) / 3000.0
            for record in prepare
        ),
        "cap_active_fraction": (
            sum(
                float(record.get("final_keypoint_count", 0)) >= 3000.0
                for record in prepare
            )
            / len(prepare)
            if prepare
            else None
        ),
        "scale_distribution": scale_counts,
        "keypoint_size_median_px": _median(sizes),
        "mutual_match_count_median": _median(
            record.get("mutual_match_count") for record in compare
        ),
        "inlier_count_median": _median(
            record.get("geometric_inlier_count") for record in compare
        ),
        "total_ms_median": _median(row.get("total_ms") for row in ok),
        "prepare_ms_median": _median(
            value
            for row in ok
            for value in (row.get("prepare_a_ms"), row.get("prepare_b_ms"))
        ),
        "compare_ms_median": _median(row.get("compare_ms") for row in ok),
        "peak_vram_allocated_bytes": max(
            _numbers(
                record.get(
                    "peak_vram_allocated",
                    record.get("peak_vram_allocated_bytes"),
                )
                for record in prepare
            ),
            default=None,
        ),
        "peak_vram_reserved_bytes": max(
            _numbers(
                record.get(
                    "peak_vram_reserved",
                    record.get("peak_vram_reserved_bytes"),
                )
                for record in prepare
            ),
            default=None,
        ),
        "physical_vram_bytes": physical_total,
        "allocated_and_reserved_summed": False,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_v3_v4_scale_normalization_comparison(
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Publish a technical ablation, separate from the supervisor report."""

    root = project_root.resolve()
    v3_root = root / V3_PILOT_RELATIVE
    v4_root = root / PILOT_RELATIVE
    conditions: dict[str, Any] = {}
    for dataset, label in CONDITIONS:
        key = f"{dataset}/{label}"
        conditions[key] = {
            "v3": _bundle_summary(
                v3_root / "runs" / dataset / label / "pairs.csv"
            ),
            "v4": _bundle_summary(
                v4_root / "runs" / dataset / label / "pairs.csv"
            ),
        }
    contract_path = (
        root
        / METHOD_RESULTS_RELATIVE
        / "preflight/physical_scale_contract_v4.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight_path = (
        root
        / METHOD_RESULTS_RELATIVE
        / "preflight/engineering_preflight_pass.json"
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "harriszplus-v3-v4-scale-comparison-v4",
        "report_type": "technical_ablation_not_supervisor_report",
        "physical_scale_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "passed": contract["passed"],
            "sd300b": contract["sd300b"],
            "sd300c": contract["sd300c"],
            "comparisons": contract["comparisons"],
            "uniform_q": contract["uniform_q"],
            "ransac": contract["ransac"],
        },
        "development_diagnostic_comparison": {
            "v4_engineering_pair_count": preflight[
                "downstream_semantic_validation"
            ]["pair_count"],
            "v4_exact_cpu_cuda_score_rate": preflight[
                "downstream_semantic_validation"
            ]["exact_score_rate"],
            "v3_reference": preflight[
                "development_diagnostic_v3_reference"
            ],
            "used_for_tuning": False,
        },
        "pilot_conditions": conditions,
        "interpretation_policy": {
            "genuine_and_negative_reported_together": True,
            "genuine_acceptance_alone_cannot_establish_superiority": True,
            "performance_threshold_applied": False,
            "parameter_tuning_performed": False,
        },
    }
    report_root = root / METHOD_RESULTS_RELATIVE / "report"
    json_path = report_root / "v3_v4_scale_normalization_comparison.json"
    md_path = report_root / "v3_v4_scale_normalization_comparison.md"
    report_root.mkdir(parents=True, exist_ok=True)
    if json_path.exists() or md_path.exists():
        raise ValueError("Immutable v3/v4 technical comparison already exists.")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# HarrisZ+ v3 מול v4 — scale normalization",
        "",
        "דוח זה הוא ablation טכני נפרד ואינו דוח המנחה.",
        "",
        f"- Physical-scale contract: `{'PASS' if contract['passed'] else 'FAIL'}`",
        "- v4 משנה רק את פרשנות הפרמטרים המרחביים לפי manifest PPI.",
        "- אין סף ביצועים בדוח זה ולא בוצע tuning מתוצאות 500.",
        "- allocated ו-reserved מוצגים בנפרד ואינם מסוכמים.",
        "",
        "## תוצאות שמונת התנאים",
        "",
        "| תנאי | גרסה | ok/total | accepted ≥4 | score median | candidates median | saturation | mutual median | inliers median | total ms median | peak allocated | peak reserved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, versions in conditions.items():
        for version in ("v3", "v4"):
            row = versions[version]
            lines.append(
                "| "
                + " | ".join(
                    (
                        key,
                        version,
                        f"{row['ok_count']}/{row['row_count']}",
                        str(row["accepted_at_threshold_4"]),
                        _fmt(row["raw_score_median"]),
                        _fmt(row["candidate_count_median"]),
                        _fmt(row["saturation_factor_median"]),
                        _fmt(row["mutual_match_count_median"]),
                        _fmt(row["inlier_count_median"]),
                        _fmt(row["total_ms_median"]),
                        _fmt(row["peak_vram_allocated_bytes"]),
                        _fmt(row["peak_vram_reserved_bytes"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Physical support",
            "",
            "| scale | parameter | B mm | C mm | delta mm | pass |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in contract["comparisons"]:
        lines.append(
            f"| {item['scale_index']} | {item['parameter']} | "
            f"{item['b_mm']:.6f} | {item['c_mm']:.6f} | "
            f"{item['absolute_delta_mm']:.6f} | {item['passed']} |"
        )
    lines.extend(
        [
            "",
            "יש לקרוא genuine ו-negative יחד; genuine acceptance לבדו אינו "
            "בסיס לטענה ש-v4 טוב יותר.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        **payload,
        "json": {
            "path": str(json_path),
            "sha256": file_sha256(json_path),
        },
        "markdown": {
            "path": str(md_path),
            "sha256": file_sha256(md_path),
        },
    }


__all__ = ["build_v3_v4_scale_normalization_comparison"]
