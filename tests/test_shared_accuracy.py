from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fingerprint_benchmark.contract import (
    OK,
    PREPARE_B_FAILURE,
    CompareOutcome,
    MethodExecutionError,
    PrepareOutcome,
    PreparedRepresentation,
)
from fingerprint_benchmark.manifest import PairRecord
from fingerprint_benchmark import shared_accuracy
from fingerprint_benchmark.shared_accuracy import (
    LEGACY_THRESHOLDS,
    AccuracyPair,
    SharedAccuracyError,
    _accuracy_pair_from_row,
    _accuracy_pair_row,
    _condition_from_preflight,
    _paired_resolution,
    _source_preflight_comparison,
    _target_key,
    generate_logical_impostors,
    materialize_genuine,
    materialize_impostors,
    project_reused_genuine_scores,
    run_prepared_scores,
    split_genuine_pairs,
    validate_logical_impostors,
    validate_shared_split,
)


def _pair_record(
    root: Path,
    dataset: str,
    subject: str,
    finger: int,
) -> PairRecord:
    dataset_code = "b" if dataset == "sd300b" else "c"
    return PairRecord(
        pair_id=f"{dataset}_{subject}_{finger:02d}",
        dataset=dataset,
        protocol="plain_roll",
        subject_id=subject,
        canonical_finger_position=finger,
        ppi=500 if dataset == "sd300b" else 1000,
        raw_frgp_a=finger,
        raw_frgp_b=finger + 10,
        path_a=root / dataset_code / subject / f"plain-{finger}.wsq",
        path_b=root / dataset_code / subject / f"roll-{finger}.wsq",
    )


def _accuracy_pair(
    root: Path,
    pair_id: str,
    left_subject: str,
    right_subject: str,
    *,
    dataset: str = "sd300b",
    split: str = "development",
    finger: int = 1,
    left_path: Path | None = None,
    right_path: Path | None = None,
    raw_frgp_a: int = 1,
    raw_frgp_b: int = 11,
) -> AccuracyPair:
    return AccuracyPair(
        accuracy_pair_id=pair_id,
        pair_label="genuine" if left_subject == right_subject else "impostor",
        dataset=dataset,
        split=split,
        canonical_finger_position=finger,
        subject_id_a=left_subject,
        subject_id_b=right_subject,
        ppi=500,
        raw_frgp_a=raw_frgp_a,
        raw_frgp_b=raw_frgp_b,
        path_a=left_path or root / dataset / left_subject / f"plain-{finger}.wsq",
        path_b=right_path or root / dataset / right_subject / f"roll-{finger}.wsq",
        source_pair_id_a=f"source-{dataset}-{left_subject}-{finger}",
        source_pair_id_b=f"source-{dataset}-{right_subject}-{finger}",
    )


class _FakeAdapter:
    def __init__(self, *, fail_paths: set[Path] | None = None) -> None:
        self.fail_paths = {path.resolve() for path in fail_paths or set()}
        self.prepare_calls: list[tuple[Path, dict[str, Any]]] = []
        self.compare_calls: list[tuple[str, str]] = []

    def prepare(self, image_path: Path, image_metadata: dict[str, Any]) -> PrepareOutcome:
        resolved = image_path.resolve()
        self.prepare_calls.append((resolved, dict(image_metadata)))
        if resolved in self.fail_paths:
            raise MethodExecutionError(
                "synthetic_prepare_failure",
                "synthetic preparation failure",
                diagnostics={"path": str(resolved), "timing_ms": 3.0},
            )
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method="sourceafis",
                method_version="fake-v1",
                representation_format="fake-template",
                representation_version="v1",
                payload=str(resolved),
            ),
            diagnostics={"stable": "ignored-for-sourceafis", "elapsed_ms": 1.0},
        )

    def compare(
        self,
        representation_a: PreparedRepresentation,
        representation_b: PreparedRepresentation,
    ) -> CompareOutcome:
        self.compare_calls.append((representation_a.payload, representation_b.payload))
        return CompareOutcome(
            raw_score=float(len(self.compare_calls)),
            diagnostics={"stable": "ignored-for-sourceafis", "timing_ms": 2.0},
        )


def _run(adapter: _FakeAdapter, pairs: list[AccuracyPair], **overrides: str):
    arguments = {
        "adapter": adapter,
        "pairs": pairs,
        "method": "sourceafis",
        "method_version": "fake-v1",
        "frozen_config_hash": "config-hash",
        "implementation_hash": "implementation-hash",
        "accuracy_runner_sha256": "runner-hash",
        "cache_scope_id": "scope-1",
    }
    arguments.update(overrides)
    return run_prepared_scores(**arguments)


def test_validate_shared_split_requires_exact_deterministic_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path = tmp_path / "results/sift_geometric/development/subject_split.json"
    split_path.parent.mkdir(parents=True)
    payload = {
        "split_version": "sift-split-v1",
        "development_subjects": ["s1", "s2"],
        "evaluation_subjects": ["s3", "s4"],
        "development_count": 2,
        "evaluation_count": 2,
        "dataset_subject_alignment": True,
        "subject_lists_sha256": "subject-hash",
        "split_rule": "deterministic-test-rule",
    }
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(shared_accuracy, "build_subject_split", lambda _root: dict(payload))

    validated = validate_shared_split(tmp_path)

    assert validated["development_subjects"] == ["s1", "s2"]
    assert validated["evaluation_subjects"] == ["s3", "s4"]
    assert validated["validation"]["deterministic_rebuild_exact"] is True
    assert validated["validation"]["no_subject_leakage"] is True

    changed = dict(payload, evaluation_subjects=["s3", "different"])
    monkeypatch.setattr(shared_accuracy, "build_subject_split", lambda _root: changed)
    with pytest.raises(SharedAccuracyError, match="does not exactly match"):
        validate_shared_split(tmp_path)


def test_split_genuine_pairs_partitions_subjects_and_preserves_bc_order(tmp_path: Path) -> None:
    identities = [("s1", 1), ("s1", 2), ("s2", 1), ("s3", 1)]
    pairs = {
        dataset: [
            _pair_record(tmp_path, dataset, subject, finger) for subject, finger in identities
        ]
        for dataset in ("sd300b", "sd300c")
    }

    partitioned = split_genuine_pairs(
        pairs,
        {"development_subjects": ["s1", "s3"], "evaluation_subjects": ["s2"]},
    )

    assert [pair.subject_id for pair in partitioned[("sd300b", "development")]] == [
        "s1",
        "s1",
        "s3",
    ]
    assert [pair.subject_id for pair in partitioned[("sd300b", "evaluation")]] == ["s2"]
    for split in ("development", "evaluation"):
        b = [
            (pair.subject_id, pair.canonical_finger_position)
            for pair in partitioned[("sd300b", split)]
        ]
        c = [
            (pair.subject_id, pair.canonical_finger_position)
            for pair in partitioned[("sd300c", split)]
        ]
        assert b == c


@pytest.mark.parametrize(
    ("split_reference", "mutate_c", "message"),
    [
        (
            {"development_subjects": ["s1"], "evaluation_subjects": ["s1", "s2"]},
            False,
            "Subject leakage",
        ),
        (
            {"development_subjects": ["s1"], "evaluation_subjects": []},
            False,
            "Unassigned subject",
        ),
        (
            {"development_subjects": ["s1", "s2"], "evaluation_subjects": []},
            True,
            "B/C genuine alignment",
        ),
    ],
)
def test_split_genuine_pairs_rejects_leakage_unassigned_and_bc_misalignment(
    tmp_path: Path,
    split_reference: dict[str, list[str]],
    mutate_c: bool,
    message: str,
) -> None:
    b = [_pair_record(tmp_path, "sd300b", subject, 1) for subject in ("s1", "s2")]
    c = [_pair_record(tmp_path, "sd300c", subject, 1) for subject in ("s1", "s2")]
    if mutate_c:
        c.reverse()

    with pytest.raises(SharedAccuracyError, match=message):
        split_genuine_pairs({"sd300b": b, "sd300c": c}, split_reference)


def test_impostor_generation_is_deterministic_balanced_and_has_no_reciprocals() -> None:
    development = {f"d{index:02d}" for index in range(50)}
    evaluation = {f"e{index:02d}" for index in range(50)}
    identities = [
        (subject, finger)
        for subject in sorted(development | evaluation)
        for finger in (1, 2)
    ]
    split_subjects = {"development": development, "evaluation": evaluation}

    first = generate_logical_impostors(identities, split_subjects)
    second = generate_logical_impostors(list(reversed(identities)), split_subjects)

    assert first == second
    for split, subjects in split_subjects.items():
        rows = first[split]
        counts = Counter(
            (row["plain_subject_id"], row["canonical_finger_position"]) for row in rows
        )
        assert len(rows) == len(subjects) * 2 * 10
        assert set(counts.values()) == {10}
        assert all(row["plain_subject_id"] != row["roll_subject_id"] for row in rows)
        assert all(row["plain_subject_id"] in subjects for row in rows)
        assert all(row["roll_subject_id"] in subjects for row in rows)
        unordered = {
            (
                row["canonical_finger_position"],
                *sorted((row["plain_subject_id"], row["roll_subject_id"])),
            )
            for row in rows
        }
        assert len(unordered) == len(rows)


def test_impostor_generation_uses_all_legal_pairs_when_ten_are_unavailable() -> None:
    development = {f"d{index:02d}" for index in range(6)}
    evaluation = {f"e{index:02d}" for index in range(6)}
    identities = [(subject, 1) for subject in sorted(development | evaluation)]

    generated = generate_logical_impostors(
        identities,
        {"development": development, "evaluation": evaluation},
    )

    for rows in generated.values():
        counts = Counter(row["plain_subject_id"] for row in rows)
        assert len(rows) == 15  # Every unordered edge among six identities appears once.
        assert sum(counts.values()) == 15
        assert min(counts.values()) >= 1
        assert max(counts.values()) < 10

    contaminated = [dict(row) for row in generated["development"]]
    contaminated[0]["roll_subject_id"] = contaminated[0]["plain_subject_id"]
    with pytest.raises(SharedAccuracyError, match="Genuine contamination"):
        validate_logical_impostors(
            contaminated,
            "development",
            {(subject, 1) for subject in development},
        )


def test_materialization_keeps_exact_logical_ids_aligned_across_b_and_c(tmp_path: Path) -> None:
    logical = [
        {
            "accuracy_pair_id": "shared_imp_development_01_s1_s2",
            "split": "development",
            "plain_subject_id": "s1",
            "roll_subject_id": "s2",
            "canonical_finger_position": 1,
            "selection_sha256": "selection-hash",
            "selection_rank": 1,
        }
    ]
    base = {
        dataset: [_pair_record(tmp_path, dataset, subject, 1) for subject in ("s1", "s2")]
        for dataset in ("sd300b", "sd300c")
    }

    b_impostor = materialize_impostors(logical, base["sd300b"], "sd300b")
    c_impostor = materialize_impostors(logical, base["sd300c"], "sd300c")
    assert [pair.accuracy_pair_id for pair in b_impostor] == [
        "shared_imp_development_01_s1_s2"
    ]
    assert [pair.accuracy_pair_id for pair in b_impostor] == [
        pair.accuracy_pair_id for pair in c_impostor
    ]
    assert (b_impostor[0].subject_id_a, b_impostor[0].subject_id_b) == ("s1", "s2")
    assert (c_impostor[0].subject_id_a, c_impostor[0].subject_id_b) == ("s1", "s2")
    assert b_impostor[0].path_a != c_impostor[0].path_a
    assert b_impostor[0].path_b != c_impostor[0].path_b

    b_genuine = materialize_genuine(base["sd300b"], "development")
    c_genuine = materialize_genuine(base["sd300c"], "development")
    assert [pair.accuracy_pair_id for pair in b_genuine] == [
        "shared_gen_s1_01",
        "shared_gen_s2_01",
    ]
    assert [pair.accuracy_pair_id for pair in b_genuine] == [
        pair.accuracy_pair_id for pair in c_genuine
    ]
    assert _accuracy_pair_from_row(_accuracy_pair_row(b_impostor[0])) == b_impostor[0]


def test_prepared_scoring_prepares_each_unique_image_exactly_once(tmp_path: Path) -> None:
    plain_a = tmp_path / "images/plain-a.wsq"
    roll_b = tmp_path / "images/roll-b.wsq"
    roll_c = tmp_path / "images/roll-c.wsq"
    plain_d = tmp_path / "images/plain-d.wsq"
    pairs = [
        _accuracy_pair(
            tmp_path,
            "pair-1",
            "a",
            "b",
            left_path=plain_a,
            right_path=roll_b,
        ),
        _accuracy_pair(
            tmp_path,
            "pair-2",
            "a",
            "c",
            left_path=plain_a,
            right_path=roll_c,
        ),
        _accuracy_pair(
            tmp_path,
            "pair-3",
            "d",
            "b",
            left_path=plain_d,
            right_path=roll_b,
        ),
    ]
    adapter = _FakeAdapter()

    rows, metadata = _run(adapter, pairs)

    prepared_paths = Counter(path for path, _metadata in adapter.prepare_calls)
    assert prepared_paths == Counter(
        {plain_a.resolve(): 1, roll_b.resolve(): 1, roll_c.resolve(): 1, plain_d.resolve(): 1}
    )
    assert [row["accuracy_pair_id"] for row in rows] == ["pair-1", "pair-2", "pair-3"]
    assert [row["status"] for row in rows] == [OK, OK, OK]
    assert len(adapter.compare_calls) == 3
    assert metadata["unique_image_count"] == 4
    assert metadata["prepare_invocation_count"] == 4
    assert metadata["prepare_failure_count"] == 0
    assert metadata["comparison_count"] == 3


def test_prepared_scoring_caches_one_prepare_failure_for_all_consumers(tmp_path: Path) -> None:
    failed_roll = tmp_path / "images/shared-failed-roll.wsq"
    pairs = [
        _accuracy_pair(
            tmp_path,
            "pair-1",
            "a",
            "failed",
            right_path=failed_roll,
        ),
        _accuracy_pair(
            tmp_path,
            "pair-2",
            "c",
            "failed",
            right_path=failed_roll,
        ),
    ]
    adapter = _FakeAdapter(fail_paths={failed_roll})

    rows, metadata = _run(adapter, pairs)

    assert Counter(path for path, _metadata in adapter.prepare_calls)[failed_roll.resolve()] == 1
    assert [row["status"] for row in rows] == [PREPARE_B_FAILURE, PREPARE_B_FAILURE]
    assert {row["error_code"] for row in rows} == {"synthetic_prepare_failure"}
    assert adapter.compare_calls == []
    assert metadata["unique_image_count"] == 3
    assert metadata["prepare_invocation_count"] == 3
    assert metadata["prepare_failure_count"] == 1


def test_prepared_cache_is_new_for_each_method_dataset_and_invocation(tmp_path: Path) -> None:
    common_plain = tmp_path / "images/plain.wsq"
    common_roll = tmp_path / "images/roll.wsq"
    b_pair = _accuracy_pair(
        tmp_path,
        "pair-b",
        "s1",
        "s2",
        left_path=common_plain,
        right_path=common_roll,
    )
    c_pair = _accuracy_pair(
        tmp_path,
        "pair-c",
        "s1",
        "s2",
        dataset="sd300c",
        left_path=common_plain,
        right_path=common_roll,
    )
    adapter = _FakeAdapter()

    first_rows, first_metadata = _run(adapter, [b_pair], cache_scope_id="b-sourceafis-1")
    second_rows, second_metadata = _run(
        adapter,
        [b_pair],
        method="other_method",
        cache_scope_id="b-other-method-2",
    )
    third_rows, third_metadata = _run(adapter, [c_pair], cache_scope_id="c-sourceafis-3")

    assert len(adapter.prepare_calls) == 6
    assert [
        first_rows[0]["representation_cache_scope_id"],
        second_rows[0]["representation_cache_scope_id"],
        third_rows[0]["representation_cache_scope_id"],
    ] == [
        "b-sourceafis-1",
        "b-other-method-2",
        "c-sourceafis-3",
    ]
    for metadata in (first_metadata, second_metadata, third_metadata):
        assert metadata["prepare_invocation_count"] == 2
        assert metadata["cache_instance_policy"] == "new_empty_in_memory_cache_each_invocation"
        assert metadata["cache_shared_across_methods"] is False
        assert metadata["cache_shared_across_datasets"] is False
        assert metadata["cache_shared_across_runs"] is False

    with pytest.raises(SharedAccuracyError, match="mixed dataset or split"):
        _run(adapter, [b_pair, c_pair], cache_scope_id="illegal-mixed-scope")


def test_source_preflight_comparison_accepts_numeric_text_equivalence_only() -> None:
    primary = {
        "pair_id": "primary-1",
        "raw_score": "1.0",
        "status": OK,
        "error_code": "",
        "prepare_a_diagnostics": "{}",
        "prepare_b_diagnostics": "{}",
        "compare_diagnostics": "{}",
    }
    rerun = dict(primary, raw_score="1")

    comparison = _source_preflight_comparison(primary, rerun)

    assert comparison["exact_numeric_score_equal"] is True
    assert comparison["passed"] is True
    assert _source_preflight_comparison(
        primary, dict(rerun, compare_diagnostics='{"different":true}')
    )["passed"] is False

    condition = {"method": "sourceafis", "dataset": "sd300b", "passed": True}
    assert (
        _condition_from_preflight({"conditions": [condition]}, "sourceafis", "sd300b")
        is condition
    )
    with pytest.raises(SharedAccuracyError, match="Missing passed preflight condition"):
        _condition_from_preflight(
            {"conditions": [dict(condition, passed=False)]}, "sourceafis", "sd300b"
        )


def test_projected_genuine_score_preserves_primary_result_and_marks_its_origin(
    tmp_path: Path,
) -> None:
    base = _pair_record(tmp_path, "sd300b", "s1", 1)
    pair = materialize_genuine([base], "evaluation")[0]
    primary_row = {
        "pair_id": base.pair_id,
        "raw_score": "42.5",
        "status": OK,
        "error_code": "",
        "error_message": "",
        "prepare_a_diagnostics": '{"elapsed_ms":1}',
        "prepare_b_diagnostics": '{"elapsed_ms":2}',
        "compare_diagnostics": '{"elapsed_ms":3}',
    }
    bundle = SimpleNamespace(
        rows=[primary_row],
        metadata={"config_hash": "primary-config", "implementation_hash": "primary-impl"},
        bundle_path=tmp_path / "primary-bundle",
    )
    runtime = {
        "method_version": "runtime-version",
        "frozen_config_hash": "runtime-config",
        "implementation_hash": "runtime-impl",
        "accuracy_runner_sha256": "runtime-runner",
    }

    projected = project_reused_genuine_scores(
        pairs=[pair], bundle=bundle, method="sourceafis", runtime=runtime
    )[0]

    assert projected["raw_score"] == "42.5"
    assert projected["status"] == OK
    assert projected["score_origin"] == (
        "reused_validated_primary_plain_roll_after_100_pair_preflight"
    )
    assert projected["source_primary_pair_id"] == base.pair_id
    assert projected["source_primary_bundle"] == str(bundle.bundle_path)
    assert projected["source_primary_config_hash"] == "primary-config"
    assert projected["source_primary_implementation_hash"] == "primary-impl"
    assert projected["frozen_config_hash"] == "runtime-config"
    assert projected["prepare_a_diagnostics_json"] == "{}"
    assert projected["compare_diagnostics_json"] == "{}"

    mismatched = AccuracyPair(**{**pair.__dict__, "source_pair_id_b": "different-primary-id"})
    with pytest.raises(SharedAccuracyError, match="Missing primary genuine score"):
        project_reused_genuine_scores(
            pairs=[mismatched], bundle=bundle, method="sourceafis", runtime=runtime
        )


def test_paired_resolution_requires_alignment_and_counts_all_decision_cases() -> None:
    def row(pair_id: str, score: float) -> dict[str, str]:
        return {
            "accuracy_pair_id": pair_id,
            "status": OK,
            "raw_score": repr(score),
            "canonical_finger_position": "1",
        }

    b_rows = [row("both-accept", 6), row("both-reject", 1), row("b-only", 6), row("c-only", 1)]
    c_rows = [row("both-accept", 7), row("both-reject", 2), row("b-only", 2), row("c-only", 8)]

    report = _paired_resolution(b_rows, c_rows, threshold=5)

    assert report["paired_scored_count"] == 4
    assert report["accepted_in_both"] == 1
    assert report["rejected_in_both"] == 1
    assert report["accepted_only_in_sd300b"] == 1
    assert report["accepted_only_in_sd300c"] == 1
    assert report["decision_agreement"] == pytest.approx(0.5)
    assert report["mcnemar_discordant_b_only"] == 1
    assert report["mcnemar_discordant_c_only"] == 1
    assert report["per_finger"]["1"]["count"] == 4

    with pytest.raises(SharedAccuracyError, match="not exactly aligned"):
        _paired_resolution(b_rows, list(reversed(c_rows)), threshold=5)


def test_legacy_thresholds_are_separate_from_shared_far_target_keys() -> None:
    assert _target_key(0.01) == "far_1_percent"
    assert _target_key(0.001) == "far_0_1_percent"
    assert _target_key(0.01) != _target_key(0.001)
    assert LEGACY_THRESHOLDS == {"sourceafis": 40.0, "sift_geometric": 4.0}
    assert "legacy" not in {_target_key(0.01), _target_key(0.001)}


def test_synthetic_calibration_and_evaluation_report_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "shared-accuracy"
    monkeypatch.setattr(shared_accuracy, "TARGET_FARS", (0.5,))
    target_key = _target_key(0.5)
    assert target_key == "far_0.5"
    runner_sha256 = shared_accuracy.file_sha256(
        Path(shared_accuracy.__file__).resolve()
    )

    def publish_scores(
        method: str,
        dataset: str,
        split: str,
        pair_label: str,
        scores: list[float],
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for index, score in enumerate(scores):
            left_subject = f"{split}-{pair_label}-left-{index:03d}"
            right_subject = (
                left_subject
                if pair_label == "genuine"
                else f"{split}-{pair_label}-right-{index:03d}"
            )
            pair = _accuracy_pair(
                tmp_path,
                f"synthetic_{split}_{pair_label}_{index:03d}",
                left_subject,
                right_subject,
                dataset=dataset,
                split=split,
                finger=index % 2 + 1,
            )
            row = shared_accuracy._score_base(
                pair,
                method=method,
                method_version=f"{method}-synthetic-v1",
                frozen_config_hash=f"{method}-config",
                implementation_hash=f"{method}-implementation",
                accuracy_runner_sha256=runner_sha256,
                cache_scope_id=f"{method}-{dataset}-{split}",
            )
            numeric_score = int(score) if method == "sift_geometric" else float(score)
            row.update(raw_score=str(numeric_score), status=OK)
            rows.append(row)
        path = shared_accuracy._score_file(
            output_root, method, dataset, split, pair_label
        )
        shared_accuracy._publish_immutable_csv(
            path, rows, shared_accuracy.SCORE_COLUMNS
        )
        return rows

    development_impostors = [float(score) for score in range(20)]
    development_genuine = [12.0, 8.0, 15.0, 5.0]
    evaluation_genuine = {
        "sd300b": [12.0, 8.0, 15.0, 5.0],
        "sd300c": [11.0, 9.0, 7.0, 16.0],
    }
    # If evaluation impostors were incorrectly included in calibration, the
    # lowest common target-0.5 threshold would move above the expected 10.
    evaluation_impostors = {
        "sd300b": [100.0, 100.0, 100.0, 1.0],
        "sd300c": [100.0, 100.0, 1.0, 100.0],
    }
    split_conditions: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "evaluation": [],
    }
    for method in shared_accuracy.METHODS:
        for dataset in shared_accuracy.DATASETS:
            for split, genuine_scores, impostor_scores in (
                ("development", development_genuine, development_impostors),
                (
                    "evaluation",
                    evaluation_genuine[dataset],
                    evaluation_impostors[dataset],
                ),
            ):
                published_rows = {
                    "genuine": publish_scores(
                        method,
                        dataset,
                        split,
                        "genuine",
                        genuine_scores,
                    ),
                    "impostor": publish_scores(
                        method,
                        dataset,
                        split,
                        "impostor",
                        impostor_scores,
                    ),
                }
                condition_dir = output_root / f"scores/{method}/{dataset}/{split}"
                metadata: dict[str, Any] = {
                    "schema_version": "shared-accuracy-score-run-v1",
                    "protocol_version": shared_accuracy.PROTOCOL_VERSION,
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "score_direction": shared_accuracy.HIGHER_IS_MORE_SIMILAR,
                    "thresholding_during_scoring": False,
                    "frozen_config_hash": f"{method}-config",
                    "implementation_hash": f"{method}-implementation",
                    "accuracy_runner_sha256": runner_sha256,
                    "genuine_score_origin": "synthetic_fixture",
                    "genuine_reuse_reason": "synthetic fixture without real data",
                    "cache": {"mode": "synthetic_fixture"},
                    "score_files": {},
                    "validation": {},
                }
                for pair_label, rows in published_rows.items():
                    score_path = condition_dir / f"{pair_label}.csv"
                    metadata["score_files"][pair_label] = {
                        "filename": score_path.name,
                        "sha256": shared_accuracy.file_sha256(score_path),
                        "size": score_path.stat().st_size,
                    }
                    metadata["validation"][pair_label] = {
                        "planned_count": len(rows),
                        "scoreable_count": len(rows),
                        "failure_count": 0,
                        "failure_counts": {},
                    }
                shared_accuracy._publish_immutable_json(
                    condition_dir / "run_metadata.json", metadata
                )
                split_conditions[split].append(metadata)

    for split in shared_accuracy.SPLITS:
        shared_accuracy._publish_immutable_json(
            output_root / f"scores/{split}_summary.json",
            {
                "schema_version": "shared-accuracy-split-score-summary-v1",
                "split": split,
                "conditions": split_conditions[split],
                "thresholding_during_scoring": False,
                "completed": True,
            },
        )
    definition_path = output_root / "protocol_definition.json"
    shared_accuracy._publish_immutable_json(
        definition_path,
        {
            "schema_version": "synthetic-shared-accuracy-protocol-definition-v1",
            "accuracy_execution_identity": {
                "runner_source_sha256": runner_sha256,
            },
        },
    )
    definition_sha256 = shared_accuracy.file_sha256(definition_path)
    shared_accuracy._publish_immutable_json(
        output_root / "provenance/genuine_preflight.json",
        {
            "passed": True,
            "protocol_definition_sha256": definition_sha256,
        },
    )

    original_read_json = shared_accuracy._read_json

    def tampered_definition(path: Path) -> dict[str, Any]:
        payload = original_read_json(path)
        if Path(path).resolve() == definition_path.resolve():
            return {
                **payload,
                "accuracy_execution_identity": {
                    **payload["accuracy_execution_identity"],
                    "runner_source_sha256": "0" * 64,
                },
            }
        return payload

    with monkeypatch.context() as definition_patch:
        definition_patch.setattr(
            shared_accuracy, "_read_json", tampered_definition
        )
        with pytest.raises(SharedAccuracyError, match="runner changed"):
            shared_accuracy.calibrate_thresholds(
                project_root=tmp_path, output_root=output_root
            )

    thresholds = shared_accuracy.calibrate_thresholds(
        project_root=tmp_path, output_root=output_root
    )

    assert thresholds["schema_version"] == "shared-accuracy-frozen-thresholds-v1"
    assert thresholds["selected_on_split"] == "development"
    assert thresholds["evaluation_scores_read_during_selection"] is False
    assert thresholds["target_fars"] == [0.5]
    for method in shared_accuracy.METHODS:
        calibration = thresholds["methods"][method][target_key]
        assert calibration["threshold"] == 10
        assert calibration["target_far"] == 0.5
        assert calibration["per_dataset"]["sd300b"]["calibration_impostors"]["far"] == 0.5
        assert calibration["per_dataset"]["sd300c"]["calibration_impostors"]["far"] == 0.5

    gate_path = output_root / "calibration/evaluation_safety_gate.json"

    def fail_gate(path: Path) -> dict[str, Any]:
        payload = original_read_json(path)
        if Path(path).resolve() == gate_path.resolve():
            return {**payload, "passed": False}
        return payload

    with monkeypatch.context() as gate_patch:
        gate_patch.setattr(shared_accuracy, "_read_json", fail_gate)
        with pytest.raises(SharedAccuracyError, match="threshold safety gate"):
            shared_accuracy.evaluate_and_report(
                project_root=tmp_path, output_root=output_root
            )

    report = shared_accuracy.evaluate_and_report(
        project_root=tmp_path, output_root=output_root
    )

    assert report["schema_version"] == "shared-accuracy-evaluation-report-v1"
    assert report["thresholds_sha256"] == shared_accuracy.file_sha256(
        output_root / "calibration/frozen_thresholds.json"
    )
    for method in shared_accuracy.METHODS:
        assert report["calibrated_evaluation"][method][target_key]["sd300b"][
            "threshold"
        ] == 10
        assert report["legacy_evaluation"][method]["sd300b"]["threshold"] == (
            LEGACY_THRESHOLDS[method]
        )
        assert LEGACY_THRESHOLDS[method] != 10

    assert report["calibrated_evaluation"]["sourceafis"][target_key]["sd300b"][
        "metrics"
    ]["true_accept_count"] == 2
    assert report["legacy_evaluation"]["sourceafis"]["sd300b"]["metrics"][
        "true_accept_count"
    ] == 0
    assert report["calibrated_evaluation"]["sift_geometric"][target_key]["sd300b"][
        "metrics"
    ]["true_accept_count"] == 2
    assert report["legacy_evaluation"]["sift_geometric"]["sd300b"]["metrics"][
        "true_accept_count"
    ] == 4

    paired = report["paired_resolution_analysis"]["sourceafis"][target_key]
    assert paired["paired_conditions_not_independent"] is True
    assert paired["genuine"]["planned_pair_count"] == 4
    assert paired["genuine"]["accepted_in_both"] == 1
    assert paired["genuine"]["rejected_in_both"] == 1
    assert paired["genuine"]["accepted_only_in_sd300b"] == 1
    assert paired["genuine"]["accepted_only_in_sd300c"] == 1
    assert paired["impostor"]["accepted_only_in_sd300b"] == 1
    assert paired["impostor"]["accepted_only_in_sd300c"] == 1

    for method in shared_accuracy.METHODS:
        for dataset in shared_accuracy.DATASETS:
            curve = report["curves"][method][dataset]
            roc_path = Path(curve["roc_csv"])
            det_path = Path(curve["det_csv"])
            assert roc_path.is_file()
            assert det_path.is_file()
            assert roc_path.read_bytes() == det_path.read_bytes()
            curve_rows = shared_accuracy._read_table(roc_path)
            assert len(curve_rows) == curve["point_count"]
            assert curve_rows[0]["tar"] == "0.0"
            assert curve_rows[0]["far"] == "0.0"
            assert curve_rows[-1]["tar"] == "1.0"
            assert curve_rows[-1]["far"] == "1.0"
            assert curve["smoothing"] is False

    legacy_rows = shared_accuracy._read_table(
        Path(report["legacy_comparison_table"]["path"])
    )
    assert {row["operating_point"] for row in legacy_rows} == {
        "legacy",
        target_key,
    }
    assert len(legacy_rows) == 8
    assert shared_accuracy._read_json(
        output_root / "reports/evaluation_metrics.json"
    ) == report
    assert shared_accuracy._read_json(
        output_root / "reports/resolution_analysis.json"
    ) == report["paired_resolution_analysis"]
