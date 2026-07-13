import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

import fingerprint_benchmark.bundle as bundle_module
from fingerprint_benchmark.bundle import (
    BundlePublicationError,
    create_candidate_directory,
    discard_candidate_directory,
    publish_candidate_directory,
)
from fingerprint_benchmark.contract import (
    BENCHMARK_CONTRACT_VERSION,
    HIGHER_IS_MORE_SIMILAR,
    BenchmarkRunSpec,
    CompareOutcome,
    MethodExecutionError,
    MethodMetadata,
    PreparedRepresentation,
    PrepareOutcome,
)
from fingerprint_benchmark.hashing import file_sha256, stable_config_hash
from fingerprint_benchmark.io import ArtifactWriteError, write_csv_atomic
from fingerprint_benchmark.manifest import PairRecord, read_pair_manifest
from fingerprint_benchmark.preflight import BenchmarkPreflightError
from fingerprint_benchmark.runner import (
    METADATA_FILENAME,
    OK,
    RESULT_COLUMNS,
    RESULT_FILENAME,
    BundleValidationError,
    ResultValidationError,
    prepare_run_context,
    run_benchmark_manifest,
    score_payload_sha256,
    validate_result_bundle,
    validate_result_contract,
)


DATASET = "sd300b"
PROTOCOL = "plain_self"
SCORE_SEMANTICS = "Fake unnormalized similarity score"


def test_runner_publishes_v2_bundle_in_manifest_order_with_warmup_and_provenance(tmp_path):
    artifacts = _published_run(tmp_path, pair_ids=("pair-b", "pair-a"))

    assert [row["pair_id"] for row in artifacts.rows] == ["pair-b", "pair-a"]
    assert all(row["status"] == OK for row in artifacts.rows)
    assert all(row["manifest_sha256"] == file_sha256(artifacts.manifest_path) for row in artifacts.rows)
    assert artifacts.bundle_directory == (
        artifacts.results_root
        / DATASET
        / PROTOCOL
        / "fake"
        / BENCHMARK_CONTRACT_VERSION
        / artifacts.metadata["config_hash"]
    ).resolve()
    assert artifacts.result_path.is_file()
    assert artifacts.metadata_path.is_file()
    assert artifacts.metadata["benchmark_contract_version"] == BENCHMARK_CONTRACT_VERSION
    assert artifacts.metadata["score_direction"] == HIGHER_IS_MORE_SIMILAR
    assert artifacts.metadata["score_semantics"] == SCORE_SEMANTICS
    assert artifacts.metadata["manifest"]["dedicated_validator_result"]["status"] == "ok"
    assert artifacts.metadata["warm_up"]["pair_ids"] == ["pair-b"]
    assert artifacts.metadata["warm_up"]["operation_count"] == 3
    assert artifacts.metadata["warm_up"]["included_in_result_rows"] is False
    assert artifacts.adapter.prepared_pair_sides == [
        ("pair-b", "a"),
        ("pair-b", "b"),
        ("pair-b", "a"),
        ("pair-b", "b"),
        ("pair-a", "a"),
        ("pair-a", "b"),
    ]

    validated = validate_result_bundle(artifacts.bundle_directory, **artifacts.validation_kwargs)
    assert validated["result"]["sha256"] == file_sha256(artifacts.result_path)
    assert validated["result"]["score_payload_sha256"] == score_payload_sha256(artifacts.rows)


def test_result_contract_rejects_wrong_pair_ids_with_the_right_row_count(tmp_path):
    artifacts = _published_run(tmp_path, pair_ids=("pair-a", "pair-b"))
    rows = copy.deepcopy(artifacts.rows)
    rows[1]["pair_id"] = "pair-not-in-manifest"

    result_path = _write_mutated_result(tmp_path, rows)
    with pytest.raises(ResultValidationError, match="pair_id sequence"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


def test_result_contract_rejects_duplicate_pair_ids(tmp_path):
    artifacts = _published_run(tmp_path, pair_ids=("pair-a", "pair-b"))
    rows = copy.deepcopy(artifacts.rows)
    rows[1]["pair_id"] = rows[0]["pair_id"]

    result_path = _write_mutated_result(tmp_path, rows)
    with pytest.raises(ResultValidationError):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


@pytest.mark.parametrize(
    ("column", "stale_value"),
    [
        ("dataset", "sd300c"),
        ("protocol", "roll_self"),
        ("subject_id", "99999999"),
        ("canonical_finger_position", "10"),
    ],
)
def test_result_contract_rejects_wrong_per_pair_identity_metadata(tmp_path, column, stale_value):
    artifacts = _published_run(tmp_path)
    rows = copy.deepcopy(artifacts.rows)
    rows[0][column] = stale_value

    result_path = _write_mutated_result(tmp_path, rows)
    with pytest.raises(ResultValidationError, match=rf"{column} mismatch"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


@pytest.mark.parametrize(
    "column",
    ["manifest_sha256", "config_hash", "implementation_hash"],
)
def test_result_contract_rejects_stale_run_identity_hashes(tmp_path, column):
    artifacts = _published_run(tmp_path)
    rows = copy.deepcopy(artifacts.rows)
    rows[0][column] = "0" * 64

    result_path = _write_mutated_result(tmp_path, rows)
    with pytest.raises(ResultValidationError, match=rf"{column} mismatch"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


@pytest.mark.parametrize(
    ("metadata_path", "match"),
    [
        (("manifest", "sha256"), "manifest SHA-256 is stale"),
        (("config_hash",), "config_hash mismatch"),
        (("implementation_hash",), "implementation_hash mismatch"),
        (("result", "sha256"), "result SHA-256 does not match"),
    ],
)
def test_bundle_rejects_stale_metadata_and_result_relationships(
    tmp_path,
    metadata_path,
    match,
):
    artifacts = _published_run(tmp_path)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    _set_nested(metadata, metadata_path, "0" * 64)
    artifacts.metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BundleValidationError, match=match):
        validate_result_bundle(artifacts.bundle_directory, **artifacts.validation_kwargs)


def test_semantically_invalid_candidate_cannot_replace_existing_bundle(tmp_path):
    artifacts = _published_run(tmp_path, pair_ids=("pair-a", "pair-b"))
    before = _bundle_bytes(artifacts.bundle_directory)
    candidate = create_candidate_directory(artifacts.bundle_directory)
    try:
        shutil.copy2(artifacts.metadata_path, candidate / METADATA_FILENAME)
        bad_rows = copy.deepcopy(artifacts.rows)
        bad_rows[0]["pair_id"] = "wrong-pair"
        _write_result(candidate / RESULT_FILENAME, bad_rows)

        with pytest.raises(ResultValidationError, match="pair_id sequence"):
            validate_result_bundle(candidate, **artifacts.validation_kwargs)
        with pytest.raises(BundlePublicationError, match="already exists"):
            publish_candidate_directory(candidate, artifacts.bundle_directory)

        assert _bundle_bytes(artifacts.bundle_directory) == before
    finally:
        discard_candidate_directory(candidate)


def test_metadata_write_failure_leaves_no_final_or_candidate_bundle(tmp_path, monkeypatch):
    manifest_path, results_root = _manifest_fixture(tmp_path)
    adapter = FakeAdapter()
    context = prepare_run_context(
        manifest_path=manifest_path,
        expected_dataset=DATASET,
        expected_protocol=PROTOCOL,
        adapter=adapter,
        results_root=results_root,
    )

    def fail_metadata_write(*args, **kwargs):
        raise ArtifactWriteError("forced metadata write failure")

    monkeypatch.setattr("fingerprint_benchmark.runner.write_json_atomic", fail_metadata_write)
    with pytest.raises(ArtifactWriteError, match="forced metadata write failure"):
        _run_manifest(manifest_path, results_root, adapter)

    _assert_no_bundle_or_candidate(context.bundle_directory)


def test_publish_failure_leaves_no_mixed_bundle(tmp_path, monkeypatch):
    manifest_path, results_root = _manifest_fixture(tmp_path)
    adapter = FakeAdapter()
    context = prepare_run_context(
        manifest_path=manifest_path,
        expected_dataset=DATASET,
        expected_protocol=PROTOCOL,
        adapter=adapter,
        results_root=results_root,
    )

    def fail_publish(*args, **kwargs):
        raise BundlePublicationError("forced publish failure")

    monkeypatch.setattr("fingerprint_benchmark.runner.publish_candidate_directory", fail_publish)
    with pytest.raises(BundlePublicationError, match="forced publish failure"):
        _run_manifest(manifest_path, results_root, adapter)

    _assert_no_bundle_or_candidate(context.bundle_directory)


def test_directory_publish_rolls_back_if_move_raises_after_moving(tmp_path, monkeypatch):
    final_directory = tmp_path / "final"
    candidate = create_candidate_directory(final_directory)
    (candidate / "sentinel.txt").write_text("candidate", encoding="utf-8")
    real_replace = bundle_module.os.replace
    call_count = 0

    def move_then_fail_once(source, destination):
        nonlocal call_count
        call_count += 1
        real_replace(source, destination)
        if call_count == 1:
            raise OSError("forced post-move failure")

    monkeypatch.setattr(bundle_module.os, "replace", move_then_fail_once)
    with pytest.raises(BundlePublicationError, match="forced post-move failure"):
        publish_candidate_directory(candidate, final_directory)

    assert not final_directory.exists()
    assert (candidate / "sentinel.txt").read_text(encoding="utf-8") == "candidate"
    discard_candidate_directory(candidate)


def test_skip_existing_rejects_same_count_manifest_with_changed_pair_ids(tmp_path):
    artifacts = _published_run(tmp_path, pair_ids=("pair-a",))
    before = _bundle_bytes(artifacts.bundle_directory)
    _write_manifest(
        artifacts.manifest_path,
        [_manifest_row("pair-b", artifacts.image_path, artifacts.image_path)],
    )
    adapter = FakeAdapter()

    with pytest.raises((ResultValidationError, BundleValidationError)):
        run_benchmark_manifest(
            manifest_path=artifacts.manifest_path,
            adapter=adapter,
            expected_dataset=DATASET,
            expected_protocol=PROTOCOL,
            results_root=artifacts.results_root,
            data_root=tmp_path / "data",
            dedicated_validator=fake_dedicated_validator,
            skip_existing=True,
        )

    assert adapter.prepared_pair_sides == []
    assert _bundle_bytes(artifacts.bundle_directory) == before


def test_raw_score_is_written_with_round_trip_safe_full_precision(tmp_path):
    score = 1.2345678901234567
    artifacts = _published_run(tmp_path, adapter=FakeAdapter(raw_score=score))
    stored = artifacts.rows[0]["raw_score"]

    assert stored == repr(score)
    assert float(stored) == score
    assert stored != format(score, ".9g")


def test_score_payload_hash_is_deterministic_and_ignores_all_timings(tmp_path):
    artifacts = _published_run(tmp_path)
    first = copy.deepcopy(artifacts.rows)
    second = copy.deepcopy(first)
    for row in second:
        row["prepare_a_ms"] = "9001"
        row["prepare_b_ms"] = "9002"
        row["compare_ms"] = "9003"
        row["method_prepare_a_ms"] = "8001"
        row["method_prepare_b_ms"] = "8002"
        row["method_compare_ms"] = "8003"
        row["total_ms"] = "99999"
        row["prepare_a_diagnostics"] = '{"runtime":"different"}'

    assert score_payload_sha256(first) == score_payload_sha256(second)
    second[0]["raw_score"] = repr(float(second[0]["raw_score"]) + 1.0)
    assert score_payload_sha256(first) != score_payload_sha256(second)


def test_score_direction_and_semantics_are_required_and_persisted(tmp_path):
    artifacts = _published_run(tmp_path)

    assert artifacts.metadata["score_direction"] == HIGHER_IS_MORE_SIMILAR
    assert artifacts.metadata["score_semantics"] == SCORE_SEMANTICS
    assert all(row["score_direction"] == HIGHER_IS_MORE_SIMILAR for row in artifacts.rows)
    assert all(row["score_semantics"] == SCORE_SEMANTICS for row in artifacts.rows)
    with pytest.raises(ValueError, match="Invalid score_direction"):
        MethodMetadata(
            method="fake",
            method_version="fake-1",
            score_direction="sideways",
            score_semantics=SCORE_SEMANTICS,
            implementation_provenance={"implementation": "fake"},
            config={},
        )


def test_representation_payload_can_be_a_non_string_object(tmp_path):
    adapter = FakeAdapter(non_string_payload=True)
    artifacts = _published_run(tmp_path, adapter=adapter)

    assert artifacts.rows[0]["status"] == OK
    assert adapter.compared_payload_types
    assert set(adapter.compared_payload_types) == {(dict, dict)}


@pytest.mark.parametrize(
    ("expected_dataset", "expected_protocol"),
    [("sd300c", PROTOCOL), (DATASET, "roll_self")],
)
def test_preflight_rejects_cli_identity_that_disagrees_with_manifest(
    tmp_path,
    expected_dataset,
    expected_protocol,
):
    manifest_path, results_root = _manifest_fixture(tmp_path)
    adapter = FakeAdapter()

    with pytest.raises(BenchmarkPreflightError, match="Manifest identity mismatch"):
        run_benchmark_manifest(
            manifest_path=manifest_path,
            adapter=adapter,
            expected_dataset=expected_dataset,
            expected_protocol=expected_protocol,
            results_root=results_root,
            data_root=tmp_path / "data",
            dedicated_validator=fake_dedicated_validator,
        )

    assert adapter.prepared_pair_sides == []
    assert not results_root.exists()


def test_failure_rows_preserve_stage_specific_timings_diagnostics_and_errors(tmp_path):
    adapter = FakeAdapter(
        fail_prepare_a={"prepare-a-fails"},
        fail_prepare_b={"prepare-b-fails"},
        fail_compare={"compare-fails"},
    )
    artifacts = _published_run(
        tmp_path,
        pair_ids=(
            "warmup",
            "prepare-a-fails",
            "prepare-b-fails",
            "compare-fails",
            "success",
        ),
        adapter=adapter,
    )
    rows = {row["pair_id"]: row for row in artifacts.rows}

    prepare_a = rows["prepare-a-fails"]
    assert prepare_a["status"] == "prepare_a_failure"
    assert prepare_a["raw_score"] == ""
    assert prepare_a["prepare_a_ms"]
    assert prepare_a["method_prepare_a_ms"]
    assert json.loads(prepare_a["prepare_a_diagnostics"]) == {"forced": "a"}
    assert all(
        prepare_a[column] == ""
        for column in (
            "prepare_b_ms",
            "method_prepare_b_ms",
            "prepare_b_diagnostics",
            "compare_ms",
            "method_compare_ms",
            "compare_diagnostics",
        )
    )
    assert prepare_a["error_code"] == "forced_prepare_a"
    assert prepare_a["error_message"] == "forced prepare A failure"

    prepare_b = rows["prepare-b-fails"]
    assert prepare_b["status"] == "prepare_b_failure"
    assert prepare_b["prepare_a_ms"] and prepare_b["prepare_b_ms"]
    assert prepare_b["compare_ms"] == ""
    assert json.loads(prepare_b["prepare_b_diagnostics"]) == {"forced": "b"}

    compare = rows["compare-fails"]
    assert compare["status"] == "comparison_failure"
    assert compare["prepare_a_ms"] and compare["prepare_b_ms"] and compare["compare_ms"]
    assert compare["raw_score"] == ""
    assert json.loads(compare["compare_diagnostics"]) == {"forced": "compare"}

    success = rows["success"]
    assert success["status"] == OK
    assert success["raw_score"]
    assert success["error_code"] == success["error_message"] == ""


def test_result_contract_rejects_outputs_from_operations_after_an_earlier_failure(tmp_path):
    adapter = FakeAdapter(fail_prepare_a={"prepare-a-fails"})
    artifacts = _published_run(
        tmp_path,
        pair_ids=("warmup", "prepare-a-fails"),
        adapter=adapter,
    )
    rows = copy.deepcopy(artifacts.rows)
    failed = next(row for row in rows if row["pair_id"] == "prepare-a-fails")
    failed["compare_ms"] = "0"

    result_path = _write_mutated_result(tmp_path, rows)
    with pytest.raises(ResultValidationError, match="after an earlier failure"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


def test_result_contract_rejects_inconsistent_success_and_failure_error_fields(tmp_path):
    adapter = FakeAdapter(fail_compare={"compare-fails"})
    artifacts = _published_run(
        tmp_path,
        pair_ids=("warmup", "compare-fails"),
        adapter=adapter,
    )

    success_error_rows = copy.deepcopy(artifacts.rows)
    success_error_rows[0]["error_code"] = "unexpected"
    result_path = _write_mutated_result(tmp_path, success_error_rows, name="success-error.csv")
    with pytest.raises(ResultValidationError, match="Successful pair .* contains error fields"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)

    missing_failure_error_rows = copy.deepcopy(artifacts.rows)
    failed = next(row for row in missing_failure_error_rows if row["pair_id"] == "compare-fails")
    failed["error_message"] = ""
    result_path = _write_mutated_result(tmp_path, missing_failure_error_rows, name="failure-error.csv")
    with pytest.raises(ResultValidationError, match="must contain error_code and error_message"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


def test_result_contract_enforces_finite_nonnegative_and_total_sum_timing_rules(tmp_path):
    artifacts = _published_run(tmp_path)

    negative_rows = copy.deepcopy(artifacts.rows)
    negative_rows[0]["prepare_a_ms"] = "-0.1"
    result_path = _write_mutated_result(tmp_path, negative_rows, name="negative.csv")
    with pytest.raises(ResultValidationError, match="finite and non-negative"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)

    nonfinite_rows = copy.deepcopy(artifacts.rows)
    nonfinite_rows[0]["total_ms"] = "nan"
    result_path = _write_mutated_result(tmp_path, nonfinite_rows, name="nonfinite.csv")
    with pytest.raises(ResultValidationError, match="finite and non-negative"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)

    bad_sum_rows = copy.deepcopy(artifacts.rows)
    bad_sum_rows[0]["prepare_a_ms"] = "1.0"
    bad_sum_rows[0]["prepare_b_ms"] = "1.0"
    bad_sum_rows[0]["compare_ms"] = "1.0"
    bad_sum_rows[0]["total_ms"] = "2.5"
    result_path = _write_mutated_result(tmp_path, bad_sum_rows, name="sum.csv")
    with pytest.raises(ResultValidationError, match=r"prepare_a_ms \+ prepare_b_ms \+ compare_ms"):
        validate_result_contract(result_path, **artifacts.validation_kwargs)


def test_config_hash_is_deterministic_for_same_method_config():
    first = stable_config_hash({"b": 2, "nested": {"z": 3, "a": 1}})
    second = stable_config_hash({"nested": {"a": 1, "z": 3}, "b": 2})
    assert first == second


def test_csv_writer_does_not_replace_existing_result_when_replace_fails(tmp_path, monkeypatch):
    result_path = tmp_path / "pairs.csv"
    result_path.write_bytes(b"existing result\n")

    def fail_replace(*args, **kwargs):
        raise OSError("forced replace failure")

    monkeypatch.setattr("fingerprint_benchmark.io.os.replace", fail_replace)
    with pytest.raises(ArtifactWriteError, match="forced replace failure"):
        write_csv_atomic([{column: "" for column in RESULT_COLUMNS}], result_path, RESULT_COLUMNS)

    assert result_path.read_bytes() == b"existing result\n"
    assert list(tmp_path.glob(".pairs.csv.*.tmp")) == []


@dataclass
class RunArtifacts:
    manifest_path: Path
    results_root: Path
    image_path: Path
    adapter: "FakeAdapter"
    metadata: dict[str, Any]
    bundle_directory: Path
    result_path: Path
    metadata_path: Path
    rows: list[dict[str, str]]
    pairs: list[PairRecord]
    run_spec: BenchmarkRunSpec

    @property
    def validation_kwargs(self) -> dict[str, Any]:
        return {
            "manifest_records": self.pairs,
            "run_spec": self.run_spec,
            "score_direction": self.metadata["score_direction"],
            "score_semantics": self.metadata["score_semantics"],
        }


class FakeAdapter:
    def __init__(
        self,
        *,
        fail_prepare_a: set[str] | None = None,
        fail_prepare_b: set[str] | None = None,
        fail_compare: set[str] | None = None,
        raw_score: float = 123.5,
        non_string_payload: bool = False,
    ) -> None:
        self.fail_prepare_a = fail_prepare_a or set()
        self.fail_prepare_b = fail_prepare_b or set()
        self.fail_compare = fail_compare or set()
        self.raw_score = raw_score
        self.non_string_payload = non_string_payload
        self.prepared_pair_sides: list[tuple[str, str]] = []
        self.compared_payload_types: list[tuple[type[Any], type[Any]]] = []

    def metadata(self) -> MethodMetadata:
        return MethodMetadata(
            method="fake",
            method_version="fake-1",
            score_direction=HIGHER_IS_MORE_SIMILAR,
            score_semantics=SCORE_SEMANTICS,
            implementation_provenance={"implementation": "fake"},
            config={"normalization": "none", "thresholding": "none"},
            runtime={"runtime": "in-process-test"},
        )

    def prepare(self, image_path: Path, image_metadata) -> PrepareOutcome:
        pair_id = str(image_metadata["pair_id"])
        side = str(image_metadata["side"])
        self.prepared_pair_sides.append((pair_id, side))
        if side == "a" and pair_id in self.fail_prepare_a:
            raise MethodExecutionError(
                "forced_prepare_a",
                "forced prepare A failure",
                method_internal_ms=0.000001,
                diagnostics={"forced": "a"},
            )
        if side == "b" and pair_id in self.fail_prepare_b:
            raise MethodExecutionError(
                "forced_prepare_b",
                "forced prepare B failure",
                method_internal_ms=0.000001,
                diagnostics={"forced": "b"},
            )
        payload: Any
        if self.non_string_payload:
            payload = {"pair_id": pair_id, "side": side}
        else:
            payload = f"{pair_id}:{side}"
        return PrepareOutcome(
            representation=PreparedRepresentation(
                method="fake",
                method_version="fake-1",
                representation_format="fake-representation",
                representation_version="fake-1",
                payload=payload,
            ),
            method_internal_ms=0.000001,
            diagnostics={"operation": "prepare", "side": side},
        )

    def compare(self, representation_a, representation_b) -> CompareOutcome:
        self.compared_payload_types.append((type(representation_a.payload), type(representation_b.payload)))
        pair_id = _payload_pair_id(representation_a.payload)
        if pair_id in self.fail_compare:
            raise MethodExecutionError(
                "forced_compare",
                "forced comparison failure",
                method_internal_ms=0.000001,
                diagnostics={"forced": "compare"},
            )
        return CompareOutcome(
            raw_score=self.raw_score,
            method_internal_ms=0.000001,
            diagnostics={"operation": "compare"},
        )

    def close(self):
        pass


def fake_dedicated_validator(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    return {
        "status": "ok",
        "row_count": row_count,
        "data_root": str(data_root),
    }


def _published_run(
    tmp_path: Path,
    *,
    pair_ids: tuple[str, ...] = ("pair-a",),
    adapter: FakeAdapter | None = None,
) -> RunArtifacts:
    manifest_path, results_root = _manifest_fixture(tmp_path, pair_ids=pair_ids)
    adapter = adapter or FakeAdapter()
    metadata = _run_manifest(manifest_path, results_root, adapter)
    result_path = Path(metadata["result"]["path"])
    bundle_directory = result_path.parent
    raw_spec = dict(metadata["run_spec"])
    raw_spec["manifest_path"] = Path(raw_spec["manifest_path"])
    run_spec = BenchmarkRunSpec(**raw_spec)
    return RunArtifacts(
        manifest_path=manifest_path,
        results_root=results_root,
        image_path=manifest_path.parent / "image.png",
        adapter=adapter,
        metadata=metadata,
        bundle_directory=bundle_directory,
        result_path=result_path,
        metadata_path=bundle_directory / METADATA_FILENAME,
        rows=_read_rows(result_path),
        pairs=read_pair_manifest(manifest_path),
        run_spec=run_spec,
    )


def _manifest_fixture(
    tmp_path: Path,
    *,
    pair_ids: tuple[str, ...] = ("pair-a",),
) -> tuple[Path, Path]:
    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / "image.png"
    image_path.write_bytes(b"fake image bytes")
    manifest_path = case_dir / "manifest.csv"
    _write_manifest(
        manifest_path,
        [
            _manifest_row(
                pair_id,
                image_path,
                image_path,
                subject_id=f"{1000 + index:08d}",
            )
            for index, pair_id in enumerate(pair_ids)
        ],
    )
    return manifest_path, case_dir / "results"


def _run_manifest(
    manifest_path: Path,
    results_root: Path,
    adapter: FakeAdapter,
    *,
    skip_existing: bool = False,
) -> dict[str, Any]:
    return run_benchmark_manifest(
        manifest_path=manifest_path,
        adapter=adapter,
        expected_dataset=DATASET,
        expected_protocol=PROTOCOL,
        results_root=results_root,
        data_root=manifest_path.parent / "data",
        dedicated_validator=fake_dedicated_validator,
        skip_existing=skip_existing,
    )


def _payload_pair_id(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload["pair_id"])
    return str(payload).split(":", 1)[0]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "pair_id",
        "dataset",
        "protocol",
        "subject_id",
        "canonical_finger_position",
        "ppi",
        "raw_frgp_a",
        "raw_frgp_b",
        "path_a",
        "path_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(
    pair_id: str,
    path_a: Path,
    path_b: Path,
    *,
    subject_id: str = "00001000",
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "dataset": DATASET,
        "protocol": PROTOCOL,
        "subject_id": subject_id,
        "canonical_finger_position": "1",
        "ppi": "1000",
        "raw_frgp_a": "11",
        "raw_frgp_b": "11",
        "path_a": str(path_a),
        "path_b": str(path_b),
    }


def _write_result(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_mutated_result(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    name: str = "mutated-pairs.csv",
) -> Path:
    path = tmp_path / name
    _write_result(path, rows)
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _set_nested(payload: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def _bundle_bytes(bundle_directory: Path) -> dict[str, bytes]:
    return {
        RESULT_FILENAME: (bundle_directory / RESULT_FILENAME).read_bytes(),
        METADATA_FILENAME: (bundle_directory / METADATA_FILENAME).read_bytes(),
    }


def _assert_no_bundle_or_candidate(bundle_directory: Path) -> None:
    assert not bundle_directory.exists()
    candidates = list(bundle_directory.parent.glob(f".{bundle_directory.name}.candidate-*"))
    assert candidates == []
