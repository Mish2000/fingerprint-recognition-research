import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

import fingerprint_benchmark.detector_only_joint500 as joint500
from fingerprint_benchmark.cli import parse_args
from fingerprint_benchmark.contract import (
    BENCHMARK_CONTRACT_VERSION,
    HIGHER_IS_MORE_SIMILAR,
    RESULT_SCHEMA_VERSION,
    BenchmarkRunSpec,
)
from fingerprint_benchmark.detector_only_joint500 import (
    COHORT_SIZE,
    PROTOCOL_DIRECTORY,
    PROTOCOL_NAME,
    SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION,
    Joint500ProtocolError,
    build_protocol_artifacts,
    report_joint500,
    validate_joint_dataset_preflight,
    validate_protocol_artifacts,
    validate_sourceafis_preflight,
)
from fingerprint_benchmark.detectors.opencv_gftt_harris import (
    METHOD_NAME as HARRIS_METHOD,
    METHOD_VERSION as HARRIS_VERSION,
)
from fingerprint_benchmark.detectors.sourceafis_final_minutiae import (
    DETECTOR_VERSION as SOURCEAFIS_DETECTOR_VERSION,
    METHOD_NAME as SOURCEAFIS_METHOD,
    METHOD_VERSION as SOURCEAFIS_VERSION,
)
from fingerprint_benchmark.hashing import file_sha256, stable_config_hash, stable_hash
from fingerprint_benchmark.manifest import read_pair_manifest
from fingerprint_benchmark.runner import (
    RESULT_COLUMNS,
    RUN_METADATA_SCHEMA_VERSION,
    score_payload_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_committed_joint500_cohort_is_balanced_unique_neutral_and_byte_exact():
    report = validate_protocol_artifacts(repository_root=REPOSITORY_ROOT)
    assert report["status"] == "ok"
    assert report["cohort_size"] == COHORT_SIZE
    assert report["unique_subject_count"] == COHORT_SIZE
    assert report["per_position_counts"] == {str(position): 50 for position in range(1, 11)}
    assert report["same_identities_b_c"] is True
    assert report["impostor_bijection"] is True
    assert report["same_impostor_logic_b_c"] is True
    assert report["self_filtering"] is False
    assert report["method_score_result_dependency"] is False
    assert build_protocol_artifacts(repository_root=REPOSITORY_ROOT, check=True)["byte_exact"] is True

    metadata = json.loads(
        (REPOSITORY_ROOT / PROTOCOL_DIRECTORY / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["method_references"] == []
    assert metadata["score_references"] == []
    assert metadata["result_references"] == []
    assert "timestamp" not in metadata


def test_joint500_validator_rejects_artifact_and_base_manifest_tampering(tmp_path):
    root = _copy_protocol_inputs(tmp_path)
    build_protocol_artifacts(repository_root=root)
    artifact_root = root / PROTOCOL_DIRECTORY
    artifacts = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    assert len(artifacts) == 11
    for artifact in artifacts:
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"tamper\n")
        with pytest.raises(Joint500ProtocolError, match="mismatch"):
            validate_protocol_artifacts(repository_root=root)
        artifact.write_bytes(original)
    assert validate_protocol_artifacts(repository_root=root)["status"] == "ok"

    root = _copy_protocol_inputs(tmp_path / "base")
    build_protocol_artifacts(repository_root=root)
    base_manifest = root / "protocols" / "sd300b" / "plain_self.csv"
    base_manifest.write_bytes(base_manifest.read_bytes() + b"\n")
    with pytest.raises(Joint500ProtocolError, match="mismatch"):
        validate_protocol_artifacts(repository_root=root)


def test_joint500_cli_exposes_all_required_phases_and_filters():
    for phase in ("build", "validate", "preflight-sourceafis", "run", "report"):
        args = parse_args(["detector-joint500", phase])
        assert args.joint_phase == phase
    run = parse_args(
        [
            "detector-joint500",
            "run",
            "--dataset",
            "sd300c",
            "--pair-kind",
            "plain_roll_impostor",
            "--method",
            "sourceafis_final_minutiae_rootsift_geometric",
        ]
    )
    assert run.dataset == "sd300c"
    assert run.pair_kind == "plain_roll_impostor"
    report = parse_args(["detector-joint500", "report", "--allow-partial"])
    assert report.allow_partial is True


def test_joint_dataset_preflight_uses_cli_data_root_and_all_six_validators(tmp_path, monkeypatch):
    data_root = tmp_path / "fingerprint-datasets"
    data_root.mkdir()
    calls = []

    def fake_validator_for(dataset, protocol):
        def validate(path, supplied_data_root):
            calls.append((dataset, protocol, path, supplied_data_root))
            return {"status": "ok"}

        return validate

    monkeypatch.setattr(joint500, "validator_for", fake_validator_for)
    monkeypatch.setattr(joint500, "_validate_derived_dataset_paths", lambda **kwargs: 8000)
    report = validate_joint_dataset_preflight(
        repository_root=REPOSITORY_ROOT,
        data_root=data_root,
    )
    assert report["dataset_preflight_status"] == "ok"
    assert report["data_root"] == str(data_root.resolve())
    assert report["derived_dataset_path_count"] == 8000
    assert {(dataset, protocol) for dataset, protocol, _, _ in calls} == {
        (dataset, protocol)
        for dataset in ("sd300b", "sd300c")
        for protocol in ("plain_self", "roll_self", "plain_roll")
    }
    assert all(root == data_root.resolve() for _, _, _, root in calls)


def test_joint_dataset_preflight_rejects_wrong_root_and_base_validator_failure(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    with pytest.raises(Joint500ProtocolError, match="data root does not exist"):
        validate_joint_dataset_preflight(repository_root=REPOSITORY_ROOT, data_root=missing)

    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(
        joint500,
        "validator_for",
        lambda dataset, protocol: lambda path, supplied_root: (_ for _ in ()).throw(
            ValueError("synthetic base failure")
        ),
    )
    with pytest.raises(Joint500ProtocolError, match="Base manifest validation failed"):
        validate_joint_dataset_preflight(repository_root=REPOSITORY_ROOT, data_root=data_root)


def test_joint_run_rejects_dataset_preflight_before_runner_warmup(tmp_path, monkeypatch):
    runner_calls = []
    monkeypatch.setattr(
        joint500,
        "validate_joint_dataset_preflight",
        lambda **kwargs: (_ for _ in ()).throw(Joint500ProtocolError("missing dataset file")),
    )
    monkeypatch.setattr(
        "fingerprint_benchmark.runner.run_benchmark_manifest",
        lambda **kwargs: runner_calls.append(kwargs),
    )
    with pytest.raises(Joint500ProtocolError, match="missing dataset file"):
        joint500.run_joint500(
            method=HARRIS_METHOD,
            dataset="sd300b",
            pair_kind="plain_self",
            results_root=tmp_path / "results",
            data_root=tmp_path / "data",
            repository_root=REPOSITORY_ROOT,
        )
    assert runner_calls == []


def test_derived_dataset_reference_rejects_missing_and_outside_paths_and_accepts_valid(tmp_path):
    data_root = tmp_path / "fingerprint-datasets"
    valid = (
        data_root
        / "NIST"
        / "sd300b"
        / "images"
        / "1000"
        / "png"
        / "plain"
        / "00000001_plain_1000_11.png"
    )
    valid.parent.mkdir(parents=True)
    with pytest.raises(Joint500ProtocolError, match="missing"):
        joint500._validate_derived_image_reference(
            path=valid,
            data_root=data_root,
            dataset="sd300b",
            impression="plain",
            expected_ppi=1000,
            expected_frgp=11,
            expected_position=1,
            label="missing",
        )
    valid.touch()
    joint500._validate_derived_image_reference(
        path=valid,
        data_root=data_root,
        dataset="sd300b",
        impression="plain",
        expected_ppi=1000,
        expected_frgp=11,
        expected_position=1,
        label="valid",
    )
    outside = tmp_path / "outside" / "plain" / valid.name
    outside.parent.mkdir(parents=True)
    outside.touch()
    with pytest.raises(Joint500ProtocolError, match="outside"):
        joint500._validate_derived_image_reference(
            path=outside,
            data_root=data_root,
            dataset="sd300b",
            impression="plain",
            expected_ppi=1000,
            expected_frgp=11,
            expected_position=1,
            label="outside",
        )


def test_sourceafis_preflight_schema_is_strict_and_returns_path_and_sha(tmp_path):
    results = tmp_path / "results"
    payload = _valid_sourceafis_preflight()
    path = _write_sourceafis_preflight(results, payload)
    validated = validate_sourceafis_preflight(
        results_root=results,
        protocol_sha256="a" * 64,
        jar_sha256="b" * 64,
    )
    assert validated["preflight_path"] == str(path.resolve())
    assert validated["preflight_sha256"] == file_sha256(path)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "status",
        "protocol_name",
        "protocol_sha256",
        "detector_version",
        "sidecar_jar_sha256",
        "image_count",
        "identities_per_dataset",
        "impressions",
        "biometric_bytes_persisted",
        "items",
    ],
)
def test_sourceafis_preflight_rejects_tampered_top_level_fields(tmp_path, field):
    results = tmp_path / field
    payload = _valid_sourceafis_preflight()
    if field == "items":
        payload[field] = payload[field][:-1]
    elif field == "impressions":
        payload[field] = ["plain"]
    elif field == "biometric_bytes_persisted":
        payload[field] = True
    elif field in {"image_count", "identities_per_dataset"}:
        payload[field] += 1
    else:
        payload[field] = "tampered"
    _write_sourceafis_preflight(results, payload)
    with pytest.raises(Joint500ProtocolError):
        validate_sourceafis_preflight(
            results_root=results,
            protocol_sha256="a" * 64,
            jar_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "field",
    [
        "logical_image_id",
        "dataset",
        "subject_id",
        "canonical_finger_position",
        "impression",
        "ppi",
        "template_sha256",
        "minutia_count",
        "parity",
        "repeated_raw_payload_equal",
    ],
)
def test_sourceafis_preflight_rejects_tampered_item_fields(tmp_path, field):
    results = tmp_path / field
    payload = _valid_sourceafis_preflight()
    item = payload["items"][0]
    if field in {"canonical_finger_position", "ppi", "minutia_count"}:
        item[field] = -1
    elif field in {"parity", "repeated_raw_payload_equal"}:
        item[field] = False
    else:
        item[field] = "tampered"
    _write_sourceafis_preflight(results, payload)
    with pytest.raises(Joint500ProtocolError):
        validate_sourceafis_preflight(
            results_root=results,
            protocol_sha256="a" * 64,
            jar_sha256="b" * 64,
        )


def test_sourceafis_preflight_rejects_tampering_in_every_item(tmp_path):
    for index in range(20):
        results = tmp_path / str(index)
        payload = _valid_sourceafis_preflight()
        payload["items"][index]["parity"] = False
        _write_sourceafis_preflight(results, payload)
        with pytest.raises(Joint500ProtocolError):
            validate_sourceafis_preflight(
                results_root=results,
                protocol_sha256="a" * 64,
                jar_sha256="b" * 64,
            )


def test_sourceafis_preflight_rejects_extra_top_level_and_item_fields(tmp_path):
    for location in ("top", "item"):
        results = tmp_path / location
        payload = _valid_sourceafis_preflight()
        if location == "top":
            payload["image_base64"] = "biometric-content"
        else:
            payload["items"][0]["pixels"] = "biometric-content"
        _write_sourceafis_preflight(results, payload)
        with pytest.raises(Joint500ProtocolError, match="schema mismatch"):
            validate_sourceafis_preflight(
                results_root=results,
                protocol_sha256="a" * 64,
                jar_sha256="b" * 64,
            )


def test_sourceafis_startup_metadata_contains_preflight_binding(tmp_path):
    results = tmp_path / "results"
    path = _write_sourceafis_preflight(results, _valid_sourceafis_preflight())
    validated = validate_sourceafis_preflight(
        results_root=results,
        protocol_sha256="a" * 64,
        jar_sha256="b" * 64,
    )
    startup = SimpleNamespace(
        managed_by_runner=True,
        service_url="http://127.0.0.1:8765",
        startup_ms=1.0,
        validation_result="ok",
        command=["java", "-jar", "sidecar.jar"],
        jar_path="sidecar.jar",
        jar_sha256="b" * 64,
        java_executable="java",
    )
    metadata = joint500._sidecar_startup_dict(
        startup,
        {"status": "ok"},
        sourceafis_preflight=validated,
    )
    assert metadata["sourceafis_preflight"] == {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "schema_version": SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION,
        "protocol_sha256": "a" * 64,
        "detector_version": SOURCEAFIS_DETECTOR_VERSION,
        "sidecar_jar_sha256": "b" * 64,
        "image_count": 20,
    }


def test_report_rejects_sourceafis_bundle_when_bound_preflight_is_tampered(tmp_path):
    results = tmp_path / "results"
    protocol_sha = validate_protocol_artifacts(repository_root=REPOSITORY_ROOT)["protocol_sha256"]
    payload = _valid_sourceafis_preflight()
    payload["protocol_sha256"] = protocol_sha
    path = _write_sourceafis_preflight(results, payload)
    _write_bundle(results, "sd300b", "plain_self", SOURCEAFIS_METHOD)
    payload["items"][0]["parity"] = False
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Joint500ProtocolError):
        report_joint500(
            results_root=results,
            repository_root=REPOSITORY_ROOT,
            allow_partial=True,
        )


def test_report_requires_and_validates_complete_16_bundle_matrix_with_failure_metrics(tmp_path):
    results = tmp_path / "results"
    _write_complete_bundle_matrix(results, include_failures=True)
    outcome = report_joint500(results_root=results, repository_root=REPOSITORY_ROOT)
    assert outcome["status"] == "ok"
    assert outcome["complete_protocol_matrix"] is True
    assert outcome["validated_bundle_count"] == 16
    report = json.loads(
        (results / PROTOCOL_NAME / "report" / "report.json").read_text(encoding="utf-8")
    )
    assert report["complete_protocol_matrix"] is True
    assert report["validated_bundle_count"] == 16
    assert report["far_resolution"] == 0.002
    assert report["reported_far_operating_points"] == [0.01]
    for item in report["genuine_impostor_screening"]:
        assert item["conditional_on_success"] is True
        assert item["conditional_auc"] > 0.5
        assert item["conditional_screening_eer"] >= 0
        assert item["genuine_failure_count"] == 1
        assert item["genuine_failure_rate"] == pytest.approx(1 / 500)
        assert item["impostor_failure_count"] == 2
        assert item["impostor_failure_rate"] == pytest.approx(2 / 500)
        assert item["failure_policy"] == "fail_closed"
        assert item["selected_threshold"] == item["conditional_far_1_percent_threshold"]
        assert item["operational_tar_at_selected_threshold"] <= item["conditional_tar_at_far_1_percent"]
        assert item["operational_far_at_selected_threshold"] <= item["conditional_actual_far_at_1_percent"]
        assert "tar_at_far_0_1_percent" not in item
    markdown = (results / PROTOCOL_NAME / "report" / "report.md").read_text(encoding="utf-8")
    assert "conditional" in markdown.lower()
    assert "fail-closed" in markdown.lower()


def test_report_rejects_bare_partial_and_tampered_bundles(tmp_path):
    results = tmp_path / "bare"
    bare = results / PROTOCOL_NAME / "sd300b" / "plain_self" / HARRIS_METHOD
    bare.mkdir(parents=True)
    (bare / "pairs.csv").write_text("bare\n", encoding="utf-8")
    with pytest.raises(Joint500ProtocolError, match="Partial bundle"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)

    results = tmp_path / "tampered"
    bundle = _write_bundle(results, "sd300b", "plain_self", HARRIS_METHOD)
    metadata_path = bundle / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(Joint500ProtocolError, match="bundle validation failed"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)


def test_report_rejects_unknown_and_duplicate_or_misplaced_bundle_artifacts(tmp_path):
    results = tmp_path / "unknown"
    unknown = results / PROTOCOL_NAME / "sd300b" / "plain_self" / "unknown_method"
    unknown.mkdir(parents=True)
    (unknown / "pairs.csv").write_text("unknown\n", encoding="utf-8")
    (unknown / "run_metadata.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Joint500ProtocolError, match="Unknown joint-500 bundle method"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)

    results = tmp_path / "duplicate"
    original = _write_bundle(results, "sd300b", "plain_self", HARRIS_METHOD)
    misplaced = results / PROTOCOL_NAME / "duplicate" / "sd300b" / "plain_self" / HARRIS_METHOD
    shutil.copytree(original, misplaced)
    with pytest.raises(Joint500ProtocolError, match="misplaced"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)


def test_report_partial_mode_and_mixed_config_version_implementation_rejection(tmp_path):
    results = tmp_path / "partial"
    _write_bundle(results, "sd300b", "plain_self", HARRIS_METHOD)
    with pytest.raises(Joint500ProtocolError, match="requires 16"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT)
    outcome = report_joint500(
        results_root=results,
        repository_root=REPOSITORY_ROOT,
        allow_partial=True,
    )
    assert outcome["complete_protocol_matrix"] is False
    assert outcome["validated_bundle_count"] == 1

    for component in ("config", "implementation"):
        results = tmp_path / component
        kwargs_first = {f"{component}_suffix": "first"}
        kwargs_second = {f"{component}_suffix": "second"}
        _write_bundle(results, "sd300b", "plain_self", HARRIS_METHOD, **kwargs_first)
        _write_bundle(results, "sd300b", "roll_self", HARRIS_METHOD, **kwargs_second)
        with pytest.raises(Joint500ProtocolError, match="Mixed method version/config/implementation"):
            report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)

    results = tmp_path / "version"
    _write_bundle(
        results,
        "sd300b",
        "plain_self",
        HARRIS_METHOD,
        method_version_override="tampered-version",
    )
    with pytest.raises(Joint500ProtocolError, match="method identity mismatch"):
        report_joint500(results_root=results, repository_root=REPOSITORY_ROOT, allow_partial=True)


def _copy_protocol_inputs(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for dataset in ("sd300b", "sd300c"):
        shutil.copytree(
            REPOSITORY_ROOT / "protocols" / dataset,
            root / "protocols" / dataset,
        )
    return root


def _valid_sourceafis_preflight():
    items = []
    for dataset, ppi in (("sd300b", 1000), ("sd300c", 2000)):
        for identity in range(1, 6):
            subject = f"{identity:08d}"
            for impression in ("plain", "roll"):
                items.append(
                    {
                        "logical_image_id": f"{dataset}:{subject}:{identity:02d}:{impression}",
                        "dataset": dataset,
                        "subject_id": subject,
                        "canonical_finger_position": identity,
                        "impression": impression,
                        "ppi": ppi,
                        "template_sha256": f"{identity:x}" * 64,
                        "minutia_count": identity,
                        "parity": True,
                        "repeated_raw_payload_equal": True,
                    }
                )
    return {
        "schema_version": SOURCEAFIS_PREFLIGHT_SCHEMA_VERSION,
        "status": "ok",
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": "a" * 64,
        "detector_version": SOURCEAFIS_DETECTOR_VERSION,
        "sidecar_jar_sha256": "b" * 64,
        "image_count": 20,
        "identities_per_dataset": 5,
        "impressions": ["plain", "roll"],
        "items": items,
        "biometric_bytes_persisted": False,
    }


def _write_sourceafis_preflight(results: Path, payload) -> Path:
    path = results / PROTOCOL_NAME / "preflight_sourceafis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_complete_bundle_matrix(results: Path, *, include_failures: bool) -> None:
    protocol_sha = validate_protocol_artifacts(repository_root=REPOSITORY_ROOT)["protocol_sha256"]
    payload = _valid_sourceafis_preflight()
    payload["protocol_sha256"] = protocol_sha
    _write_sourceafis_preflight(results, payload)
    for method in (HARRIS_METHOD, SOURCEAFIS_METHOD):
        for dataset in ("sd300b", "sd300c"):
            for pair_kind in ("plain_self", "roll_self", "plain_roll_genuine", "plain_roll_impostor"):
                _write_bundle(
                    results,
                    dataset,
                    pair_kind,
                    method,
                    include_failures=include_failures,
                )


def _write_bundle(
    results: Path,
    dataset: str,
    pair_kind: str,
    method: str,
    *,
    include_failures: bool = False,
    config_suffix: str = "shared",
    implementation_suffix: str = "shared",
    method_version_override: str | None = None,
) -> Path:
    versions = {HARRIS_METHOD: HARRIS_VERSION, SOURCEAFIS_METHOD: SOURCEAFIS_VERSION}
    method_version = method_version_override or versions[method]
    manifest = REPOSITORY_ROOT / PROTOCOL_DIRECTORY / dataset / f"{pair_kind}.csv"
    records = read_pair_manifest(manifest)
    config = {"synthetic_bundle": True, "method": method, "identity_suffix": config_suffix}
    components = {
        "synthetic_bundle": True,
        "method": method,
        "identity_suffix": implementation_suffix,
    }
    if method == SOURCEAFIS_METHOD:
        components["sidecar_jar_sha256"] = "b" * 64
    config_hash = stable_config_hash(config)
    implementation_hash = stable_hash(components)
    score_semantics = "Synthetic geometric inlier count; no decision threshold."
    run_spec = BenchmarkRunSpec(
        expected_dataset=dataset,
        expected_protocol=f"{PROTOCOL_NAME}_{pair_kind}",
        manifest_path=manifest.resolve(),
        manifest_sha256=file_sha256(manifest),
        method=method,
        method_version=method_version,
        benchmark_contract_version=BENCHMARK_CONTRACT_VERSION,
        config_hash=config_hash,
        implementation_hash=implementation_hash,
    )
    rows = []
    for index, record in enumerate(records):
        failed = include_failures and (
            (pair_kind == "plain_roll_genuine" and index == 0)
            or (pair_kind == "plain_roll_impostor" and index < 2)
        )
        score = (
            10.0 - index / 1000
            if pair_kind == "plain_roll_genuine"
            else 1.0 + index / 10000
            if pair_kind == "plain_roll_impostor"
            else 12.0 - index / 1000
        )
        diagnostics = json.dumps(
            {"detector_point_count": 20, "representation_descriptor_count": 18},
            separators=(",", ":"),
        )
        compare = json.dumps(
            {
                "mutual_match_count": 8,
                "geometric_inlier_count": score,
                "inlier_ratio": 0.5,
                "score_components": {"inliers_over_min_keypoints": 0.4},
            },
            separators=(",", ":"),
        )
        row = {field: "" for field in RESULT_COLUMNS}
        row.update(
            {
                "pair_id": record.pair_id,
                "dataset": record.dataset,
                "protocol": record.protocol,
                "subject_id": record.subject_id,
                "canonical_finger_position": str(record.canonical_finger_position),
                "method": method,
                "method_version": method_version,
                "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "config_hash": config_hash,
                "implementation_hash": implementation_hash,
                "manifest_sha256": file_sha256(manifest),
                "score_direction": HIGHER_IS_MORE_SIMILAR,
                "score_semantics": score_semantics,
                "raw_score": "" if failed else repr(float(score)),
                "prepare_a_ms": "1.0",
                "prepare_a_diagnostics": diagnostics,
                "prepare_b_ms": "" if failed else "1.0",
                "prepare_b_diagnostics": "" if failed else diagnostics,
                "compare_ms": "" if failed else "1.0",
                "compare_diagnostics": "" if failed else compare,
                "total_ms": "1.0" if failed else "3.0",
                "status": "prepare_a_failure" if failed else "ok",
                "error_code": "synthetic_prepare" if failed else "",
                "error_message": "synthetic failure" if failed else "",
            }
        )
        rows.append(row)
    bundle = results / PROTOCOL_NAME / dataset / pair_kind / method
    bundle.mkdir(parents=True, exist_ok=True)
    result_path = bundle / "pairs.csv"
    _write_rows(result_path, rows, RESULT_COLUMNS)
    startup_validation = {}
    if method == SOURCEAFIS_METHOD:
        protocol_sha = validate_protocol_artifacts(repository_root=REPOSITORY_ROOT)["protocol_sha256"]
        validated = validate_sourceafis_preflight(
            results_root=results,
            protocol_sha256=protocol_sha,
            jar_sha256="b" * 64,
        )
        startup_validation["jar_sha256"] = "b" * 64
        startup_validation["sourceafis_preflight"] = {
            "path": validated["preflight_path"],
            "sha256": validated["preflight_sha256"],
            "schema_version": validated["schema_version"],
            "protocol_sha256": validated["protocol_sha256"],
            "detector_version": validated["detector_version"],
            "sidecar_jar_sha256": validated["sidecar_jar_sha256"],
            "image_count": validated["image_count"],
        }
    metadata = {
        "metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": dataset,
        "protocol": f"{PROTOCOL_NAME}_{pair_kind}",
        "method": method,
        "method_version": method_version,
        "score_direction": HIGHER_IS_MORE_SIMILAR,
        "score_semantics": score_semantics,
        "config": config,
        "config_hash": config_hash,
        "implementation_hash": implementation_hash,
        "implementation_hash_components": components,
        "run_spec": run_spec.as_dict(),
        "manifest": {
            "path": str(manifest.resolve()),
            "row_count": len(records),
            "sha256": file_sha256(manifest),
            "dedicated_validator_result": {"status": "ok"},
        },
        "result": {
            "path": str(result_path.resolve()),
            "relative_path": "pairs.csv",
            "row_count": len(rows),
            "sha256": file_sha256(result_path),
            "score_payload_sha256": score_payload_sha256(rows),
        },
        "startup_validation": startup_validation,
    }
    (bundle / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle


def _write_rows(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
