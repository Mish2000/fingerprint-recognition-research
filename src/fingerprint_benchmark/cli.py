"""Command line entry points for pairwise benchmark execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .manifest import read_pair_manifest
from .detector_only_joint500 import (
    DATASETS as JOINT_DATASETS,
    PAIR_KINDS as JOINT_PAIR_KINDS,
    build_protocol_artifacts,
    report_joint500,
    run_joint500,
    run_sourceafis_preflight,
    validate_protocol_artifacts,
)
from .detectors.opencv_gftt_harris import METHOD_NAME as HARRIS_DETECTOR_METHOD
from .detectors.sourceafis_final_minutiae import METHOD_NAME as SOURCEAFIS_DETECTOR_METHOD
from .runner import run_benchmark_manifest
from .sift.parity import run_parity
from .sift.restored import RestoredSiftGeometricAdapter, restoration_provenance
from .sourceafis_adapter import SourceAfisAdapter
from .sourceafis_client import SourceAfisSidecarClient, validate_health
from .sourceafis_sidecar import ManagedSourceAfisSidecar, SidecarStartup, unmanaged_startup
from .summary import summarize_result_file


DATASETS = ("sd300b", "sd300c")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
DEFAULT_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_SIDECAR_JAR = (
    Path("apps") / "sourceafis-sidecar" / "target" / "sourceafis-sidecar-0.4.0.jar"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hardened pairwise fingerprint benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("sourceafis-smoke", help="Validate SourceAFIS and one manifest pair.")
    _add_sourceafis_connection_args(smoke, allow_unmanaged=True)
    smoke.add_argument("--manifest", type=Path, default=default_manifest_path("sd300b", "plain_self"))

    run_one = subparsers.add_parser("run-sourceafis", help="Run one isolated SourceAFIS benchmark-v2 bundle.")
    _add_sourceafis_connection_args(run_one)
    _add_benchmark_io_args(run_one)
    run_one.add_argument("--dataset", choices=DATASETS, required=True)
    run_one.add_argument("--protocol", choices=PROTOCOLS, required=True)
    run_one.add_argument("--manifest", type=Path)
    run_one.add_argument("--skip-existing", action="store_true")

    run_all = subparsers.add_parser(
        "run-sourceafis-all",
        help="Run all six bundles with a fresh managed JVM for every dataset/protocol.",
    )
    _add_sourceafis_connection_args(run_all)
    _add_benchmark_io_args(run_all)
    run_all.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a bundle only after full manifest/result/metadata validation.",
    )

    sift_smoke = subparsers.add_parser(
        "sift-geometric-smoke",
        help="Compare one manifest pair with the restored SIFT geometric baseline.",
    )
    sift_smoke.add_argument("--manifest", type=Path, default=default_manifest_path("sd300b", "plain_self"))
    sift_smoke.add_argument("--pair-id", help="Manifest pair_id; defaults to the first row.")

    sift_run = subparsers.add_parser(
        "run-sift-geometric",
        help="Run one restored SIFT geometric benchmark-v2 bundle.",
    )
    _add_benchmark_io_args(sift_run)
    sift_run.add_argument("--dataset", choices=DATASETS, required=True)
    sift_run.add_argument("--protocol", choices=PROTOCOLS, required=True)
    sift_run.add_argument("--manifest", type=Path)
    sift_run.add_argument("--skip-existing", action="store_true")

    sift_parity = subparsers.add_parser(
        "sift-geometric-parity",
        help="Prove restored/historical parity against the six historical bundles.",
    )
    sift_parity.add_argument(
        "--historical-results-root",
        type=Path,
        required=True,
        help="results/ directory of a read-only worktree at the historical source commit.",
    )
    sift_parity.add_argument("--repository-root", type=Path, default=Path("."))
    sift_parity.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "restoration_preflight" / "sift_geometric_v1" / "parity_report.json",
    )
    sift_parity.add_argument("--historical-source-commit")
    sift_parity.add_argument("--current-commit")

    summarize = subparsers.add_parser("summarize", help="Summarize one or more result CSV files.")
    summarize.add_argument("results", nargs="+", type=Path)

    joint = subparsers.add_parser("detector-joint500", help="Build, validate, run, or report joint-500 screening.")
    joint_phases = joint.add_subparsers(dest="joint_phase", required=True)
    joint_build = joint_phases.add_parser("build", help="Build deterministic joint-500 protocol artifacts.")
    joint_build.add_argument("--repository-root", type=Path, default=Path("."))
    joint_build.add_argument("--check", action="store_true")
    joint_validate = joint_phases.add_parser("validate", help="Validate all joint-500 artifacts without detectors.")
    joint_validate.add_argument("--repository-root", type=Path, default=Path("."))
    joint_preflight = joint_phases.add_parser(
        "preflight-sourceafis",
        help="Check raw-template/final-minutiae parity and encoded-ingestion diagnostics on 20 cohort images.",
    )
    _add_sourceafis_connection_args(joint_preflight)
    joint_preflight.add_argument("--results-root", type=Path, default=Path("results"))
    joint_preflight.add_argument("--repository-root", type=Path, default=Path("."))
    joint_run = joint_phases.add_parser("run", help="Run selected joint-500 benchmark bundles.")
    _add_sourceafis_connection_args(joint_run)
    _add_benchmark_io_args(joint_run)
    joint_run.add_argument("--repository-root", type=Path, default=Path("."))
    joint_run.add_argument("--dataset", choices=JOINT_DATASETS)
    joint_run.add_argument("--pair-kind", choices=JOINT_PAIR_KINDS)
    joint_run.add_argument(
        "--method",
        choices=(HARRIS_DETECTOR_METHOD, SOURCEAFIS_DETECTOR_METHOD),
        default=HARRIS_DETECTOR_METHOD,
    )
    joint_run.add_argument("--skip-existing", action="store_true")
    joint_report = joint_phases.add_parser("report", help="Report screening metrics from existing bundles only.")
    joint_report.add_argument("--results-root", type=Path, default=Path("results"))
    joint_report.add_argument("--output-directory", type=Path)
    joint_report.add_argument("--repository-root", type=Path, default=Path("."))
    joint_report.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a validated subset of the 16 bundles for debugging only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "sourceafis-smoke":
            summary = _sourceafis_smoke_command(args)
            print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "run-sourceafis":
            manifest = args.manifest or default_manifest_path(args.dataset, args.protocol)
            metadata = _run_sourceafis_managed(
                dataset=args.dataset,
                protocol=args.protocol,
                manifest_path=manifest,
                args=args,
                skip_existing=args.skip_existing,
            )
            print(json.dumps(_metadata_brief(metadata), ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "run-sourceafis-all":
            metadata_items = [
                _run_sourceafis_managed(
                    dataset=dataset,
                    protocol=protocol,
                    manifest_path=default_manifest_path(dataset, protocol),
                    args=args,
                    skip_existing=args.skip_existing,
                )
                for dataset, protocol in _all_runs()
            ]
            print(
                json.dumps(
                    [_metadata_brief(metadata) for metadata in metadata_items],
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "sift-geometric-smoke":
            print(json.dumps(_sift_smoke_command(args), ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "run-sift-geometric":
            manifest = args.manifest or default_manifest_path(args.dataset, args.protocol)
            adapter = RestoredSiftGeometricAdapter()
            try:
                metadata = run_benchmark_manifest(
                    manifest_path=manifest,
                    adapter=adapter,
                    expected_dataset=args.dataset,
                    expected_protocol=args.protocol,
                    results_root=args.results_root,
                    data_root=args.data_root,
                    skip_existing=args.skip_existing,
                    progress_callback=lambda completed, total: print(
                        f"[{args.dataset}/{args.protocol}] {completed}/{total} measured pairs",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
            finally:
                adapter.close()
            print(json.dumps(_metadata_brief(metadata), ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "sift-geometric-parity":
            report = run_parity(
                historical_results_root=args.historical_results_root,
                repository_root=args.repository_root,
                output_path=args.output,
                **(
                    {"historical_source_commit": args.historical_source_commit}
                    if args.historical_source_commit
                    else {}
                ),
                current_commit=args.current_commit,
            )
            print(json.dumps(_parity_brief(report), ensure_ascii=True, indent=2, sort_keys=True))
            return 0 if report["status"] == "pass" else 1
        if args.command == "summarize":
            summaries = [summarize_result_file(path) for path in args.results]
            print(json.dumps(summaries, ensure_ascii=True, indent=2, sort_keys=True))
            return 0
        if args.command == "detector-joint500":
            if args.joint_phase == "build":
                payload = build_protocol_artifacts(
                    repository_root=args.repository_root,
                    check=args.check,
                )
            elif args.joint_phase == "validate":
                payload = validate_protocol_artifacts(repository_root=args.repository_root)
            elif args.joint_phase == "preflight-sourceafis":
                payload = _joint_sourceafis_preflight_command(args)
            elif args.joint_phase == "run":
                payload = run_joint500(
                    method=args.method,
                    dataset=args.dataset,
                    pair_kind=args.pair_kind,
                    results_root=args.results_root,
                    data_root=args.data_root,
                    repository_root=args.repository_root,
                    sidecar_jar=args.sidecar_jar,
                    service_url=args.service_url,
                    timeout_seconds=args.timeout_seconds,
                    skip_existing=args.skip_existing,
                )
            elif args.joint_phase == "report":
                payload = report_joint500(
                    results_root=args.results_root,
                    output_directory=args.output_directory,
                    repository_root=args.repository_root,
                    allow_partial=args.allow_partial,
                )
            else:
                raise ValueError(f"Unsupported detector-joint500 phase: {args.joint_phase!r}")
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Error: unsupported command {args.command!r}", file=sys.stderr)
    return 1


def _add_sourceafis_connection_args(parser: argparse.ArgumentParser, *, allow_unmanaged: bool = False) -> None:
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL)
    parser.add_argument("--sidecar-jar", type=Path, default=DEFAULT_SIDECAR_JAR)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    if allow_unmanaged:
        parser.add_argument(
            "--unmanaged-sidecar",
            action="store_true",
            help="Use an already-running loopback service (smoke test only).",
        )


def _add_benchmark_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def _sourceafis_smoke_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.unmanaged_sidecar:
        return _sourceafis_smoke(
            args.manifest,
            args.service_url,
            _startup_dict(unmanaged_startup(args.service_url)),
            args.timeout_seconds,
        )
    with ManagedSourceAfisSidecar(
        args.sidecar_jar,
        args.service_url,
        timeout_seconds=args.timeout_seconds,
    ) as sidecar:
        return _sourceafis_smoke(
            args.manifest,
            args.service_url,
            _startup_dict(sidecar.startup),
            args.timeout_seconds,
        )


def _joint_sourceafis_preflight_command(args: argparse.Namespace) -> dict[str, Any]:
    with ManagedSourceAfisSidecar(
        args.sidecar_jar,
        args.service_url,
        timeout_seconds=args.timeout_seconds,
    ) as sidecar:
        if sidecar.startup is None or sidecar.startup.jar_sha256 is None:
            raise ValueError("Managed SourceAFIS preflight requires a validated JAR SHA-256.")
        client = SourceAfisSidecarClient(args.service_url, timeout_seconds=args.timeout_seconds)
        try:
            health = client.health()
            validate_health(health)
            return run_sourceafis_preflight(
                client=client,
                jar_sha256=sidecar.startup.jar_sha256,
                results_root=args.results_root,
                repository_root=args.repository_root,
            )
        finally:
            client.close()


def _sourceafis_smoke(
    manifest_path: Path,
    service_url: str,
    startup: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    client = SourceAfisSidecarClient(service_url, timeout_seconds=timeout_seconds)
    try:
        health = client.health()
        validate_health(health)
        adapter = SourceAfisAdapter(client, health=health)
        pair = read_pair_manifest(manifest_path)[0]
        prepared_a = adapter.prepare(pair.path_a, pair.image_metadata_a())
        prepared_b = adapter.prepare(pair.path_b, pair.image_metadata_b())
        comparison = adapter.compare(prepared_a.representation, prepared_b.representation)
        return {
            "status": "ok",
            "manifest": str(manifest_path),
            "pair_id": pair.pair_id,
            "raw_score": comparison.raw_score,
            "adapter_prepare_a_method_internal_ms": prepared_a.method_internal_ms,
            "adapter_prepare_b_method_internal_ms": prepared_b.method_internal_ms,
            "adapter_compare_method_internal_ms": comparison.method_internal_ms,
            "health": health.raw,
            "startup_validation": startup,
            "health_requests_before_pair_execution": client.health_request_count,
        }
    finally:
        client.close()


def _sift_smoke_command(args: argparse.Namespace) -> dict[str, Any]:
    pairs = read_pair_manifest(args.manifest)
    if args.pair_id:
        matching = [pair for pair in pairs if pair.pair_id == args.pair_id]
        if not matching:
            raise ValueError(f"Manifest has no pair_id {args.pair_id!r}.")
        pair = matching[0]
    else:
        pair = pairs[0]
    adapter = RestoredSiftGeometricAdapter()
    try:
        metadata = adapter.metadata()
        prepared_a = adapter.prepare(pair.path_a, pair.image_metadata_a())
        prepared_b = adapter.prepare(pair.path_b, pair.image_metadata_b())
        comparison = adapter.compare(prepared_a.representation, prepared_b.representation)
        return {
            "status": "ok",
            "manifest": str(args.manifest),
            "pair_id": pair.pair_id,
            "method": metadata.method,
            "method_version": metadata.method_version,
            "score_direction": metadata.score_direction,
            "raw_score": comparison.raw_score,
            "matches_submitted_to_geometry": comparison.diagnostics["matches_submitted_to_geometry"],
            "geometric_inlier_count": comparison.diagnostics["geometric_inlier_count"],
            "geometry_failure_reason": comparison.diagnostics["geometry_failure_reason"],
            "restoration_provenance": restoration_provenance(),
        }
    finally:
        adapter.close()


def _parity_brief(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "historical_source_commit": report["historical_source_commit"],
        "bundle_count": report["bundle_count"],
        "total_pair_count": report["total_pair_count"],
        "matched_pair_count": report["matched_pair_count"],
        "mismatch_count": report["mismatch_count"],
        "config_hash_reproduced_in_all_bundles": report["config_hash_reproduced_in_all_bundles"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
        "bundles": [
            {
                "dataset": bundle["dataset"],
                "protocol": bundle["protocol"],
                "status": bundle["status"],
                "pair_count": bundle["pair_count"],
                "matched_pair_count": bundle["matched_pair_count"],
            }
            for bundle in report["bundles"]
        ],
    }


def _run_sourceafis_managed(
    *,
    dataset: str,
    protocol: str,
    manifest_path: Path,
    args: argparse.Namespace,
    skip_existing: bool,
) -> dict[str, Any]:
    # The managed lifecycle is intentionally inside this function: run-all
    # invokes it six times, preventing undocumented cross-protocol JIT state.
    with ManagedSourceAfisSidecar(
        args.sidecar_jar,
        args.service_url,
        timeout_seconds=args.timeout_seconds,
    ) as sidecar:
        startup = _startup_dict(sidecar.startup)
        client = SourceAfisSidecarClient(args.service_url, timeout_seconds=args.timeout_seconds)
        try:
            health = client.health()
            validate_health(health)
            adapter = SourceAfisAdapter(client, health=health)
            return run_benchmark_manifest(
                manifest_path=manifest_path,
                adapter=adapter,
                expected_dataset=dataset,
                expected_protocol=protocol,
                results_root=args.results_root,
                startup_validation={
                    **startup,
                    "health": health.raw,
                    "health_requests_before_pair_execution": client.health_request_count,
                },
                data_root=args.data_root,
                skip_existing=skip_existing,
                progress_callback=lambda completed, total: print(
                    f"[{dataset}/{protocol}] {completed}/{total} measured pairs",
                    file=sys.stderr,
                    flush=True,
                ),
            )
        finally:
            client.close()


def default_manifest_path(dataset: str, protocol: str) -> Path:
    return Path("protocols") / dataset / f"{protocol}.csv"


def _all_runs() -> Iterable[tuple[str, str]]:
    for dataset in DATASETS:
        for protocol in PROTOCOLS:
            yield dataset, protocol


def _startup_dict(startup: SidecarStartup | None) -> dict[str, Any]:
    if startup is None:
        return {}
    return {
        "managed_by_runner": startup.managed_by_runner,
        "service_url": startup.service_url,
        "startup_ms": startup.startup_ms,
        "validation_result": startup.validation_result,
        "command": startup.command,
        "jar_path": startup.jar_path,
        "jar_sha256": startup.jar_sha256,
        "java_executable": startup.java_executable,
    }


def _metadata_brief(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": metadata["dataset"],
        "protocol": metadata["protocol"],
        "method": metadata["method"],
        "method_version": metadata["method_version"],
        "benchmark_contract_version": metadata["benchmark_contract_version"],
        "manifest_rows": metadata["manifest"]["row_count"],
        "result_rows": metadata["result"]["row_count"],
        "ok": metadata["success_count"],
        "failure_counts": metadata["failure_counts"],
        "result_path": metadata["result"]["path"],
        "result_sha256": metadata["result"]["sha256"],
        "score_payload_sha256": metadata["result"]["score_payload_sha256"],
        "config_hash": metadata["config_hash"],
        "implementation_hash": metadata["implementation_hash"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
