"""Command line entry points for pairwise benchmark execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

from fingerprint_data_discovery.nist_sd300 import DEFAULT_DATA_ROOT

from .manifest import read_pair_manifest
from .runner import run_benchmark_manifest
from .sourceafis_adapter import SourceAfisAdapter
from .sourceafis_client import SourceAfisSidecarClient, validate_health
from .sourceafis_sidecar import ManagedSourceAfisSidecar, SidecarStartup, unmanaged_startup
from .summary import summarize_result_file


DATASETS = ("sd300b", "sd300c")
PROTOCOLS = ("plain_self", "roll_self", "plain_roll")
DEFAULT_SERVICE_URL = "http://127.0.0.1:8765"
DEFAULT_SIDECAR_JAR = (
    Path("apps") / "sourceafis-sidecar" / "target" / "sourceafis-sidecar-0.2.0.jar"
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

    summarize = subparsers.add_parser("summarize", help="Summarize one or more result CSV files.")
    summarize.add_argument("results", nargs="+", type=Path)
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
        if args.command == "summarize":
            summaries = [summarize_result_file(path) for path in args.results]
            print(json.dumps(summaries, ensure_ascii=True, indent=2, sort_keys=True))
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
