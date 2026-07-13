"""Lifecycle helper for the loopback-only SourceAFIS Java sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import TextIO

from .sourceafis_client import (
    SourceAfisClientError,
    SourceAfisContractError,
    SourceAfisSidecarClient,
    parse_sourceafis_service_url,
    validate_health,
)


OUTPUT_TAIL_LIMIT = 16_384
_PAYLOAD_ASSIGNMENT = re.compile(
    r'''(?i)(["']?(?:image_base64|template_base64|template_a_base64|template_b_base64)["']?\s*[:=]\s*)'''
    r'''(?:"[^"]*"|'[^']*'|[^\s,}\]]+)'''
)
_LONG_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/=])")


@dataclass(frozen=True)
class SidecarStartup:
    managed_by_runner: bool
    service_url: str
    startup_ms: float | None
    validation_result: str
    command: list[str]
    jar_path: str | None
    jar_sha256: str | None
    java_executable: str | None


class _BoundedOutputTail:
    def __init__(self, max_chars: int = OUTPUT_TAIL_LIMIT) -> None:
        self._max_chars = max_chars
        self._text = ""
        self._lock = threading.Lock()

    def append(self, value: str) -> None:
        if not value:
            return
        with self._lock:
            self._text = (self._text + value)[-self._max_chars :]

    def safe_excerpt(self) -> str:
        with self._lock:
            raw = self._text
        return _sanitize_process_output(raw)[-self._max_chars :].strip()


class ManagedSourceAfisSidecar:
    """Start one JVM for one dataset/protocol run and always stop it."""

    def __init__(
        self,
        jar_path: Path,
        service_url: str,
        *,
        timeout_seconds: float = 60.0,
    ) -> None:
        endpoint = parse_sourceafis_service_url(service_url)
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SourceAFIS sidecar startup timeout must be a positive finite number.")
        self.jar_path = Path(jar_path)
        self.service_url = endpoint.service_url
        self.timeout_seconds = timeout
        self._endpoint = endpoint
        self.process: subprocess.Popen[str] | None = None
        self.startup: SidecarStartup | None = None
        self.failure_output_excerpt: dict[str, str] = {}
        self._stdout_tail = _BoundedOutputTail()
        self._stderr_tail = _BoundedOutputTail()
        self._output_threads: list[threading.Thread] = []

    def __enter__(self) -> "ManagedSourceAfisSidecar":
        jar_path = self.jar_path.resolve()
        if not jar_path.is_file():
            raise FileNotFoundError(f"SourceAFIS sidecar jar does not exist: {jar_path}")
        java_command = shutil.which("java")
        if java_command is None:
            raise FileNotFoundError("Java executable was not found on PATH.")
        java_executable = str(Path(java_command).resolve())
        jar_sha256 = _file_sha256(jar_path)
        command = [java_executable, "-jar", str(jar_path)]
        env = {
            **os.environ,
            "SOURCEAFIS_HOST": self._endpoint.host,
            "SOURCEAFIS_PORT": str(self._endpoint.port),
        }
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self._start_output_drains()
            self._wait_until_ready()
            if self.process.poll() is not None:
                raise RuntimeError("SourceAFIS sidecar exited immediately after readiness validation.")
            startup_ms = (time.perf_counter() - started) * 1000.0
            self.jar_path = jar_path
            self.startup = SidecarStartup(
                managed_by_runner=True,
                service_url=self.service_url,
                startup_ms=startup_ms,
                validation_result="ok",
                command=command,
                jar_path=str(jar_path),
                jar_sha256=jar_sha256,
                java_executable=java_executable,
            )
            return self
        except BaseException as exc:
            self._stop_process()
            self._finish_output_drains()
            self._record_failure_output(exc)
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        unexpected_exit = self.process is not None and self.process.poll() is not None
        self._stop_process()
        self._finish_output_drains()
        if exc is not None or unexpected_exit:
            self._record_failure_output(exc)

    def safe_output_excerpt(self) -> dict[str, str]:
        excerpts = {
            "stdout": self._stdout_tail.safe_excerpt(),
            "stderr": self._stderr_tail.safe_excerpt(),
        }
        return {name: value for name, value in excerpts.items() if value}

    def _start_output_drains(self) -> None:
        if self.process is None:
            return
        for name, stream, tail in (
            ("stdout", self.process.stdout, self._stdout_tail),
            ("stderr", self.process.stderr, self._stderr_tail),
        ):
            if stream is None:
                continue
            thread = threading.Thread(
                target=_drain_stream,
                args=(stream, tail),
                name=f"sourceafis-sidecar-{name}",
                daemon=True,
            )
            thread.start()
            self._output_threads.append(thread)

    def _finish_output_drains(self) -> None:
        for thread in self._output_threads:
            thread.join(timeout=2.0)
        if self.process is not None:
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self._output_threads.clear()

    def _stop_process(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _wait_until_ready(self) -> None:
        deadline = time.perf_counter() + self.timeout_seconds
        last_error: SourceAfisClientError | None = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("SourceAFIS sidecar exited during startup.")
            try:
                client = SourceAfisSidecarClient(self.service_url, timeout_seconds=min(2.0, self.timeout_seconds))
                try:
                    health = client.health()
                    validate_health(health)
                finally:
                    client.close()
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("SourceAFIS sidecar exited during readiness validation.")
                return
            except SourceAfisContractError:
                raise
            except SourceAfisClientError as exc:
                last_error = exc
                time.sleep(min(0.25, max(0.0, deadline - time.perf_counter())))
        detail = f": {last_error}" if last_error is not None else ""
        raise TimeoutError(f"SourceAFIS sidecar did not become ready{detail}")

    def _record_failure_output(self, exc: BaseException | None) -> None:
        self.failure_output_excerpt = self.safe_output_excerpt()
        if exc is not None and self.failure_output_excerpt:
            exc.add_note(f"Safe SourceAFIS sidecar output excerpt: {self.failure_output_excerpt!r}")


def unmanaged_startup(service_url: str, command: list[str] | None = None) -> SidecarStartup:
    endpoint = parse_sourceafis_service_url(service_url)
    return SidecarStartup(
        managed_by_runner=False,
        service_url=endpoint.service_url,
        startup_ms=None,
        validation_result="ok",
        command=command or [],
        jar_path=None,
        jar_sha256=None,
        java_executable=None,
    )


def _drain_stream(stream: TextIO, tail: _BoundedOutputTail) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            tail.append(chunk)
    except (OSError, ValueError):
        return


def _sanitize_process_output(value: str) -> str:
    cleaned = "".join(character for character in value if character in "\n\r\t" or ord(character) >= 32)
    cleaned = _PAYLOAD_ASSIGNMENT.sub(lambda match: f"{match.group(1)}<redacted>", cleaned)
    return _LONG_BASE64_TOKEN.sub("<redacted-base64>", cleaned)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
