import hashlib
import io
import sys

import pytest

import fingerprint_benchmark.sourceafis_sidecar as sidecar_module
from fingerprint_benchmark.sourceafis_client import SourceAfisClientError
from fingerprint_benchmark.sourceafis_sidecar import ManagedSourceAfisSidecar


class FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.pid = 4242
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise sidecar_module.subprocess.TimeoutExpired("java", timeout)
        return self.returncode


def test_managed_sidecar_rejects_remote_host_before_process_creation(monkeypatch, tmp_path):
    popen_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal popen_called
        popen_called = True
        raise AssertionError("Popen must not be called for a remote URL")

    monkeypatch.setattr(sidecar_module.subprocess, "Popen", fail_if_called)

    with pytest.raises(SourceAfisClientError) as exc_info:
        ManagedSourceAfisSidecar(tmp_path / "sidecar.jar", "http://192.168.1.5:8765")

    assert exc_info.value.error_code == "remote_transport_forbidden"
    assert popen_called is False


def test_startup_failure_always_stops_process_and_keeps_only_safe_bounded_excerpt(monkeypatch, tmp_path):
    jar_path = tmp_path / "sidecar.jar"
    jar_path.write_bytes(b"fake jar")
    secret = "A" * 256
    process = FakeProcess(
        stdout=f'SourceAFIS startup image_base64="{secret}"\n',
        stderr=f"template_a_base64={secret}\nstartup failed",
    )
    monkeypatch.setattr(sidecar_module.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(sidecar_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        sidecar_module.subprocess,
        "run",
        lambda *args, **kwargs: (process.kill(), sidecar_module.subprocess.CompletedProcess(args[0], 0))[1],
    )
    monkeypatch.setattr(
        ManagedSourceAfisSidecar,
        "_wait_until_ready",
        lambda self: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    manager = ManagedSourceAfisSidecar(jar_path, "http://127.0.0.1:8765")

    with pytest.raises(TimeoutError):
        manager.__enter__()

    assert process.terminated or process.killed
    assert process.poll() is not None
    assert manager.failure_output_excerpt
    joined = "\n".join(manager.failure_output_excerpt.values())
    assert secret not in joined
    assert "<redacted>" in joined
    assert all(len(value) <= sidecar_module.OUTPUT_TAIL_LIMIT for value in manager.failure_output_excerpt.values())


def test_successful_startup_records_resolved_java_command_and_jar_sha(monkeypatch, tmp_path):
    jar_path = tmp_path / "sidecar.jar"
    jar_bytes = b"sidecar artifact bytes"
    jar_path.write_bytes(jar_bytes)
    process = FakeProcess(stdout="ready\n")
    monkeypatch.setattr(sidecar_module.shutil, "which", lambda name: sys.executable)
    monkeypatch.setattr(sidecar_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        sidecar_module.subprocess,
        "run",
        lambda *args, **kwargs: (process.kill(), sidecar_module.subprocess.CompletedProcess(args[0], 0))[1],
    )
    monkeypatch.setattr(ManagedSourceAfisSidecar, "_wait_until_ready", lambda self: None)

    with ManagedSourceAfisSidecar(jar_path, "http://localhost:8765") as manager:
        assert manager.startup is not None
        assert manager.startup.jar_path == str(jar_path.resolve())
        assert manager.startup.jar_sha256 == hashlib.sha256(jar_bytes).hexdigest()
        assert manager.startup.java_executable == str(sidecar_module.Path(sys.executable).resolve())
        assert manager.startup.command == [manager.startup.java_executable, "-jar", manager.startup.jar_path]

    assert process.terminated or process.killed
