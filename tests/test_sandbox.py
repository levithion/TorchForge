from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from torchforge.sandbox import (
    _BEGIN,
    _END,
    reset_docker_cache,
    sandbox_enabled,
)
from torchforge.validator import (
    ValidationStatus,
    validate_artifact_directory,
)

GOOD_CODE = """import torch
from torch import nn

class RuntimeModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=8):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, features):
        return self.out(torch.relu(self.layer(features)))
"""

TOPOLOGY = {
    "schema_version": "1.0",
    "architecture_name": "Tiny",
    "task": "demo",
    "inputs": [{"name": "features", "shape": [None, 16, 8], "dtype": "float32"}],
    "layers": [
        {
            "id": "layer",
            "layer_type": "linear",
            "inputs": ["features"],
            "parameters": {"hidden_size": 8},
            "confidence": 0.9,
        }
    ],
    "connections": [],
    "outputs": [{"name": "out", "shape": [None, 16, 8], "dtype": "float32"}],
    "assumptions": [],
    "source_images": [],
    "overall_confidence": 0.9,
}


def _write_artifact(tmp_path: Path, code: str):
    root = tmp_path / "artifacts"
    output = tmp_path / "output_code"
    root.mkdir()
    output.mkdir()
    (root / "topology.json").write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    manifest = {
        "status": "completed",
        "compilation": {"class_name": "RuntimeModel"},
        "artifacts": {"generated_code": str(output / "model.py")},
        "warnings": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "model.py").write_text(code, encoding="utf-8")
    return root, output


def test_sandbox_runner_reports_completed_payload(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)
    runner = Path(__file__).resolve().parents[1] / "src" / "torchforge" / "sandbox_runner.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--code",
            str(output / "model.py"),
            "--class-name",
            "RuntimeModel",
            "--topology",
            str(root / "topology.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    begin = completed.stdout.index(_BEGIN) + len(_BEGIN)
    end = completed.stdout.index(_END)
    payload = json.loads(completed.stdout[begin:end])
    assert payload["status"] == "completed"
    assert payload["input_shapes"] == [[1, 16, 8]]
    assert payload["output_shapes"] == [[1, 16, 8]]
    assert payload["finite_outputs"] is True
    assert payload["gradient_flow"] is True
    assert payload["latency_ms_mean"] > 0


def test_sandbox_runner_reports_failure_traceback(tmp_path: Path) -> None:
    broken = GOOD_CODE.replace("return self.out(", "return self.missing(")
    _, output = _write_artifact(tmp_path, broken)
    runner = Path(__file__).resolve().parents[1] / "src" / "torchforge" / "sandbox_runner.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--code",
            str(output / "model.py"),
            "--class-name",
            "RuntimeModel",
            "--topology",
            str(tmp_path / "artifacts" / "topology.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    begin = completed.stdout.index(_BEGIN) + len(_BEGIN)
    end = completed.stdout.index(_END)
    payload = json.loads(completed.stdout[begin:end])
    assert payload["status"] == "failed"
    assert "missing" in payload["error"]


def test_sandboxed_validation_builds_isolated_report(tmp_path: Path, monkeypatch) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)

    canned = {
        "status": "completed",
        "constructor_kwargs": {"input_size": 8, "hidden_size": 8},
        "input_shapes": [[1, 16, 8]],
        "output_shapes": [[1, 16, 8]],
        "finite_outputs": True,
        "gradient_flow": True,
        "latency_ms_mean": 1.5,
        "throughput_samples_per_sec": 600.0,
        "error": None,
    }

    monkeypatch.setenv("TORCHFORGE_SANDBOX", "docker")
    monkeypatch.setattr("torchforge.validator.sandbox_enabled", lambda: True)
    monkeypatch.setattr("torchforge.validator.docker_available", lambda: True)
    monkeypatch.setattr(
        "torchforge.validator.run_sandboxed_validation",
        lambda *args, **kwargs: dict(canned),
    )

    report = validate_artifact_directory(root, output, device_name="cpu", max_repairs=0)

    assert report.status is ValidationStatus.COMPLETED
    assert report.sandboxed is True
    assert report.device == "cpu"
    assert {check.name for check in report.conformance_checks} >= {
        "runtime.forward",
        "runtime.finite_outputs",
        "runtime.gradient_flow",
    }
    assert report.performance is not None
    assert report.performance.latency_ms_mean == 1.5
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["performance"]["throughput_samples_per_sec"] == 600.0


def test_sandboxed_failure_triggers_repair_path(tmp_path: Path, monkeypatch) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)
    calls: list[dict] = []

    def failing_runner(*args, **kwargs):
        calls.append(kwargs)
        return {
            "status": "failed",
            "error": "RuntimeError: boom inside container",
        }

    monkeypatch.setattr("torchforge.validator.sandbox_enabled", lambda: True)
    monkeypatch.setattr("torchforge.validator.docker_available", lambda: True)
    monkeypatch.setattr(
        "torchforge.validator.run_sandboxed_validation", failing_runner
    )
    monkeypatch.setattr(
        "torchforge.validator.compile_artifact_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no repair available")),
    )

    report = validate_artifact_directory(root, output, device_name="cpu", max_repairs=1)

    assert report.status is ValidationStatus.FAILED
    assert report.sandboxed is True
    assert "boom inside container" in (report.attempts[0].error or "")
    assert len(calls) == 1


def test_sandbox_requested_without_docker_falls_back_to_process(
    tmp_path: Path, monkeypatch
) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)
    monkeypatch.setenv("TORCHFORGE_SANDBOX", "docker")
    monkeypatch.setattr("torchforge.validator.sandbox_enabled", lambda: True)
    monkeypatch.setattr("torchforge.validator.docker_available", lambda: False)

    report = validate_artifact_directory(root, output, device_name="cpu", max_repairs=0)

    assert report.status is ValidationStatus.COMPLETED
    assert report.sandboxed is False
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert any("Docker" in warning for warning in manifest.get("warnings", []))


def test_sandbox_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TORCHFORGE_SANDBOX", raising=False)
    reset_docker_cache()
    assert sandbox_enabled() is False
