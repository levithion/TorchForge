from __future__ import annotations

import json
from pathlib import Path

import pytest
from torch import nn

from torchforge.compiler import CompilationResponse
from torchforge.validator import (
    RuntimeValidationError,
    ValidationStatus,
    infer_constructor_kwargs,
    run_forward_validation,
    validate_artifact_directory,
)

TOPOLOGY = {
    "schema_version": "1.0",
    "architecture_name": "Runtime model",
    "task": "runtime validation",
    "inputs": [{"name": "features", "shape": [None, None, 8], "dtype": "float32"}],
    "layers": [
        {
            "id": "projection",
            "layer_type": "linear",
            "inputs": ["features"],
            "parameters": {"hidden_size": 8, "num_heads": 2},
            "confidence": 0.95,
        }
    ],
    "connections": [],
    "outputs": [{"name": "features", "shape": [None, None, 8]}],
    "assumptions": [],
    "source_images": [],
    "overall_confidence": 0.95,
}

GOOD_CODE = """import torch
from torch import nn

class RuntimeModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)

    def forward(self, features):
        return self.projection(features)
"""

BAD_RUNTIME_CODE = GOOD_CODE.replace(
    "return self.projection(features)",
    "return self.projection(features) + torch.ones((2, 3))",
)


def _write_artifact(tmp_path: Path, source: str) -> tuple[Path, Path]:
    root = tmp_path / "paper-hash"
    output = tmp_path / "output"
    root.mkdir()
    output.mkdir()
    code_path = output / "paper_hash.py"
    code_path.write_text(source, encoding="utf-8")
    (root / "topology.json").write_text(json.dumps(TOPOLOGY), encoding="utf-8")
    (root / "paper.md").write_text("A small runtime model.", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "generated_code": str(code_path),
                    "pymupdf_text": "paper.md",
                },
                "compilation": {
                    "class_name": "RuntimeModel",
                    "model": "initial",
                },
            }
        ),
        encoding="utf-8",
    )
    return root, output


def test_infer_constructor_kwargs_uses_topology_dimensions() -> None:
    class NeedsDimensions(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_heads: int):
            super().__init__()

        def forward(self, features):
            return features

    assert infer_constructor_kwargs(NeedsDimensions, TOPOLOGY) == {
        "input_size": 8,
        "hidden_size": 8,
        "num_heads": 2,
    }


def test_infer_constructor_kwargs_rejects_unknown_required_argument() -> None:
    class Unsupported(nn.Module):
        def __init__(self, mystery: int):
            super().__init__()

        def forward(self, features):
            return features

    with pytest.raises(RuntimeValidationError, match="mystery"):
        infer_constructor_kwargs(Unsupported, TOPOLOGY)


def test_run_forward_validation_on_cpu(tmp_path: Path) -> None:
    code_path = tmp_path / "model.py"
    code_path.write_text(GOOD_CODE, encoding="utf-8")

    kwargs, input_shapes, output_shapes, performance = run_forward_validation(
        code_path, "RuntimeModel", TOPOLOGY, device_name="cpu"
    )

    assert kwargs == {"input_size": 8, "hidden_size": 8}
    assert input_shapes == [[1, 16, 8]]
    assert output_shapes == [[1, 16, 8]]
    assert performance.latency_ms_mean > 0
    assert performance.latency_ms_p50 >= performance.latency_ms_mean * 0.1
    assert performance.latency_ms_p95 >= performance.latency_ms_p50
    assert performance.throughput_samples_per_sec > 0
    assert performance.measured_forward_passes == 8


def test_successful_validation_updates_manifest(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)

    report = validate_artifact_directory(
        root, output, device_name="cpu", max_repairs=0
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert report.status is ValidationStatus.COMPLETED
    assert report.attempts[0].succeeded
    assert manifest["artifacts"]["validation_report"] == "validation.json"
    assert manifest["validation"]["status"] == "completed"
    assert manifest["validation"]["conformance_passed"] is True
    assert report.conformance_checks[0].name == "runtime.forward"
    assert report.performance is not None
    assert report.performance.latency_ms_mean > 0
    assert manifest["validation"]["performance"]["latency_ms_p50"] > 0


def test_toy_transformer_is_rejected_when_topology_claims_bert(tmp_path: Path) -> None:
    bert_code = """import torch
from torch import nn

class BERT(nn.Module):
    def __init__(self, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32)
        self.encoder = nn.TransformerEncoderLayer(32, 4, batch_first=True)

    def forward(self, input_ids):
        return self.encoder(self.embedding(input_ids))
"""
    root, output = _write_artifact(tmp_path, bert_code)
    topology = dict(TOPOLOGY)
    topology["architecture_name"] = "BERT"
    (root / "topology.json").write_text(json.dumps(topology), encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["compilation"]["class_name"] = "BERT"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_artifact_directory(
        root, output, device_name="cpu", max_repairs=0
    )

    assert report.status is ValidationStatus.FAILED
    assert report.architecture_profile == "bert_base"
    assert "Architecture conformance failed" in (report.attempts[0].error or "")
    assert "input_ids, attention_mask, and token_type_ids" in (
        report.attempts[0].error or ""
    )


def test_runtime_failure_is_recompiled_with_traceback(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, BAD_RUNTIME_CODE)

    class RepairCompiler:
        model = "repair-model"
        feedback: str | None = None

        def compile(self, topology, paper_text, validation_feedback=None):
            self.feedback = validation_feedback
            return CompilationResponse(
                code=GOOD_CODE,
                class_name="RuntimeModel",
                assumptions=["Corrected the runtime shape mismatch."],
            )

    compiler = RepairCompiler()
    report = validate_artifact_directory(
        root, output, device_name="cpu", max_repairs=1, compiler=compiler
    )

    assert report.status is ValidationStatus.REPAIRED
    assert [attempt.succeeded for attempt in report.attempts] == [False, True]
    assert compiler.feedback is not None
    assert "RuntimeError" in compiler.feedback


def test_failed_validation_records_traceback(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, BAD_RUNTIME_CODE)

    report = validate_artifact_directory(
        root, output, device_name="cpu", max_repairs=0
    )

    assert report.status is ValidationStatus.FAILED
    assert "RuntimeError" in (report.attempts[0].error or "")


def test_failed_repair_compilation_is_reported_instead_of_crashing(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, BAD_RUNTIME_CODE)

    class BrokenRepairCompiler:
        model = "broken-repair"

        def compile(self, topology, paper_text, validation_feedback=None):
            return CompilationResponse(
                code="import torch\nthis is not valid",
                class_name="RuntimeModel",
            )

    report = validate_artifact_directory(
        root,
        output,
        device_name="cpu",
        max_repairs=1,
        compiler=BrokenRepairCompiler(),
    )

    assert report.status is ValidationStatus.FAILED
    assert "REPAIR COMPILATION FAILED" in (report.attempts[0].error or "")
    assert "failed static validation" in (report.attempts[0].error or "")


def test_validation_rejects_unknown_device(tmp_path: Path) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)
    with pytest.raises(RuntimeValidationError, match="'auto'.*'cuda'.*'mps'.*'cpu'"):
        validate_artifact_directory(root, output, device_name="vulkan")


def test_auto_device_falls_back_to_cpu(tmp_path: Path, monkeypatch) -> None:
    root, output = _write_artifact(tmp_path, GOOD_CODE)
    monkeypatch.setattr("torchforge.validator.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torchforge.validator.torch.backends.mps.is_built", lambda: False)

    report = validate_artifact_directory(
        root, output, device_name="auto", max_repairs=0
    )

    assert report.status is ValidationStatus.COMPLETED
    assert report.device == "cpu"
