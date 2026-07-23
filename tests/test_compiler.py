from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_topology import valid_topology_payload

from torchforge.compiler import (
    CompilationResponse,
    CompilerError,
    OllamaCodeCompiler,
    build_compiler_prompt,
    compile_artifact_directory,
    validate_pytorch_source,
)

VALID_CODE = """import torch
from torch import nn

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)

    def forward(self, x):
        return self.projection(x)
"""


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_static_validation_accepts_module_and_strips_fence() -> None:
    source, class_name = validate_pytorch_source(f"```python\n{VALID_CODE.rstrip()}\n```")
    assert class_name == "TinyTransformer"
    assert source.startswith("import torch")


@pytest.mark.parametrize(
    "code, message",
    [
        (VALID_CODE + "\nprint('side effect')\n", "top-level"),
        (VALID_CODE.replace("return self.projection(x)", "return self.missing(x)"), "not initialized"),
        (
            VALID_CODE.replace("return self.projection(x)", "return torch.nn.Sequential('relu')(x)"),
            "not an nn.Module",
        ),
        (
            VALID_CODE.replace(
                "self.projection = nn.Linear(8, 8)", "self.projection = 'not callable'"
            ),
            "not initialized as modules",
        ),
        (VALID_CODE.replace("return self.projection(x)", "return self.projection(x.cuda())"), "CUDA"),
        (
            VALID_CODE.replace(
                "from torch import nn",
                "from torch import nn\nfrom torch.nn.functional import F",
            ),
            "Invalid functional import",
        ),
        (VALID_CODE.replace("        super().__init__()\n", ""), "super"),
        ("class Nothing:\n    pass\n", "import torch"),
    ],
)
def test_static_validation_rejects_invalid_contracts(code: str, message: str) -> None:
    with pytest.raises(CompilerError, match=message):
        validate_pytorch_source(code)


def test_ollama_compiler_uses_structured_output(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        content = CompilationResponse(
            code=VALID_CODE, class_name="TinyTransformer", assumptions=[]
        ).model_dump_json()
        return FakeResponse({"message": {"content": content}})

    monkeypatch.setattr("torchforge.compiler.urlopen", fake_urlopen)
    compiler = OllamaCodeCompiler(model="coder-test", timeout=12)
    response = compiler.compile(valid_topology_payload(), "paper text")

    assert response.class_name == "TinyTransformer"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "coder-test"
    assert captured["payload"]["format"]["title"] == "CompilationResponse"
    assert captured["payload"]["options"] == {
        "temperature": 0,
        "num_ctx": 8_192,
        "num_predict": 4_096,
    }


def test_bert_prompt_contains_architecture_contract() -> None:
    topology = valid_topology_payload()
    topology["architecture_name"] = "BERT"

    prompt = build_compiler_prompt(topology, "BERT paper")

    assert "ARCHITECTURE CONTRACT — BERT BASE" in prompt
    assert "intermediate_size=3072" in prompt
    assert "attention_mask=None" in prompt
    assert "pooler_output" in prompt


def test_unknown_architecture_does_not_claim_bert_contract() -> None:
    prompt = build_compiler_prompt(valid_topology_payload(), "paper")
    assert "ARCHITECTURE CONTRACT — BERT BASE" not in prompt


def test_bert_compilation_uses_deterministic_reference(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "bert-paper"
    artifact_dir.mkdir()
    topology = valid_topology_payload()
    topology["architecture_name"] = "BERT"
    (artifact_dir / "topology.json").write_text(json.dumps(topology), encoding="utf-8")
    (artifact_dir / "paper.md").write_text("BERT paper", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"pymupdf_text": "paper.md"}}), encoding="utf-8"
    )

    class MustNotRun:
        model = "unused"

        def compile(self, *args, **kwargs):
            raise AssertionError("Ollama should not generate a canonical BERT implementation")

    destination = compile_artifact_directory(
        artifact_dir, tmp_path / "output", compiler=MustNotRun()
    )
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    source = destination.read_text(encoding="utf-8")

    assert "self.position_embeddings" in source
    assert "self.token_type_embeddings" in source
    assert "intermediate_size=3072" in source
    assert "dim_feedforward=intermediate_size" in source
    assert manifest["compilation"]["model"] == "torchforge-reference"
    assert manifest["compilation"]["architecture_profile"] == "bert_base"


def test_compile_artifacts_writes_code_and_manifest(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper-hash"
    artifact_dir.mkdir()
    (artifact_dir / "topology.json").write_text(
        json.dumps(valid_topology_payload()), encoding="utf-8"
    )
    (artifact_dir / "pymupdf.md").write_text("paper text", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"pymupdf_text": "pymupdf.md", "nougat_text": None}}),
        encoding="utf-8",
    )

    class FakeCompiler:
        model = "fake-coder"

        def compile(self, topology, paper_text):
            assert topology["architecture_name"] == "Tiny Transformer"
            assert paper_text == "paper text"
            return CompilationResponse(
                code=VALID_CODE,
                class_name="TinyTransformer",
                assumptions=["Used eight hidden features."],
            )

    destination = compile_artifact_directory(
        artifact_dir, tmp_path / "output", compiler=FakeCompiler()
    )
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    normalized, _ = validate_pytorch_source(VALID_CODE)
    assert destination.read_text(encoding="utf-8") == normalized
    assert manifest["artifacts"]["generated_code"] == str(destination)
    assert manifest["compilation"]["class_name"] == "TinyTransformer"
    assert len(manifest["compilation"]["source_sha256"]) == 64


def test_compile_artifacts_requires_phase_two(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompilerError, match="Phase 2 artifact"):
        compile_artifact_directory(artifact_dir, tmp_path / "output")


def test_compile_rejects_low_confidence_topology(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper"
    artifact_dir.mkdir()
    topology = valid_topology_payload()
    topology["overall_confidence"] = 0.4
    (artifact_dir / "topology.json").write_text(json.dumps(topology), encoding="utf-8")
    (artifact_dir / "paper.md").write_text("paper", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"pymupdf_text": "paper.md"}}), encoding="utf-8"
    )

    class MustNotRun:
        model = "unused"

        def compile(self, *args, **kwargs):
            raise AssertionError("Low-confidence topology must not reach code generation")

    with pytest.raises(CompilerError, match="confidence 0.40.*required 0.60"):
        compile_artifact_directory(
            artifact_dir,
            tmp_path / "output",
            compiler=MustNotRun(),
        )


def test_response_class_name_must_match_source(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper"
    artifact_dir.mkdir()
    (artifact_dir / "topology.json").write_text(
        json.dumps(valid_topology_payload()), encoding="utf-8"
    )
    (artifact_dir / "text.md").write_text("text", encoding="utf-8")
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"pymupdf_text": "text.md"}}), encoding="utf-8"
    )

    class MismatchCompiler:
        model = "fake"

        def compile(self, topology, paper_text, **kwargs):
            return CompilationResponse(code=VALID_CODE, class_name="WrongName")

    with pytest.raises(CompilerError, match="class_name"):
        compile_artifact_directory(artifact_dir, tmp_path / "output", compiler=MismatchCompiler())
