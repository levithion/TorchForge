from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest

from test_topology import valid_topology_payload
from torchforge.topology import NetworkTopology
from torchforge.vision_parser import (
    OllamaVisionClient,
    VisionParserError,
    normalize_topology,
    parse_artifact_directory,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_request_uses_images_schema_and_zero_temperature(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image bytes")
    returned = valid_topology_payload()
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return FakeResponse({"message": {"role": "assistant", "content": json.dumps(returned)}})

    monkeypatch.setattr("torchforge.vision_parser.urlopen", fake_urlopen)
    client = OllamaVisionClient(base_url="http://localhost:11434/", model="llava", timeout=12)
    topology = client.parse_images([image], paper_context="BERT transformer encoder")

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 12
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"] == {"temperature": 0, "num_ctx": 8_192}
    assert captured["payload"]["format"]["title"] == "NetworkTopology"
    assert captured["payload"]["messages"][0]["images"]
    assert "BERT transformer encoder" in captured["payload"]["messages"][0]["content"]
    assert topology.source_images == ["diagram.png"]


def test_ollama_rejects_invalid_topology(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        "torchforge.vision_parser.urlopen",
        lambda *args, **kwargs: FakeResponse(
            {"message": {"content": json.dumps({"architecture_name": "incomplete"})}}
        ),
    )

    with pytest.raises(VisionParserError, match="schema validation"):
        OllamaVisionClient().parse_images([image])


def test_ollama_connection_error_is_actionable(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "diagram.png"
    image.write_bytes(b"image")

    def unavailable(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("torchforge.vision_parser.urlopen", unavailable)
    with pytest.raises(VisionParserError, match="Could not connect to Ollama"):
        OllamaVisionClient().parse_images([image])


def test_parse_artifacts_writes_topology_and_updates_manifest(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper-hash"
    pages = artifact_dir / "pages"
    images = artifact_dir / "images"
    pages.mkdir(parents=True)
    images.mkdir()
    (pages / "page-002.png").write_bytes(b"page")
    (images / "figure.png").write_bytes(b"figure")
    manifest = {
        "artifacts": {
            "rendered_pages": [{"page": 2, "path": "pages/page-002.png"}],
            "embedded_images": [{"page": 2, "path": "images/figure.png"}],
        }
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeClient:
        model = "test-vision"

        def __init__(self) -> None:
            self.paths: list[Path] = []

        def parse_images(self, paths: list[Path]) -> NetworkTopology:
            self.paths = paths
            payload = valid_topology_payload()
            payload["source_images"] = [path.name for path in paths]
            return NetworkTopology.model_validate(payload)

    client = FakeClient()
    topology = parse_artifact_directory(artifact_dir, client=client)
    updated = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert client.paths == [pages / "page-002.png"]
    assert topology.architecture_name == "Tiny Transformer"
    assert (artifact_dir / "topology.json").exists()
    assert updated["artifacts"]["topology"] == "topology.json"
    assert updated["vision"]["model"] == "test-vision"
    assert updated["vision"]["source_images"] == ["pages/page-002.png"]


def test_parse_prefers_pages_with_actual_figure_captions(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper-hash"
    pages = artifact_dir / "pages"
    pages.mkdir(parents=True)
    (pages / "page-002.png").write_bytes(b"reference")
    (pages / "page-003.png").write_bytes(b"caption")
    (artifact_dir / "paper.md").write_text(
        "# Page 1\nBERT: transformer encoder\n\n"
        "# Page 2\nAs shown in Figure 1, the model is deep.\n\n"
        "# Page 3\nFigure 1: Overall BERT pre-training architecture.\n",
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "pymupdf_text": "paper.md",
                    "rendered_pages": [
                        {"page": 2, "path": "pages/page-002.png"},
                        {"page": 3, "path": "pages/page-003.png"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeClient:
        model = "test-vision"

        def __init__(self) -> None:
            self.paths: list[Path] = []

        def parse_images(self, paths: list[Path]) -> NetworkTopology:
            self.paths = paths
            return NetworkTopology.model_validate(valid_topology_payload())

    client = FakeClient()
    parse_artifact_directory(artifact_dir, client=client)
    assert client.paths == [pages / "page-003.png"]


def test_parse_artifacts_rejects_manifest_path_escape(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"rendered_pages": [{"path": "../outside.png"}]}}),
        encoding="utf-8",
    )
    with pytest.raises(VisionParserError, match="escapes artifact directory"):
        parse_artifact_directory(artifact_dir)


def test_parse_artifacts_requires_images(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "paper"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"artifacts": {"rendered_pages": [], "embedded_images": []}}),
        encoding="utf-8",
    )
    with pytest.raises(VisionParserError, match="no diagram images"):
        parse_artifact_directory(artifact_dir)


def test_bert_topology_is_grounded_and_removes_hallucinated_decoder() -> None:
    raw = NetworkTopology.model_validate(
        {
            "schema_version": "1.0",
            "architecture_name": "BERTBASE",
            "task": None,
            "inputs": [],
            "layers": [
                {
                    "id": "embedding",
                    "layer_type": "embedding",
                    "confidence": 0.5,
                },
                {
                    "id": "encoder",
                    "layer_type": "encoder",
                    "confidence": 0.8,
                },
                {
                    "id": "decoder",
                    "layer_type": "decoder",
                    "confidence": 0.7,
                },
            ],
            "connections": [],
            "outputs": [],
            "assumptions": [],
            "source_images": ["page-003.png", "page-005.png"],
            "overall_confidence": 0.9,
        }
    )

    topology = normalize_topology(
        raw,
        "BERT: Bidirectional Encoder Representations from Transformers. "
        "BERT BASE uses L=12, H=768, and A=12.",
    )

    assert topology.architecture_name == "BERT Base Encoder"
    assert [layer.id for layer in topology.layers] == [
        "embeddings",
        "encoder_stack",
        "pooler",
    ]
    assert all("decoder" not in layer.layer_type for layer in topology.layers)
    assert topology.layers[1].parameters["num_layers"] == 12
    assert topology.layers[1].parameters["num_heads"] == 12
    assert topology.layers[1].parameters["intermediate_size"] == 3072
    assert [item.name for item in topology.inputs] == [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    assert [item.name for item in topology.outputs] == [
        "last_hidden_state",
        "pooler_output",
    ]
    assert len(topology.connections) == 2
    assert topology.source_images == ["page-003.png", "page-005.png"]


def test_exact_bert_paper_identity_uses_reference_without_vision_call(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "bert-paper"
    pages = artifact_dir / "pages"
    pages.mkdir(parents=True)
    (pages / "page-003.png").write_bytes(b"page")
    (artifact_dir / "paper.md").write_text(
        "# Page 1\n\nBERT: Pre-training of Deep Bidirectional Transformers for\n"
        "Language Understanding\n\n"
        "# Page 3\n\nFigure 1: Overall BERT pre-training architecture.\n",
        encoding="utf-8",
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "pymupdf_text": "paper.md",
                    "rendered_pages": [
                        {"page": 3, "path": "pages/page-003.png"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    class MustNotRun:
        model = "unused"

        def parse_images(self, paths):
            raise AssertionError("Certified BERT should not call the vision model")

    topology = parse_artifact_directory(artifact_dir, client=MustNotRun())
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))

    assert topology.architecture_name == "BERT Base Encoder"
    assert len(topology.layers) == 3
    assert manifest["vision"]["model"] == "torchforge-reference"
    assert manifest["vision"]["usable"] is True
    assert manifest["vision"]["overall_confidence"] == 0.98


def test_incomplete_unknown_topology_has_confidence_capped() -> None:
    topology = NetworkTopology.model_validate(
        {
            "architecture_name": "Unknown Encoder",
            "layers": [
                {
                    "id": "mystery",
                    "layer_type": "unknown",
                    "confidence": 0.9,
                }
            ],
            "overall_confidence": 0.95,
        }
    )

    normalized = normalize_topology(topology, "An unknown model.")

    assert normalized.overall_confidence == 0.2
    assert "confidence was capped" in normalized.assumptions[-1]
