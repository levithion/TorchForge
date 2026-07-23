from __future__ import annotations

import json
from pathlib import Path

from torchforge.cli import main
from torchforge.topology import NetworkTopology


def test_extract_cli_success(sample_pdf: Path, tmp_path: Path, capsys) -> None:
    exit_code = main(
        ["extract", str(sample_pdf), "--assets-root", str(tmp_path / "assets"), "--no-nougat"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "completed"


def test_extract_cli_failure_is_nonzero(tmp_path: Path, capsys) -> None:
    exit_code = main(["extract", str(tmp_path / "missing.pdf"), "--no-nougat"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "failed"


def test_parse_cli_failure_is_nonzero(tmp_path: Path, capsys) -> None:
    exit_code = main(["parse", str(tmp_path / "missing-artifacts")])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "manifest does not exist" in payload["error"]


def test_parse_cli_success_outputs_topology(tmp_path: Path, capsys, monkeypatch) -> None:
    topology = NetworkTopology.model_validate(
        {
            "architecture_name": "CLI Transformer",
            "layers": [
                {
                    "id": "encoder",
                    "layer_type": "transformer_encoder",
                    "confidence": 0.8,
                }
            ],
            "overall_confidence": 0.8,
        }
    )
    captured: dict = {}

    def fake_parse(artifact_dir, *, client, max_images):
        captured["artifact_dir"] = artifact_dir
        captured["model"] = client.model
        captured["max_images"] = max_images
        return topology

    monkeypatch.setattr("torchforge.cli.parse_artifact_directory", fake_parse)
    exit_code = main(
        ["parse", str(tmp_path), "--model", "vision-test", "--max-images", "3"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["architecture_name"] == "CLI Transformer"
    assert captured == {
        "artifact_dir": tmp_path,
        "model": "vision-test",
        "max_images": 3,
    }


def test_compile_cli_failure_is_nonzero(tmp_path: Path, capsys) -> None:
    exit_code = main(["compile", str(tmp_path / "missing")])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"


def test_validate_cli_failure_is_nonzero(tmp_path: Path, capsys) -> None:
    exit_code = main(["validate", str(tmp_path / "missing"), "--device", "cpu"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "Phase 3 validation inputs" in payload["error"]
