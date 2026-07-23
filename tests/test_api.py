from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from test_compiler import VALID_CODE
from test_topology import valid_topology_payload

from torchforge.api import app


def test_health_and_empty_paper_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    monkeypatch.setattr("torchforge.api._ollama_health", lambda: (False, []))
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/papers").json() == {"papers": []}


def test_upload_list_and_read_artifacts(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    client = TestClient(app)
    response = client.post(
        "/api/papers",
        content=sample_pdf.read_bytes(),
        headers={"x-filename": "Research Paper.pdf", "content-type": "application/pdf"},
    )

    assert response.status_code == 200
    paper = response.json()
    assert paper["stages"]["extract"]
    assert not paper["stages"]["parse"]
    assert client.get("/api/papers").json()["papers"][0]["id"] == paper["id"]
    assert client.get(f"/api/papers/{paper['id']}/artifacts/text").status_code == 200
    assert client.get(f"/api/papers/{paper['id']}/artifacts/manifest").status_code == 200


def test_upload_rejects_non_pdf(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    response = TestClient(app).post(
        "/api/papers",
        content=b"not a pdf",
        headers={"x-filename": "notes.txt"},
    )
    assert response.status_code == 415


def test_paper_id_cannot_escape_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    response = TestClient(app).get("/api/papers/%2E%2E%2Fsecret")
    assert response.status_code == 404


def test_hosted_frontend_can_preflight_local_engine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    response = TestClient(app).options(
        "/api/health",
        headers={
            "origin": "https://torchforge-studio.shshank-work.chatgpt.site",
            "access-control-request-method": "GET",
            "access-control-request-private-network": "true",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://torchforge-studio.shshank-work.chatgpt.site"
    )
    assert response.headers["access-control-allow-private-network"] == "true"


def _upload(client: TestClient, sample_pdf: Path, filename: str = "Paper.pdf") -> dict:
    response = client.post(
        "/api/papers",
        content=sample_pdf.read_bytes(),
        headers={"x-filename": filename, "content-type": "application/pdf"},
    )
    assert response.status_code == 200
    return response.json()


def test_paper_management_evidence_and_recoverable_delete(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    client = TestClient(app)
    paper = _upload(client, sample_pdf)
    paper_id = paper["id"]

    updated = client.patch(
        f"/api/papers/{paper_id}",
        json={"title": "Renamed paper", "tags": ["bert", "local"], "archived": True},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Renamed paper"
    assert updated.json()["tags"] == ["bert", "local"]
    assert updated.json()["archived"]

    evidence = client.get(f"/api/papers/{paper_id}/evidence")
    assert evidence.status_code == 200
    assert evidence.json()["sourceAvailable"]
    assert client.get(f"/api/papers/{paper_id}/source").headers["content-type"] == (
        "application/pdf"
    )

    duplicate = client.post(f"/api/papers/{paper_id}/duplicate")
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] != paper_id

    deleted = client.delete(f"/api/papers/{paper_id}")
    assert deleted.status_code == 200
    assert Path(deleted.json()["recoverableFrom"]).is_dir()
    assert client.get(f"/api/papers/{paper_id}").status_code == 404


def test_editable_topology_and_code_keep_revisions(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    client = TestClient(app)
    paper = _upload(client, sample_pdf)
    paper_id = paper["id"]
    root = tmp_path / "temp_assets" / paper_id

    first = client.put(
        f"/api/papers/{paper_id}/artifacts/topology",
        json=valid_topology_payload(),
    )
    assert first.status_code == 200
    payload = valid_topology_payload()
    payload["task"] = "edited task"
    second = client.put(f"/api/papers/{paper_id}/artifacts/topology", json=payload)
    assert second.status_code == 200

    output = tmp_path / "output_code" / f"{paper_id}.py"
    output.parent.mkdir()
    output.write_text(VALID_CODE, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["generated_code"] = str(output)
    manifest["compilation"] = {"class_name": "TinyTransformer"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    saved = client.put(
        f"/api/papers/{paper_id}/artifacts/code",
        content=VALID_CODE,
        headers={"content-type": "text/plain"},
    )
    assert saved.status_code == 200
    revisions = client.get(f"/api/papers/{paper_id}/revisions").json()["revisions"]
    assert {revision["artifact"] for revision in revisions} == {"topology", "code"}


def test_jobs_complete_and_bundle_export_contains_reproduction_file(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TORCHFORGE_ROOT", str(tmp_path))
    monkeypatch.setattr("torchforge.api._run_configured_stage", lambda *_args: None)
    client = TestClient(app)
    paper = _upload(client, sample_pdf)

    created = client.post(
        "/api/jobs",
        json={"paper_ids": [paper["id"]], "stages": ["parse", "compile"]},
    )
    assert created.status_code == 200
    job_id = created.json()["jobs"][0]["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "completed":
            break
        time.sleep(0.01)
    assert job["status"] == "completed"
    assert job["progress"] == 100

    bundle = client.get(f"/api/papers/{paper['id']}/exports/bundle")
    assert bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert "MODEL_CARD.md" in archive.namelist()
        assert "REPRODUCE.txt" in archive.namelist()
