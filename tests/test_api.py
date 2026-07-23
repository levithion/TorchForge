from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
