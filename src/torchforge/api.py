"""Local HTTP API exposing the TorchForge Phase 1–4 workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlopen

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from torchforge.compiler import (
    DEFAULT_CODE_MODEL,
    CompilerError,
    OllamaCodeCompiler,
    compile_artifact_directory,
)
from torchforge.extractor import extract_pdf
from torchforge.validator import RuntimeValidationError, validate_artifact_directory
from torchforge.vision_parser import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_MODEL,
    OllamaVisionClient,
    VisionParserError,
    parse_artifact_directory,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _project_root() -> Path:
    configured = os.environ.get("TORCHFORGE_ROOT")
    return Path(configured or Path.cwd()).expanduser().resolve()


def _assets_root() -> Path:
    return _project_root() / "temp_assets"


def _output_root() -> Path:
    return _project_root() / "output_code"


def _input_root() -> Path:
    return _project_root() / "input_papers"


def _safe_name(filename: str) -> str:
    stem = Path(filename).stem
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("._-")
    return (normalized[:80] or "paper") + ".pdf"


def _paper_root(paper_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", paper_id):
        raise HTTPException(status_code=404, detail="Paper was not found.")
    root = (_assets_root() / paper_id).resolve()
    try:
        root.relative_to(_assets_root().resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Paper was not found.") from exc
    if not (root / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="Paper was not found.")
    return root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {path.name}.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=500, detail=f"{path.name} is malformed.")
    return value


def _paper_payload(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    artifacts = manifest.get("artifacts", {})
    metadata = manifest.get("metadata", {})
    title = metadata.get("title") or Path(manifest.get("source_path", root.name)).stem
    stages = {
        "extract": manifest.get("status") in {"completed", "completed_with_warnings"},
        "parse": bool(artifacts.get("topology"))
        and manifest.get("vision", {}).get("usable") is not False,
        "compile": bool(artifacts.get("generated_code")),
        "validate": manifest.get("validation", {}).get("status")
        in {"completed", "repaired"},
    }
    available_artifacts = ["manifest"]
    for key, name in (
        ("text", artifacts.get("nougat_text") or artifacts.get("pymupdf_text")),
        ("topology", artifacts.get("topology")),
        ("validation", artifacts.get("validation_report")),
        ("code", artifacts.get("generated_code")),
    ):
        if isinstance(name, str):
            available_artifacts.append(key)
    return {
        "id": root.name,
        "title": title,
        "source": Path(manifest.get("source_path", root.name)).name,
        "pageCount": manifest.get("page_count", 0),
        "status": manifest.get("status", "unknown"),
        "warnings": manifest.get("warnings", []),
        "errors": manifest.get("errors", []),
        "stages": stages,
        "availableArtifacts": available_artifacts,
        "visionModel": manifest.get("vision", {}).get("model"),
        "codeModel": manifest.get("compilation", {}).get("model"),
        "validation": manifest.get("validation"),
    }


def _all_papers() -> list[dict[str, Any]]:
    root = _assets_root()
    if not root.exists():
        return []
    papers: list[dict[str, Any]] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            papers.append(_paper_payload(manifest_path.parent))
        except HTTPException:
            continue
    return sorted(papers, key=lambda item: item["id"], reverse=True)


def _ollama_health() -> tuple[bool, list[str]]:
    try:
        with urlopen(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return False, []
    models = [
        entry.get("name", "")
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ]
    return True, models


async def _run_stage(operation: Callable[[], Any]) -> Any:
    try:
        return await asyncio.to_thread(operation)
    except (VisionParserError, CompilerError, RuntimeValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app = FastAPI(title="TorchForge API", version="0.5.0")
origins = [
    value.strip()
    for value in os.environ.get(
        "TORCHFORGE_ALLOWED_ORIGINS",
        (
            "http://localhost:3000,http://127.0.0.1:3000,"
            "https://torchforge-studio.shshank-work.chatgpt.site"
        ),
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_private_network=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Filename"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    ollama_ready, models = await asyncio.to_thread(_ollama_health)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return {
        "status": "ready",
        "device": device,
        "ollama": {"ready": ollama_ready, "models": models},
        "defaults": {
            "visionModel": DEFAULT_VISION_MODEL,
            "codeModel": DEFAULT_CODE_MODEL,
        },
    }


@app.get("/api/papers")
async def list_papers() -> dict[str, Any]:
    return {"papers": await asyncio.to_thread(_all_papers)}


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str) -> dict[str, Any]:
    return await asyncio.to_thread(_paper_payload, _paper_root(paper_id))


@app.post("/api/papers")
async def upload_paper(request: Request) -> dict[str, Any]:
    filename = request.headers.get("x-filename", "paper.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")
    content = await request.body()
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF must be between 1 byte and 100 MB.",
        )
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The uploaded file is not a PDF.")
    input_root = _input_root()
    input_root.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(filename)
    source = input_root / safe
    source.write_bytes(content)
    result = await asyncio.to_thread(extract_pdf, source, _assets_root(), False)
    if not result.succeeded or result.artifact_dir is None:
        raise HTTPException(
            status_code=422,
            detail=result.errors[0] if result.errors else "PDF extraction failed.",
        )
    return _paper_payload(Path(result.artifact_dir))


@app.post("/api/papers/{paper_id}/parse")
async def parse_paper(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    client = OllamaVisionClient()
    await _run_stage(lambda: parse_artifact_directory(root, client=client))
    return _paper_payload(root)


@app.post("/api/papers/{paper_id}/compile")
async def compile_paper(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    compiler = OllamaCodeCompiler()
    await _run_stage(
        lambda: compile_artifact_directory(root, _output_root(), compiler=compiler)
    )
    return _paper_payload(root)


@app.post("/api/papers/{paper_id}/validate")
async def validate_paper(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    compiler = OllamaCodeCompiler()
    report = await _run_stage(
        lambda: validate_artifact_directory(
            root,
            _output_root(),
            device_name="auto",
            max_repairs=2,
            compiler=compiler,
        )
    )
    if not report.succeeded:
        raise HTTPException(status_code=422, detail="Runtime validation failed.")
    return _paper_payload(root)


@app.get("/api/papers/{paper_id}/artifacts/{artifact_name}")
async def get_artifact(paper_id: str, artifact_name: str):
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    artifacts = manifest.get("artifacts", {})
    references = {
        "manifest": root / "manifest.json",
        "text": root / (artifacts.get("nougat_text") or artifacts.get("pymupdf_text", "")),
        "topology": root / artifacts.get("topology", ""),
        "validation": root / artifacts.get("validation_report", ""),
        "code": Path(artifacts.get("generated_code", "")),
    }
    candidate = references.get(artifact_name)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    candidate = candidate.resolve()
    allowed_root = _output_root().resolve() if artifact_name == "code" else root
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact was not found.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return FileResponse(candidate)
    return PlainTextResponse(candidate.read_text(encoding="utf-8"))
