"""Local HTTP API exposing the TorchForge Phase 1–4 workflow."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import URLError
from urllib.request import urlopen

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from torchforge.compiler import (
    DEFAULT_CODE_CONTEXT,
    DEFAULT_CODE_MODEL,
    DEFAULT_CODE_OUTPUT_TOKENS,
    DEFAULT_MAX_TEXT_CHARS,
    CompilerError,
    OllamaCodeCompiler,
    compile_artifact_directory,
    validate_pytorch_source,
)
from torchforge.extractor import extract_pdf
from torchforge.topology import NetworkTopology
from torchforge.validator import (
    RuntimeValidationError,
    _dummy_inputs,
    _load_generated_class,
    infer_constructor_kwargs,
    validate_artifact_directory,
)
from torchforge.vision_parser import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_VISION_CONTEXT,
    DEFAULT_VISION_MODEL,
    OllamaVisionClient,
    VisionParserError,
    parse_artifact_directory,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
StageName = Literal["parse", "compile", "validate"]


class StageOptions(BaseModel):
    """User-selectable controls for local pipeline stages."""

    vision_model: str = DEFAULT_VISION_MODEL
    code_model: str = DEFAULT_CODE_MODEL
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_images: int = Field(default=8, ge=1, le=32)
    context_window: int = Field(default=DEFAULT_CODE_CONTEXT, ge=512, le=131_072)
    max_output_tokens: int = Field(default=DEFAULT_CODE_OUTPUT_TOKENS, ge=256, le=65_536)
    max_text_chars: int = Field(default=DEFAULT_MAX_TEXT_CHARS, ge=500, le=100_000)
    max_repairs: int = Field(default=2, ge=0, le=5)
    timeout: float = Field(default=600, ge=5, le=3600)


class JobRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1, max_length=50)
    stages: list[StageName] = Field(min_length=1, max_length=3)
    options: StageOptions = Field(default_factory=StageOptions)


class PaperUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=180)
    tags: list[str] | None = Field(default=None, max_length=20)
    archived: bool | None = None


_jobs: dict[str, dict[str, Any]] = {}
_job_cancellations: dict[str, threading.Event] = {}
_job_lock = threading.Lock()


def _project_root() -> Path:
    configured = os.environ.get("TORCHFORGE_ROOT")
    return Path(configured or Path.cwd()).expanduser().resolve()


def _assets_root() -> Path:
    return _project_root() / "temp_assets"


def _output_root() -> Path:
    return _project_root() / "output_code"


def _input_root() -> Path:
    return _project_root() / "input_papers"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _source_pdf(root: Path, manifest: dict[str, Any]) -> Path | None:
    source = Path(manifest.get("source_path", "")).expanduser().resolve()
    try:
        source.relative_to(_input_root().resolve())
    except ValueError:
        return None
    return source if source.is_file() and source.suffix.lower() == ".pdf" else None


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact was not found.") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact was not found.")
    return candidate


def _snapshot_artifact(root: Path, name: str, source: Path) -> str:
    revisions = root / "revisions"
    revisions.mkdir(exist_ok=True)
    suffix = source.suffix or ".txt"
    revision = revisions / f"{name}-{int(time.time() * 1000)}{suffix}"
    shutil.copy2(source, revision)
    return revision.relative_to(root).as_posix()


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
    studio = manifest.get("studio", {})
    title = (
        studio.get("title")
        or metadata.get("title")
        or Path(manifest.get("source_path", root.name)).stem
    )
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
        "tags": studio.get("tags", []),
        "archived": bool(studio.get("archived", False)),
        "createdAt": studio.get("created_at"),
        "updatedAt": studio.get("updated_at"),
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


def _configured_components(
    options: StageOptions,
) -> tuple[OllamaVisionClient, OllamaCodeCompiler]:
    vision = OllamaVisionClient(
        model=options.vision_model,
        timeout=options.timeout,
        context_window=min(options.context_window, DEFAULT_VISION_CONTEXT * 4),
    )
    compiler = OllamaCodeCompiler(
        model=options.code_model,
        timeout=options.timeout,
        context_window=options.context_window,
        max_output_tokens=options.max_output_tokens,
    )
    return vision, compiler


def _run_configured_stage(root: Path, stage: StageName, options: StageOptions) -> None:
    vision, compiler = _configured_components(options)
    if stage == "parse":
        parse_artifact_directory(root, client=vision, max_images=options.max_images)
    elif stage == "compile":
        compile_artifact_directory(
            root,
            _output_root(),
            compiler=compiler,
            max_text_chars=options.max_text_chars,
        )
    else:
        report = validate_artifact_directory(
            root,
            _output_root(),
            device_name=options.device,
            max_repairs=options.max_repairs,
            compiler=compiler,
        )
        if not report.succeeded:
            raise RuntimeValidationError("Runtime validation failed.")


def _job_public(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def _update_job(job_id: str, **values: Any) -> None:
    with _job_lock:
        if job_id in _jobs:
            _jobs[job_id].update(values)


def _append_job_log(job_id: str, message: str) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["logs"].append({"time": _now(), "message": message})
        job["logs"] = job["logs"][-200:]


def _run_job(job_id: str, paper_id: str, stages: list[StageName], options: StageOptions) -> None:
    cancellation = _job_cancellations[job_id]
    started = time.perf_counter()
    _update_job(job_id, status="running", startedAt=_now())
    try:
        root = _paper_root(paper_id)
        for index, stage in enumerate(stages):
            if cancellation.is_set():
                _update_job(job_id, status="cancelled", finishedAt=_now())
                _append_job_log(job_id, "Cancelled before the next stage.")
                return
            _update_job(
                job_id,
                stage=stage,
                progress=round((index / len(stages)) * 100),
            )
            _append_job_log(job_id, f"Starting {stage}.")
            stage_started = time.perf_counter()
            _run_configured_stage(root, stage, options)
            elapsed = time.perf_counter() - stage_started
            _append_job_log(job_id, f"Completed {stage} in {elapsed:.2f}s.")
            _update_job(job_id, progress=round(((index + 1) / len(stages)) * 100))
        _update_job(
            job_id,
            status="completed",
            stage=None,
            progress=100,
            finishedAt=_now(),
            durationMs=round((time.perf_counter() - started) * 1000),
            paper=_paper_payload(root),
        )
    except Exception as exc:
        _append_job_log(job_id, str(exc))
        _update_job(
            job_id,
            status="failed",
            stage=None,
            error=str(exc),
            finishedAt=_now(),
            durationMs=round((time.perf_counter() - started) * 1000),
        )


def _create_job(paper_id: str, stages: list[StageName], options: StageOptions) -> dict[str, Any]:
    _paper_root(paper_id)
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "paperId": paper_id,
        "stages": stages,
        "stage": None,
        "status": "queued",
        "progress": 0,
        "logs": [{"time": _now(), "message": "Queued pipeline job."}],
        "error": None,
        "createdAt": _now(),
        "startedAt": None,
        "finishedAt": None,
        "durationMs": None,
        "options": options.model_dump(),
    }
    with _job_lock:
        _jobs[job_id] = job
        _job_cancellations[job_id] = threading.Event()
        if len(_jobs) > 200:
            completed = [
                key
                for key, value in _jobs.items()
                if value["status"] in {"completed", "failed", "cancelled"}
            ]
            for old_id in completed[: len(_jobs) - 200]:
                _jobs.pop(old_id, None)
                _job_cancellations.pop(old_id, None)
    threading.Thread(
        target=_run_job,
        args=(job_id, paper_id, stages, options),
        daemon=True,
        name=f"torchforge-job-{job_id[:8]}",
    ).start()
    return _job_public(job)


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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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
        "capabilities": {
            "jobs": True,
            "topologyEditing": True,
            "codeEditing": True,
            "paperManagement": True,
            "bundleExport": True,
            "onnxExport": True,
        },
        "presets": {
            "fast": {
                "max_images": 4,
                "context_window": 4096,
                "max_output_tokens": 2048,
                "max_repairs": 1,
            },
            "balanced": StageOptions().model_dump(),
            "thorough": {
                "max_images": 16,
                "context_window": 16_384,
                "max_output_tokens": 8192,
                "max_repairs": 3,
            },
        },
    }


@app.get("/api/papers")
async def list_papers() -> dict[str, Any]:
    return {"papers": await asyncio.to_thread(_all_papers)}


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str) -> dict[str, Any]:
    return await asyncio.to_thread(_paper_payload, _paper_root(paper_id))


@app.patch("/api/papers/{paper_id}")
async def update_paper(paper_id: str, update: PaperUpdate) -> dict[str, Any]:
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    studio = manifest.setdefault("studio", {})
    if update.title is not None:
        studio["title"] = update.title.strip()
    if update.tags is not None:
        studio["tags"] = sorted(
            {
                tag.strip()[:40]
                for tag in update.tags
                if isinstance(tag, str) and tag.strip()
            }
        )
    if update.archived is not None:
        studio["archived"] = update.archived
    studio["updated_at"] = _now()
    _write_json(root / "manifest.json", manifest)
    return _paper_payload(root)


@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    trash = _project_root() / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / f"{root.name}-{int(time.time())}"
    shutil.move(str(root), destination)
    return {"deleted": paper_id, "recoverableFrom": str(destination)}


@app.post("/api/papers/{paper_id}/duplicate")
async def duplicate_paper(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    duplicate_id = f"{root.name}-run-{uuid.uuid4().hex[:6]}"
    destination = _assets_root() / duplicate_id
    shutil.copytree(root, destination)
    manifest = _read_json(destination / "manifest.json")
    studio = manifest.setdefault("studio", {})
    studio.update(
        {
            "title": f"{_paper_payload(root)['title']} — new run",
            "created_at": _now(),
            "updated_at": _now(),
            "archived": False,
        }
    )
    generated = manifest.get("artifacts", {}).get("generated_code")
    if isinstance(generated, str) and Path(generated).is_file():
        copied_code = _output_root() / f"{duplicate_id}.py"
        copied_code.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, copied_code)
        manifest["artifacts"]["generated_code"] = str(copied_code)
    _write_json(destination / "manifest.json", manifest)
    return _paper_payload(destination)


@app.post("/api/jobs")
async def create_jobs(request: JobRequest) -> dict[str, Any]:
    jobs = [
        _create_job(paper_id, list(dict.fromkeys(request.stages)), request.options)
        for paper_id in request.paper_ids
    ]
    return {"jobs": jobs}


@app.get("/api/jobs")
async def list_jobs(paper_id: str | None = None) -> dict[str, Any]:
    with _job_lock:
        values = [
            _job_public(job)
            for job in _jobs.values()
            if paper_id is None or job["paperId"] == paper_id
        ]
    return {"jobs": sorted(values, key=lambda job: job["createdAt"], reverse=True)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    with _job_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job was not found.")
        return _job_public(job)


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict[str, Any]:
    with _job_lock:
        job = _jobs.get(job_id)
        cancellation = _job_cancellations.get(job_id)
        if job is None or cancellation is None:
            raise HTTPException(status_code=404, detail="Job was not found.")
        if job["status"] in {"completed", "failed", "cancelled"}:
            return _job_public(job)
        cancellation.set()
        job["status"] = "cancelling"
        job["logs"].append(
            {
                "time": _now(),
                "message": "Cancellation requested; the active local operation will finish safely.",
            }
        )
        return _job_public(job)


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
    root = Path(result.artifact_dir)
    manifest = _read_json(root / "manifest.json")
    manifest["studio"] = {
        "created_at": _now(),
        "updated_at": _now(),
        "tags": [],
        "archived": False,
    }
    _write_json(root / "manifest.json", manifest)
    return _paper_payload(root)


@app.post("/api/papers/{paper_id}/parse")
async def parse_paper(
    paper_id: str, options: StageOptions | None = None
) -> dict[str, Any]:
    root = _paper_root(paper_id)
    settings = options or StageOptions()
    await _run_stage(lambda: _run_configured_stage(root, "parse", settings))
    return _paper_payload(root)


@app.post("/api/papers/{paper_id}/compile")
async def compile_paper(
    paper_id: str, options: StageOptions | None = None
) -> dict[str, Any]:
    root = _paper_root(paper_id)
    settings = options or StageOptions()
    await _run_stage(lambda: _run_configured_stage(root, "compile", settings))
    return _paper_payload(root)


@app.post("/api/papers/{paper_id}/validate")
async def validate_paper(
    paper_id: str, options: StageOptions | None = None
) -> dict[str, Any]:
    root = _paper_root(paper_id)
    settings = options or StageOptions()
    await _run_stage(lambda: _run_configured_stage(root, "validate", settings))
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


@app.put("/api/papers/{paper_id}/artifacts/topology")
async def update_topology(paper_id: str, topology: NetworkTopology) -> dict[str, Any]:
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    topology_path = root / "topology.json"
    if topology_path.is_file():
        revision = _snapshot_artifact(root, "topology", topology_path)
        manifest.setdefault("studio", {}).setdefault("revisions", []).append(
            {"artifact": "topology", "path": revision, "created_at": _now()}
        )
    topology_path.write_text(topology.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest.setdefault("artifacts", {})["topology"] = topology_path.name
    manifest.setdefault("vision", {})["usable"] = topology.usable
    manifest["vision"]["overall_confidence"] = topology.overall_confidence
    manifest.setdefault("studio", {})["updated_at"] = _now()
    _write_json(root / "manifest.json", manifest)
    return _paper_payload(root)


@app.put("/api/papers/{paper_id}/artifacts/code")
async def update_code(paper_id: str, request: Request) -> dict[str, Any]:
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    try:
        source = (await request.body()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Code must be UTF-8 text.") from exc
    expected = manifest.get("compilation", {}).get("class_name")
    try:
        validated, class_name = validate_pytorch_source(
            source,
            expected_class_name=expected if isinstance(expected, str) else None,
        )
    except CompilerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    code_reference = manifest.get("artifacts", {}).get("generated_code")
    if not isinstance(code_reference, str):
        raise HTTPException(status_code=404, detail="Generated code was not found.")
    destination = Path(code_reference).resolve()
    try:
        destination.relative_to(_output_root().resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Generated code was not found.") from exc
    if destination.is_file():
        revision = _snapshot_artifact(root, "code", destination)
        manifest.setdefault("studio", {}).setdefault("revisions", []).append(
            {"artifact": "code", "path": revision, "created_at": _now()}
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(validated, encoding="utf-8")
    compilation = manifest.setdefault("compilation", {})
    compilation["class_name"] = class_name
    compilation["source_sha256"] = hashlib.sha256(validated.encode("utf-8")).hexdigest()
    manifest.setdefault("studio", {})["updated_at"] = _now()
    _write_json(root / "manifest.json", manifest)
    return _paper_payload(root)


@app.get("/api/papers/{paper_id}/revisions")
async def list_revisions(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    return {"revisions": manifest.get("studio", {}).get("revisions", [])}


@app.get("/api/papers/{paper_id}/revisions/{revision_name}")
async def get_revision(paper_id: str, revision_name: str):
    root = _paper_root(paper_id)
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", revision_name):
        raise HTTPException(status_code=404, detail="Revision was not found.")
    candidate = _safe_artifact_path(root, f"revisions/{revision_name}")
    return PlainTextResponse(candidate.read_text(encoding="utf-8"))


@app.get("/api/papers/{paper_id}/source")
async def get_source_pdf(paper_id: str):
    root = _paper_root(paper_id)
    source = _source_pdf(root, _read_json(root / "manifest.json"))
    if source is None:
        raise HTTPException(status_code=404, detail="Source PDF is not available.")
    return FileResponse(
        source,
        media_type="application/pdf",
        filename=source.name,
        content_disposition_type="inline",
    )


@app.get("/api/papers/{paper_id}/evidence")
async def get_evidence(paper_id: str) -> dict[str, Any]:
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    rendered_page_numbers: set[int] = set()
    for collection in ("rendered_pages", "embedded_images"):
        for item in manifest.get("artifacts", {}).get(collection, []):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            page_number = item.get("page")
            if collection == "embedded_images" and page_number in rendered_page_numbers:
                continue
            relative = item["path"]
            if relative in seen:
                continue
            try:
                _safe_artifact_path(root, relative)
            except HTTPException:
                continue
            seen.add(relative)
            entries.append(
                {
                    "path": relative,
                    "page": item.get("page"),
                    "kind": "page" if collection == "rendered_pages" else "image",
                }
            )
            if collection == "rendered_pages" and isinstance(page_number, int):
                rendered_page_numbers.add(page_number)
    return {
        "sourceAvailable": _source_pdf(root, manifest) is not None,
        "images": entries,
        "visionSources": manifest.get("vision", {}).get("source_images", []),
    }


@app.get("/api/papers/{paper_id}/evidence/{artifact_path:path}")
async def get_evidence_image(paper_id: str, artifact_path: str):
    root = _paper_root(paper_id)
    candidate = _safe_artifact_path(root, artifact_path)
    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=404, detail="Evidence image was not found.")
    return FileResponse(candidate)


def _model_card(root: Path, manifest: dict[str, Any]) -> str:
    paper = _paper_payload(root)
    validation = manifest.get("validation") or {}
    return (
        f"# {paper['title']} — TorchForge model card\n\n"
        "## Source\n\n"
        f"- Paper file: `{paper['source']}`\n"
        f"- Artifact ID: `{paper['id']}`\n"
        f"- Pages: {paper['pageCount']}\n\n"
        "## Generation\n\n"
        f"- Vision model: `{paper['visionModel'] or 'not run'}`\n"
        f"- Code model: `{paper['codeModel'] or 'not run'}`\n"
        f"- Architecture profile: `{validation.get('architecture_profile') or 'none'}`\n\n"
        "## Validation\n\n"
        f"- Status: `{validation.get('status') or 'not run'}`\n"
        f"- Device: `{validation.get('device') or 'not run'}`\n"
        f"- Attempts: {validation.get('attempt_count', 0)}\n"
        f"- Output shapes: `{validation.get('output_shapes', [])}`\n\n"
        "## Limitations\n\n"
        "Generated source must be reviewed before use. Runtime validation is not a "
        "security sandbox and does not establish training quality or checkpoint equivalence.\n"
    )


@app.get("/api/papers/{paper_id}/exports/model-card")
async def export_model_card(paper_id: str):
    root = _paper_root(paper_id)
    content = _model_card(root, _read_json(root / "manifest.json"))
    return PlainTextResponse(
        content,
        headers={
            "Content-Disposition": f'attachment; filename="{paper_id}-MODEL_CARD.md"'
        },
    )


@app.get("/api/papers/{paper_id}/exports/bundle")
async def export_bundle(paper_id: str):
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in root.rglob("*"):
            if candidate.is_file():
                archive.write(candidate, f"artifacts/{candidate.relative_to(root).as_posix()}")
        generated = manifest.get("artifacts", {}).get("generated_code")
        if isinstance(generated, str) and Path(generated).is_file():
            archive.write(generated, f"generated/{Path(generated).name}")
        source = _source_pdf(root, manifest)
        if source is not None:
            archive.write(source, f"source/{source.name}")
        archive.writestr("MODEL_CARD.md", _model_card(root, manifest))
        archive.writestr(
            "REPRODUCE.txt",
            (
                f"uv run torchforge parse temp_assets/{paper_id}\n"
                f"uv run torchforge compile temp_assets/{paper_id}\n"
                f"uv run torchforge validate temp_assets/{paper_id}\n"
            ),
        )
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{paper_id}-bundle.zip"'},
    )


@app.get("/api/papers/{paper_id}/exports/onnx")
async def export_onnx(paper_id: str):
    root = _paper_root(paper_id)
    manifest = _read_json(root / "manifest.json")
    try:
        topology = _read_json(root / "topology.json")
        class_name = manifest["compilation"]["class_name"]
        code_path = Path(manifest["artifacts"]["generated_code"]).resolve()
        model_class = _load_generated_class(code_path, class_name)
        constructor_kwargs = infer_constructor_kwargs(model_class, topology)
        model = model_class(**constructor_kwargs).cpu().eval()
        inputs = _dummy_inputs(model, topology, constructor_kwargs, torch.device("cpu"))
        destination = _output_root() / f"{paper_id}.onnx"
        torch.onnx.export(
            model,
            tuple(inputs),
            destination,
            input_names=[f"input_{index}" for index in range(len(inputs))],
            dynamo=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "ONNX export failed for this generated module. Install the optional "
                f"ONNX dependencies and review dynamic inputs: {exc}"
            ),
        ) from exc
    return FileResponse(
        destination,
        media_type="application/octet-stream",
        filename=destination.name,
    )
