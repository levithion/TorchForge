"""Ollama vision integration for strict architecture-topology extraction."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from torchforge.architecture_profiles import (
    canonical_topology,
    canonical_topology_from_paper,
)
from torchforge.topology import NetworkTopology

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_VISION_MODEL = "llava"
DEFAULT_VISION_CONTEXT = 8_192


class VisionParserError(RuntimeError):
    """Raised when topology parsing cannot produce a valid result."""


def build_topology_prompt(
    schema: dict[str, Any], paper_context: str | None = None
) -> str:
    """Build a conservative prompt grounded in the enforced output schema."""

    prompt = (
        "You are analyzing architecture diagrams from one transformer or NLP research paper. "
        "Return only a network topology matching the supplied JSON schema. Identify operations, "
        "directed data flow, tensor shapes, layer parameters, residual/skip connections, model "
        "inputs, and model outputs. Use stable snake_case layer IDs. Every layer input must reference "
        "either a declared model input name or a declared layer ID; every connection endpoint must "
        "reference a declared layer ID. Do not invent values that are not "
        "visible: use null for unknown optional values, omit unknown parameters, and record material "
        "interpretations in assumptions. Confidence values must be between 0 and 1. The source_images "
        "field will be replaced by the caller after validation. Distinguish encoder-only, decoder-only, "
        "and encoder-decoder architectures explicitly. Do not label prediction heads or fine-tuning "
        "outputs as a decoder. Do not report high confidence when inputs, outputs, parameters, or graph "
        "connections are missing."
    )
    if paper_context:
        prompt += (
            "\n\nPAPER IDENTITY AND TEXT CONTEXT:\n"
            + paper_context
            + "\n\nThe inferred architecture must be consistent with this paper identity and context."
        )
    return prompt + "\n\nJSON schema:\n" + json.dumps(
        schema, separators=(",", ":"), sort_keys=True
    )


class OllamaVisionClient:
    """Minimal client for Ollama's local `/api/chat` vision endpoint."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_VISION_MODEL,
        timeout: float = 300,
        context_window: int = DEFAULT_VISION_CONTEXT,
    ) -> None:
        if context_window < 1:
            raise ValueError("context_window must be positive.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.context_window = context_window

    def parse_images(
        self, image_paths: list[Path], *, paper_context: str | None = None
    ) -> NetworkTopology:
        if not image_paths:
            raise VisionParserError("No diagram images were provided.")

        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise VisionParserError(f"Diagram images do not exist: {', '.join(missing)}")

        schema = NetworkTopology.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_topology_prompt(schema, paper_context),
                    "images": [
                        base64.b64encode(path.read_bytes()).decode("ascii")
                        for path in image_paths
                    ],
                }
            ],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_ctx": self.context_window},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise VisionParserError(
                f"Ollama returned HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise VisionParserError(
                f"Could not connect to Ollama at {self.base_url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise VisionParserError(f"Ollama timed out after {self.timeout:g} seconds.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionParserError("Ollama returned an invalid JSON response envelope.") from exc

        try:
            content = response_payload["message"]["content"]
            topology = NetworkTopology.model_validate_json(content)
        except (KeyError, TypeError) as exc:
            raise VisionParserError("Ollama response did not contain message.content.") from exc
        except ValidationError as exc:
            raise VisionParserError(f"Ollama topology failed schema validation: {exc}") from exc

        topology.source_images = [path.name for path in image_paths]
        return topology


def _manifest_images(
    artifact_dir: Path,
    manifest: dict[str, Any],
    *,
    max_images: int,
    preferred_pages: set[int] | None = None,
) -> list[Path]:
    artifacts = manifest.get("artifacts", {})
    rendered = artifacts.get("rendered_pages") or []
    embedded = artifacts.get("embedded_images") or []
    candidates = rendered if rendered else embedded
    if preferred_pages:
        preferred = [
            entry
            for entry in candidates
            if isinstance(entry, dict) and entry.get("page") in preferred_pages
        ]
        if preferred:
            candidates = preferred

    paths: list[Path] = []
    seen: set[Path] = set()
    for entry in candidates:
        relative = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative, str):
            continue
        candidate = (artifact_dir / relative).resolve()
        try:
            candidate.relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise VisionParserError(f"Manifest image escapes artifact directory: {relative}") from exc
        if candidate not in seen:
            paths.append(candidate)
            seen.add(candidate)
        if len(paths) >= max_images:
            break
    return paths


CAPTION_DEFINITION = re.compile(
    r"^\s*fig(?:ure)?\.?\s*\d+\s*[:.]",
    re.IGNORECASE | re.MULTILINE,
)


def _paper_context(
    root: Path, manifest: dict[str, Any]
) -> tuple[str | None, set[int]]:
    artifacts = manifest.get("artifacts", {})
    text_name = artifacts.get("nougat_text") or artifacts.get("pymupdf_text")
    if not isinstance(text_name, str):
        return None, set()
    text_path = (root / text_name).resolve()
    try:
        text_path.relative_to(root)
        text = text_path.read_text(encoding="utf-8")
    except (ValueError, OSError):
        return None, set()

    caption_pages: set[int] = set()
    for chunk in text.split("# Page ")[1:]:
        page_line, _, page_text = chunk.partition("\n")
        try:
            page_number = int(page_line.strip())
        except ValueError:
            continue
        if CAPTION_DEFINITION.search(page_text):
            caption_pages.add(page_number)
    identity = text[:2_000]
    evidence_lines = [
        line.strip()
        for line in text.splitlines()
        if re.search(
            r"\b(model architecture|input representation|BERT_BASE|BERTBASE|"
            r"number of layers|hidden size|self-attention heads|feed-forward|"
            r"token embeddings|segment embeddings|position embeddings)\b",
            line,
            re.IGNORECASE,
        )
    ]
    evidence = "\n".join(dict.fromkeys(evidence_lines))[:2_000]
    context = identity + (
        "\n\nARCHITECTURE EVIDENCE FROM THE PAPER:\n" + evidence if evidence else ""
    )
    return context, caption_pages


def _completeness_score(topology: NetworkTopology) -> float:
    """Estimate whether a topology is usable as a directed implementation contract."""

    multi_layer_connections = bool(topology.connections) or len(topology.layers) == 1
    parameterized = sum(bool(layer.parameters) for layer in topology.layers) / len(
        topology.layers
    )
    described = sum(bool(layer.description) for layer in topology.layers) / len(
        topology.layers
    )
    components = [
        bool(topology.inputs),
        bool(topology.outputs),
        multi_layer_connections,
        parameterized,
        described,
    ]
    return round(sum(float(component) for component in components) / len(components), 2)


def normalize_topology(
    topology: NetworkTopology, paper_context: str | None
) -> NetworkTopology:
    """Ground known models and calibrate incomplete model-generated results."""

    canonical = canonical_topology(topology.model_dump(mode="json"), paper_context)
    if canonical is not None:
        return NetworkTopology.model_validate(canonical)

    completeness = _completeness_score(topology)
    if topology.overall_confidence > completeness:
        topology.overall_confidence = completeness
        topology.assumptions.append(
            f"Overall confidence was capped at {completeness:.2f} because the topology "
            "is missing implementation-relevant graph details."
        )
    return topology


def parse_artifact_directory(
    artifact_dir: str | Path,
    *,
    client: OllamaVisionClient | None = None,
    max_images: int = 8,
) -> NetworkTopology:
    """Parse Phase 1 diagram artifacts and attach `topology.json` to the manifest."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if max_images < 1:
        raise VisionParserError("max_images must be at least 1.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VisionParserError(f"Artifact manifest does not exist: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VisionParserError(f"Could not read artifact manifest: {exc}") from exc

    paper_context, caption_pages = _paper_context(root, manifest)
    images = _manifest_images(
        root,
        manifest,
        max_images=max_images,
        preferred_pages=caption_pages,
    )
    if not images:
        raise VisionParserError("Artifact manifest contains no diagram images to parse.")

    canonical = canonical_topology_from_paper(
        paper_context,
        [path.name for path in images],
    )
    parser = client or OllamaVisionClient()
    if canonical is not None:
        topology = NetworkTopology.model_validate(canonical)
        parser_model = "torchforge-reference"
    elif isinstance(parser, OllamaVisionClient):
        topology = parser.parse_images(images, paper_context=paper_context)
        parser_model = parser.model
    else:
        topology = parser.parse_images(images)
        parser_model = parser.model
    topology = normalize_topology(topology, paper_context)
    topology_path = root / "topology.json"
    topology_path.write_text(topology.model_dump_json(indent=2) + "\n", encoding="utf-8")

    artifacts = manifest.setdefault("artifacts", {})
    artifacts["topology"] = topology_path.name
    manifest["vision"] = {
        "model": parser_model,
        "image_count": len(images),
        "source_images": [path.relative_to(root).as_posix() for path in images],
        "schema_version": topology.schema_version,
        "usable": topology.usable,
        "overall_confidence": topology.overall_confidence,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return topology
