"""Deterministic PDF extraction and artifact generation."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import unicodedata
from typing import Any

import fitz

from torchforge.models import ExtractionResult, ExtractionStatus
from torchforge.nougat import run_nougat

LOGGER = logging.getLogger(__name__)
FIGURE_CAPTION = re.compile(
    r"^\s*fig(?:ure)?\.?\s*\d+\s*[:.]",
    re.IGNORECASE | re.MULTILINE,
)
RENDER_DPI = 150


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stem(path: Path) -> str:
    normalized = unicodedata.normalize("NFKD", path.stem).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug[:64] or "paper"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_manifest(result: ExtractionResult) -> None:
    manifest = result.manifest_path
    if manifest is None:
        return
    manifest.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in metadata.items() if value not in (None, "")}


def extract_pdf(
    pdf_path: str | Path,
    assets_root: str | Path,
    nougat_enabled: bool = True,
    *,
    nougat_timeout: float = 1200,
) -> ExtractionResult:
    """Extract text and diagrams from one PDF into a content-addressed directory."""

    source = Path(pdf_path).expanduser().resolve()
    root = Path(assets_root).expanduser().resolve()

    if not source.is_file():
        return ExtractionResult(
            source_path=str(source),
            source_sha256="",
            status=ExtractionStatus.FAILED,
            errors=["Source PDF does not exist or is not a file."],
        )
    if source.suffix.lower() != ".pdf":
        return ExtractionResult(
            source_path=str(source),
            source_sha256="",
            status=ExtractionStatus.FAILED,
            errors=["Source file must have a .pdf extension."],
        )

    try:
        source_hash = sha256_file(source)
    except OSError as exc:
        return ExtractionResult(
            source_path=str(source),
            source_sha256="",
            status=ExtractionStatus.FAILED,
            errors=[f"Could not read source PDF: {exc}"],
        )

    artifact_dir = root / f"{_safe_stem(source)}-{source_hash[:12]}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for managed_directory in ("images", "pages"):
        managed_path = artifact_dir / managed_directory
        if managed_path.is_dir():
            shutil.rmtree(managed_path)
    for managed_file in ("pymupdf.md", "topology.json", "validation.json"):
        managed_path = artifact_dir / managed_file
        if managed_path.is_file():
            managed_path.unlink()
    result = ExtractionResult(
        source_path=str(source),
        source_sha256=source_hash,
        status=ExtractionStatus.FAILED,
        artifact_dir=str(artifact_dir),
    )

    try:
        document = fitz.open(source)
        if not document.is_pdf:
            raise ValueError("File content is not a PDF.")
        if document.needs_pass:
            raise ValueError("Encrypted PDFs requiring a password are not supported.")
        if document.page_count == 0:
            raise ValueError("PDF contains no pages.")

        result.page_count = document.page_count
        result.metadata = _clean_metadata(document.metadata)
        images_dir = artifact_dir / "images"
        pages_dir = artifact_dir / "pages"
        extracted_images: list[dict[str, Any]] = []
        rendered_pages: list[dict[str, Any]] = []
        text_sections: list[str] = []
        extracted_xrefs: dict[int, str] = {}

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_text = page.get_text("text").strip()
            text_sections.append(f"# Page {page_number}\n\n{page_text}".rstrip())
            page_images = page.get_images(full=True)

            for image_index, image_info in enumerate(page_images, start=1):
                xref = image_info[0]
                if xref in extracted_xrefs:
                    relative_path = extracted_xrefs[xref]
                else:
                    image = document.extract_image(xref)
                    extension = image.get("ext", "bin")
                    images_dir.mkdir(exist_ok=True)
                    destination = images_dir / f"page-{page_number:03d}-img-{image_index:02d}-xref-{xref}.{extension}"
                    destination.write_bytes(image["image"])
                    relative_path = _relative(destination, artifact_dir)
                    extracted_xrefs[xref] = relative_path
                extracted_images.append(
                    {"page": page_number, "xref": xref, "path": relative_path}
                )

            if page_images or FIGURE_CAPTION.search(page_text):
                pages_dir.mkdir(exist_ok=True)
                pixmap = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
                destination = pages_dir / f"page-{page_number:03d}.png"
                pixmap.save(destination)
                rendered_pages.append(
                    {"page": page_number, "dpi": RENDER_DPI, "path": _relative(destination, artifact_dir)}
                )

        document.close()
        text_path = artifact_dir / "pymupdf.md"
        text_path.write_text("\n\n".join(text_sections) + "\n", encoding="utf-8")
        result.artifacts = {
            "pymupdf_text": text_path.name,
            "nougat_text": None,
            "embedded_images": extracted_images,
            "rendered_pages": rendered_pages,
        }
        result.status = ExtractionStatus.COMPLETED
    except Exception as exc:
        result.errors.append(f"PDF extraction failed: {exc}")
        LOGGER.exception("Failed to extract %s", source)
        _write_manifest(result)
        return result

    if nougat_enabled:
        nougat = run_nougat(source, artifact_dir, timeout_seconds=nougat_timeout)
        if nougat.succeeded:
            result.ocr_provider = "nougat"
            result.artifacts["nougat_text"] = nougat.output
        elif nougat.warning:
            result.warnings.append(nougat.warning)
            result.status = ExtractionStatus.COMPLETED_WITH_WARNINGS

    _write_manifest(result)
    return result
