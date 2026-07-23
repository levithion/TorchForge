from __future__ import annotations

import json
from pathlib import Path
import shutil

from torchforge.extractor import extract_pdf
from torchforge.models import ExtractionStatus


def test_extracts_text_images_and_candidate_pages(sample_pdf: Path, tmp_path: Path) -> None:
    result = extract_pdf(sample_pdf, tmp_path / "assets", nougat_enabled=False)

    assert result.status is ExtractionStatus.COMPLETED
    assert result.page_count == 3
    assert result.source_sha256
    assert result.artifact_dir is not None
    artifact_dir = Path(result.artifact_dir)
    assert (artifact_dir / "pymupdf.md").read_text(encoding="utf-8").startswith("# Page 1")
    assert result.artifacts["embedded_images"]
    assert result.artifacts["rendered_pages"] == [
        {"page": 2, "dpi": 150, "path": "pages/page-002.png"},
        {"page": 3, "dpi": 150, "path": "pages/page-003.png"},
    ]
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["source_sha256"] == result.source_sha256


def test_content_hash_prevents_same_name_collisions(sample_pdf: Path, tmp_path: Path) -> None:
    first = tmp_path / "one" / "paper.pdf"
    second = tmp_path / "two" / "paper.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    shutil.copyfile(sample_pdf, first)
    shutil.copyfile(sample_pdf, second)
    with second.open("ab") as target:
        target.write(b"\n% distinct but still valid PDF\n")

    first_result = extract_pdf(first, tmp_path / "assets", nougat_enabled=False)
    second_result = extract_pdf(second, tmp_path / "assets", nougat_enabled=False)

    assert first_result.succeeded and second_result.succeeded
    assert first_result.artifact_dir != second_result.artifact_dir


def test_repeat_extraction_removes_stale_managed_artifacts(
    sample_pdf: Path, tmp_path: Path
) -> None:
    first = extract_pdf(sample_pdf, tmp_path / "assets", nougat_enabled=False)
    artifact_dir = Path(first.artifact_dir or "")
    stale_page = artifact_dir / "pages" / "page-999.png"
    stale_image = artifact_dir / "images" / "stale.jpeg"
    stale_page.write_bytes(b"stale")
    stale_image.write_bytes(b"stale")
    (artifact_dir / "topology.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "validation.json").write_text("{}", encoding="utf-8")

    second = extract_pdf(sample_pdf, tmp_path / "assets", nougat_enabled=False)

    assert second.succeeded
    assert not stale_page.exists()
    assert not stale_image.exists()
    assert not (artifact_dir / "topology.json").exists()
    assert not (artifact_dir / "validation.json").exists()


def test_missing_nougat_completes_with_warning(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("torchforge.nougat.shutil.which", lambda _: None)

    result = extract_pdf(sample_pdf, tmp_path / "assets")

    assert result.status is ExtractionStatus.COMPLETED_WITH_WARNINGS
    assert result.ocr_provider == "pymupdf"
    assert "not installed" in result.warnings[0]


def test_corrupt_pdf_fails_and_writes_manifest(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a pdf")

    result = extract_pdf(source, tmp_path / "assets", nougat_enabled=False)

    assert result.status is ExtractionStatus.FAILED
    assert result.errors
    assert result.manifest_path is not None and result.manifest_path.exists()
