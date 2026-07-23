from __future__ import annotations

from pathlib import Path

from torchforge.models import ExtractionResult, ExtractionStatus
from torchforge.watcher import PaperProcessor, wait_for_stable_file


def test_stability_check_accepts_unchanged_file(tmp_path: Path) -> None:
    source = tmp_path / "stable.pdf"
    source.write_bytes(b"stable")

    assert wait_for_stable_file(source, timeout=0.1, interval=0.001, stable_checks=2)


def test_stability_check_times_out_for_missing_file(tmp_path: Path) -> None:
    assert not wait_for_stable_file(
        tmp_path / "missing.pdf", timeout=0.01, interval=0.001, stable_checks=2
    )


def test_processor_ignores_non_pdf(tmp_path: Path) -> None:
    processor = PaperProcessor(tmp_path)
    assert processor.process_path(tmp_path / "notes.txt") is None


def test_processor_deduplicates_and_continues(sample_pdf: Path, tmp_path: Path, monkeypatch) -> None:
    calls: list[Path] = []

    def fake_extract(source, assets_root, nougat_enabled, **kwargs):
        calls.append(Path(source))
        return ExtractionResult(
            source_path=str(source),
            source_sha256="hash",
            status=ExtractionStatus.COMPLETED,
        )

    monkeypatch.setattr("torchforge.watcher.wait_for_stable_file", lambda *args, **kwargs: True)
    processor = PaperProcessor(tmp_path / "assets", extractor=fake_extract)

    first = processor.process_path(sample_pdf)
    duplicate = processor.process_path(sample_pdf)

    assert first is not None and first.succeeded
    assert duplicate is None
    assert calls == [sample_pdf]


def test_processor_contains_unexpected_extractor_error(
    sample_pdf: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("torchforge.watcher.wait_for_stable_file", lambda *args, **kwargs: True)

    def explode(*args, **kwargs):
        raise RuntimeError("unexpected")

    processor = PaperProcessor(tmp_path / "assets", extractor=explode)
    assert processor.process_path(sample_pdf) is None
