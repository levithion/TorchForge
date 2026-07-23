from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "transformer-paper.pdf"
    document = fitz.open()

    first = document.new_page()
    first.insert_text((72, 72), "Transformer architecture overview")

    second = document.new_page()
    second.insert_text((72, 72), "Figure 1. Encoder and decoder attention blocks")
    second.draw_rect(fitz.Rect(72, 100, 260, 220), color=(0, 0, 0))
    second.draw_line(fitz.Point(100, 160), fitz.Point(230, 160), color=(0, 0, 0))

    third = document.new_page()
    third.insert_text((72, 72), "An embedded attention heatmap")
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pixmap.clear_with(180)
    third.insert_image(fitz.Rect(72, 100, 112, 140), stream=pixmap.tobytes("png"))

    document.save(path)
    document.close()
    return path
