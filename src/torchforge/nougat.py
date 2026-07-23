"""Optional subprocess adapter for Meta's Nougat OCR CLI."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class NougatResult:
    succeeded: bool
    output: str | None = None
    warning: str | None = None


def run_nougat(
    pdf_path: Path,
    artifact_dir: Path,
    *,
    timeout_seconds: float = 1200,
) -> NougatResult:
    """Run Nougat when installed and copy its non-empty MMD result."""

    executable = shutil.which("nougat")
    if executable is None:
        return NougatResult(False, warning="Nougat is not installed; using PyMuPDF text only.")

    try:
        with tempfile.TemporaryDirectory(prefix="nougat-", dir=artifact_dir) as temp_dir:
            completed = subprocess.run(
                [executable, str(pdf_path), "-o", temp_dir],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
                return NougatResult(
                    False,
                    warning=f"Nougat exited with status {completed.returncode}: {detail}",
                )

            candidates = sorted(Path(temp_dir).rglob("*.mmd"))
            source = next((candidate for candidate in candidates if candidate.stat().st_size), None)
            if source is None:
                return NougatResult(False, warning="Nougat completed without a non-empty .mmd file.")

            destination = artifact_dir / "nougat.mmd"
            shutil.copyfile(source, destination)
            return NougatResult(True, output=destination.name)
    except subprocess.TimeoutExpired:
        return NougatResult(False, warning=f"Nougat timed out after {timeout_seconds:g} seconds.")
    except OSError as exc:
        return NougatResult(False, warning=f"Nougat could not be started: {exc}")
