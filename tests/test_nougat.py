from __future__ import annotations

import subprocess
from pathlib import Path

from torchforge.nougat import run_nougat


def test_nougat_success_copies_mmd(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.touch()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("torchforge.nougat.shutil.which", lambda _: "/bin/nougat")

    def fake_run(command, **kwargs):
        output_dir = Path(command[command.index("-o") + 1])
        (output_dir / "paper.mmd").write_text("# Equation\n$x+y$", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("torchforge.nougat.subprocess.run", fake_run)
    result = run_nougat(pdf, artifacts)

    assert result.succeeded
    assert result.output == "nougat.mmd"
    assert (artifacts / "nougat.mmd").read_text(encoding="utf-8").endswith("$x+y$")


def test_nougat_nonzero_exit_is_warning(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("torchforge.nougat.shutil.which", lambda _: "/bin/nougat")
    monkeypatch.setattr(
        "torchforge.nougat.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 7, "", "model failed"),
    )

    result = run_nougat(tmp_path / "paper.pdf", artifacts)

    assert not result.succeeded
    assert "status 7" in (result.warning or "")


def test_nougat_empty_output_is_warning(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("torchforge.nougat.shutil.which", lambda _: "/bin/nougat")
    monkeypatch.setattr(
        "torchforge.nougat.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""),
    )

    result = run_nougat(tmp_path / "paper.pdf", artifacts)

    assert not result.succeeded
    assert "non-empty" in (result.warning or "")


def test_nougat_timeout_is_warning(tmp_path: Path, monkeypatch) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setattr("torchforge.nougat.shutil.which", lambda _: "/bin/nougat")

    def time_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("torchforge.nougat.subprocess.run", time_out)
    result = run_nougat(tmp_path / "paper.pdf", artifacts, timeout_seconds=3)

    assert not result.succeeded
    assert "3 seconds" in (result.warning or "")
