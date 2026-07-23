"""Data models shared by TorchForge ingestion components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ExtractionStatus(StrEnum):
    """Terminal state for a PDF extraction attempt."""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


@dataclass(slots=True)
class ExtractionResult:
    """Machine-readable record of an extraction attempt."""

    source_path: str
    source_sha256: str
    status: ExtractionStatus
    page_count: int = 0
    artifact_dir: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    ocr_provider: str = "pymupdf"
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status is not ExtractionStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @property
    def manifest_path(self) -> Path | None:
        if self.artifact_dir is None:
            return None
        return Path(self.artifact_dir) / "manifest.json"
