"""Resilient directory monitoring for incoming papers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import threading
import time
from typing import Callable

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from torchforge.extractor import ExtractionResult, extract_pdf, sha256_file

LOGGER = logging.getLogger(__name__)


def wait_for_stable_file(
    path: Path,
    *,
    timeout: float = 60,
    interval: float = 0.5,
    stable_checks: int = 3,
) -> bool:
    """Return once file size and mtime remain unchanged for several checks."""

    deadline = time.monotonic() + timeout
    previous: tuple[int, int] | None = None
    unchanged = 0
    while time.monotonic() < deadline:
        try:
            stat = path.stat()
            current = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            current = None

        if current is not None and current == previous and current[0] > 0:
            unchanged += 1
            if unchanged >= stable_checks:
                return True
        else:
            unchanged = 0
        previous = current
        time.sleep(interval)
    return False


class PaperProcessor:
    """Stabilize and deduplicate filesystem events before extraction."""

    def __init__(
        self,
        assets_root: Path,
        *,
        nougat_enabled: bool = True,
        nougat_timeout: float = 1200,
        stability_timeout: float = 60,
        extractor: Callable[..., ExtractionResult] = extract_pdf,
    ) -> None:
        self.assets_root = assets_root
        self.nougat_enabled = nougat_enabled
        self.nougat_timeout = nougat_timeout
        self.stability_timeout = stability_timeout
        self.extractor = extractor
        self._seen_hashes: set[str] = set()
        self._lock = threading.Lock()

    def process_path(self, path: str | Path) -> ExtractionResult | None:
        source = Path(path)
        if source.suffix.lower() != ".pdf":
            return None
        if not wait_for_stable_file(source, timeout=self.stability_timeout):
            LOGGER.error("PDF did not stabilize before timeout: %s", source)
            return None

        try:
            source_hash = sha256_file(source)
        except OSError as exc:
            LOGGER.error("Could not fingerprint %s: %s", source, exc)
            return None

        with self._lock:
            if source_hash in self._seen_hashes:
                LOGGER.info("Ignoring duplicate PDF event for %s", source)
                return None
            self._seen_hashes.add(source_hash)

        try:
            result = self.extractor(
                source,
                self.assets_root,
                self.nougat_enabled,
                nougat_timeout=self.nougat_timeout,
            )
            if result.succeeded:
                LOGGER.info("Extracted %s to %s", source, result.artifact_dir)
            else:
                LOGGER.error("Extraction failed for %s: %s", source, "; ".join(result.errors))
            return result
        except Exception:
            LOGGER.exception("Unexpected extraction error for %s", source)
            return None


class PDFEventHandler(FileSystemEventHandler):
    def __init__(self, processor: PaperProcessor) -> None:
        self.processor = processor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torchforge")

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            self._submit(event.dest_path)

    def _submit(self, path: str) -> None:
        if Path(path).suffix.lower() == ".pdf":
            self._executor.submit(self.processor.process_path, path)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def watch_directory(
    input_dir: str | Path,
    assets_root: str | Path,
    *,
    nougat_enabled: bool = True,
    nougat_timeout: float = 1200,
    stability_timeout: float = 60,
) -> None:
    """Watch a directory until interrupted by the user."""

    watched = Path(input_dir).expanduser().resolve()
    watched.mkdir(parents=True, exist_ok=True)
    processor = PaperProcessor(
        Path(assets_root).expanduser().resolve(),
        nougat_enabled=nougat_enabled,
        nougat_timeout=nougat_timeout,
        stability_timeout=stability_timeout,
    )
    handler = PDFEventHandler(processor)
    # The polling backend works consistently on local, network, and sandboxed
    # macOS paths where an FSEvents stream may fail after startup.
    observer = PollingObserver(timeout=0.5)
    observer.schedule(handler, str(watched), recursive=False)
    observer.start()
    LOGGER.info("Watching %s for PDFs", watched)
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        LOGGER.info("Stopping watcher")
    finally:
        observer.stop()
        observer.join()
        handler.close()
