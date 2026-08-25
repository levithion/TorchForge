"""Optional Docker isolation for generated-code execution.

Generated Python is untrusted model output; Phase 4 normally imports and
executes it inside the TorchForge process. When ``TORCHFORGE_SANDBOX=docker``
is set and a Docker daemon is reachable, runtime validation instead runs the
generated module inside a hardened, network-isolated container and consumes a
JSON result. If Docker is missing or fails, validation falls back to the
in-process path and records that decision.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_SANDBOX_IMAGE = "pytorch/pytorch:2.5.1-cpu"
DEFAULT_SANDBOX_MEMORY = "4g"
DEFAULT_SANDBOX_CPUS = "2"
DEFAULT_SANDBOX_TIMEOUT = 600.0

_BEGIN = "-----TORCHFORGE-SANDBOX-RESULT-BEGIN-----"
_END = "-----TORCHFORGE-SANDBOX-RESULT-END-----"

_docker_available_cache: bool | None = None


def sandbox_enabled() -> bool:
    return os.environ.get("TORCHFORGE_SANDBOX", "").strip().lower() == "docker"


def docker_available() -> bool:
    global _docker_available_cache
    if _docker_available_cache is None:
        executable = shutil.which("docker")
        if executable is None:
            _docker_available_cache = False
        else:
            try:
                probe = subprocess.run(
                    [executable, "version", "--format", "ok"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                _docker_available_cache = probe.returncode == 0
            except (OSError, subprocess.SubprocessError):
                _docker_available_cache = False
    return _docker_available_cache


def reset_docker_cache() -> None:
    """Forget the cached daemon probe; used by tests and CLI diagnostics."""

    global _docker_available_cache
    _docker_available_cache = None


def run_sandboxed_validation(
    code_path: Path,
    class_name: str,
    topology_path: Path,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Execute one generated module inside a hardened container.

    The container has no network, a read-only filesystem, dropped
    capabilities, no new privileges, and bounded CPU/memory. The generated
    file is bind-mounted read-only; results arrive as delimited JSON on
    stdout from the bundled runner script.
    """

    runner_path = Path(__file__).resolve().parent / "sandbox_runner.py"
    image = os.environ.get("TORCHFORGE_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
    memory = os.environ.get("TORCHFORGE_SANDBOX_MEMORY", DEFAULT_SANDBOX_MEMORY)
    cpus = os.environ.get("TORCHFORGE_SANDBOX_CPUS", DEFAULT_SANDBOX_CPUS)
    effective_timeout = (
        float(os.environ.get("TORCHFORGE_SANDBOX_TIMEOUT", timeout or DEFAULT_SANDBOX_TIMEOUT))
    )

    work_dir = code_path.parent
    command = [
        shutil.which("docker") or "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        f"--memory={memory}",
        f"--cpus={cpus}",
        "--tmpfs=/tmp:rw,size=64m",
        "-v",
        f"{work_dir}:/work:ro",
        "-v",
        f"{runner_path}:/runner.py:ro",
        "-v",
        f"{topology_path}:/topology.json:ro",
        image,
        "python",
        "/runner.py",
        "--code",
        f"/work/{code_path.name}",
        "--class-name",
        class_name,
        "--topology",
        "/topology.json",
    ]
    completed = subprocess.run(  # noqa: S603 - argv list, fixed binary
        command,
        capture_output=True,
        text=True,
        timeout=effective_timeout,
        check=False,
    )
    stdout = completed.stdout or ""
    start = stdout.find(_BEGIN)
    end = stdout.find(_END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(
            "The sandboxed runner did not return a parseable result.\n"
            f"exit={completed.returncode}\n{stdout[-2000:]}\n{(completed.stderr or '')[-2000:]}"
        )
    payload = json.loads(stdout[start + len(_BEGIN) : end])
    if not isinstance(payload, dict):
        raise RuntimeError("The sandboxed runner returned an unexpected payload type.")
    return payload
