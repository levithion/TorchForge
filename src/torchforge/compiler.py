"""Compile extracted paper artifacts into statically validated PyTorch source."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from torchforge.architecture_profiles import (
    generation_contract,
    identify_architecture,
    reference_implementation,
)
from torchforge.topology import MIN_USABLE_CONFIDENCE, NetworkTopology

DEFAULT_CODE_MODEL = "qwen2.5-coder:3b"
DEFAULT_CODE_CONTEXT = 8_192
DEFAULT_CODE_OUTPUT_TOKENS = 4_096
DEFAULT_MAX_TEXT_CHARS = 6_000
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class CompilerError(RuntimeError):
    """Raised when code generation or static validation fails."""


class CompilationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    class_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    assumptions: list[str] = Field(default_factory=list)


def _strip_fences(code: str) -> str:
    cleaned = code.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", cleaned, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else cleaned).strip() + "\n"


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _base_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def validate_pytorch_source(
    code: str, *, expected_class_name: str | None = None
) -> tuple[str, str]:
    """Validate syntax and the minimum nn.Module contract without importing code."""

    source = _strip_fences(code)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CompilerError(f"Generated code is not valid Python: {exc}") from exc

    tree.body = [statement for statement in tree.body if not _is_main_guard(statement)]
    source = ast.unparse(tree).strip() + "\n"

    allowed_top_level = (ast.Import, ast.ImportFrom, ast.ClassDef)
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if not isinstance(statement, allowed_top_level):
            raise CompilerError("Generated code contains executable top-level statements.")

    imported_torch = any(
        (isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "torch")
        for node in tree.body
    )
    if not imported_torch:
        raise CompilerError("Generated code must import torch.")
    for statement in tree.body:
        if (
            isinstance(statement, ast.ImportFrom)
            and statement.module == "torch.nn.functional"
            and any(alias.name == "F" for alias in statement.names)
        ):
            raise CompilerError(
                "Invalid functional import: use `import torch.nn.functional as F`, "
                "not `from torch.nn.functional import F`."
            )

    module_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(_base_name(base) in {"nn.Module", "torch.nn.Module", "Module"} for base in node.bases)
    ]
    if not module_classes:
        raise CompilerError("Generated code must define an nn.Module subclass.")
    if expected_class_name is None:
        if len(module_classes) != 1:
            raise CompilerError("Generated code defines multiple nn.Module subclasses without a root class.")
        module_class = module_classes[0]
    else:
        matches = [node for node in module_classes if node.name == expected_class_name]
        if len(matches) != 1:
            raise CompilerError(
                f"Response class_name {expected_class_name!r} does not identify one generated nn.Module."
            )
        module_class = matches[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "cuda" in node.value.lower():
            raise CompilerError("Generated code must be device-agnostic and cannot target CUDA.")
        if isinstance(node, ast.Attribute) and node.attr.lower() == "cuda":
            raise CompilerError("Generated code must be device-agnostic and cannot call CUDA APIs.")
        if (
            isinstance(node, ast.Call)
            and _base_name(node.func) in {"nn.Sequential", "torch.nn.Sequential"}
            and any(not isinstance(argument, (ast.Call, ast.Starred)) for argument in node.args)
        ):
            raise CompilerError("nn.Sequential contains an expression that is not an nn.Module.")

    for candidate in module_classes:
        methods = {
            node.name: node
            for node in candidate.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "__init__" not in methods or "forward" not in methods:
            raise CompilerError(f"nn.Module {candidate.name!r} must define __init__ and forward.")
        if isinstance(methods["forward"], ast.AsyncFunctionDef):
            raise CompilerError("forward must be synchronous.")
        calls_super_init = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "__init__"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            for node in ast.walk(methods["__init__"])
        )
        if not calls_super_init:
            raise CompilerError(f"nn.Module {candidate.name!r} must call super().__init__().")

        initialized: set[str] = set()
        initialized_values: dict[str, ast.expr | None] = {}
        for node in ast.walk(methods["__init__"]):
            targets: list[tuple[ast.expr, ast.expr | None]] = []
            if isinstance(node, ast.Assign):
                targets = [(target, node.value) for target in node.targets]
            elif isinstance(node, ast.AnnAssign):
                targets = [(node.target, node.value)]
            for target, value in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    initialized.add(target.attr)
                    initialized_values[target.attr] = value
        called_attributes = {
            node.func.attr
            for node in ast.walk(methods["forward"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        }
        method_names = set(methods)
        non_module_callables = {
            name
            for name in called_attributes - method_names
            if name in initialized_values and not isinstance(initialized_values[name], ast.Call)
        }
        if non_module_callables:
            raise CompilerError(
                f"nn.Module {candidate.name!r} calls attributes not initialized as modules: "
                f"{sorted(non_module_callables)}"
            )
        referenced_attributes = {
            node.attr
            for node in ast.walk(methods["forward"])
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Load)
        }
        framework_attributes = {"training"}
        missing = referenced_attributes - initialized - framework_attributes
        if missing:
            raise CompilerError(
                f"nn.Module {candidate.name!r} uses attributes not initialized in __init__: {sorted(missing)}"
            )
    return source, module_class.name


def build_compiler_prompt(
    topology: dict[str, Any], paper_text: str, validation_feedback: str | None = None
) -> str:
    contract = generation_contract(topology)
    prompt = (
        "Generate one complete, importable Python file implementing the supplied architecture as a "
        "PyTorch nn.Module. Return the response schema only. The code field must contain only Python "
        "source: no Markdown fences, prose, tests, example execution, or top-level side effects. "
        "Define exactly one nn.Module subclass with __init__ and forward. Initialize all trainable "
        "layers and mathematical parameters in __init__; implement the directed computational graph, "
        "including residual and skip connections, in forward. Keep the module device-agnostic: never "
        "use CUDA or move tensors to a device inside the module. If functional operations are needed, "
        "use `import torch.nn.functional as F`; never use `from torch.nn.functional import F`. "
        "Preserve batch and sequence semantics, implement documented masks, and return all principal "
        "architecture outputs. When the paper is ambiguous, choose a runnable conservative default "
        "and list it in assumptions."
        + (f"\n\n{contract}" if contract else "")
        + "\n\nTOPOLOGY JSON:\n"
        + json.dumps(topology, indent=2, sort_keys=True)
        + "\n\nPAPER TEXT AND EQUATIONS:\n"
        + paper_text
    )
    if validation_feedback:
        prompt += (
            "\n\nThe previous candidate failed validation or runtime execution. Return a corrected full "
            "file that directly fixes the traceback below; do not repeat the failing implementation. "
            f"\n\nVALIDATION TRACEBACK:\n{validation_feedback}"
        )
    return prompt


class OllamaCodeCompiler:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_CODE_MODEL,
        timeout: float = 600,
        context_window: int = DEFAULT_CODE_CONTEXT,
        max_output_tokens: int = DEFAULT_CODE_OUTPUT_TOKENS,
    ) -> None:
        if context_window < 1:
            raise ValueError("context_window must be positive.")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens

    def compile(
        self,
        topology: dict[str, Any],
        paper_text: str,
        *,
        validation_feedback: str | None = None,
    ) -> CompilationResponse:
        schema = CompilationResponse.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_compiler_prompt(topology, paper_text, validation_feedback),
                }
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": self.context_window,
                "num_predict": self.max_output_tokens,
            },
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise CompilerError(f"Ollama returned HTTP {exc.code}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise CompilerError(f"Could not connect to Ollama at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise CompilerError(f"Ollama timed out after {self.timeout:g} seconds.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompilerError("Ollama returned an invalid JSON response envelope.") from exc
        try:
            return CompilationResponse.model_validate_json(envelope["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise CompilerError("Ollama response did not contain message.content.") from exc
        except ValidationError as exc:
            raise CompilerError(f"Ollama compilation response failed validation: {exc}") from exc


def compile_artifact_directory(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    compiler: OllamaCodeCompiler | None = None,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    max_attempts: int = 2,
    runtime_feedback: str | None = None,
) -> Path:
    """Compile a Phase 2 artifact directory and update its manifest."""

    root = Path(artifact_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if max_text_chars < 1:
        raise CompilerError("max_text_chars must be at least 1.")
    if max_attempts < 1:
        raise CompilerError("max_attempts must be at least 1.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        topology = json.loads((root / "topology.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CompilerError(f"Required Phase 2 artifact does not exist: {exc.filename}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CompilerError(f"Could not read compilation inputs: {exc}") from exc
    try:
        validated_topology = NetworkTopology.model_validate(topology)
    except ValidationError as exc:
        raise CompilerError(f"Phase 2 topology is invalid: {exc}") from exc
    if not validated_topology.usable:
        raise CompilerError(
            "Phase 2 topology is not usable for code generation: confidence "
            f"{validated_topology.overall_confidence:.2f} is below the required "
            f"{MIN_USABLE_CONFIDENCE:.2f}. Re-run or review architecture parsing."
        )
    topology = validated_topology.model_dump(mode="json")

    artifacts = manifest.get("artifacts", {})
    text_name = artifacts.get("nougat_text") or artifacts.get("pymupdf_text")
    if not isinstance(text_name, str):
        raise CompilerError("Manifest does not reference extracted paper text.")
    text_path = (root / text_name).resolve()
    try:
        text_path.relative_to(root)
    except ValueError as exc:
        raise CompilerError("Manifest text path escapes the artifact directory.") from exc
    paper_text = text_path.read_text(encoding="utf-8")[:max_text_chars]

    engine = compiler or OllamaCodeCompiler()
    reference = reference_implementation(topology)
    if reference is not None:
        reference_source, reference_class, reference_assumptions = reference
        source, discovered_class = validate_pytorch_source(
            reference_source, expected_class_name=reference_class
        )
        response = CompilationResponse(
            code=source,
            class_name=discovered_class,
            assumptions=reference_assumptions,
        )
        compiler_name = "torchforge-reference"
    else:
        feedback: str | None = runtime_feedback
        response: CompilationResponse | None = None
        source = ""
        discovered_class = ""
        for attempt in range(max_attempts):
            if feedback is None:
                response = engine.compile(topology, paper_text)
            else:
                response = engine.compile(topology, paper_text, validation_feedback=feedback)
            try:
                source, discovered_class = validate_pytorch_source(
                    response.code, expected_class_name=response.class_name
                )
                break
            except CompilerError as exc:
                feedback = str(exc)
                if attempt == max_attempts - 1:
                    raise CompilerError(
                        f"Generated code failed static validation after {max_attempts} attempts: {exc}"
                    ) from exc
        assert response is not None
        compiler_name = engine.model

    destination_root = Path(output_dir).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", root.name).strip("_").lower() or "model"
    destination = destination_root / f"{safe_name}.py"
    destination.write_text(source, encoding="utf-8")

    architecture_profile = identify_architecture(topology)
    manifest.setdefault("artifacts", {})["generated_code"] = str(destination)
    manifest["compilation"] = {
        "model": compiler_name,
        "class_name": discovered_class,
        "architecture_profile": architecture_profile.key if architecture_profile else None,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "assumptions": response.assumptions,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
