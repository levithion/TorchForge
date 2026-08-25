"""Runtime validation and compiler-assisted repair for generated PyTorch modules."""

from __future__ import annotations

import importlib.util
import inspect
import json
import statistics
import sys
import time
import traceback
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from torchforge.architecture_profiles import ArchitectureProfile, identify_architecture
from torchforge.compiler import OllamaCodeCompiler, compile_artifact_directory
from torchforge.sandbox import docker_available, run_sandboxed_validation, sandbox_enabled


class RuntimeValidationError(RuntimeError):
    """Raised for a deterministic validation setup or execution failure."""


class ValidationStatus(StrEnum):
    COMPLETED = "completed"
    REPAIRED = "repaired"
    FAILED = "failed"


class ValidationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    code_path: str
    succeeded: bool
    error: str | None = None


class ConformanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str


class PerformanceReport(BaseModel):
    """Device-level runtime measurements for one validated module."""

    model_config = ConfigDict(extra="forbid")

    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    throughput_samples_per_sec: float
    measured_forward_passes: int
    peak_memory_bytes: int | None = None
    estimated_flops: int | None = None


_PERFORMANCE_WARMUP_PASSES = 2
_PERFORMANCE_TIMED_PASSES = 8


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_bytes(device: torch.device) -> int | None:
    try:
        if device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
        if device.type == "cpu":
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is reported in bytes on macOS and kibibytes on Linux.
            scale = 1 if sys.platform == "darwin" else 1024
            return int(peak) * scale
    except (AttributeError, RuntimeError, OSError, ImportError):
        return None
    return None


def _estimate_flops(
    model: nn.Module, inputs: list[torch.Tensor]
) -> int | None:
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as profiler:
            with torch.no_grad():
                model(*inputs)
        total = sum(event.flops for event in profiler.key_averages())
        return int(total) if total > 0 else None
    except Exception:
        return None


def _measure_performance(
    model: nn.Module, inputs: list[torch.Tensor], device: torch.device
) -> PerformanceReport:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch_size = 1
    for tensor in inputs:
        if tensor.dim() >= 1:
            batch_size = max(batch_size, int(tensor.shape[0]))
            break
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(_PERFORMANCE_WARMUP_PASSES):
            model(*inputs)
        _synchronize(device)
        for _ in range(_PERFORMANCE_TIMED_PASSES):
            started = time.perf_counter()
            model(*inputs)
            _synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    total_seconds = sum(timings) / 1000
    return PerformanceReport(
        latency_ms_mean=statistics.fmean(timings),
        latency_ms_p50=ordered[len(ordered) // 2],
        latency_ms_p95=ordered[p95_index],
        throughput_samples_per_sec=(
            batch_size * len(timings) / total_seconds if total_seconds > 0 else 0.0
        ),
        measured_forward_passes=len(timings),
        peak_memory_bytes=_peak_memory_bytes(device),
        estimated_flops=_estimate_flops(model, inputs),
    )


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ValidationStatus
    device: str
    class_name: str | None = None
    constructor_kwargs: dict[str, Any] = Field(default_factory=dict)
    input_shapes: list[list[int]] = Field(default_factory=list)
    output_shapes: list[list[int]] = Field(default_factory=list)
    architecture_profile: str | None = None
    conformance_checks: list[ConformanceCheck] = Field(default_factory=list)
    attempts: list[ValidationAttempt] = Field(default_factory=list)
    performance: PerformanceReport | None = None
    sandboxed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status is not ValidationStatus.FAILED


def _load_generated_class(code_path: Path, class_name: str) -> type[nn.Module]:
    module_name = f"torchforge_generated_{abs(hash(code_path))}"
    spec = importlib.util.spec_from_file_location(module_name, code_path)
    if spec is None or spec.loader is None:
        raise RuntimeValidationError(f"Could not create an import spec for {code_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    candidate = getattr(module, class_name, None)
    if not inspect.isclass(candidate) or not issubclass(candidate, nn.Module):
        raise RuntimeValidationError(
            f"Generated class {class_name!r} is missing or is not an nn.Module subclass."
        )
    return candidate


def _numeric_parameters(topology: dict[str, Any]) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for layer in topology.get("layers", []):
        for key, value in layer.get("parameters", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, value)
    return values


def _last_known_dimension(specs: list[dict[str, Any]]) -> int | None:
    for spec in specs:
        shape = spec.get("shape")
        if isinstance(shape, list):
            for dimension in reversed(shape):
                if isinstance(dimension, int) and dimension > 0:
                    return dimension
    return None


def infer_constructor_kwargs(
    model_class: type[nn.Module], topology: dict[str, Any]
) -> dict[str, Any]:
    """Resolve common required constructor arguments from the topology."""

    parameters = _numeric_parameters(topology)
    hidden_size = int(
        parameters.get("hidden_size")
        or parameters.get("d_model")
        or parameters.get("embedding_dim")
        or _last_known_dimension(topology.get("inputs", []))
        or 64
    )
    num_heads = int(parameters.get("num_heads") or parameters.get("heads") or 8)
    while num_heads > 1 and hidden_size % num_heads:
        num_heads -= 1
    input_size = int(_last_known_dimension(topology.get("inputs", [])) or hidden_size)
    output_size = int(_last_known_dimension(topology.get("outputs", [])) or hidden_size)
    common: dict[str, Any] = {
        "input_size": input_size,
        "input_dim": input_size,
        "in_features": input_size,
        "d_model": hidden_size,
        "model_dim": hidden_size,
        "hidden_size": hidden_size,
        "embedding_dim": hidden_size,
        "dim": hidden_size,
        "num_heads": num_heads,
        "nhead": num_heads,
        "heads": num_heads,
        "num_layers": max(1, len(topology.get("layers", []))),
        "n_layers": max(1, len(topology.get("layers", []))),
        "output_size": output_size,
        "output_dim": output_size,
        "num_classes": output_size,
        "vocab_size": int(parameters.get("vocab_size") or 32_000),
        "in_channels": 3,
        "channels": 3,
    }
    signature = inspect.signature(model_class.__init__)
    resolved: dict[str, Any] = {}
    unsupported: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is not inspect.Parameter.empty:
            continue
        if name in common:
            resolved[name] = common[name]
        else:
            unsupported.append(name)
    if unsupported:
        raise RuntimeValidationError(
            f"Cannot infer required constructor arguments: {sorted(unsupported)}"
        )
    return resolved


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeValidationError("This PyTorch build does not include MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeValidationError("MPS is not available on this Mac.")
    elif name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeValidationError("CUDA is not available on this computer.")
    elif name != "cpu":
        raise RuntimeValidationError(
            "Validation device must be 'auto', 'cuda', 'mps', or 'cpu'."
        )
    return torch.device(name)


def _dummy_inputs(
    model: nn.Module,
    topology: dict[str, Any],
    constructor_kwargs: dict[str, Any],
    device: torch.device,
) -> list[torch.Tensor]:
    signature = inspect.signature(model.forward)
    forward_parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and parameter.default is inspect.Parameter.empty
    ]
    if not forward_parameters:
        raise RuntimeValidationError("forward does not accept a required model input.")

    input_size = int(
        constructor_kwargs.get("input_size")
        or constructor_kwargs.get("input_dim")
        or constructor_kwargs.get("hidden_size")
        or constructor_kwargs.get("d_model")
        or _last_known_dimension(topology.get("inputs", []))
        or 64
    )
    embedding_vocabularies = [
        module.num_embeddings
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.num_embeddings > 2
    ]
    vocab_size = int(
        constructor_kwargs.get("vocab_size")
        or (max(embedding_vocabularies) if embedding_vocabularies else 32_000)
    )
    topology_inputs = topology.get("inputs", [])
    tensors: list[torch.Tensor] = []
    for index, parameter in enumerate(forward_parameters):
        spec = topology_inputs[index] if index < len(topology_inputs) else {}
        dtype = str(spec.get("dtype") or "").lower()
        name = parameter.name.lower()
        if "mask" in name:
            tensor = torch.zeros((1, 16), dtype=torch.bool, device=device)
        elif "token" in name or "ids" in name or "int" in dtype or "long" in dtype:
            tensor = torch.randint(0, vocab_size, (1, 16), dtype=torch.long, device=device)
        else:
            tensor = torch.randn((1, 16, input_size), dtype=torch.float32, device=device)
        tensors.append(tensor)
    return tensors


def _bert_inputs(
    device: torch.device, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (2, 8), dtype=torch.long, device=device)
    attention_mask = torch.ones((2, 8), dtype=torch.long, device=device)
    attention_mask[1, -2:] = 0
    token_type_ids = torch.zeros((2, 8), dtype=torch.long, device=device)
    token_type_ids[:, 4:] = 1
    return input_ids, attention_mask, token_type_ids


def _named_outputs(value: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if isinstance(value, dict):
        sequence = value.get("last_hidden_state")
        pooled = value.get("pooler_output")
        return (
            sequence if isinstance(sequence, torch.Tensor) else None,
            pooled if isinstance(pooled, torch.Tensor) else None,
        )
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (
            value[0] if isinstance(value[0], torch.Tensor) else None,
            value[1] if isinstance(value[1], torch.Tensor) else None,
        )
    sequence = getattr(value, "last_hidden_state", None)
    pooled = getattr(value, "pooler_output", None)
    return (
        sequence if isinstance(sequence, torch.Tensor) else None,
        pooled if isinstance(pooled, torch.Tensor) else None,
    )


def _check(name: str, condition: bool, detail: str) -> ConformanceCheck:
    return ConformanceCheck(name=name, passed=condition, detail=detail)


def _bert_conformance(
    model: nn.Module,
    constructor_kwargs: dict[str, Any],
    device: torch.device,
) -> list[ConformanceCheck]:
    """Validate BERT Base structure and observable tensor semantics."""

    modules = list(model.modules())
    embeddings = [module for module in modules if isinstance(module, nn.Embedding)]
    encoder_layers = [
        module for module in modules if isinstance(module, nn.TransformerEncoderLayer)
    ]
    layer_norms = [module for module in modules if isinstance(module, nn.LayerNorm)]
    tanh_modules = [module for module in modules if isinstance(module, nn.Tanh)]
    signature = inspect.signature(model.forward)
    parameter_names = set(signature.parameters)

    checks = [
        _check(
            "bert.forward_contract",
            {"input_ids", "attention_mask", "token_type_ids"} <= parameter_names,
            "forward accepts input_ids, attention_mask, and token_type_ids",
        ),
        _check(
            "bert.embeddings",
            len(embeddings) >= 3
            and any(module.num_embeddings >= 512 for module in embeddings)
            and any(module.num_embeddings == 2 for module in embeddings),
            "word, position (>=512), and token-type (size 2) embeddings are present",
        ),
        _check(
            "bert.embedding_normalization",
            bool(layer_norms) and any(isinstance(module, nn.Dropout) for module in modules),
            "embedding/encoder normalization and dropout modules are present",
        ),
        _check(
            "bert.layer_norm_epsilon",
            bool(layer_norms)
            and all(abs(float(module.eps) - 1e-12) < 1e-15 for module in layer_norms),
            "all BERT LayerNorm modules use epsilon 1e-12",
        ),
        _check(
            "bert.encoder_depth",
            len(encoder_layers) == 12,
            f"expected 12 Transformer encoder layers, found {len(encoder_layers)}",
        ),
        _check(
            "bert.attention_shape",
            bool(encoder_layers)
            and all(
                layer.self_attn.embed_dim == 768 and layer.self_attn.num_heads == 12
                for layer in encoder_layers
            ),
            "all encoder layers use hidden size 768 and 12 attention heads",
        ),
        _check(
            "bert.feed_forward",
            bool(encoder_layers)
            and all(layer.linear1.out_features == 3072 for layer in encoder_layers),
            "all encoder layers use intermediate size 3072",
        ),
        _check(
            "bert.activation",
            bool(encoder_layers)
            and all(
                getattr(layer.activation, "__name__", "").lower() == "gelu"
                for layer in encoder_layers
            ),
            "all encoder feed-forward blocks use GELU activation",
        ),
        _check(
            "bert.dropout_configuration",
            bool(encoder_layers)
            and all(
                layer.dropout.p == layer.dropout1.p == layer.dropout2.p
                for layer in encoder_layers
            ),
            "encoder residual and feed-forward dropout share the hidden-dropout setting",
        ),
        _check(
            "bert.batch_semantics",
            bool(encoder_layers)
            and all(bool(layer.self_attn.batch_first) for layer in encoder_layers),
            "encoder attention uses [batch, sequence, hidden] semantics",
        ),
        _check(
            "bert.pooler",
            bool(tanh_modules)
            and any(
                isinstance(module, nn.Linear)
                and module.in_features == 768
                and module.out_features == 768
                for module in modules
            ),
            "a 768-to-768 projection and tanh pooler are present",
        ),
        _check(
            "bert.parameter_count",
            sum(parameter.numel() for parameter in model.parameters()) == 109_482_240,
            "BERT Base encoder with pooler has exactly 109,482,240 parameters",
        ),
        _check(
            "bert.checkpoint_adapter",
            callable(getattr(model, "load_huggingface_state_dict", None)),
            "a Hugging Face BertModel weight adapter is available",
        ),
    ]

    if not checks[0].passed:
        return checks

    vocabulary_embeddings = [
        module.num_embeddings
        for module in embeddings
        if module.embedding_dim == 768 and module.num_embeddings > 512
    ]
    vocab_size = int(
        constructor_kwargs.get("vocab_size")
        or (max(vocabulary_embeddings) if vocabulary_embeddings else 30_522)
    )
    input_ids, attention_mask, token_type_ids = _bert_inputs(device, vocab_size)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    sequence, pooled = _named_outputs(output)
    outputs_valid = (
        sequence is not None
        and pooled is not None
        and tuple(sequence.shape) == (2, 8, 768)
        and tuple(pooled.shape) == (2, 768)
    )
    checks.append(
        _check(
            "bert.outputs",
            outputs_valid,
            "returns last_hidden_state [batch, sequence, 768] and pooler_output [batch, 768]",
        )
    )
    finite = bool(
        sequence is not None
        and pooled is not None
        and outputs_valid
        and torch.isfinite(sequence).all().item()
        and torch.isfinite(pooled).all().item()
    )
    checks.append(_check("runtime.finite_outputs", finite, "all returned values are finite"))

    if sequence is not None:
        model.zero_grad(set_to_none=True)
        loss = sequence.float().square().mean()
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients_ok = bool(trainable) and any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
        )
    else:
        gradients_ok = False
    checks.append(
        _check(
            "runtime.gradient_flow",
            gradients_ok,
            "a backward pass produces finite trainable-parameter gradients",
        )
    )

    if sequence is not None and outputs_valid:
        model.eval()
        with torch.no_grad():
            single_output = model(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1],
                token_type_ids=token_type_ids[:1],
            )
        single_sequence, _ = _named_outputs(single_output)
        independent = bool(
            single_sequence is not None
            and torch.allclose(sequence[:1], single_sequence, atol=2e-5, rtol=2e-4)
        )
    else:
        independent = False
    checks.append(
        _check(
            "runtime.batch_independence",
            independent,
            "one sample is unchanged when evaluated alone versus in a batch",
        )
    )
    return checks


def _gpt2_inputs(device: torch.device, vocab_size: int) -> torch.Tensor:
    return torch.randint(0, vocab_size, (2, 8), dtype=torch.long, device=device)


def _gpt2_conformance(
    model: nn.Module,
    constructor_kwargs: dict[str, Any],
    device: torch.device,
) -> list[ConformanceCheck]:
    """Validate GPT-2 Small structure and observable tensor semantics."""

    blocks = list(getattr(model, "h", []))
    embeddings = [module for module in model.modules() if isinstance(module, nn.Embedding)]
    layer_norms = [module for module in model.modules() if isinstance(module, nn.LayerNorm)]
    signature = inspect.signature(model.forward)
    parameter_names = set(signature.parameters)

    token_embedding = next((m for m in embeddings if m.num_embeddings > 1024), None)
    position_embedding = next((m for m in embeddings if 256 <= m.num_embeddings <= 4096), None)

    checks = [
        _check(
            "gpt2.forward_contract",
            "input_ids" in parameter_names,
            "forward accepts input_ids",
        ),
        _check(
            "gpt2.embeddings",
            token_embedding is not None
            and token_embedding.num_embeddings == 50_257
            and token_embedding.embedding_dim == 768
            and position_embedding is not None
            and position_embedding.num_embeddings == 1024,
            "token (50,257 x 768) and learned position (1024 x 768) embeddings are present",
        ),
        _check(
            "gpt2.decoder_depth",
            len(blocks) == 12,
            f"expected 12 decoder blocks, found {len(blocks)}",
        ),
        _check(
            "gpt2.attention_shape",
            bool(blocks)
            and all(
                getattr(block, "attn", None) is not None
                and getattr(block.attn, "n_head", 0) == 12
                and getattr(block.attn, "head_dim", 0) == 64
                and getattr(block.attn.c_attn, "out_features", 0) == 3 * 768
                for block in blocks
            ),
            "all decoder blocks use fused 2304-output projections with 12 heads of size 64",
        ),
        _check(
            "gpt2.feed_forward",
            bool(blocks)
            and all(
                getattr(block.mlp.c_fc, "out_features", 0) == 3072 for block in blocks
            ),
            "all decoder blocks use intermediate width 3072",
        ),
        _check(
            "gpt2.activation",
            bool(blocks)
            and all(callable(getattr(block.mlp, "_gelu_new", None)) for block in blocks),
            "all feed-forward blocks implement the GELU tanh approximation",
        ),
        _check(
            "gpt2.layer_norm_epsilon",
            bool(layer_norms)
            and all(abs(float(module.eps) - 1e-5) < 1e-9 for module in layer_norms),
            "all LayerNorm modules use epsilon 1e-5",
        ),
        _check(
            "gpt2.pre_normalization",
            bool(blocks)
            and all(hasattr(block, "ln_1") and hasattr(block, "ln_2") for block in blocks)
            and hasattr(model, "ln_f"),
            "pre-normalization LayerNorms precede attention and feed-forward sub-blocks",
        ),
        _check(
            "gpt2.parameter_count",
            sum(parameter.numel() for parameter in model.parameters()) == 124_439_808,
            "GPT-2 Small base decoder has exactly 124,439,808 parameters",
        ),
        _check(
            "gpt2.checkpoint_adapter",
            callable(getattr(model, "load_huggingface_state_dict", None)),
            "a Hugging Face GPT2Model weight adapter is available",
        ),
    ]

    if not checks[0].passed:
        return checks

    vocab_size = int(constructor_kwargs.get("vocab_size") or 50_257)
    input_ids = _gpt2_inputs(device, min(vocab_size, 1_000))
    output = model(input_ids=input_ids)
    sequence = output.get("last_hidden_state") if isinstance(output, dict) else None
    outputs_valid = isinstance(sequence, torch.Tensor) and tuple(sequence.shape) == (2, 8, 768)
    checks.append(
        _check(
            "gpt2.outputs",
            outputs_valid,
            "returns last_hidden_state [batch, sequence, 768]",
        )
    )
    finite = bool(
        sequence is not None
        and outputs_valid
        and torch.isfinite(sequence).all().item()
    )
    checks.append(_check("runtime.finite_outputs", finite, "all returned values are finite"))

    if sequence is not None:
        model.zero_grad(set_to_none=True)
        loss = sequence.float().square().mean()
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradients_ok = bool(trainable) and any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
        )
    else:
        gradients_ok = False
    checks.append(
        _check(
            "runtime.gradient_flow",
            gradients_ok,
            "a backward pass produces finite trainable-parameter gradients",
        )
    )

    if sequence is not None:
        probed = input_ids.clone()
        # 503 never maps a value back to itself modulo 1000, so the probe
        # token is guaranteed to differ from the original.
        probed[:, -1] = (probed[:, -1] + 503) % 1000
        with torch.no_grad():
            probe_output = model(input_ids=probed)
        probe_sequence = (
            probe_output.get("last_hidden_state")
            if isinstance(probe_output, dict)
            else None
        )
        causal = bool(
            isinstance(probe_sequence, torch.Tensor)
            and torch.allclose(sequence[:, :-1], probe_sequence[:, :-1], atol=1e-5, rtol=1e-4)
        )
    else:
        causal = False
    checks.append(
        _check(
            "runtime.causal_masking",
            causal,
            "changing the final token leaves every earlier position unchanged",
        )
    )

    if sequence is not None:
        model.eval()
        with torch.no_grad():
            single_output = model(input_ids=input_ids[:1])
        single_sequence = (
            single_output.get("last_hidden_state")
            if isinstance(single_output, dict)
            else None
        )
        independent = bool(
            isinstance(single_sequence, torch.Tensor)
            and torch.allclose(sequence[:1], single_sequence, atol=2e-5, rtol=2e-4)
        )
    else:
        independent = False
    checks.append(
        _check(
            "runtime.batch_independence",
            independent,
            "one sample is unchanged when evaluated alone versus in a batch",
        )
    )
    return checks


def evaluate_architecture_conformance(
    model: nn.Module,
    profile: ArchitectureProfile | None,
    constructor_kwargs: dict[str, Any],
    device: torch.device,
) -> list[ConformanceCheck]:
    if profile is None:
        return [
            _check(
                "runtime.forward",
                True,
                "forward completed; no strict profile exists for this architecture",
            )
        ]
    if profile.key == "bert_base":
        return _bert_conformance(model, constructor_kwargs, device)
    if profile.key == "gpt2_small":
        return _gpt2_conformance(model, constructor_kwargs, device)
    raise RuntimeValidationError(f"Unsupported architecture profile: {profile.key}")


def _tensor_shapes(value: Any) -> list[list[int]]:
    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, (list, tuple)):
        shapes: list[list[int]] = []
        for item in value:
            shapes.extend(_tensor_shapes(item))
        return shapes
    if isinstance(value, dict):
        shapes = []
        for item in value.values():
            shapes.extend(_tensor_shapes(item))
        return shapes
    return []


def run_forward_validation(
    code_path: Path,
    class_name: str,
    topology: dict[str, Any],
    *,
    device_name: str = "auto",
) -> tuple[dict[str, Any], list[list[int]], list[list[int]], PerformanceReport]:
    """Import, instantiate, execute, and benchmark one generated module."""

    device = _select_device(device_name)
    model_class = _load_generated_class(code_path, class_name)
    constructor_kwargs = infer_constructor_kwargs(model_class, topology)
    model = model_class(**constructor_kwargs).to(device)
    model.eval()
    inputs = _dummy_inputs(model, topology, constructor_kwargs, device)
    with torch.no_grad():
        output = model(*inputs)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    output_shapes = _tensor_shapes(output)
    if not output_shapes:
        raise RuntimeValidationError("forward did not return a tensor or tensor collection.")
    performance = _measure_performance(model, inputs, device)
    return (
        constructor_kwargs,
        [list(tensor.shape) for tensor in inputs],
        output_shapes,
        performance,
    )


def _run_complete_validation(
    code_path: Path,
    class_name: str,
    topology: dict[str, Any],
    *,
    device_name: str,
) -> tuple[
    dict[str, Any],
    list[list[int]],
    list[list[int]],
    list[ConformanceCheck],
    PerformanceReport,
]:
    kwargs, input_shapes, output_shapes, performance = run_forward_validation(
        code_path, class_name, topology, device_name=device_name
    )
    device = _select_device(device_name)
    model_class = _load_generated_class(code_path, class_name)
    model = model_class(**kwargs).to(device)
    model.eval()
    profile = identify_architecture(topology)
    checks = evaluate_architecture_conformance(model, profile, kwargs, device)
    failures = [check.detail for check in checks if not check.passed]
    if failures:
        raise RuntimeValidationError(
            "Architecture conformance failed:\n- " + "\n- ".join(failures)
        )
    return kwargs, input_shapes, output_shapes, checks, performance


def _run_sandboxed_attempt(
    code_path: Path,
    class_name: str,
    topology_path: Path,
) -> tuple[dict[str, Any], list[list[int]], list[list[int]], list[ConformanceCheck], PerformanceReport]:
    """Validate one generated module inside an isolated container."""

    payload = run_sandboxed_validation(code_path, class_name, topology_path)
    if payload.get("status") != "completed":
        error = payload.get("error") or "The sandboxed runner reported an unspecified failure."
        raise RuntimeValidationError(error)
    checks = [
        _check(
            "runtime.forward",
            True,
            "forward completed inside a network-isolated container",
        ),
        _check(
            "runtime.finite_outputs",
            bool(payload.get("finite_outputs")),
            "all returned values are finite",
        ),
        _check(
            "runtime.gradient_flow",
            bool(payload.get("gradient_flow")),
            "a backward pass produces finite trainable-parameter gradients",
        ),
    ]
    latency = payload.get("latency_ms_mean")
    performance = PerformanceReport(
        latency_ms_mean=float(latency or 0.0),
        latency_ms_p50=float(latency or 0.0),
        latency_ms_p95=float(latency or 0.0),
        throughput_samples_per_sec=float(payload.get("throughput_samples_per_sec") or 0.0),
        measured_forward_passes=5,
    )
    constructor_kwargs = payload.get("constructor_kwargs")
    return (
        constructor_kwargs if isinstance(constructor_kwargs, dict) else {},
        payload.get("input_shapes") or [],
        payload.get("output_shapes") or [],
        checks,
        performance,
    )


def _write_report(root: Path, manifest: dict[str, Any], report: ValidationReport) -> None:
    report_path = root / "validation.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    manifest.setdefault("artifacts", {})["validation_report"] = report_path.name
    manifest["validation"] = {
        "status": report.status.value,
        "device": report.device,
        "attempt_count": len(report.attempts),
        "output_shapes": report.output_shapes,
        "architecture_profile": report.architecture_profile,
        "conformance_passed": all(check.passed for check in report.conformance_checks),
        "performance": (
            {
                "latency_ms_p50": round(report.performance.latency_ms_p50, 3),
                "throughput_samples_per_sec": round(
                    report.performance.throughput_samples_per_sec, 3
                ),
                "estimated_flops": report.performance.estimated_flops,
                "peak_memory_bytes": report.performance.peak_memory_bytes,
            }
            if report.performance is not None
            else None
        ),
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_artifact_directory(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    max_repairs: int = 2,
    compiler: OllamaCodeCompiler | None = None,
) -> ValidationReport:
    """Validate generated code and repair runtime failures through the compiler."""

    root = Path(artifact_dir).expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        topology = json.loads((root / "topology.json").read_text(encoding="utf-8"))
        class_name = manifest["compilation"]["class_name"]
        code_path = Path(manifest["artifacts"]["generated_code"]).resolve()
    except (FileNotFoundError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeValidationError(f"Could not read Phase 3 validation inputs: {exc}") from exc
    if max_repairs < 0:
        raise RuntimeValidationError("max_repairs cannot be negative.")
    selected_device = _select_device(device_name)
    selected_device_name = selected_device.type
    architecture_profile = identify_architecture(topology)
    architecture_profile_key = architecture_profile.key if architecture_profile else None
    sandbox_requested = sandbox_enabled()
    sandbox_active = sandbox_requested and docker_available()
    if sandbox_requested and not sandbox_active:
        manifest.setdefault("warnings", []).append(
            "TORCHFORGE_SANDBOX=docker was requested but no Docker daemon is "
            "reachable; validation ran in-process."
        )

    attempts: list[ValidationAttempt] = []
    engine = compiler or OllamaCodeCompiler()
    device_name_for_report = selected_device_name
    for attempt_number in range(1, max_repairs + 2):
        try:
            if sandbox_active:
                kwargs, input_shapes, output_shapes, checks, performance = (
                    _run_sandboxed_attempt(
                        code_path,
                        class_name,
                        root / "topology.json",
                    )
                )
                device_name_for_report = "cpu"
            else:
                kwargs, input_shapes, output_shapes, checks, performance = (
                    _run_complete_validation(
                        code_path, class_name, topology, device_name=selected_device_name
                    )
                )
                device_name_for_report = selected_device_name
            attempts.append(
                ValidationAttempt(
                    attempt=attempt_number, code_path=str(code_path), succeeded=True
                )
            )
            report = ValidationReport(
                status=(
                    ValidationStatus.COMPLETED
                    if attempt_number == 1
                    else ValidationStatus.REPAIRED
                ),
                device=device_name_for_report,
                class_name=class_name,
                constructor_kwargs=kwargs,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                architecture_profile=architecture_profile_key,
                conformance_checks=checks,
                attempts=attempts,
                performance=performance,
                sandboxed=sandbox_active,
            )
            _write_report(root, manifest, report)
            return report
        except Exception:
            error = traceback.format_exc()
            attempts.append(
                ValidationAttempt(
                    attempt=attempt_number,
                    code_path=str(code_path),
                    succeeded=False,
                    error=error,
                )
            )
            if attempt_number > max_repairs:
                report = ValidationReport(
                    status=ValidationStatus.FAILED,
                    device=device_name_for_report,
                    class_name=class_name,
                    architecture_profile=architecture_profile_key,
                    attempts=attempts,
                    sandboxed=sandbox_active,
                )
                _write_report(root, manifest, report)
                return report
            try:
                code_path = compile_artifact_directory(
                    root,
                    output_dir,
                    compiler=engine,
                    runtime_feedback=error[-12_000:],
                )
            except Exception:
                repair_error = traceback.format_exc()
                attempts[-1].error = (
                    (attempts[-1].error or "")
                    + "\n\nREPAIR COMPILATION FAILED:\n"
                    + repair_error
                )
                if attempt_number >= max_repairs:
                    report = ValidationReport(
                        status=ValidationStatus.FAILED,
                        device=device_name_for_report,
                        class_name=class_name,
                        architecture_profile=architecture_profile_key,
                        attempts=attempts,
                        sandboxed=sandbox_active,
                    )
                    _write_report(root, manifest, report)
                    return report
                continue
            refreshed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest.update(refreshed)
            class_name = manifest["compilation"]["class_name"]

    raise AssertionError("validation loop ended unexpectedly")
