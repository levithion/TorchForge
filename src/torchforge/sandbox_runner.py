"""Standalone in-container runner for sandboxed validation of generated modules.

This script intentionally depends on nothing but PyTorch and the standard
library so it can execute inside a minimal PyTorch container. It imports one
generated module, runs bounded forward and backward probes, and prints a
single JSON result object delimited by sentinel markers on stdout.

The host process never imports or executes the generated code when the
sandbox is active; it consumes this JSON instead.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
import traceback

BEGIN_MARKER = "-----TORCHFORGE-SANDBOX-RESULT-BEGIN-----"
END_MARKER = "-----TORCHFORGE-SANDBOX-RESULT-END-----"

WARMUP_PASSES = 2
TIMED_PASSES = 5


def _load_class(path: str, class_name: str):
    spec = importlib.util.spec_from_file_location("torchforge_sandbox_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["torchforge_sandbox_module"] = module
    spec.loader.exec_module(module)
    candidate = getattr(module, class_name, None)
    if not inspect.isclass(candidate):
        raise RuntimeError(f"Generated class {class_name!r} is missing.")
    import torch.nn as nn

    if not issubclass(candidate, nn.Module):
        raise RuntimeError(f"Generated class {class_name!r} is not an nn.Module subclass.")
    return candidate


def _numeric_parameters(topology: dict) -> dict:
    values: dict = {}
    for layer in topology.get("layers", []):
        for key, value in layer.get("parameters", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, value)
    return values


def _last_dimension(specs: list) -> int | None:
    for spec in specs:
        shape = spec.get("shape")
        if isinstance(shape, list):
            for dimension in reversed(shape):
                if isinstance(dimension, int) and dimension > 0:
                    return dimension
    return None


def _infer_constructor_kwargs(model_class, topology: dict) -> dict:
    parameters = _numeric_parameters(topology)
    hidden_size = int(
        parameters.get("hidden_size")
        or parameters.get("d_model")
        or parameters.get("embedding_dim")
        or _last_dimension(topology.get("inputs", []))
        or 64
    )
    num_heads = int(parameters.get("num_heads") or parameters.get("heads") or 8)
    while num_heads > 1 and hidden_size % num_heads:
        num_heads -= 1
    input_size = int(_last_dimension(topology.get("inputs", [])) or hidden_size)
    output_size = int(_last_dimension(topology.get("outputs", [])) or hidden_size)
    common = {
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
    resolved: dict = {}
    unsupported: list = []
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
        raise RuntimeError(f"Cannot infer required constructor arguments: {sorted(unsupported)}")
    return resolved


def _dummy_inputs(model, topology: dict, constructor_kwargs: dict):
    import torch

    signature = inspect.signature(model.forward)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and parameter.default is inspect.Parameter.empty
    ]
    if not required:
        raise RuntimeError("forward does not accept a required model input.")
    input_size = int(
        constructor_kwargs.get("input_size")
        or constructor_kwargs.get("hidden_size")
        or _last_dimension(topology.get("inputs", []))
        or 64
    )
    vocabularies = [
        module.num_embeddings
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding) and module.num_embeddings > 2
    ]
    vocab_size = int(
        constructor_kwargs.get("vocab_size") or (max(vocabularies) if vocabularies else 32_000)
    )
    topology_inputs = topology.get("inputs", [])
    tensors = []
    for index, parameter in enumerate(required):
        spec = topology_inputs[index] if index < len(topology_inputs) else {}
        dtype = str(spec.get("dtype") or "").lower()
        name = parameter.name.lower()
        if "mask" in name:
            tensor = torch.zeros((1, 16), dtype=torch.bool)
        elif "token" in name or "ids" in name or "int" in dtype or "long" in dtype:
            tensor = torch.randint(0, min(vocab_size, 1000), (1, 16), dtype=torch.long)
        else:
            tensor = torch.randn((1, 16, input_size), dtype=torch.float32)
        tensors.append(tensor)
    return tensors


def _shapes(value) -> list:
    import torch

    if isinstance(value, torch.Tensor):
        return [list(value.shape)]
    if isinstance(value, (list, tuple)):
        shapes = []
        for item in value:
            shapes.extend(_shapes(item))
        return shapes
    if isinstance(value, dict):
        shapes = []
        for item in value.values():
            shapes.extend(_shapes(item))
        return shapes
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one generated module in isolation.")
    parser.add_argument("--code", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--topology", required=True)
    args = parser.parse_args()

    result: dict = {
        "status": "failed",
        "class_name": args.class_name,
        "device": "cpu",
        "constructor_kwargs": {},
        "input_shapes": [],
        "output_shapes": [],
        "finite_outputs": False,
        "gradient_flow": False,
        "latency_ms_mean": None,
        "throughput_samples_per_sec": None,
        "error": None,
    }

    try:
        import json as json_module

        import torch

        with open(args.topology, encoding="utf-8") as handle:
            topology = json_module.load(handle)
        model_class = _load_class(args.code, args.class_name)
        constructor_kwargs = _infer_constructor_kwargs(model_class, topology)
        model = model_class(**constructor_kwargs).cpu().eval()
        inputs = _dummy_inputs(model, topology, constructor_kwargs)

        with torch.no_grad():
            output = model(*inputs)
        output_shapes = _shapes(output)
        if not output_shapes:
            raise RuntimeError("forward did not return a tensor or tensor collection.")

        flat_tensors: list = []

        def _collect(value) -> None:
            if isinstance(value, torch.Tensor):
                flat_tensors.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _collect(item)

        _collect(output)
        finite_outputs = bool(flat_tensors) and all(
            torch.isfinite(tensor).all().item() for tensor in flat_tensors
        )

        # The shape/finite pass runs under no_grad, so recompute a fresh
        # forward with gradients enabled before probing backward flow.
        gradient_flow = False
        try:
            with torch.enable_grad():
                fresh_output = model(*inputs)
            fresh_tensors: list = []

            def _collect_fresh(value) -> None:
                if isinstance(value, torch.Tensor):
                    fresh_tensors.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        _collect_fresh(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        _collect_fresh(item)

            _collect_fresh(fresh_output)
            floating = [t for t in fresh_tensors if t.is_floating_point()]
            if floating:
                model.zero_grad(set_to_none=True)
                loss = torch.stack([tensor.float().square().mean() for tensor in floating]).sum()
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                gradient_flow = bool(trainable) and any(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in trainable
                )
        except RuntimeError:
            gradient_flow = False

        timings: list = []
        with torch.no_grad():
            for _ in range(WARMUP_PASSES):
                model(*inputs)
            for _ in range(TIMED_PASSES):
                started = time.perf_counter()
                model(*inputs)
                timings.append(time.perf_counter() - started)

        batch_size = 1
        for tensor in inputs:
            if tensor.dim() >= 1:
                batch_size = max(batch_size, int(tensor.shape[0]))
                break
        total_seconds = sum(timings)

        result.update(
            {
                "status": "completed",
                "constructor_kwargs": constructor_kwargs,
                "input_shapes": [list(tensor.shape) for tensor in inputs],
                "output_shapes": output_shapes,
                "finite_outputs": finite_outputs,
                "gradient_flow": gradient_flow,
                "latency_ms_mean": round(1000 * sum(timings) / len(timings), 3)
                if timings
                else None,
                "throughput_samples_per_sec": (
                    round(batch_size * len(timings) / total_seconds, 3)
                    if total_seconds > 0
                    else None
                ),
            }
        )
    except Exception:
        result["error"] = traceback.format_exc()

    sys.stdout.write(f"{BEGIN_MARKER}{json.dumps(result)}{END_MARKER}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
