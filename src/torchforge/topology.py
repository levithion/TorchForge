"""Strict, validated schema for vision-derived network topologies."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_USABLE_CONFIDENCE = 0.60


class TensorSpec(BaseModel):
    """A named model input or output and its paper-reported shape."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    shape: list[int | None] | None = None
    dtype: str | None = None
    description: str | None = None


class LayerSpec(BaseModel):
    """One operation or module visible in an architecture diagram."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    layer_type: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    input_shape: list[int | None] | None = None
    output_shape: list[int | None] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    confidence: float = Field(ge=0, le=1)


class ConnectionSpec(BaseModel):
    """A directed edge between two layer IDs."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: Literal[
        "sequential",
        "skip",
        "residual",
        "concat",
        "cross_attention",
        "other",
    ]
    description: str | None = None


class NetworkTopology(BaseModel):
    """Complete structured interpretation returned by the vision model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    architecture_name: str = Field(min_length=1)
    task: str | None = None
    inputs: list[TensorSpec] = Field(default_factory=list, max_length=32)
    layers: list[LayerSpec] = Field(min_length=1, max_length=64)
    connections: list[ConnectionSpec] = Field(default_factory=list, max_length=256)
    outputs: list[TensorSpec] = Field(default_factory=list, max_length=32)
    assumptions: list[str] = Field(default_factory=list)
    source_images: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "NetworkTopology":
        layer_ids = [layer.id for layer in self.layers]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("layer IDs must be unique")

        input_names = [tensor.name for tensor in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("model input names must be unique")

        known = set(layer_ids)
        valid_inputs = known | set(input_names)
        inferred_inputs: set[str] = set()
        for layer in self.layers:
            unknown_inputs = set(layer.inputs) - valid_inputs
            inferred_inputs.update(unknown_inputs)
        for name in sorted(inferred_inputs):
            self.inputs.append(
                TensorSpec(
                    name=name,
                    description="Inferred from a layer input reference omitted by the vision model.",
                )
            )
            self.assumptions.append(
                f"Model input {name!r} was inferred from an undeclared layer input reference."
            )
        for connection in self.connections:
            if connection.source not in known or connection.target not in known:
                raise ValueError(
                    "connection endpoints must reference declared layer IDs: "
                    f"{connection.source!r} -> {connection.target!r}"
                )
        return self

    @property
    def usable(self) -> bool:
        return self.overall_confidence >= MIN_USABLE_CONFIDENCE
