from __future__ import annotations

import pytest
from pydantic import ValidationError

from torchforge.topology import NetworkTopology


def valid_topology_payload() -> dict:
    return {
        "schema_version": "1.0",
        "architecture_name": "Tiny Transformer",
        "task": "language modeling",
        "inputs": [{"name": "token_ids", "shape": [None, None], "dtype": "int64"}],
        "layers": [
            {
                "id": "token_embedding",
                "layer_type": "embedding",
                "inputs": [],
                "parameters": {"embedding_dim": 128},
                "confidence": 0.95,
            },
            {
                "id": "self_attention",
                "layer_type": "multi_head_attention",
                "inputs": ["token_embedding"],
                "parameters": {"num_heads": 4},
                "confidence": 0.9,
            },
        ],
        "connections": [
            {"source": "token_embedding", "target": "self_attention", "kind": "sequential"}
        ],
        "outputs": [{"name": "hidden_states", "shape": [None, None, 128]}],
        "assumptions": [],
        "source_images": [],
        "overall_confidence": 0.9,
    }


def test_valid_topology_schema() -> None:
    topology = NetworkTopology.model_validate(valid_topology_payload())
    assert topology.layers[1].inputs == ["token_embedding"]


def test_topology_rejects_duplicate_layer_ids() -> None:
    payload = valid_topology_payload()
    payload["layers"][1]["id"] = "token_embedding"
    with pytest.raises(ValidationError, match="layer IDs must be unique"):
        NetworkTopology.model_validate(payload)


def test_topology_rejects_unknown_connection_endpoint() -> None:
    payload = valid_topology_payload()
    payload["connections"][0]["target"] = "missing"
    with pytest.raises(ValidationError, match="connection endpoints"):
        NetworkTopology.model_validate(payload)


def test_first_layer_can_reference_declared_model_input() -> None:
    payload = valid_topology_payload()
    payload["layers"][0]["inputs"] = ["token_ids"]
    topology = NetworkTopology.model_validate(payload)
    assert topology.layers[0].inputs == ["token_ids"]


def test_undeclared_layer_input_is_normalized_to_model_input() -> None:
    payload = valid_topology_payload()
    payload["layers"][0]["inputs"] = ["input_tensor"]
    topology = NetworkTopology.model_validate(payload)
    assert topology.inputs[-1].name == "input_tensor"
    assert "inferred" in topology.assumptions[-1]


def test_topology_forbids_unrecognized_fields() -> None:
    payload = valid_topology_payload()
    payload["invented"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NetworkTopology.model_validate(payload)


def test_topology_rejects_pathological_layer_expansion() -> None:
    payload = valid_topology_payload()
    payload["layers"] = [payload["layers"][0] | {"id": f"layer_{index}"} for index in range(65)]
    payload["connections"] = []
    with pytest.raises(ValidationError, match="at most 64"):
        NetworkTopology.model_validate(payload)
