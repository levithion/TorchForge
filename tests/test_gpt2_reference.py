from __future__ import annotations

import pytest
import torch

from torchforge.architecture_profiles import reference_implementation


def _generated_gpt2_class():
    reference = reference_implementation({"architecture_name": "GPT-2"})
    assert reference is not None
    source, class_name, _ = reference
    namespace: dict[str, object] = {}
    exec(compile(source, "<torchforge-gpt2-reference>", "exec"), namespace)
    return namespace[class_name]


def test_gpt2_reference_parameter_count_matches_canonical_config() -> None:
    gpt2_class = _generated_gpt2_class()
    model = gpt2_class()

    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == 124_439_808


def test_gpt2_attention_is_causal() -> None:
    gpt2_class = _generated_gpt2_class()
    torch.manual_seed(11)
    model = gpt2_class(
        vocab_size=101,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        embd_pdrop=0.0,
    ).eval()

    input_ids = torch.randint(0, 101, (1, 8), dtype=torch.long)
    with torch.no_grad():
        base = model(input_ids)["last_hidden_state"]
        probed_ids = input_ids.clone()
        probed_ids[0, -1] = (probed_ids[0, -1] + 37) % 101
        probed = model(probed_ids)["last_hidden_state"]

    torch.testing.assert_close(
        base[:, :-1],
        probed[:, :-1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_generated_gpt2_rejects_huggingface_shape_mismatch() -> None:
    pytest.importorskip("transformers")
    transformers = __import__("transformers", fromlist=["GPT2Config", "GPT2Model"])
    GPT2Config = transformers.GPT2Config
    GPT2Model = transformers.GPT2Model

    config = GPT2Config(
        vocab_size=101,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
    )
    oracle = GPT2Model(config).eval()
    gpt2_class = _generated_gpt2_class()
    mismatched = gpt2_class(
        vocab_size=config.vocab_size + 10,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=128,
    )
    with pytest.raises(ValueError):
        mismatched.load_huggingface_state_dict(oracle.state_dict())


def test_generated_gpt2_matches_huggingface_gpt2_model() -> None:
    transformers = __import__("transformers", fromlist=["GPT2Config", "GPT2Model"])
    GPT2Config = transformers.GPT2Config
    GPT2Model = transformers.GPT2Model
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=101,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
    )
    oracle = GPT2Model(config).eval()
    gpt2_class = _generated_gpt2_class()
    generated = gpt2_class(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_inner=config.n_inner if config.n_inner is not None else 4 * config.n_embd,
        embd_pdrop=config.embd_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
    ).eval()
    generated.load_huggingface_state_dict(oracle.state_dict())

    input_ids = torch.tensor([[2, 7, 11, 13, 0, 5], [3, 5, 17, 19, 23, 29]], dtype=torch.long)

    with torch.no_grad():
        expected = oracle(input_ids=input_ids).last_hidden_state
        actual = generated(input_ids=input_ids)["last_hidden_state"]

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
