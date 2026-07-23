from __future__ import annotations

import torch

from torchforge.architecture_profiles import reference_implementation


def _generated_bert_class():
    reference = reference_implementation({"architecture_name": "BERT"})
    assert reference is not None
    source, class_name, _ = reference
    namespace: dict[str, object] = {}
    exec(compile(source, "<torchforge-bert-reference>", "exec"), namespace)
    return namespace[class_name]


def test_bert_keeps_hidden_and_attention_dropout_independent() -> None:
    bert_class = _generated_bert_class()
    model = bert_class(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.05,
    )

    for layer in model.encoder.layers:
        assert layer.self_attn.dropout == 0.05
        assert layer.dropout.p == 0.2
        assert layer.dropout1.p == 0.2
        assert layer.dropout2.p == 0.2


def test_generated_bert_matches_huggingface_bert_model() -> None:
    transformers = __import__("transformers", fromlist=["BertConfig", "BertModel"])
    BertConfig = transformers.BertConfig
    BertModel = transformers.BertModel
    torch.manual_seed(7)
    config = BertConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=32,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
    )
    oracle = BertModel(config).eval()
    bert_class = _generated_bert_class()
    generated = bert_class(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        type_vocab_size=config.type_vocab_size,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        layer_norm_eps=config.layer_norm_eps,
    ).eval()
    generated.load_huggingface_state_dict(oracle.state_dict())

    input_ids = torch.tensor(
        [[2, 7, 11, 13, 0, 0], [3, 5, 17, 19, 23, 29]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=torch.long
    )

    with torch.no_grad():
        expected = oracle(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        actual = generated(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    torch.testing.assert_close(
        actual["last_hidden_state"],
        expected.last_hidden_state,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        actual["pooler_output"],
        expected.pooler_output,
        rtol=2e-5,
        atol=2e-6,
    )
