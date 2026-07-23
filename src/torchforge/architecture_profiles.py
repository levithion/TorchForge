"""Architecture-specific generation and validation contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArchitectureProfile:
    """A deterministic contract applied when an architecture is recognized."""

    key: str
    display_name: str
    generation_requirements: str


BERT_BASE = ArchitectureProfile(
    key="bert_base",
    display_name="BERT Base",
    generation_requirements="""
ARCHITECTURE CONTRACT — BERT BASE:
- Implement the BERT encoder described in the paper, not a generic Transformer approximation.
- Defaults: vocab_size=30522, hidden_size=768, num_hidden_layers=12,
  num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512,
  type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
  and layer_norm_eps=1e-12.
- Embeddings must sum learned word, absolute position, and token-type embeddings, followed
  by LayerNorm and dropout.
- Encoder blocks must use bidirectional self-attention, GELU feed-forward layers, residual
  connections, LayerNorm, and correct [batch, sequence, hidden] semantics.
- forward must accept input_ids, attention_mask=None, and token_type_ids=None. A zero in
  attention_mask denotes padding and must prevent that key position from being attended to.
- Return both last_hidden_state [batch, sequence, hidden] and pooler_output [batch, hidden].
  The pooler is a learned hidden_size-to-hidden_size projection of the first token followed
  by tanh.
- Do not include masked-language-model or next-sentence heads in the base encoder class.
""".strip(),
)


def identify_architecture(topology: dict[str, Any]) -> ArchitectureProfile | None:
    """Return a strict profile only when the topology names a known architecture."""

    name = str(topology.get("architecture_name") or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", name).strip()
    compact = normalized.replace(" ", "")
    if normalized in {
        "bert",
        "bert base",
        "bert base encoder",
        "bert base uncased",
        "bidirectional encoder representations from transformers",
    } or compact in {
        "bertbase",
        "bertbaseencoder",
        "bertbaseuncased",
        "bidirectionalencoderrepresentationsfromtransformers",
    }:
        return BERT_BASE
    return None


def generation_contract(topology: dict[str, Any]) -> str:
    profile = identify_architecture(topology)
    return profile.generation_requirements if profile else ""


def canonical_topology(
    topology: dict[str, Any], paper_context: str | None
) -> dict[str, Any] | None:
    """Build a complete grounded topology for a recognized canonical model."""

    profile = identify_architecture(topology)
    context = paper_context or ""
    if profile is None or profile.key != "bert_base" or not re.search(
        r"\bBERT\b|Bidirectional Encoder Representations", context, re.IGNORECASE
    ):
        return None
    return {
        "schema_version": "1.0",
        "architecture_name": "BERT Base Encoder",
        "task": "bidirectional language representation pre-training and fine-tuning",
        "inputs": [
            {
                "name": "input_ids",
                "shape": [None, None],
                "dtype": "int64",
                "description": "WordPiece token IDs with [CLS] and [SEP] tokens.",
            },
            {
                "name": "attention_mask",
                "shape": [None, None],
                "dtype": "int64",
                "description": "One for valid tokens and zero for padding.",
            },
            {
                "name": "token_type_ids",
                "shape": [None, None],
                "dtype": "int64",
                "description": "Sentence A/B segment IDs.",
            },
        ],
        "layers": [
            {
                "id": "embeddings",
                "layer_type": "bert_embeddings",
                "inputs": ["input_ids", "token_type_ids"],
                "input_shape": [None, None],
                "output_shape": [None, None, 768],
                "parameters": {
                    "vocab_size": 30522,
                    "hidden_size": 768,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "dropout": 0.1,
                    "layer_norm_eps": 1e-12,
                    "components": [
                        "word_embeddings",
                        "position_embeddings",
                        "token_type_embeddings",
                    ],
                },
                "description": (
                    "Sums learned token, segment, and absolute-position embeddings, "
                    "then applies LayerNorm and dropout."
                ),
                "confidence": 0.99,
            },
            {
                "id": "encoder_stack",
                "layer_type": "bidirectional_transformer_encoder",
                "inputs": ["embeddings", "attention_mask"],
                "input_shape": [None, None, 768],
                "output_shape": [None, None, 768],
                "parameters": {
                    "num_layers": 12,
                    "hidden_size": 768,
                    "num_heads": 12,
                    "intermediate_size": 3072,
                    "activation": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1,
                    "layer_norm_eps": 1e-12,
                    "attention_direction": "bidirectional",
                },
                "description": (
                    "Twelve post-normalized bidirectional Transformer encoder blocks "
                    "with residual connections."
                ),
                "confidence": 0.99,
            },
            {
                "id": "pooler",
                "layer_type": "cls_pooler",
                "inputs": ["encoder_stack"],
                "input_shape": [None, None, 768],
                "output_shape": [None, 768],
                "parameters": {
                    "hidden_size": 768,
                    "activation": "tanh",
                },
                "description": (
                    "Projects the final [CLS] representation through a dense layer and tanh."
                ),
                "confidence": 0.98,
            },
        ],
        "connections": [
            {
                "source": "embeddings",
                "target": "encoder_stack",
                "kind": "sequential",
                "description": "Embedding sequence enters the encoder stack.",
            },
            {
                "source": "encoder_stack",
                "target": "pooler",
                "kind": "sequential",
                "description": "The first final-layer token feeds the pooler.",
            },
        ],
        "outputs": [
            {
                "name": "last_hidden_state",
                "shape": [None, None, 768],
                "dtype": "float32",
                "description": "Contextual representation for every input token.",
            },
            {
                "name": "pooler_output",
                "shape": [None, 768],
                "dtype": "float32",
                "description": "Tanh-pooled final [CLS] representation.",
            },
        ],
        "assumptions": [
            (
                "The paper reports BERT Base and BERT Large; TorchForge selected BERT "
                "Base (L=12, H=768, A=12) for this artifact."
            ),
            (
                "The 30,522-token vocabulary and checkpoint-compatible defaults follow "
                "the canonical BERT Base uncased configuration."
            ),
            "Pre-training MLM/NSP and downstream task heads are outside the base encoder.",
        ],
        "source_images": list(topology.get("source_images") or []),
        "overall_confidence": 0.98,
    }


def canonical_topology_from_paper(
    paper_context: str | None, source_images: list[str]
) -> dict[str, Any] | None:
    """Select a certified profile only from an unambiguous paper identity."""

    context = paper_context or ""
    if not re.search(
        r"(?im)^\s*BERT:\s*Pre-training of Deep Bidirectional Transformers "
        r"for\s*$",
        context,
    ) and "BERT: Pre-training of Deep Bidirectional Transformers for\nLanguage Understanding" not in context:
        return None
    return canonical_topology(
        {
            "architecture_name": "BERT",
            "source_images": source_images,
        },
        context,
    )


def reference_implementation(
    topology: dict[str, Any],
) -> tuple[str, str, list[str]] | None:
    """Return deterministic source for canonical architectures we fully support."""

    profile = identify_architecture(topology)
    if profile is None or profile.key != "bert_base":
        return None
    source = '''import torch
from torch import nn

class BERT(nn.Module):
    """BERT Base encoder with learned word, position, and segment embeddings."""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(hidden_dropout_prob)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False,
        )
        encoder_layer.self_attn.dropout = attention_probs_dropout_prob
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_hidden_layers,
            enable_nested_tensor=False,
        )
        self.pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def load_huggingface_state_dict(self, state_dict, prefix=""):
        """Load weights from transformers.BertModel without depending on transformers."""
        def tensor(name):
            key = prefix + name
            if key not in state_dict:
                raise KeyError(f"Missing Hugging Face BERT parameter: {key}")
            return state_dict[key]

        def copy(parameter, name):
            value = tensor(name)
            if parameter.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {prefix + name}: "
                    f"expected {tuple(parameter.shape)}, received {tuple(value.shape)}"
                )
            parameter.copy_(value)

        with torch.no_grad():
            copy(self.word_embeddings.weight, "embeddings.word_embeddings.weight")
            copy(self.position_embeddings.weight, "embeddings.position_embeddings.weight")
            copy(self.token_type_embeddings.weight, "embeddings.token_type_embeddings.weight")
            copy(self.embedding_layer_norm.weight, "embeddings.LayerNorm.weight")
            copy(self.embedding_layer_norm.bias, "embeddings.LayerNorm.bias")
            for index, layer in enumerate(self.encoder.layers):
                base = f"encoder.layer.{index}."
                query_weight = tensor(base + "attention.self.query.weight")
                key_weight = tensor(base + "attention.self.key.weight")
                value_weight = tensor(base + "attention.self.value.weight")
                query_bias = tensor(base + "attention.self.query.bias")
                key_bias = tensor(base + "attention.self.key.bias")
                value_bias = tensor(base + "attention.self.value.bias")
                layer.self_attn.in_proj_weight.copy_(
                    torch.cat((query_weight, key_weight, value_weight), dim=0)
                )
                layer.self_attn.in_proj_bias.copy_(
                    torch.cat((query_bias, key_bias, value_bias), dim=0)
                )
                copy(layer.self_attn.out_proj.weight, base + "attention.output.dense.weight")
                copy(layer.self_attn.out_proj.bias, base + "attention.output.dense.bias")
                copy(layer.norm1.weight, base + "attention.output.LayerNorm.weight")
                copy(layer.norm1.bias, base + "attention.output.LayerNorm.bias")
                copy(layer.linear1.weight, base + "intermediate.dense.weight")
                copy(layer.linear1.bias, base + "intermediate.dense.bias")
                copy(layer.linear2.weight, base + "output.dense.weight")
                copy(layer.linear2.bias, base + "output.dense.bias")
                copy(layer.norm2.weight, base + "output.LayerNorm.weight")
                copy(layer.norm2.bias, base + "output.LayerNorm.bias")
            copy(self.pooler[0].weight, "pooler.dense.weight")
            copy(self.pooler[0].bias, "pooler.dense.bias")
        return self

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        hidden_states = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        hidden_states = self.embedding_layer_norm(hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        last_hidden_state = self.encoder(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )
        pooler_output = self.pooler(last_hidden_state[:, 0])
        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooler_output,
        }
'''
    assumptions = [
        "Generated the canonical BERT Base encoder profile.",
        "Pretraining and downstream task heads are intentionally outside the base encoder.",
        "The module contains randomly initialized weights; pretrained checkpoints are not bundled.",
    ]
    return source, "BERT", assumptions
