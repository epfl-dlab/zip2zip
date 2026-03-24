import types

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")
pytest.importorskip("peft")
pytest.importorskip("accelerate")
pytest.importorskip("zip2zip_compression")

from zip2zip.config import CompressionConfig
from zip2zip.nn.encoders.attention import SelfAttention
from zip2zip.nn.encoders.config import TransformerEncoderConfig
from zip2zip.nn.encoders.transformer import TransformerEncoder
import zip2zip.nn.encoders.attention as attention_module


class FakeMask:
    def __init__(self) -> None:
        self.to_calls = []
        self.unsqueeze_calls = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        self.to_calls.append(device)
        return self

    def unsqueeze(self, dim: int):
        self.unsqueeze_calls.append(dim)
        return self


def test_self_attention_moves_mask_to_query_device(monkeypatch):
    attention = SelfAttention(hidden_size=4, num_heads=2)
    input_embeddings = torch.randn(2, 3, 4)
    attn_mask = FakeMask()
    captured = {}

    def fake_scaled_dot_product_attention(query, key, value, attn_mask=None, **kwargs):
        captured["query_device"] = query.device
        captured["attn_mask"] = attn_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(
        attention_module.F,
        "scaled_dot_product_attention",
        fake_scaled_dot_product_attention,
    )

    output = attention(input_embeddings, attn_mask)

    assert output.shape == input_embeddings.shape
    assert attn_mask.to_calls == [input_embeddings.device]
    assert attn_mask.unsqueeze_calls == [1]
    assert captured["attn_mask"] is attn_mask
    assert captured["query_device"] == input_embeddings.device


def test_transformer_encoder_moves_mask_to_hidden_state_device(monkeypatch):
    encoder = TransformerEncoder(
        TransformerEncoderConfig(
            hidden_size=4,
            tie_encoders=False,
            position_encoding=None,
            num_hidden_layers=1,
            intermediate_size=8,
            num_heads=2,
        ),
        CompressionConfig(
            initial_vocab_size=8,
            max_codebook_size=4,
            max_subtokens=3,
            disabled_ids=[0],
        ),
    )
    codebook = torch.tensor([[[1, 2, 0], [3, 0, 0]]], dtype=torch.long)
    embeddings = torch.empty(8, 4, device="meta")
    captured = {}

    def fake_forward(self, x, mask):
        captured["x_device"] = x.device
        captured["mask_device"] = mask.device
        return x

    monkeypatch.setattr(
        encoder.layers[0],
        "forward",
        types.MethodType(fake_forward, encoder.layers[0]),
    )

    output = encoder(codebook, embeddings, pad_token_id=0)

    assert captured["x_device"].type == "meta"
    assert captured["mask_device"] == captured["x_device"]
    assert output.shape == (1, 2, 4)
