import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import zip2zip.model as model_module
from zip2zip.codebook import CodebookManager
from zip2zip.config import CompressionConfig, Zip2ZipConfig
from zip2zip.model import Zip2ZipModel
from zip2zip.nn.encoders.config import ResLatentAttnConfig
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.utils import hub_kwargs


ENCODER = dict(
    hidden_size=16,
    model_hidden_size=None,
    num_hidden_layers=1,
    intermediate_size=32,
    num_heads=4,
    causal=False,
    residual=True,
    tie_encoders=False,
    position_encoding=None,
)
COMPRESSION = dict(
    initial_vocab_size=10, max_codebook_size=20, max_subtokens=4, disabled_ids=[0]
)


def _write_config(tmp_path, base_model_name_or_path):
    config = {
        "format_version": 2,
        "base_model_name_or_path": base_model_name_or_path,
        "position_mode": "base_token_end",
        "encoder_type": "res_latent_attn",
        "encoder": ENCODER,
        "compression": COMPRESSION,
    }
    (tmp_path / "zip2zip_config.json").write_text(json.dumps(config))


def _manager(embedding_dim=16):
    return CodebookManager(
        initial_vocab_size=10,
        max_codebook_size=20,
        max_subtokens=4,
        embedding_dim=embedding_dim,
        pad_token_id=0,
        disabled_ids=[0],
    )


def _tiny_base():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=10,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=9,
            pad_token_id=0,
        )
    ).eval()


def test_hub_kwargs_keeps_only_download_arguments():
    selected = hub_kwargs(
        {"revision": "hf", "device_map": "auto", "dtype": "auto", "token": "t"},
        subfolder=None,
    )
    assert selected == {"revision": "hf", "token": "t"}
    assert hub_kwargs({}, subfolder="sub") == {"subfolder": "sub"}


@pytest.mark.parametrize(
    ("base", "expected"),
    [(".", {"revision": "hf"}), ("some-org/external-base", {})],
)
def test_tokenizer_forwards_hub_kwargs_only_for_self_contained_repos(
    tmp_path, monkeypatch, base, expected
):
    _write_config(tmp_path, base)
    captured = {}

    def fake_init(self, config, tokenizer=None, tokenizer_kwargs=None):
        captured["base"] = config.base_model_name_or_path
        captured["tokenizer_kwargs"] = tokenizer_kwargs

    monkeypatch.setattr(Zip2ZipTokenizer, "__init__", fake_init)
    Zip2ZipTokenizer.from_pretrained(str(tmp_path), revision="hf")

    assert captured["tokenizer_kwargs"] == expected
    assert captured["base"] == (str(tmp_path) if base == "." else base)


def test_model_forwards_hub_kwargs_to_the_codebook_tokenizer(tmp_path, monkeypatch):
    _write_config(tmp_path, ".")
    captured = {}

    def fake_from_config(cls, config, tokenizer_kwargs=None):
        captured["tokenizer_kwargs"] = tokenizer_kwargs
        return _manager()

    monkeypatch.setattr(CodebookManager, "from_config", classmethod(fake_from_config))
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _tiny_base()),
    )
    monkeypatch.setattr(
        Zip2ZipModel, "load_pretrained_hyper_encoders", lambda self, *a, **k: None
    )

    model = Zip2ZipModel.from_pretrained(str(tmp_path), revision="hf", device_map="cpu")

    # revision reaches the tokenizer; device_map does not.
    assert captured["tokenizer_kwargs"] == {"revision": "hf"}
    assert model.zip2zip_config.base_model_name_or_path == str(tmp_path)

