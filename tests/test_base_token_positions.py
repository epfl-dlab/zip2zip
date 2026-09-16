import torch
from transformers import LlamaConfig, LlamaForCausalLM

from zip2zip.codebook import CodebookManager
from zip2zip.config import CompressionConfig, Zip2ZipConfig
from zip2zip.model import Zip2ZipModel
from zip2zip.nn.encoders.config import ResLatentAttnConfig


def _manager(embedding_dim=8):
    return CodebookManager(
        initial_vocab_size=10,
        max_codebook_size=20,
        max_subtokens=4,
        embedding_dim=embedding_dim,
        pad_token_id=0,
        disabled_ids=[0],
    )


def test_base_positions_follow_hypertoken_spans_across_cached_steps():
    manager = _manager()

    # Reading 1,2,1,2 installs entries 10=[1,2], 11=[2,1].
    assert manager.prepare_input_ids(torch.tensor([[1, 2, 1, 2]])).tolist() == [
        [0, 1, 2, 3]
    ]
    # Entry 10 spans two base tokens, so it lands at position 5 after the
    # four-token prefix. Processing it also installs 12=[1,2,1] (span three).
    assert manager.prepare_input_ids(torch.tensor([[10]])).tolist() == [[5]]
    assert manager.prepare_input_ids(torch.tensor([[12]])).tolist() == [[8]]
    assert manager.prepare_input_ids(torch.tensor([[3]])).tolist() == [[9]]


def test_base_positions_ignore_left_padding():
    manager = _manager()
    ids = torch.tensor([[0, 0, 1, 2], [0, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])

    positions = manager.prepare_input_ids(ids, attention_mask=mask)

    assert positions.tolist() == [[0, 0, 0, 1], [0, 0, 1, 2]]
    assert manager.base_position_offset.tolist() == [[2], [3]]


def test_reset_clears_position_and_codebook_state():
    manager = _manager()
    manager.prepare_input_ids(torch.tensor([[1, 2, 1, 2]]))
    manager.reset()

    assert manager.prepare_input_ids(torch.tensor([[7]])).tolist() == [[0]]


def test_generate_injects_base_positions_into_transformers_decoder(monkeypatch):
    base = LlamaForCausalLM(
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
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.zero_()
    config = Zip2ZipConfig(
        format_version=2,
        base_model_name_or_path="unused",
        position_mode="base_token_end",
        encoder_type="res_latent_attn",
        encoder=ResLatentAttnConfig(
            hidden_size=16,
            model_hidden_size=None,
            num_hidden_layers=1,
            intermediate_size=32,
            num_heads=4,
            causal=False,
            residual=True,
            tie_encoders=False,
            position_encoding=None,
        ),
        compression=CompressionConfig(
            initial_vocab_size=10,
            max_codebook_size=20,
            max_subtokens=4,
            disabled_ids=[0],
        ),
    )
    monkeypatch.setattr(
        CodebookManager,
        "from_config",
        classmethod(lambda cls, config: _manager(embedding_dim=16)),
    )
    model = Zip2ZipModel(config, base_model=base).eval()
    seen_positions = []

    def capture_positions(module, args, kwargs):
        seen_positions.append(kwargs["position_ids"].detach().clone())

    handle = base.register_forward_pre_hook(capture_positions, with_kwargs=True)
    try:
        model.generate(
            input_ids=torch.tensor([[1, 2, 1, 2, 10]]),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            max_new_tokens=2,
            do_sample=False,
        )
    finally:
        handle.remove()

    assert seen_positions[0].tolist() == [[0, 1, 2, 3, 5]]
    assert seen_positions[1].tolist() == [[6]]
