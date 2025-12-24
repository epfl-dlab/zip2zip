import os
import mlx.core as mx
from typing import Tuple
from argparse import ArgumentParser
from mlx_lm.models.phi3 import Model
from mlx.utils import tree_unflatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from huggingface_hub import snapshot_download
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.utils import linear_to_lora_layers

from configs import Config
from paper_prompts import PROMPTS
from fast_compression import encode
from utils import unflatten, dataclass_from_dict
from inference.mlx.codebook import CodebookManager
from inference.mlx.layers.linear import HyperLinear
from inference.mlx.layers.embedding import HyperEmbedding
from inference.mlx.layers.encoder import TransformerEncoder
from inference.mlx.tokenizer import LZWSPMStreamingDetokenizer

parser = ArgumentParser()
parser.add_argument("--original", action="store_true", default=False)
parser.add_argument("--prompt-length", type=str, choices=PROMPTS.keys(), required=True)
args = parser.parse_args()

mx.random.seed(0)

local_folder_path = snapshot_download(
    repo_id="epfl-dlab/online-zip2zip-2",
    allow_patterns=["evqn/model_7000.safetensors"],
)
adapter_path = os.path.join(local_folder_path, "evqn/model_7000.safetensors")

adapter_weights, metadata = mx.load(adapter_path, return_metadata=True)
dict_config = unflatten(metadata)["config"]
del dict_config["early_stopping_patience"]
del dict_config["lora"]["use_rslora"]
del dict_config["epochs"]
config = dataclass_from_dict(Config, dict_config)

state: Tuple[Model, TokenizerWrapper] = load(
    "mlx-community/Phi-3.5-mini-instruct-bf16", lazy=True
)
model, tokenizer = state

if not args.original:
    codebook_manager = CodebookManager(
        initial_vocab_size=config.initial_vocab_size,
        max_codebook_size=config.extra_vocab_size,
        max_subtokens=config.compression.max_subtokens,
        pad_token_id=tokenizer.pad_token_id,
        # disabled_ids=list(tokenizer.get_added_vocab().values()),
    )

    tokenizer._detokenizer = LZWSPMStreamingDetokenizer.from_tokenizer(
        tokenizer, codebook_manager, config.initial_vocab_size
    )

    # tokenizer.add_eos_token(32007)

    keys = []
    for k in adapter_weights:
        if "lora" in k:
            keys.append(k.split("_orig_mod.model.")[1].split(".lora")[0])

    linear_to_lora_layers(
        model,
        32,
        {
            "keys": keys,
            "rank": config.lora.rank,
            "scale": config.lora.alpha / config.lora.rank,
            "dropout": 0.0,
        },
    )

    lora_weights = [
        (k.split("_orig_mod.model.")[1], adapter_weights[k].T)
        for k in adapter_weights
        if "lora" in k
    ]
    model.load_weights(lora_weights, strict=False)

    fused_linears = [
        (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    encoder1 = TransformerEncoder(config)
    encoder1.set_dtype(mx.bfloat16)

    encoder2 = TransformerEncoder(config)
    encoder2.set_dtype(mx.bfloat16)

    encoder1_weights = [
        (
            "position_embeddings",
            adapter_weights[
                "hyper_embedding._orig_mod.embedding_encoder.pos_embed.weight"
            ],
        ),
    ]

    encoder2_weights = [
        (
            "position_embeddings",
            adapter_weights[
                "hyper_lm_head._orig_mod.embedding_encoder.pos_embed.weight"
            ],
        ),
    ]

    for key in adapter_weights.keys():
        if "hyper_embedding" in key and "embedding_encoder.layers" in key:
            encoder1_weights.append(
                (key.split("embedding_encoder.")[1], adapter_weights[key])
            )

        if "hyper_lm_head" in key and "embedding_encoder.layers" in key:
            encoder2_weights.append(
                (key.split("embedding_encoder.")[1], adapter_weights[key])
            )

    encoder1.load_weights(encoder1_weights)
    encoder2.load_weights(encoder2_weights)

    model.model.embed_tokens = HyperEmbedding.from_embedding(
        model.model.embed_tokens,
        encoder=encoder1,
        pad_token_id=tokenizer.pad_token_id,
        codebook_manager=codebook_manager,
    )

    model.lm_head = HyperLinear.from_linear(
        model.lm_head,
        encoder=encoder2,
        pad_token_id=tokenizer.pad_token_id,
        codebook_manager=codebook_manager,
    )

messages = [
    {
        "role": "user",
        "content": PROMPTS[args.prompt_length],
    }
]

ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

compressed_ids = None
if not args.original:
    compressed_ids, _ = encode(
        ids,
        config.initial_vocab_size,
        config.extra_vocab_size,
        config.compression.max_subtokens,
        disabled_ids=list(tokenizer.get_added_vocab().values()),
    )

mx.eval(model.parameters())

print("=" * 10)

generated_text = ""
generated_ids = []
for response in stream_generate(
    model,
    tokenizer,
    prompt=compressed_ids or ids,
    max_tokens=256,
    sampler=make_sampler(temp=0.6),
):
    print(response.text, end="", flush=True)
    generated_text += response.text
    generated_ids.append(response.token)

print()
print("=" * 10)
print("Compressed:")
print(
    f"Prompt: {response.prompt_tokens} tokens, {response.prompt_tps:.3f} tokens-per-sec"
)
print(
    f"Generation: {response.generation_tokens} tokens, {response.generation_tps:.3f} tokens-per-sec"
)

if not args.original:
    print("=" * 10)
    print("Uncompressed:")
    print(
        f"Prompt: {len(ids)} tokens, {len(ids) * (response.prompt_tps / len(compressed_ids)):.3f} tokens-per-sec"
    )
    print(
        f"Generation: {len(tokenizer.encode(generated_text))} tokens, {len(tokenizer.encode(generated_text)) * (response.generation_tps / len(generated_ids)):.3f} tokens-per-sec"
    )
print("=" * 10)
print(f"Peak memory: {response.peak_memory:.3f} GB")
