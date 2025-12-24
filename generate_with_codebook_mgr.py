import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional


from model import OnlineZZModel

from fast_compression import lzw_compress
from zip2zip_compression import CodebookManager as FastCodebookManager
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.config import Zip2ZipConfig, CompressionConfig
import time
import os, sys
from argparse import ArgumentParser
from fast_compression import lzw_compress

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visual import legacy_contrast_colorprint_tokens

from interface import load_model
from model import OnlineZZModel
from utils import (
    setup_seed,
    describe_lzw,
    decompress,
)


@dataclass
class GenerateConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    ddp_rank: int = 0
    use_kv_cache: bool = True
    extra_vocab_size: int = None
    until: Optional[List[str]] = None
    compress_during_generation: bool = False
    do_sample: bool = False


def until_has_been_reached(generate_config: GenerateConfig, text: str) -> bool:
    if generate_config.until is None:
        return False

    return any(u in text for u in generate_config.until)


@torch.no_grad()
def z2z_generate(
    prompt: str,
    model: OnlineZZModel,
    generate_config: GenerateConfig,
    incremental_codebook_update: bool = False,
) -> Tuple[str, List[int], List[int], Dict[str, int], float]:
    model.eval()
    config = model.config
    generate_config.extra_vocab_size = (
        generate_config.extra_vocab_size
        if generate_config.extra_vocab_size is not None
        else config.extra_vocab_size
    )
    sample_rng = torch.Generator(device=model.device)
    sample_rng.manual_seed(42 + generate_config.ddp_rank)

    # disabled_ids = [32001, 32007, 32010]  # <|user|> <|assistant> <|end|>
    disabled_ids = list(model.tokenizer.get_added_vocab().values())

    codebook_manager = FastCodebookManager(
        config.initial_vocab_size,
        generate_config.extra_vocab_size,
        config.compression.max_subtokens,
        model.tokenizer.pad_token_id,
        disabled_ids=disabled_ids,
    )
    _compression_config = CompressionConfig(
        initial_vocab_size=config.initial_vocab_size,
        max_codebook_size=generate_config.extra_vocab_size,
        max_subtokens=config.compression.max_subtokens,
        disabled_ids=disabled_ids,
    )
    zip2zip_config = Zip2ZipConfig(
        compression=_compression_config,
        encoder_type=None,
        encoder=None,
    )

    tokenizer = Zip2ZipTokenizer(model.tokenizer, zip2zip_config)

    prompt_lzw_ids = tokenizer.encode(prompt, add_special_tokens=False)

    token_ids = prompt_lzw_ids
    num_input_lzw_tokens = len(prompt_lzw_ids)

    new_token_ids = token_ids

    metadata = {}

    codebook_list = []

    with tqdm(total=generate_config.max_new_tokens, desc="Generating text") as pbar:
        while len(token_ids) - num_input_lzw_tokens < generate_config.max_new_tokens:

            if incremental_codebook_update:
                codebook_updates, _ = codebook_manager.update_codebook(
                    new_token_ids,
                    return_all_entries=False,
                )
                codebook_list.extend(codebook_updates)
            else:
                codebook_manager.reset()
                codebook_list, _ = codebook_manager.update_codebook(
                    token_ids,
                    return_all_entries=False,
                )

            input_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
            codebook_tensor = torch.tensor(
                codebook_list, device=model.device
            ).unsqueeze(0)

            logits, metadata = model(input_ids, codebook_tensor, metadata=metadata)

            logits = logits[:, -1, :]

            # set temperature to a small value if it is 0
            tmp = (
                generate_config.temperature if generate_config.temperature > 0 else 1e-5
            )
            probs = F.softmax(logits / tmp, dim=-1)

            if generate_config.do_sample:
                if generate_config.top_k > 0:
                    topk_probs, topk_indices = torch.topk(probs, generate_config.top_k)
                    idx = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    next_lzw_token = torch.gather(topk_indices, -1, idx)
                elif generate_config.top_p > 0:
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                    # Compute cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Keep only the tokens where cumulative probability is <= p
                    cutoff_mask = cumulative_probs <= generate_config.top_p
                    cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
                    cutoff_mask[..., 0] = True  # Always keep the first token

                    # Apply the mask
                    filtered_probs = sorted_probs * cutoff_mask
                    filtered_probs /= filtered_probs.sum()  # Re-normalize probabilities

                    # Sample from the filtered distribution
                    idx = torch.multinomial(filtered_probs, 1, generator=sample_rng)
                    next_lzw_token = sorted_indices.gather(-1, idx)
                else:
                    next_lzw_token = torch.multinomial(probs, 1, generator=sample_rng)
            else:
                next_lzw_token = torch.argmax(probs, dim=-1)

            next_lzw_token_id: int = next_lzw_token.item()
            token_ids.append(next_lzw_token_id)

            new_token_ids = [next_lzw_token_id]

            pbar.update(1)

        codebook_dict = codebook_manager.codebook

    return (
        tokenizer.decode(token_ids),
        token_ids,
        token_ids[num_input_lzw_tokens:],
        codebook_dict,
        None,
    )


if __name__ == "__main__":
    setup_seed()

    parser = ArgumentParser()
    parser.add_argument("--adapter", type=str, required=False)
    parser.add_argument("--hub-adapter", type=str, required=False)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument(
        "--demo",
        default=None,
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--extra-vocab-size", type=int, default=None)
    parser.add_argument("--compress-during-generation", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--pretrained-model", type=str, required=False)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--incremental-codebook-update", action="store_true")
    args = parser.parse_args()

    args.demo = args.demo.lower() if args.demo is not None else None
    if args.demo == "rust":
        args.prompt = (
            open("fast_compression/src/lib.rs").read() + "Explain the above code"
        )
    elif args.demo == "python":
        args.prompt = (
            open("fast_compression/src/lib.rs").read()
            + "Rewrite the above code in Python"
        )
    elif args.demo == "python-completion":
        args.prompt = open("train.py").read()[:100]
    elif args.demo == "cpp":
        args.prompt = "#include <iostream>"
    elif args.demo == "transformer":
        args.prompt = "Implement a Transformer model in PyTorch"
    elif args.demo == "java":
        args.prompt = "Write a Java program for university course registration"
    elif args.demo == "eu":
        args.prompt = "Briefly explain the History of the EU"
    elif args.demo == "calc":
        args.prompt = "Calculate the sum of 1 to 100"
    elif args.demo == "epfl":
        args.prompt = "Compare the EPFL and ETH Zurich"
    elif args.demo == "french":
        args.prompt = "Expliquez l'histoire de la révolution française"
    elif args.demo == "chinese":
        args.prompt = "讲解中国的历史"
    else:
        if args.prompt is None:
            raise ValueError(
                f"Demo {args.demo} not found, pass it as a prompt with `--prompt`"
            )

    generate_config = GenerateConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=not args.disable_kv_cache,
        compress_during_generation=args.compress_during_generation,
        extra_vocab_size=args.extra_vocab_size,
        # until= ["<|end|>"],
    )

    model, tokenizer = load_model(args.pretrained_model, args.adapter, args.hub_adapter)

    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = args.prompt

    model_config = model.config

    generate_config.extra_vocab_size = (
        generate_config.extra_vocab_size
        if generate_config.extra_vocab_size is not None
        else model_config.extra_vocab_size
    )

    start_time = time.time()

    (
        full_text,
        full_lzw_token_ids,
        out_lzw_token_ids,
        codebook_dict,
        _,
    ) = z2z_generate(
        prompt,
        model,
        generate_config,
        incremental_codebook_update=args.incremental_codebook_update,
    )  # N.B. the text, lzw_token_ids, normal_token_ids don't include the prompt, but the codebook_dict does include the hypertokens from the prompt

    gen_time = time.time() - start_time

    print(full_text)

    print("=" * 100)

    legacy_contrast_colorprint_tokens(
        full_lzw_token_ids,
        codebook_dict,
        model.tokenizer,
        color_scheme="finegrained",
    )

    print("-" * 20)

    # build full_unzipped_token_ids from full_lzw_token_ids and codebook_dict
    full_unzipped_token_ids = decompress(full_lzw_token_ids, codebook_dict)

    # check if the compression is standard lzw or not
    standard_lzw_token_ids, _codebook_dict = lzw_compress(
        ids=full_unzipped_token_ids,
        initial_vocab_size=model_config.initial_vocab_size,
        extra_vocab_size=generate_config.extra_vocab_size,
        max_out_seq_length=len(full_unzipped_token_ids),
        max_subtokens=model_config.compression.max_subtokens,
    )[0]

    codebook_dict = {v: list(map(int, k.split(","))) for k, v in _codebook_dict.items()}

    legacy_contrast_colorprint_tokens(
        standard_lzw_token_ids,
        codebook_dict,
        model.tokenizer,
        color_scheme="finegrained",
    )

    print("-" * 20)

    metadata = describe_lzw(
        full_lzw_token_ids,
        model_config.initial_vocab_size,
        standard_lzw_token_ids,
        hyper_vocab_size=generate_config.extra_vocab_size,
    )

    print(metadata)

    print(f"Generation time: {gen_time:.2f} seconds")
