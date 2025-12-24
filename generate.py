import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from colorama import Fore, Style


from model import OnlineZZModel
from utils import (
    pad_codebook,
)

from fast_compression import lzw_compress
from zip2zip_compression import bounded_lzw_in_chunks


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
    do_sample: bool = True


def until_has_been_reached(generate_config: GenerateConfig, text: str) -> bool:
    if generate_config.until is None:
        return False

    return any(u in text for u in generate_config.until)


@torch.no_grad()
def z2z_generate(
    prompt: str,
    model: OnlineZZModel,
    generate_config: GenerateConfig,
    use_new_lzw: bool = False,
) -> Tuple[str, List[int], List[int], Dict[str, int], float]:
    model.eval()
    config = model.config
    tokenizer = model.tokenizer
    generate_config.extra_vocab_size = (
        generate_config.extra_vocab_size
        if generate_config.extra_vocab_size is not None
        else config.extra_vocab_size
    )
    sample_rng = torch.Generator(device=model.device)
    sample_rng.manual_seed(42 + generate_config.ddp_rank)

    input_token_ids = tokenizer.encode(prompt)
    prompt_orig_size = len(input_token_ids)
    unzipped_token_ids = input_token_ids.copy()

    disabled_ids = [32001, 32007, 32010]

    if use_new_lzw:
        lzw_token_ids, codebook_dict = bounded_lzw_in_chunks(
            ids=unzipped_token_ids,
            initial_vocab_size=config.initial_vocab_size,
            extra_vocab_size=generate_config.extra_vocab_size,
            max_out_seq_length=len(unzipped_token_ids)
            + 1,  # +1 is needed, because max_out_seq_length-1 is the length
            max_subtokens=config.compression.max_subtokens,
            disabled_ids=disabled_ids,
        )[0]
    else:
        lzw_token_ids, codebook_dict = lzw_compress(
            ids=unzipped_token_ids,
            initial_vocab_size=config.initial_vocab_size,
            extra_vocab_size=generate_config.extra_vocab_size,
            max_out_seq_length=len(unzipped_token_ids) + 1,
            max_subtokens=config.compression.max_subtokens,
            disabled_ids=disabled_ids,
        )[0]

    prompt_compressed_size = len(lzw_token_ids)
    print(
        f"the original size is {prompt_orig_size} and the compressed size is {prompt_compressed_size}"
    )

    num_input_tokens = len(lzw_token_ids)

    codebook_list, _, _ = pad_codebook(
        codebook_dict=codebook_dict,
        initial_vocab_size=config.initial_vocab_size,
        extra_vocab_size=generate_config.extra_vocab_size,
        max_subtokens=config.compression.max_subtokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(f"the vocab size is {len(codebook_list)}")

    # just to make the code more readable
    # no really overhead as the interpreter will optimize this
    token_ids = lzw_token_ids

    metadata = {}
    first_token_time = -1

    with tqdm(total=generate_config.max_new_tokens, desc="Generating text") as pbar:
        while len(token_ids) - num_input_tokens < generate_config.max_new_tokens:

            if use_new_lzw:
                standard_lzw_token_ids, codebook_dict = bounded_lzw_in_chunks(
                    ids=unzipped_token_ids,
                    initial_vocab_size=config.initial_vocab_size,
                    extra_vocab_size=generate_config.extra_vocab_size,
                    max_out_seq_length=len(input_token_ids),
                    max_subtokens=config.compression.max_subtokens,
                    disabled_ids=disabled_ids,
                )[0]
            else:
                standard_lzw_token_ids, codebook_dict = lzw_compress(
                    ids=unzipped_token_ids,
                    initial_vocab_size=config.initial_vocab_size,
                    extra_vocab_size=generate_config.extra_vocab_size,
                    max_out_seq_length=len(input_token_ids),
                    max_subtokens=config.compression.max_subtokens,
                    disabled_ids=disabled_ids,
                )[0]

            codebook_list, _, _ = pad_codebook(
                codebook_dict,
                config.initial_vocab_size,
                len(codebook_dict),
                config.compression.max_subtokens,
                tokenizer.pad_token_id,
            )

            if generate_config.compress_during_generation:
                token_ids = standard_lzw_token_ids

            input_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
            codebook_tensor = torch.tensor(
                codebook_list, device=model.device
            ).unsqueeze(0)

            logits, metadata = model(input_ids, codebook_tensor, metadata=metadata)

            if not generate_config.use_kv_cache:
                metadata["kv_cache"] = None

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
                    next_token = torch.gather(topk_indices, -1, idx)
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
                    next_token = sorted_indices.gather(-1, idx)
                else:
                    next_token = torch.multinomial(probs, 1, generator=sample_rng)
            else:
                next_token = torch.argmax(probs, dim=-1)

            next_token_id: int = next_token.item()
            token_ids.append(next_token_id)

            if first_token_time == -1:
                first_token_time = time()

            if next_token_id >= config.initial_vocab_size:
                id_to_str = {v: k for k, v in codebook_dict.items()}
                next_subtokens = list(map(int, id_to_str[next_token_id].split(",")))

            else:
                next_subtokens = [next_token_id]
            unzipped_token_ids.extend(next_subtokens)

            if next_token_id == tokenizer.eos_token_id or until_has_been_reached(
                generate_config, tokenizer.decode(next_subtokens)
            ):
                pbar.update(generate_config.max_new_tokens - pbar.n)
                print(
                    Fore.RED
                    + "Reached early exit condition. Stopping generation."
                    + Style.RESET_ALL
                )
                break
            else:

                pbar.update(1)

    return (
        tokenizer.decode(unzipped_token_ids),
        token_ids,
        token_ids[prompt_compressed_size:],
        codebook_dict,
        first_token_time,
    )
