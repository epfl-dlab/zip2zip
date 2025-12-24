messages256 = """```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
from vllm.transformers_utils.tokenizer_base import TokenizerBase, TokenizerRegistry

adapter_path = snapshot_download(repo_id="nathanrchn/ozzi")

llm = LLM(model="microsoft/Phi-3.5-mini-instruct", enable_lora=True, max_lora_rank=32)

output = llm.chat(
    messages=[
        {"role": "user", "content": f"Can you rewrite this LZW encoder from Rust to Python?\n\n```rust\n{open('fast_compression/src/lib.rs').read()}\n```"},
    ],
    sampling_params=SamplingParams(
        max_tokens=4096
    ),
    lora_request=LoRARequest("adapter", 1, adapter_path)
)

print(output)

```
Explain in detail what the code does.
"""

messages512 = (
    """```python
from custom_types import Codebook, BatchedLZWTokenization
import fast_compression


def batched_lzw_compress(
    ids,
    initial_vocab_size,
    extra_vocab_size,
    max_out_seq_length,
    max_subtokens,
    disabled_ids = None,
    pad_token_id = 0,
) -> BatchedLZWTokenization:
    if disabled_ids is None:
        disabled_ids = []
    if pad_token_id is None:
        raise ValueError()
    raw_results = fast_compression.batch_lzw_compress(
        ids,
        initial_vocab_size,
        extra_vocab_size,
        max_out_seq_length,
        max_subtokens,
        disabled_ids,
    )
    token_ids = []
    codebooks = []
    for compressed_ids, token_map in raw_results:
        codebook = Codebook.from_token_map(
            token_map, initial_vocab_size, extra_vocab_size, max_subtokens, pad_token_id
        )
        token_ids.append(compressed_ids)
        codebooks.append(codebook)
    return BatchedLZWTokenization(token_ids, codebooks)


def lzw_compress(
    ids,
    initial_vocab_size,
    extra_vocab_size,
    max_out_seq_length,
    max_subtokens,
    disabled_ids = None,
    pad_token_id = 0,
) -> BatchedLZWTokenization:
    return batched_lzw_compress(
        [ids],
        initial_vocab_size,
        extra_vocab_size,
        max_out_seq_length,
        max_subtokens,
        disabled_ids,
        pad_token_id,
    )[0]

```
Explain in detail what the code does.
""",
)

messages1024 = (
    """```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm


def visualize_attention_grid(
    attn_tensor,
    attn_mask=None,
    save_path="attention_plots/attention_grid.png",
    tokens=None,
    cmap="Reds",
):
    \"\"\"
    Visualizes a (L, H, S, S) attention tensor as a single figure with L rows and H columns of heatmaps.

    Args:
        attn_tensor (torch.Tensor or np.ndarray): Attention tensor of shape (L, H, S, S)
        save_path (str, optional): Path to save the figure (default: "attention_plots/attention_grid.png")
        tokens (list, optional): List of tokens to label axes (default: None)
        cmap (str, optional): Color map for heatmaps (default: "viridis")
    \"\"\"
    if attn_mask is not None:
        attn_tensor = attn_tensor.masked_fill(attn_mask.logical_not(), float("nan"))
    if isinstance(attn_tensor, torch.Tensor):
        attn_tensor = attn_tensor.detach().cpu().to(torch.float32).numpy()

    L, H, seq_len, _ = attn_tensor.shape  # Layers, Heads, Seq_len, Seq_len

    os.makedirs(
        os.path.dirname(save_path), exist_ok=True
    )  # Ensure save directory exists

    fig, axes = plt.subplots(
        L, H, figsize=(H * 3, L * 3)
    )  # Create L x H grid of subplots

    # Progress bar tracking
    total_plots = L * H
    with tqdm(total=total_plots, desc="Plotting Attention Heads") as pbar:
        for layer in range(L):
            for head in range(H):
                ax = (
                    axes[layer, head] if L > 1 and H > 1 else axes[max(layer, head)]
                )  # Handle edge cases
                attn_map = attn_tensor[layer, head]
                im = ax.imshow(attn_map, cmap=cmap, aspect="auto")  # Plot the heatmap

                # Add numerical values to heatmap
                for i in range(seq_len):
                    for j in range(seq_len):
                        text_color = "white" if attn_map[i, j] > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{attn_map[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=6,
                        )

                # Add a title to the plot
                ax.set_title(f"L{layer+1} H{head+1}", fontsize=8)

                if tokens and layer == L - 1:  # Only label x-axis on last row
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                else:
                    ax.set_xticks([])

                if tokens and head == 0:  # Only label y-axis on first column
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=6)
                else:
                    ax.set_yticks([])
                pbar.update(1)  # Update progress bar

    # Use a tight layout and set dpi to 300
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Attention grid saved at: {save_path}")

```
Explain in detail what the code does.
""",
)

messages2048 = (
    """```python
import torch
from time import time
from tqdm import tqdm
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

from model import OnlineZZModel
from utils import (
    pad_codebook,
)

from fast_compression import lzw_compress


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
    original_size = len(input_token_ids)
    normal_token_ids = input_token_ids.copy()

    lzw_token_ids, codebook_dict = lzw_compress(
        ids=normal_token_ids,
        initial_vocab_size=config.initial_vocab_size,
        extra_vocab_size=generate_config.extra_vocab_size,
        max_out_seq_length=len(normal_token_ids) - 1,
        max_subtokens=config.compression.max_subtokens,
    )[0]

    compressed_size = len(lzw_token_ids)
    print(
        f"the original size is {original_size} and the compressed size is {compressed_size}"
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
            standard_lzw_token_ids, codebook_dict = lzw_compress(
                ids=normal_token_ids,
                initial_vocab_size=config.initial_vocab_size,
                extra_vocab_size=generate_config.extra_vocab_size,
                max_out_seq_length=len(normal_token_ids) - 1,
                max_subtokens=config.compression.max_subtokens,
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

            probs = F.softmax(logits / generate_config.temperature, dim=-1)

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

            next_token = next_token.item()
            if next_token == tokenizer.eos_token_id or until_has_been_reached(
                generate_config, tokenizer.decode(token_ids)
            ):
                pbar.update(generate_config.max_new_tokens - pbar.n)
                break

            token_ids.append(next_token)
            pbar.update(1)

            if first_token_time == -1:
                first_token_time = time()

            if next_token >= config.initial_vocab_size:
                id_to_str = {v: k for k, v in codebook_dict.items()}
                subtokens = list(map(int, id_to_str[next_token].split(",")))
                normal_token_ids.extend(subtokens)
            else:
                normal_token_ids.append(next_token)

    return (
        tokenizer.decode(normal_token_ids),
        token_ids[compressed_size:],
        normal_token_ids[original_size:],
        codebook_dict,
        first_token_time,
    )

```
Explain in detail what the code does.
""",
)

PROMPTS = {
    "256": messages256,
    "512": messages512,
    "1024": messages1024,
    "2048": messages2048,
}
