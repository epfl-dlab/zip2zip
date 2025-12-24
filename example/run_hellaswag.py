"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import os, sys
import json
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import OnlineZZModel
from interface import load_model
from utils import PLATFORM_BEST_DTYPE

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def render_example(example, model, tokenizer):
    """
    Given the example as a dictionary, render it as three torch tensors using a Hugging Face tokenizer:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)

    Args:
        example (dict): A dictionary containing "ctx" (context), "label" (correct completion index), and "endings" (list of completions).
        tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer to tokenize the input text.

    Returns:
        tuple: (data dictionary, tokens tensor, mask tensor, label index)
    """

    use_lzw = True if isinstance(model, OnlineZZModel) else False
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    candidates = [ctx + " " + end for end in endings]

    # Tokenize context
    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)

    candidates = [ctx + " " + end for end in endings]

    candidates_tokens = tokenizer.batch_encode_plus(
        candidates,
        add_special_tokens=False,
    )["input_ids"]

    if use_lzw:
        lzw_ctx_tokens, _ = model.lzw_compress(ctx_tokens) if use_lzw else ctx_tokens
        lzw_candidates_tokens, codebook_tensor = model.lzw_compress(candidates_tokens)
        ending_token_mask = torch.zeros_like(lzw_candidates_tokens)
        ending_token_mask[:, lzw_ctx_tokens.shape[1] :] = 1
        # mask all padding tokens to 0
        ending_token_mask[lzw_candidates_tokens == model.pad_token_id] = 0
        tokens = lzw_candidates_tokens
    else:
        tok_rows = []
        mask_rows = []

        for end in endings:
            end_tokens = tokenizer.encode(
                end, add_special_tokens=False
            )  # No need to prepend space with HF tokenizers
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        # Find max length for padding
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        ending_token_mask = torch.zeros((4, max_len), dtype=torch.long)

        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, : len(tok_row)] = torch.tensor(tok_row)
            ending_token_mask[i, : len(mask_row)] = torch.tensor(mask_row)
        codebook_tensor = None
    return tokens, ending_token_mask, label, codebook_tensor


def iterate_examples(split):
    # there are 10,042 examples in total in val
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate(model, tokenizer, device, num_examples: int):

    torch.set_float32_matmul_precision("high")  # use tf32
    model.to(device).to(PLATFORM_BEST_DTYPE)

    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in tqdm(iterate_examples("val"), total=num_examples):
        tokens, ending_token_mask, label, codebook_tensor = render_example(
            example, model, tokenizer
        )
        tokens = tokens.to(device)  # shape (4, N)
        ending_token_mask = ending_token_mask.to(device)  # shape (4, N)

        # get the logits
        if isinstance(model, OnlineZZModel):
            logits, metadata = model(tokens, codebook_tensor)  # shape (4, N, V)
        else:
            logits = model(tokens)[0]  # shape (4, N, V)
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()  # shape (4, N-1, V)
        shift_tokens = (tokens[..., 1:]).contiguous()  # shape (4, N-1)
        flat_shift_logits = shift_logits.view(
            -1, shift_logits.size(-1)
        )  # shape (4*(N-1), V)
        flat_shift_tokens = shift_tokens.view(-1)  # shape (4*(N-1),)
        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )  # shape (4*(N-1),)
        shift_losses = shift_losses.view(tokens.size(0), -1)  # shape (4, N-1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (
            ending_token_mask[..., 1:]  # shape (4, N-1)
        ).contiguous()  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(
            f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f} acc: {num_correct}/{num_total}={num_correct/num_total:.4f}"
        )

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")
        if num_total >= num_examples:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-type", type=str, default=None, help="the model type to use"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the device to use"
    )
    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        default=10042,
        help="the number of examples to evaluate",
    )
    parser.add_argument(
        "--adapter", type=str, required=False, help="Optional hub adapter path."
    )
    parser.add_argument(
        "--hub-adapter", type=str, required=False, help="Optional hub adapter path."
    )
    parser.add_argument("--seed", type=int, default=40, help="Random seed")
    args = parser.parse_args()
    seed_everything(args.seed)

    model, tokenizer = load_model(args.model_type, args.adapter, args.hub_adapter)
    evaluate(model, tokenizer, args.device, args.num_examples)
