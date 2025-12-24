import pandas as pd
import seaborn as sns
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from datasets import load_dataset
from argparse import ArgumentParser
from transformers import AutoTokenizer
from typing import List, Union, Optional, Tuple, Dict, Set
from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zip2zip.tokenizer import Zip2ZipTokenizer


def codebook_contains(
    codebook: Dict[Tuple[int], int], ids_to_merge: List[int], initial_vocab_size: int
) -> bool:
    if len(ids_to_merge) == 1:
        return ids_to_merge[0] < initial_vocab_size
    return tuple(ids_to_merge) in codebook


def get_id_from_codebook(
    codebook: Dict[Tuple[int], int], ids_to_merge: List[int]
) -> int:
    if len(ids_to_merge) == 1:
        return ids_to_merge[0]
    return codebook[tuple(ids_to_merge)]


def encode_python(
    ids: List[int],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Set[int],
) -> Tuple[List[int], List[int]]:
    compressed_ids = []

    codebook = {}
    next_id = initial_vocab_size
    ids_to_merge = []

    for id in ids:
        if id in disabled_ids:
            if ids_to_merge:
                compressed_ids.append(get_id_from_codebook(codebook, ids_to_merge))
                ids_to_merge = []
            compressed_ids.append(id)
            continue

        ids_to_merge.append(id)

        is_in_codebook = codebook_contains(codebook, ids_to_merge, initial_vocab_size)
        if not is_in_codebook:
            if next_id < initial_vocab_size + max_codebook_size:
                codebook[tuple(ids_to_merge)] = next_id
                next_id += 1

            ids_to_merge.pop()
            compressed_ids.append(get_id_from_codebook(codebook, ids_to_merge))
            ids_to_merge = [id]

        if len(ids_to_merge) == max_subtokens:
            compressed_ids.append(get_id_from_codebook(codebook, ids_to_merge))
            ids_to_merge = []

    if len(ids_to_merge) > max_subtokens:
        last_id = ids_to_merge.pop()
        compressed_ids.append(get_id_from_codebook(codebook, ids_to_merge))
        ids_to_merge = [last_id]

    if ids_to_merge:
        compressed_ids.append(get_id_from_codebook(codebook, ids_to_merge))

    return compressed_ids, [1] * len(compressed_ids)


def batch_encode_python(
    ids: List[List[int]],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    compressed_ids = []
    attention_masks = []
    disabled_ids_set = set(disabled_ids) if disabled_ids else set()

    for batch in ids:
        cids, ams = encode_python(
            batch,
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            disabled_ids_set,
        )
        compressed_ids.append(cids)
        attention_masks.append(ams)

    return compressed_ids, attention_masks


def decode_python(
    compressed_ids: List[int],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[int]:
    result_ids = []
    codebook = {}

    next_id = initial_vocab_size
    previous_ids = []
    disabled_ids_set = set(disabled_ids) if disabled_ids else set()

    for id in compressed_ids:
        if id in disabled_ids_set:
            previous_ids = []
            result_ids.append(id)
            continue

        if id < initial_vocab_size:
            current_ids = [id]
        elif id in codebook:
            current_ids = codebook[id]
        elif len(previous_ids) == max_subtokens:
            current_ids = codebook[id]
        else:
            current_ids = previous_ids + [previous_ids[0]]
            codebook[id] = current_ids

        result_ids.extend(current_ids)

        if (
            previous_ids
            and next_id < initial_vocab_size + max_codebook_size
            and len(previous_ids) < max_subtokens
        ):
            new_entry = previous_ids + [current_ids[0]]
            codebook[next_id] = new_entry
            next_id += 1

        previous_ids = current_ids

    return result_ids


class PythonZip2ZipTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.initial_vocab_size = initial_vocab_size
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.disabled_ids = disabled_ids

        self.old_batch_encode_plus = self.tokenizer._batch_encode_plus
        self.tokenizer._batch_encode_plus = self._batch_encode_plus

        self.old_decode = self.tokenizer._decode
        self.tokenizer._decode = self._decode

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        encoding = self.old_batch_encode_plus(*args, **kwargs)

        encoding.input_ids, encoding.attention_mask = batch_encode_python(
            encoding.input_ids,
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            disabled_ids=self.disabled_ids,
        )

        return encoding

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        token_ids = decode_python(
            token_ids,
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            disabled_ids=self.disabled_ids,
        )

        return self.old_decode(
            token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_codebook_size: int,
        max_subtokens: int,
        disabled_ids: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        return cls(
            tokenizer,
            len(tokenizer),
            max_codebook_size,
            max_subtokens,
            disabled_ids,
        )


if True:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--batch-sizes", type=str, default="1,10,50,100,150,200")
    args = parser.parse_args()

    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]

    print(f"Benchmarking {args.model} with batch sizes {batch_sizes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lzw_tokenizer = Zip2ZipTokenizer.from_config(
        args.model,
        2048,
        4,
        disabled_ids=list(tokenizer.get_added_vocab().values()),
    )
    python_lzw_tokenizer = PythonZip2ZipTokenizer.from_pretrained(
        args.model,
        2048,
        4,
        disabled_ids=list(tokenizer.get_added_vocab().values()),
    )

    dataset = load_dataset(
        "devngho/the-stack-llm-annotations-v2",
        streaming=True,
        name="all",
        split="train",
    )

    data = {
        "time": [],
        "length": [],
        "tokenizer": [],
        "function": [],
        "batch_size": [],
    }

    TOTAL_LENGTH = 10_000_000
    for batch_size in batch_sizes:
        total_length = 0

        with tqdm(
            total=TOTAL_LENGTH, desc=f"tokenizing dataset (batch_size={batch_size})"
        ) as pbar:
            for samples in dataset.batch(batch_size):
                start = time()
                ids = tokenizer.batch_encode_plus(samples["text"])
                end = time()
                len_ids = sum(len(ids) for ids in ids.input_ids)
                data["tokenizer"].append("base_tokenizer")
                data["function"].append("encode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)
                total_length += len_ids

                start = time()
                compressed_ids = lzw_tokenizer.batch_encode_plus(samples["text"])
                end = time()
                data["tokenizer"].append("rust_lzw_tokenizer")
                data["function"].append("encode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)

                start = time()
                python_compressed_ids = python_lzw_tokenizer.batch_encode_plus(
                    samples["text"]
                )
                end = time()
                data["tokenizer"].append("python_lzw_tokenizer")
                data["function"].append("encode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)

                start = time()
                t1 = tokenizer.batch_decode(ids.input_ids)
                end = time()
                data["tokenizer"].append("base_tokenizer")
                data["function"].append("decode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)

                start = time()
                t2 = lzw_tokenizer.batch_decode(compressed_ids.input_ids)
                end = time()
                data["tokenizer"].append("rust_lzw_tokenizer")
                data["function"].append("decode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)

                start = time()
                t3 = python_lzw_tokenizer.batch_decode(python_compressed_ids.input_ids)
                end = time()
                data["tokenizer"].append("python_lzw_tokenizer")
                data["function"].append("decode")
                data["time"].append(end - start)
                data["length"].append(len_ids)
                data["batch_size"].append(batch_size)

                if total_length > TOTAL_LENGTH:
                    pbar.update(TOTAL_LENGTH - pbar.n)
                    break

                pbar.update(len_ids)

                assert t1 == t2 == t3

    df = pd.DataFrame(data)

    # df.to_csv("tokenizer_performance_comparison.csv", index=False)

    # df = pd.read_csv("tokenizer_performance_comparison.csv")

    df["time_ms"] = df["time"] * 1000

    sns.set_theme(style="whitegrid")

    fig, ax_latency = plt.subplots(figsize=(10, 7))
    latency_df = df.copy()
    latency_df["function"] = latency_df["function"].replace(
        {"encode": "Tokenize", "decode": "Detokenize"}
    )
    latency_df["tokenizer"] = latency_df["tokenizer"].replace(
        {
            "base_tokenizer": "Base Tokenizer",
            "rust_lzw_tokenizer": "Rust LZW Tokenizer",
            "python_lzw_tokenizer": "Python LZW Tokenizer",
        }
    )

    sns.barplot(
        ax=ax_latency,
        data=latency_df,
        x="tokenizer",
        y="time_ms",
        hue="function",
        palette="viridis",
    )
    ax_latency.set_title("Tokenization Latency", fontsize=20)
    ax_latency.set_xlabel("Tokenizer Type", fontsize=18)
    ax_latency.set_ylabel("Latency (milliseconds)", fontsize=18)
    ax_latency.tick_params(axis="x", rotation=15, labelsize=14)
    ax_latency.tick_params(axis="y", labelsize=14)
    ax_latency.legend(title="Operation", fontsize=14, title_fontsize=16)

    plt.tight_layout()
    plt.savefig("tokenizer_performance_comparison.png", dpi=300)
