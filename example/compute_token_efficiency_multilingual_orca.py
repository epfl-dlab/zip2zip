import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import load_dataset, IterableDataset
from tqdm import tqdm
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import Config
from utils import dataclass_from_file, compute_compression_rates, get_base_vocab_size
from fast_compression import batch_lzw_compress
from _legacy_lzw_tokenizer import Legacy_LZW_Tokenizer
from zip2zip.utils.token_efficiency_calc import TokenEfficiencyCalculator


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1_000)
    parser.add_argument("--filter_length_min", type=int, default=500)
    parser.add_argument("--use_relative", action="store_true")
    args = parser.parse_args()

    NUM_SAMPLES = args.num_samples

    tokenizer_32k_llama = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_fast=True
    )

    lzw_tokenizer_32k_llama = Legacy_LZW_Tokenizer(tokenizer_32k_llama)

    tokenizer_128k_llama = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=True
    )

    lzw_tokenizer_128k_llama = Legacy_LZW_Tokenizer(tokenizer_128k_llama)

    tokenizer_names = ["Llama-32K", "Llama-32K-LZW", "Llama-128K", "Llama-128K-LZW"]

    FILTER_LENGTH_MIN = args.filter_length_min

    # code knowledge math chat multilingual

    es_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="es_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    ar_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="ar_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    zh_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="zh_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    de_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="de_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    fr_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="fr_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    ru_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="ru_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)
    tr_dataset: IterableDataset = load_dataset(
        "multilingual/orca_dpo_pairs", split="tr_train", streaming=True
    ).filter(lambda x: len(x["question"]) > FILTER_LENGTH_MIN)

    all_datasets = [
        es_dataset,
        ar_dataset,
        zh_dataset,
        de_dataset,
        fr_dataset,
        ru_dataset,
        tr_dataset,
    ]

    # dataset_names = ["es", "ar", "zh", "de", "fr", "ru", "tr"]
    dataset_names = [
        "Spanish",
        "Arabic",
        "Chinese",
        "German",
        "French",
        "Russian",
        "Turkish",
    ]

    token_efficiency_matrix = TokenEfficiencyCalculator.compute_matrix(
        [
            tokenizer_32k_llama,
            lzw_tokenizer_32k_llama,
            tokenizer_128k_llama,
            lzw_tokenizer_128k_llama,
        ],
        all_datasets,
        num_samples=NUM_SAMPLES,
        column_name="question",
        relative=args.use_relative,
        base_index=0,
    )

    TokenEfficiencyCalculator.prettyprint_matrix(
        token_efficiency_matrix, column_names=dataset_names, row_names=tokenizer_names
    )

    TokenEfficiencyCalculator.bar_chart(
        token_efficiency_matrix,
        column_names=dataset_names,
        row_names=tokenizer_names,
        save_path=f"token_efficiency_bar_chart_multilingual_orca_{args.filter_length_min}_{'relative' if args.use_relative else ''}.png",
        title=f"{'Relative' if args.use_relative else ''} Token Efficiency",
        xlabel="Language",
        ylabel="Bytes per Token",
    )
