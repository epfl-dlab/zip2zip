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

    NUM_SAMPLES = 1_0

    tokenizer_32k_llama = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_fast=True
    )

    lzw_tokenizer_32k_llama = Legacy_LZW_Tokenizer(tokenizer_32k_llama)

    tokenizer_128k_llama = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=True
    )

    lzw_tokenizer_128k_llama = Legacy_LZW_Tokenizer(tokenizer_128k_llama)

    tokenizer_128k_deepseek = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-Base", use_fast=True
    )
    lzw_tokenizer_128k_deepseek = Legacy_LZW_Tokenizer(tokenizer_128k_deepseek)

    tokenizer_150k_qwen = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", use_fast=True
    )

    lzw_tokenizer_150k_qwen = Legacy_LZW_Tokenizer(tokenizer_150k_qwen)

    tokenizer_200k_phi = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-mini-instruct", use_fast=True
    )

    lzw_tokenizer_200k_phi = Legacy_LZW_Tokenizer(tokenizer_200k_phi)

    tokenizer_256K_gemma = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-it", use_fast=True
    )

    lzw_tokenizer_256K_gemma = Legacy_LZW_Tokenizer(tokenizer_256K_gemma)

    all_tokenizers = [
        tokenizer_32k_llama,
        lzw_tokenizer_32k_llama,
        tokenizer_128k_llama,
        lzw_tokenizer_128k_llama,
        tokenizer_200k_phi,
        lzw_tokenizer_200k_phi,
        tokenizer_256K_gemma,
        lzw_tokenizer_256K_gemma,
        tokenizer_150k_qwen,
        lzw_tokenizer_150k_qwen,
        tokenizer_128k_deepseek,
        lzw_tokenizer_128k_deepseek,
    ]

    tokenizer_names = [
        "Llama-32K",
        "Llama-32K-LZW",
        "Llama-128K",
        "Llama-128K-LZW",
        "Phi-200K",
        "Phi-200K-LZW",
        "Gemma-256K",
        "Gemma-256K-LZW",
        "Qwen-150K",
        "Qwen-150K-LZW",
        "DeepSeek-128K",
        "DeepSeek-128K-LZW",
    ]

    # code knowledge math chat multilingual

    code_dataset: IterableDataset = load_dataset(
        "epfl-dlab/zip2zip-1B", split="train", name="code", streaming=True
    )
    math_dataset: IterableDataset = load_dataset(
        "epfl-dlab/zip2zip-1B", split="train", name="math", streaming=True
    )
    chat_dataset: IterableDataset = load_dataset(
        "epfl-dlab/zip2zip-1B", split="train", name="chat", streaming=True
    )
    multilingual_dataset: IterableDataset = load_dataset(
        "epfl-dlab/zip2zip-1B", split="train", name="multilingual", streaming=True
    )
    knowledge_dataset: IterableDataset = load_dataset(
        "epfl-dlab/zip2zip-1B", split="train", name="knowledge", streaming=True
    )

    all_datasets = [
        code_dataset,
        math_dataset,
        chat_dataset,
        multilingual_dataset,
        knowledge_dataset,
    ]

    dataset_names = ["code", "math", "chat", "multilingual", "knowledge"]

    token_efficiency_matrix = TokenEfficiencyCalculator.compute_matrix(
        all_tokenizers, all_datasets, num_samples=NUM_SAMPLES
    )

    TokenEfficiencyCalculator.prettyprint_matrix(
        token_efficiency_matrix, column_names=dataset_names, row_names=tokenizer_names
    )
