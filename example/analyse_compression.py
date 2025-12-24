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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import Config
from utils import dataclass_from_file, compute_compression_rates, get_base_vocab_size
from fast_compression import batch_lzw_compress

# Constants
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_OUT_SEQ_LENGTH = 128_000
DEFAULT_LIMIT = 100
DEFAULT_FILTER_MIN_LENGTH = 512
DEFAULT_FILTER_MAX_LENGTH = 4096


# Define logarithmic decay function
def log_decay(x, a, b, c):
    return a - b * np.log(x + c)


def compress_dataset_samples(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    limit: int = DEFAULT_LIMIT,
    min_sample_length: int = DEFAULT_FILTER_MIN_LENGTH,
    max_sample_length: int = DEFAULT_FILTER_MAX_LENGTH,
    extra_vocab_size: int = None,
    max_subtokens: int = None,
) -> List[List[int]]:
    """
    Compresses samples from a dataset using LZW compression.
    """

    dataset = dataset.filter(
        lambda x: min_sample_length < len(x["input_ids"]) < max_sample_length
    )
    dataset_iter = iter(dataset.batch(DEFAULT_BATCH_SIZE))

    total_compressed_seq = []
    with tqdm(total=limit, desc="Processing dataset") as pbar:
        while True:
            try:
                batch = next(dataset_iter)
            except StopIteration:
                break

            compressed_batch: List[
                Tuple[List[int], Dict[str, int]]
            ] = batch_lzw_compress(
                ids=batch["input_ids"],
                initial_vocab_size=get_base_vocab_size(tokenizer),
                extra_vocab_size=extra_vocab_size,
                max_out_seq_length=DEFAULT_MAX_OUT_SEQ_LENGTH,
                max_subtokens=max_subtokens,
            )
            for compressed_ids, _ in compressed_batch:
                total_compressed_seq.append(compressed_ids)

            pbar.update(len(batch["input_ids"]))
            if pbar.n >= limit:
                break
    return total_compressed_seq


def plot_compression_rates(
    compression_rates: List[np.ndarray],
    filename: str,
    title: str,
    log_x_scale: bool = False,
):
    """
    Plots the compression rates as subplots.

    Args:
        compression_rates (List[np.ndarray]): List of compression rates to plot.
        filename (str): Filename to save the plot.
        title (str): Title of the plot.
    """
    if log_x_scale:
        filename = filename.replace(".png", "_log.png")

    num_plots = len(compression_rates)
    cols = 5 if num_plots > 5 else num_plots
    rows = (num_plots + cols - 1) // cols  # Calculate the number of rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten() if num_plots > 1 else [axs]

    for i, rate in enumerate(compression_rates):
        axs[i].plot(rate)
        axs[i].set_title(f"Sample {i+1}", fontsize=30)
        axs[i].set_xlabel("Sequence Length (Tokens)", fontsize=30)
        axs[i].set_ylabel("Compression Rate", fontsize=30)
        axs[i].tick_params(axis="both", which="major", labelsize=20)
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        if log_x_scale:
            axs[i].set_xscale("log")

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(
        rect=[0, 0.03, 1, 0.95]
    )  # Adjust layout to make room for the title
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")


def plot_multiple_compression_rates(
    compression_rates_dict: Dict[str, np.ndarray],
    filename: str,
    title: str,
    log_x_scale: bool = False,
):
    """
    Plots multiple compression rates on the same figure with a legend.

    Args:
        compression_rates_dict (Dict[str, np.ndarray]): Dictionary where keys are labels and values are compression rates.
        filename (str): Filename to save the plot.
        title (str): Title of the plot.
    """

    if log_x_scale:
        filename = filename.replace(".png", "_log.png")

    readable_labels = {
        "microsoft/Phi-3.5-mini-instruct": "32K-tokenizer",
        "meta-llama/Llama-2-7b-hf": "32K-tokenizer",
        "meta-llama/Llama-3.2-1B-Instruct": "128K-tokenizer",
        "microsoft/Phi-4-mini-instruct": "200K-tokenizer",
    }

    plt.figure(figsize=(10, 6))

    for label, rates in compression_rates_dict.items():
        plt.plot(
            rates, label=readable_labels.get(label, label), linewidth=2.5
        )  # Use label if not in readable_labels

        if log_x_scale:
            popt = fit_log_scale(rates)
            a, b, c = popt
            a, b, c = round(a, 2), round(b, 2), round(c, 2)
            x = range(len(rates))
            y = log_decay(x, a, b, c)
            plt.plot(
                x,
                y,
                label=f"Fitted Logarithmic Curve: {label}, a: {a}, b: {b}, c: {c}",
                linestyle="--",
            )

    plt.title(title, fontsize=30)
    plt.xlabel("Sequence Length (Tokens)", fontsize=30)
    plt.ylabel("Compression Rate", fontsize=30)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if log_x_scale:
        plt.xscale("log")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")


def fit_log_scale(compression_rates: np.ndarray):
    """
    Fits a log scale to the compression rates.
    """
    # Fit the model
    popt, _ = curve_fit(log_decay, range(len(compression_rates)), compression_rates)

    return popt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=False, default=None)
    parser.add_argument(
        "--subset", type=str, nargs="+", required=False, default="default"
    )
    parser.add_argument("--split", type=str, required=False, default="train")
    parser.add_argument("--text-column", type=str, required=False, default="text")
    parser.add_argument(
        "-t", "--tokenizer", type=str, nargs="+", required=False, default=None
    )
    parser.add_argument("-l", "--limit", type=int, required=False, default=200)
    parser.add_argument("--max-seq-length", type=int, required=False, default=16000)
    parser.add_argument(
        "-e",
        "--extra_vocab_size",
        type=int,
        required=False,
        default=None,
        help="Extra vocab size to use for compression. If not provided, it will be set to 2x the max sequence length.",
    )
    parser.add_argument(
        "--max-subtokens",
        type=int,
        required=False,
        default=8000,
        help="The maximum number of subtokens per hypertoken.",
    )
    parser.add_argument(
        "-i",
        "--individual",
        action="store_true",
        help="Whether to save the compressed sequences for each sample.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="compression_plots",
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--log-x-scale",
        action="store_true",
        help="Plot the compression rates with a logarithmic x-axis scale.",
    )

    args = parser.parse_args()

    config = (
        dataclass_from_file(Config, args.config) if args.config is not None else None
    )

    if config is not None:
        args.tokenizer = [config.pretrained_tokenizer_name_or_path]
        args.dataset = config.data.dataset_path
        args.subset = [config.data.dataset_name]
        args.split = config.data.split
        args.text_column = config.data.text_column
        args.extra_vocab_size = config.extra_vocab_size
        args.max_subtokens = config.compression.max_subtokens

    if args.extra_vocab_size is None:
        args.extra_vocab_size = args.max_seq_length

    if any(
        arg is None
        for arg in [
            args.dataset,
            args.subset,
            args.split,
            args.text_column,
            args.tokenizer,
        ]
    ):
        raise ValueError("All arguments are required unless a config file is provided.")

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Save the args to a JSON file in the output directory
    args_file_path = os.path.join(args.output_path, "args.json")
    with open(args_file_path, "w") as args_file:
        json.dump(vars(args), args_file, indent=4)

    named_compression_rates: Dict[str, Dict[str, np.ndarray]] = {
        f"{args.dataset}_{subset}": {
            tokenizer_path: None for tokenizer_path in args.tokenizer
        }
        for subset in args.subset
    }

    for subset in args.subset:
        dataset: IterableDataset = load_dataset(
            path=args.dataset,
            name=subset,
            split=args.split,
            streaming=True,
        )

        for tokenizer_path in args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = (
                tokenizer.eos_token
                if tokenizer.pad_token is None
                else tokenizer.pad_token
            )

            out_dataset = dataset.map(
                lambda x: tokenizer(x[args.text_column], return_attention_mask=False),
                batched=True,
            )

            total_compressed_seq = compress_dataset_samples(
                out_dataset,
                tokenizer,
                args.limit,
                min_sample_length=int(args.max_seq_length * 0.8),
                max_sample_length=int(args.max_seq_length * 1.2),
                extra_vocab_size=args.extra_vocab_size,
                max_subtokens=args.max_subtokens,
            )

            compression_rates = [
                compute_compression_rates(
                    compressed_ids=seq,
                    initial_vocab_size=get_base_vocab_size(tokenizer),
                    extra_vocab_size=args.extra_vocab_size,
                    max_seq_length=args.max_seq_length,
                )
                for seq in total_compressed_seq
            ]

            if args.individual:
                sampled_compression_rates = [
                    compression_rates[i]
                    for i in random.sample(range(len(compression_rates)), 30)
                ]

                plot_compression_rates(
                    sampled_compression_rates,
                    os.path.join(
                        args.output_path,
                        f"compressed_sequences_{args.dataset}_{tokenizer_path}.png",
                    ),
                    "Individual Compression Rates",
                    args.log_x_scale,
                )

            compression_array = np.array(compression_rates)
            effective_positions = np.any(compression_array != 0, axis=0)
            max_effective_positions = np.sum(effective_positions)
            compression_array = compression_array[:, :max_effective_positions]

            valid_compression_rates_mask = compression_array != 0
            mean_compression_rate = np.sum(compression_array, axis=0) / np.sum(
                valid_compression_rates_mask, axis=0
            )

            named_compression_rates[f"{args.dataset}_{subset}"][
                tokenizer_path
            ] = mean_compression_rate

    if len(args.tokenizer) == 1 and len(args.subset) == 1:
        # single tokenizer + single dataset

        key = f"{args.dataset}_{args.subset[0]}"
        plot_compression_rates(
            [named_compression_rates[key][args.tokenizer[0]]],
            os.path.join(args.output_path, f"mean_compression_rate.png"),
            "Compression Rate",
            args.log_x_scale,
        )

    elif len(args.subset) == 1:
        # single dataset but multiple tokenizers
        key = f"{args.dataset}_{args.subset[0]}"
        plot_multiple_compression_rates(
            {
                tokenizer_path: named_compression_rates[key][tokenizer_path]
                for tokenizer_path in args.tokenizer
            },
            os.path.join(
                args.output_path, f"mean_compression_rate_multiple_tokenizers.png"
            ),
            "Impact of Tokenizer Size on Compression",
            args.log_x_scale,
        )

    elif len(args.tokenizer) == 1:
        # single tokenizer but multiple datasets
        plot_multiple_compression_rates(
            {
                f"{subset}": named_compression_rates[f"{args.dataset}_{subset}"][
                    args.tokenizer[0]
                ]
                for subset in args.subset
            },
            os.path.join(
                args.output_path, f"mean_compression_rate_multiple_datasets.png"
            ),
            "Domain-wise Compression Performance",
            args.log_x_scale,
        )

    else:
        raise ValueError("Multiple datasets + multiple tokenizers are not supported.")


if __name__ == "__main__":
    main()


# python example/analyse_compression.py  -d "epfl-dlab/zip2zip-1B" --subset code knowledge math chat multilingual  -t microsoft/Phi-3.5-mini-instruct --o compression_plots/phi-3.5-32K/multi-domain --max-seq-length 6000 -l 200

# python example/analyse_compression.py  -d "epfl-dlab/zip2zip-1B" --subset code  -t  meta-llama/Llama-3.2-1B-Instruct microsoft/Phi-4-mini-instruct meta-llama/Llama-2-7b-hf  --o plots/compression_plots/multi_tokenizer/code -l 40  --max-seq-length 6000
