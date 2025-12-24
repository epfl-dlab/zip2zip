import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch
from hashlib import sha256
from argparse import ArgumentParser
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Iterable
from datasets import load_dataset, IterableDataset
from safetensors.torch import save_file, load_file

from configs import Config
from custom_types import Codebook, BatchedLZWTokenization, LZWTokenization
from utils import dataclass_from_file

from py_fast_compression import batched_lzw_compress, lzw_compress


def data_config_hash(config: Config) -> str:
    attributes = [
        config.pretrained_tokenizer_name_or_path,
        config.data.path,
        config.data.dataset_path,
        config.data.dataset_name,
        config.data.text_column,
        config.compression.compressor_function_name,
        config.compression.max_subtokens,
        config.extra_vocab_size,
        config.initial_vocab_size,
        config.seq_length,
    ]

    return sha256(str(attributes).encode()).hexdigest()


class DataLoaderLite:
    def __init__(
        self,
        config: Config,
        split: str,
        device: str,
        process_rank: int,
        num_processes: int,
    ) -> None:
        self.config = config
        self.device = device
        self.data_config = config.data
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_length = config.seq_length
        self.total_batch_size = config.total_batch_size
        self.per_device_batch_size = config.per_device_batch_size

        data_set_hash = data_config_hash(config)

        dataset_path = f"{config.data.path}/{data_set_hash}/{split}/data.safetensors"
        if process_rank == 0:
            print(f"loading data from {dataset_path}")

        self.samples = load_data(
            dataset_path,
            device="cpu",  # transfer to GPU later. If the dataset is small enough, keep it in GPU doesn't seem to improve efficiency
        )

        if config.data.shuffle:
            np_indices = np.random.permutation(self.samples["input_ids"].size(0))
        else:
            np_indices = np.arange(self.samples["input_ids"].size(0))

        self.indices = torch.from_numpy(np_indices).to("cpu")

        self.reset()

    def reset(self) -> None:
        self.current_position = self.per_device_batch_size * self.process_rank

    def next_batch(
        self,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = self.indices[
            self.current_position : self.current_position + self.per_device_batch_size
        ]

        input_ids = self.samples["input_ids"][indices][:, :-1].to(self.device)
        labels = self.samples["input_ids"][indices][:, 1:].to(self.device)

        vocab = self.samples["vocab"][indices].to(self.device)

        mask = torch.ones_like(labels, dtype=self.config.dtype, device=self.device)

        if self.config.mask_first_occurrence:
            B, S = mask.size()
            for i in range(B):
                m = set()
                for j in range(S):
                    id = labels[i, j].item()
                    if id >= self.config.initial_vocab_size and id not in m:
                        m.add(id)
                        mask[i, j] = 0

        self.current_position += self.per_device_batch_size * self.num_processes

        if (
            self.current_position + self.per_device_batch_size * self.num_processes
            > len(self.samples["input_ids"])
        ):
            self.reset()

        return input_ids, labels, vocab, mask


def compress_dataset(
    config: Config,
    dataset_iter: Iterable[Dict],
    num_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    split: str,
) -> None:
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("eos_token_id and pad_token_id must be provided")
    if config.compression.compressor_function_name == "lzw":
        compressor_function = batched_lzw_compress
    else:
        raise ValueError(
            f"Unknown compressor function: {config.compression.compressor_function_name}"
        )

    data_hash = data_config_hash(config)
    dataset_path = f"{config.data.path}/{data_hash}/{split}/data.safetensors"

    # delete in case exists
    if os.path.exists(os.path.dirname(dataset_path)):
        shutil.rmtree(os.path.dirname(dataset_path))
    os.makedirs(os.path.dirname(dataset_path))

    num_samples = 0
    total_raw_tokens = 0
    num_lzw_tokens = 0
    total_out_tokens = 0
    batched_compressed_ids: List[List[int]] = []

    all_hypertoken_activeness = []
    all_hypertoken_density = []

    # Process in chunks of a fixed size (e.g., 10000 samples)
    chunk_size = 100_000
    processed_tokenizations: List[List[int]] = []
    processed_codebooks: List[torch.Tensor] = []
    codebook_utilization_per_sample: List[float] = []
    code_sizes: List[float] = []
    chunk_idx = 0

    with tqdm(desc="Processing dataset", total=num_tokens) as pbar:
        while True:  # Instead of checking against num_tokens
            try:
                batch = next(dataset_iter)  # type: ignore
            except StopIteration:
                print(f"Dataset processing complete. Processed {pbar.n} tokens.")
                if num_tokens is not None and pbar.n < num_tokens:
                    print(
                        f"Warning: Dataset exhausted after processing {pbar.n}/{num_tokens} tokens"
                    )
                break

            if num_tokens is not None and pbar.n >= num_tokens:
                print(f"Dataset processing complete. Processed {pbar.n} tokens")
                break

            raw_tokens = sum([len(ids) for ids in batch["input_ids"]])
            total_raw_tokens += raw_tokens

            num_samples += len(batch["input_ids"])

            batched_tokenizations: BatchedLZWTokenization = compressor_function(
                ids=batch["input_ids"],
                initial_vocab_size=config.initial_vocab_size,
                extra_vocab_size=config.extra_vocab_size,
                max_out_seq_length=config.seq_length
                + 1,  # +1 to account for the shift in the labels
                max_subtokens=config.compression.max_subtokens,
                pad_token_id=pad_token_id,
            )

            batched_compressed_ids = batched_tokenizations.get_padded_token_ids(
                config.seq_length + 1
            )
            codebooks = [codebook.pad() for codebook in batched_tokenizations.codebooks]
            codebook_utilization_per_sample.extend(
                [
                    codebook.codebook_stats["utilization"]
                    for codebook in batched_tokenizations.codebooks
                ]
            )
            code_sizes.extend(
                [
                    codebook.codebook_stats["mean_code_size"]
                    for codebook in batched_tokenizations.codebooks
                ]
            )

            all_hypertoken_activeness.append(
                batched_tokenizations.hypertoken_activeness
            )
            all_hypertoken_density.append(batched_tokenizations.hypertoken_density)

            num_lzw_tokens += batched_tokenizations.num_tokens

            # Add to current chunk
            processed_tokenizations.extend(batched_compressed_ids)
            processed_codebooks.extend(codebooks)

            pbar_update = len(batched_compressed_ids) * config.seq_length
            pbar.update(pbar_update)
            total_out_tokens += len(batched_compressed_ids) * config.seq_length

            # If chunk is full, save it and reset
            if len(processed_tokenizations) >= chunk_size:
                save_chunk(
                    dataset_path,
                    processed_tokenizations,
                    processed_codebooks,
                    chunk_idx,
                )
                chunk_idx += 1
                processed_tokenizations.clear()
                processed_codebooks.clear()

    # Save any remaining data
    if processed_tokenizations:
        save_chunk(
            dataset_path, processed_tokenizations, processed_codebooks, chunk_idx
        )
        chunk_idx += 1

    if num_tokens is not None and total_out_tokens < num_tokens:
        # print in red
        print(
            f"\033[91mWarning\033[0m: Dataset exhausted after processing {pbar.n}/{num_tokens} tokens"
        )

    metadata = {
        "saved_to": dataset_path,
        "num_data_chunk": str(chunk_idx),
        "num_token_target": str(num_tokens),
        "num_input_tokens": str(total_raw_tokens),
        "num_output_tokens": str(total_out_tokens),
        "num_lzw_tokens": str(num_lzw_tokens),
        "compression_rate": f"{num_lzw_tokens / total_raw_tokens:.2%}"
        if total_raw_tokens > 0
        else "N/A",
        "hypertoken_activeness": f"{sum(all_hypertoken_activeness) / len(all_hypertoken_activeness):.2%}",
        "hypertoken_density": f"{sum(all_hypertoken_density) / len(all_hypertoken_density):.2%}",
        "padding_efficiency": f"{num_lzw_tokens/total_out_tokens:.2%}",
        "codebook_utilization": f"{sum(codebook_utilization_per_sample) / len(codebook_utilization_per_sample):.2%}",
        "avg_code_length": f"{sum(code_sizes) / len(code_sizes):.2f}",
    }

    # Print structured metadata
    print("\nDataset Compression Summary")
    for key, value in metadata.items():
        print(f"{key:25}: {value:>12}")
    print("\n")


def save_chunk(
    dataset_path: str,
    compressed_ids: List[List[int]],
    codebooks: List[torch.Tensor],
    chunk_idx: int,
) -> None:
    # Convert just this chunk to tensors and save
    input_ids = torch.stack(
        [
            torch.tensor(ids)
            for ids in tqdm(compressed_ids, desc="concatenating input_ids")
        ]
    )
    codebook_tensor = torch.stack(
        [v for v in tqdm(codebooks, desc="concatenating codebooks")]
    )

    dataset_chunk_path = os.path.join(
        os.path.dirname(dataset_path), f"data_{chunk_idx}.safetensors"
    )

    save_file(
        {"input_ids": input_ids, "vocab": codebook_tensor},
        dataset_chunk_path,
    )


def load_chunk(dataset_path, chunk_idx, device):
    dataset_chunk_path = os.path.join(
        os.path.dirname(dataset_path), f"data_{chunk_idx}.safetensors"
    )
    return load_file(
        dataset_chunk_path,
        device=device,
    )


def load_data(dataset_path, device) -> Dict[str, torch.Tensor]:
    # if the dataset_path exists, load the data from the dataset_path, old format of a single file
    if os.path.exists(dataset_path):
        return load_file(dataset_path, device=device)
    # if the dataset_path does not exist, load the chunks and concatenate them, this is the new format
    else:
        dir_path = os.path.dirname(dataset_path)
        chunk_files = os.listdir(dir_path)
        if len(chunk_files) == 0:
            raise ValueError(f"No chunks found in {dir_path}")

        chunk_ids = sorted(
            [int(chunk_file.split("_")[-1].split(".")[0]) for chunk_file in chunk_files]
        )
        chunks: List[Dict[str, torch.Tensor]] = [
            load_chunk(dataset_path, chunk_id, device) for chunk_id in chunk_ids
        ]

        concat = {
            "input_ids": torch.cat([chunk["input_ids"] for chunk in chunks], dim=0),
            "vocab": torch.cat([chunk["vocab"] for chunk in chunks], dim=0),
        }

        return concat


def tokenize_and_compress_dataset(
    config: Config,
    num_tokens: int,
    batch_size: int,
    split: str = "train",
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_tokenizer_name_or_path)
    # set the padding token to the eos token if it is not set
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    dataset: IterableDataset = load_dataset(
        path=config.data.dataset_path,
        name=config.data.dataset_name,
        split=split,
        streaming=True,
    )

    dataset = dataset.map(
        lambda x: tokenizer(x[config.data.text_column], return_attention_mask=False),
        batched=True,
        remove_columns=[config.data.text_column],
    )

    dataset = dataset.shuffle()

    dataset_iter = iter(dataset.batch(batch_size))

    compress_dataset(
        config,
        dataset_iter,
        num_tokens,
        tokenizer.eos_token_id,  # type: ignore
        tokenizer.pad_token_id,  # type: ignore
        split,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--validation-tokens", type=int, default=None)
    args = parser.parse_args()

    config = dataclass_from_file(Config, args.config)

    if args.train_tokens is None:
        num_train_tokens = (
            config.total_batch_size * config.max_steps
            if config.max_steps is not None
            else None
        )
    else:
        num_train_tokens = args.train_tokens
    if args.validation_tokens is None:
        num_val_tokens = (
            (
                config.total_batch_size
                * (config.max_steps // config.val_interval)
                * config.val_steps
            )
            if config.max_steps is not None
            else None
        )
    else:
        num_val_tokens = args.validation_tokens

    # check if the dataset already exists
    data_hash = data_config_hash(config)
    dataset_path = f"{config.data.path}/{data_hash}"
    if os.path.exists(dataset_path):
        print(f"Dataset already exists in {dataset_path}, loading dataset...")
        # load the dataset
        train_loader = DataLoaderLite(
            config,
            "train",
            "cpu",
            0,
            1,
        )
        num_tokens = train_loader.samples["input_ids"].numel()
        print(
            f"Num train tokens in dataset: {num_tokens} VS target train tokens: {num_train_tokens}"
        )
        val_loader = DataLoaderLite(
            config,
            "validation",
            "cpu",
            0,
            1,
        )
        num_tokens = val_loader.samples["input_ids"].numel()
        print(
            f"Num val tokens in dataset: {num_tokens} VS target val tokens: {num_val_tokens}"
        )
        exit()

    tokenize_and_compress_dataset(
        config, num_train_tokens, args.batch_size, split="train"
    )
    tokenize_and_compress_dataset(
        config, num_val_tokens, args.batch_size, split="validation"
    )
