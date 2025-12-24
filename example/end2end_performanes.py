import os
import sys
import torch
import random
import argparse
from time import time
from typing import List
from datasets import load_dataset
from transformers import PreTrainedTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs import Config
from utils import get_device
from model import OnlineZZModel
from generate import z2z_generate, GenerateConfig


LANGS = [
    "c",
    "cpp",
    "go",
    "html",
    "java",
    "javascript",
    "lua",
    "powershell",
    "python",
    "rust",
    "swift",
    "typescript",
]


def median(values: List[float]) -> float:
    if len(values) == 0:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def is_within_context_size_range(tokens: List[int], context_size: int) -> bool:
    return int(context_size * 0.9) <= len(tokens) <= int(context_size * 1.1)


def get_prompt(text: str, task: str, lang: str, tokenizer: PreTrainedTokenizer) -> str:
    messages = []
    if task == "splg":
        samples = [
            "Write a complete python program to create a neural network using PyTorch to solve the MNIST classification problem.",
            "Write a complete java program to create a UI application using JavaFX to display a ping pong game.",
            "Write a complete c++ program to create a web server using the socket library.",
            "Write a complete javascript program to create a todo list application using the React library.",
        ]
        messages.append({"role": "user", "content": random.choice(samples)})
    elif task == "lpsg":
        messages.append(
            {
                "role": "user",
                "content": f"Summarize the following code snippet in two sentences:\n\n{text}",
            }
        )
    elif task == "lplg":
        messages.append(
            {
                "role": "user",
                "content": f"Rewrite the following code snippet in {random.choice(list(set(LANGS) - {lang}))}:\n\n{text}",
            }
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--extra-vocab-size", type=int, default=None)
    parser.add_argument(
        "--task", type=str, choices=["splg", "lpsg", "lplg"]
    )  # {long or short} prompt and {long or short} generation (short prompt and short generation is useless)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    config = Config.from_file(args.config)

    model = OnlineZZModel(config, get_device()).to(config.dtype)
    compiled_model = torch.compile(model)

    print(f"device: {torch.cuda.get_device_name()}")

    dataset = load_dataset(
        path="devngho/the-stack-llm-annotations-v2",
        name="all",
        split="train",
        streaming=True,
    )

    dataset = dataset.filter(
        lambda x: is_within_context_size_range(
            model.tokenizer.tokenize(x["text"]), args.context_size
        )
        and x["lang"] in LANGS
    )

    # warm up
    for x in dataset.take(2):
        prompt = get_prompt(x["text"], args.task, x["lang"], model.tokenizer)
        z2z_generate(
            prompt,
            compiled_model,
            GenerateConfig(
                max_new_tokens=args.max_tokens, extra_vocab_size=args.extra_vocab_size
            ),
        )

    # benchmark
    times = []
    hyper_token_throughputs = []
    normal_token_throughputs = []
    first_token_latencies = []
    for x in dataset.skip(2).take(10):
        prompt = get_prompt(x["text"], args.task, x["lang"], model.tokenizer)

        start = time()
        (
            _,
            full_lzw_token_ids,
            out_lzw_tokens_ids,
            _,
            first_token_time,
        ) = z2z_generate(
            prompt,
            compiled_model,
            GenerateConfig(
                max_new_tokens=args.max_tokens, extra_vocab_size=args.extra_vocab_size
            ),
        )
        end = time()
        times.append(end - start)
        first_token_latencies.append(first_token_time - start)
        hyper_token_throughputs.append(len(out_lzw_tokens_ids) / (end - start))
        normal_token_throughputs.append(len(out_unzipped_token_ids) / (end - start))

    print(f"median generation time: {median(times) * 1000:.2f} ms")
    print(f"median first token latency: {median(first_token_latencies) * 1000:.2f} ms")
    print(f"median hyper token throughput: {median(hyper_token_throughputs):.2f} t/s")
    print(f"median normal token throughput: {median(normal_token_throughputs):.2f} t/s")
