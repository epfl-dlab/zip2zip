import os
import sys
import torch
import argparse
from time import time

from configs import Config
from utils import get_device
from model import OnlineZZModel
from generate import z2z_generate, GenerateConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--context-size", type=int, required=True)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    config = Config.from_file(args.config)

    # disable float8
    config.float8_base_model = False

    model = OnlineZZModel(config, get_device()).to(config.dtype).eval()
    compiled_model = torch.compile(model)

    print(f"device: {torch.cuda.get_device_name()}")

    # end2end latency
    prompt = model.tokenizer.decode(
        torch.randint(0, model.tokenizer.vocab_size, (args.context_size,))
    )

    # warmup
    for _ in range(5):
        z2z_generate(
            prompt,
            compiled_model,
            GenerateConfig(max_new_tokens=args.max_tokens, do_sample=False),
        )
        torch.cuda.synchronize()

    # benchmark
    start = time()
    for _ in range(10):
        z2z_generate(
            prompt,
            compiled_model,
            GenerateConfig(max_new_tokens=args.max_tokens, do_sample=False),
        )
        torch.cuda.synchronize()
    end = time()

    print(
        f"time taken to generate {args.max_tokens} tokens: {((end - start) / 10) * 1000} ms, {args.max_tokens / ((end - start) / 10)}t/s"
    )

    compiled_model = torch.compile(model)

    # prompt processing latency with and without additional encoder
    input_ids = torch.randint(
        0,
        config.initial_vocab_size + config.extra_vocab_size,
        (1, args.context_size),
        device=model.device,
    )
    codebook_tensor = torch.randint(
        0,
        config.initial_vocab_size,
        (1, config.extra_vocab_size, config.compression.max_subtokens),
        device=model.device,
    )

    # warmup
    for _ in range(10):
        compiled_model(input_ids, codebook_tensor)
        torch.cuda.synchronize()

    # benchmark
    start = time()
    for _ in range(100):
        compiled_model(input_ids, codebook_tensor)
        torch.cuda.synchronize()
    end = time()

    print(f"time taken to process prompt: {((end - start) / 100) * 1000} ms")

    # remove the extra vocab
    model.config.extra_vocab_size = 0
    model.hyper_embedding.extra_vocab_size = 0

    input_ids = torch.randint(
        0,
        config.initial_vocab_size,
        (1, args.context_size),
        device=model.device,
    )
    codebook_tensor = torch.randint(
        0,
        config.initial_vocab_size,
        (1, config.extra_vocab_size, config.compression.max_subtokens),
        device=model.device,
    )

    # warmup
    for _ in range(100):
        compiled_model(input_ids, codebook_tensor)
        torch.cuda.synchronize()

    # benchmark
    start = time()
    for _ in range(100):
        compiled_model(input_ids, codebook_tensor)
        torch.cuda.synchronize()
    end = time()

    print(
        f"time taken to process prompt without additional encoder: {((end - start) / 100) * 1000} ms"
    )
