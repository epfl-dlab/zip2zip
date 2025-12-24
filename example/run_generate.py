import time
import os, sys
from argparse import ArgumentParser
from fast_compression import lzw_compress

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from visual import legacy_contrast_colorprint_tokens

from generate import z2z_generate, GenerateConfig
from interface import load_model
from model import OnlineZZModel
from utils import (
    setup_seed,
    describe_lzw,
    decompress,
)

if __name__ == "__main__":
    setup_seed()

    parser = ArgumentParser()
    parser.add_argument("--adapter", type=str, required=False)
    parser.add_argument("--hub-adapter", type=str, required=False)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument(
        "--demo",
        default=None,
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--extra-vocab-size", type=int, default=None)
    parser.add_argument("--compress-during-generation", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--pretrained-model", type=str, required=False)
    parser.add_argument("--use-new-lzw", action="store_true")
    parser.add_argument("--chat", action="store_true")
    args = parser.parse_args()

    args.demo = args.demo.lower() if args.demo is not None else None
    if args.demo == "rust":
        args.prompt = (
            open("fast_compression/src/lib.rs").read() + "Explain the above code"
        )
    elif args.demo == "python":
        args.prompt = (
            open("fast_compression/src/lib.rs").read()
            + "Rewrite the above code in Python"
        )
    elif args.demo == "python-completion":
        args.prompt = open("train.py").read()[:100]
    elif args.demo == "cpp":
        args.prompt = "#include <iostream>"
    elif args.demo == "transformer":
        args.prompt = "Implement a Transformer model in PyTorch"
    elif args.demo == "java":
        args.prompt = "Write a Java program for university course registration"
    elif args.demo == "eu":
        args.prompt = "Briefly explain the History of the EU"
    elif args.demo == "calc":
        args.prompt = "Calculate the sum of 1 to 100"
    elif args.demo == "epfl":
        args.prompt = "Compare the EPFL and ETH Zurich"
    elif args.demo == "french":
        args.prompt = "Expliquez l'histoire de la révolution française"
    elif args.demo == "chinese":
        args.prompt = "讲解中国的历史"
    else:
        if args.prompt is None:
            raise ValueError(
                f"Demo {args.demo} not found, pass it as a prompt with `--prompt`"
            )

    generate_config = GenerateConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=not args.disable_kv_cache,
        compress_during_generation=args.compress_during_generation,
        extra_vocab_size=args.extra_vocab_size,
        # until= ["<|end|>"],
    )

    model, tokenizer = load_model(args.pretrained_model, args.adapter, args.hub_adapter)

    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = args.prompt

    if not isinstance(model, OnlineZZModel):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=generate_config.max_new_tokens,
            min_new_tokens=generate_config.max_new_tokens,
            use_cache=True,
        )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(text)
        print(f"Generation time: {time.time() - start:.2f} seconds")

    else:
        model_config = model.config

        generate_config.extra_vocab_size = (
            generate_config.extra_vocab_size
            if generate_config.extra_vocab_size is not None
            else model_config.extra_vocab_size
        )

        start_time = time.time()

        (
            full_text,
            full_lzw_token_ids,
            out_lzw_token_ids,
            codebook_dict,
            _,
        ) = z2z_generate(
            prompt,
            model,
            generate_config,
            use_new_lzw=args.use_new_lzw,
        )  # N.B. the text, lzw_token_ids, normal_token_ids don't include the prompt, but the codebook_dict does include the hypertokens from the prompt

        gen_time = time.time() - start_time

        print(full_text)

        print("=" * 100)

        codebook_dict = {
            v: list(map(int, k.split(","))) for k, v in codebook_dict.items()
        }

        legacy_contrast_colorprint_tokens(
            full_lzw_token_ids,
            codebook_dict,
            model.tokenizer,
            color_scheme="finegrained",
        )

        print("-" * 20)

        # build full_unzipped_token_ids from full_lzw_token_ids and codebook_dict
        full_unzipped_token_ids = decompress(full_lzw_token_ids, codebook_dict)

        # check if the compression is standard lzw or not
        standard_lzw_token_ids, _codebook_dict = lzw_compress(
            ids=full_unzipped_token_ids,
            initial_vocab_size=model_config.initial_vocab_size,
            extra_vocab_size=generate_config.extra_vocab_size,
            max_out_seq_length=len(full_unzipped_token_ids) - 1,
            max_subtokens=model_config.compression.max_subtokens,
        )[0]

        codebook_dict = {
            v: list(map(int, k.split(","))) for k, v in _codebook_dict.items()
        }

        legacy_contrast_colorprint_tokens(
            standard_lzw_token_ids,
            codebook_dict,
            model.tokenizer,
            color_scheme="finegrained",
        )

        print("-" * 20)

        metadata = describe_lzw(
            full_lzw_token_ids,
            model_config.initial_vocab_size,
            standard_lzw_token_ids,
            hyper_vocab_size=generate_config.extra_vocab_size,
        )

        print(metadata)

        print(f"Generation time: {gen_time:.2f} seconds")
