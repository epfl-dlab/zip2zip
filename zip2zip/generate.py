import os, sys
import torch
from time import time
from safetensors import safe_open
from argparse import ArgumentParser
from zip2zip_compression import decode
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


from configs import Config
from paper_prompts import PROMPTS
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from utils import (
    get_device,
    unflatten,
    dataclass_from_dict,
    adapt_model,
    setup_seed,
    PLATFORM_BEST_DTYPE,
)


class TimeLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()
        self.timestamps = []

    def __call__(
        self, _: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.add_timestamp()
        return scores

    def add_timestamp(self):
        self.timestamps.append(time())


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="Write a MultiHeadAttention layer in PyTorch"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    messages = [
        {
            "role": "user",
            "content": args.prompt
            # "content":  PROMPTS[args.prompt_length],
        }
    ]

    setup_seed()
    device = get_device()
    torch.set_float32_matmul_precision("high")

    zip2zip_adapter_path = args.adapter

    metadata = {}
    adapter_state_dict = {}
    with safe_open(zip2zip_adapter_path, framework="pt", device=device) as f:
        for k in f.keys():
            adapter_state_dict[k.replace("_orig_mod.", "")] = f.get_tensor(k)

        for k, v in f.metadata().items():
            metadata[k] = v

    dict_config = unflatten(metadata)["config"]

    old_zip2zip_config = dataclass_from_dict(Config, dict_config)

    original_tokenizer = AutoTokenizer.from_pretrained(
        old_zip2zip_config.pretrained_tokenizer_name_or_path,
    )

    original_model = (
        AutoModelForCausalLM.from_pretrained(
            old_zip2zip_config.pretrained_model_name_or_path,
        )
        .to(device)
        .to(PLATFORM_BEST_DTYPE)
    )

    disabled_ids = list(original_tokenizer.get_added_vocab().values())

    new_zip2zip_model_config = old_zip2zip_config.to_zip2zip_config()

    # set disabled_ids to zip2zip_config
    if new_zip2zip_model_config.compression.disabled_ids is None:
        new_zip2zip_model_config.compression.disabled_ids = disabled_ids
    else:
        original_disabled_ids = new_zip2zip_model_config.compression.disabled_ids
        new_zip2zip_model_config.compression.disabled_ids.extend(disabled_ids)
        print(
            f"Adding {len(disabled_ids)} disabled ids to the {len(original_disabled_ids)} existing ones."
        )

    tokenizer = Zip2ZipTokenizer(original_tokenizer, new_zip2zip_model_config)

    model = Zip2ZipModel(new_zip2zip_model_config, model=original_model)

    adapt_model(model.base_hf_transformer, old_zip2zip_config, merge=False)

    model._load_zip2zip_adaptor_state_dict(adapter_state_dict)

    model.eval()
    # model.compile()

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    start_tokenize = time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokenize_time = time() - start_tokenize

    input_length = inputs["input_ids"].shape[1]

    time_logits_processor = TimeLogitsProcessor()
    time_logits_processor.add_timestamp()

    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=1.0,
        # top_p=0.9,
        max_new_tokens=args.max_tokens,
        cache_implementation="dynamic",
        logits_processor=[time_logits_processor],
    )

    time_logits_processor.add_timestamp()

    generated_outputs = outputs[:, input_length:]

    generated_outputs_length = generated_outputs.shape[1]

    tokenizer.color_decode(
        outputs[0].tolist(), model.codebook_manager.codebook_manager.codebook
    )

    prefill_time = (
        time_logits_processor.timestamps[1] - time_logits_processor.timestamps[0]
    )
    generation_time = (
        time_logits_processor.timestamps[-1] - time_logits_processor.timestamps[1]
    )

    prompt_tps = input_length / prefill_time
    generation_tps = generated_outputs_length / generation_time

    print("=" * 10)
    print("Compressed:")
    print(f"Prompt: {input_length} tokens, {prompt_tps:.3f} tokens-per-sec")
    print(
        f"Generation: {generated_outputs_length} tokens, {generation_tps:.3f} tokens-per-sec"
    )

    original_input_length = input_length

    original_inputs = decode(
        inputs["input_ids"][0].tolist(),
        old_zip2zip_config.initial_vocab_size,
        old_zip2zip_config.extra_vocab_size,
        old_zip2zip_config.compression.max_subtokens,
    )
    original_input_length = len(original_inputs)

    original_generated_outputs = decode(
        generated_outputs[0].tolist(),
        old_zip2zip_config.initial_vocab_size,
        old_zip2zip_config.extra_vocab_size,
        old_zip2zip_config.compression.max_subtokens,
    )

    original_generated_outputs_length = len(original_generated_outputs)
    print("=" * 10)
    print("Uncompressed:")
    print(
        f"Prompt: {original_input_length} tokens, {original_input_length * (prompt_tps / input_length):.3f} tokens-per-sec"
    )
    print(
        f"Generation: {original_generated_outputs_length} tokens, {original_generated_outputs_length * (generation_tps / generated_outputs_length):.3f} tokens-per-sec"
    )

    print("=" * 10)
    print(
        f"PT: {prefill_time:.3f}s, GT: {generation_time:.3f}s, TPS: {prefill_time + generation_time:.3f}, TT: {original_input_length / tokenize_time:.3f} tokens-per-sec"
    )
