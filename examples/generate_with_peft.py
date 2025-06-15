import os, sys
import torch
from time import time
from safetensors import safe_open
from argparse import ArgumentParser
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.utils import (
    get_device,
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
    parser.add_argument("--hub-url", type=str, required=True)
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

    # zip2zip_adapter_path = args.adapter

    hf_hub_url = args.hub_url

    from zip2zip.config import Zip2ZipConfig

    zip2zip_config = Zip2ZipConfig.from_pretrained(hf_hub_url)

    print(zip2zip_config)

    original_tokenizer = AutoTokenizer.from_pretrained(
        zip2zip_config.base_model_name_or_path,
    )

    original_model = (
        AutoModelForCausalLM.from_pretrained(
            zip2zip_config.base_model_name_or_path,
        )
        .to(device)
        .to(PLATFORM_BEST_DTYPE)
    )

    disabled_ids = list(original_tokenizer.get_added_vocab().values())

    model = Zip2ZipModel.from_pretrained(
        hf_hub_url,
        base_model=original_model,
        with_peft=True,
        use_4bit=True,
    )

    tokenizer = Zip2ZipTokenizer.from_pretrained(hf_hub_url)
    # tokenizer.set_disabled_ids(disabled_ids)
    # TODO, needs to set the disabled ids to the model as well

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

    input_codebooks = inputs.pop("codebooks")

    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=1.0,
        # top_p=0.9,
        max_new_tokens=args.max_tokens,
        # cache_implementation="dynamic",
        use_cache=True,
        logits_processor=[time_logits_processor],
        reset_codebook=False,
    )

    time_logits_processor.add_timestamp()

    generated_outputs = outputs[:, input_length:]

    generated_outputs_length = generated_outputs.shape[1]

    tokenizer.color_decode(
        outputs[0].tolist(), model.codebook_manager.codebook_manager.codebook
    )

    model.codebook_manager.reset()

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
