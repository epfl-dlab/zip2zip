"""
This module provides a unified interface for both zip2zip and HF models.
"""

from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import OnlineZZModel
from generate import z2z_generate, GenerateConfig
from utils import PLATFORM_BEST_DTYPE


def generate(
    model, tokenizer, input_text, generate_kwargs: dict, disable_kv_cache=False
):
    input_obj = tokenizer(
        input_text,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = input_obj.input_ids.cuda()
    attention_mask = input_obj.attention_mask.cuda()

    if isinstance(model, OnlineZZModel):
        (text, full_lzw_token_ids, out_lzw_token_ids, codebook_dict, _,) = z2z_generate(
            input_text,
            model,
            GenerateConfig(use_kv_cache=not disable_kv_cache, **generate_kwargs),
        )
        output_text = text[len(input_text) :]
    else:
        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
        )
        output_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1] :],
            skip_special_tokens=True,
            ignore_tokenization_space=True,
        )

    return output_text


def load_model(
    model_name_or_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    hub_adapter: Optional[str] = None,
):
    if model_name_or_path:
        print(f"Loading model from {model_name_or_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=PLATFORM_BEST_DTYPE,
            trust_remote_code=True,
        )
    else:
        if not adapter_path and not hub_adapter:
            raise ValueError("Adapter path or model name is required")
        print(f"Loading model from {adapter_path} ...")
        model = OnlineZZModel.load_pretrained(adapter_path, hub_adapter)
        tokenizer = model.tokenizer

    model.to(dtype=PLATFORM_BEST_DTYPE, device=model.device)

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
