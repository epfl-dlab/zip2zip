import os
from typing import Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from lm_eval.api.model import TemplateLM
from typing import List, Dict, Tuple, Any
from lm_eval.api.instance import Instance
from huggingface_hub import snapshot_download
from torchao.float8 import convert_to_float8_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table
from lm_eval.models.huggingface import HFLM
from zip2zip.config import Zip2ZipConfig
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.model import Zip2ZipModel


class Zip2ZipForLMEval(TemplateLM):
    def __init__(
        self, zip2zip_model: Zip2ZipModel, zip2zip_tokenizer: Zip2ZipTokenizer
    ):
        self.hf_model_for_lmeval = HFLM(
            pretrained=zip2zip_model, tokenizer=zip2zip_tokenizer
        )
        # self.pad_token_id = zip2zip_tokenizer.pad_token_id
        # self.device = zip2zip_model.device
        # self.dtype = zip2zip_model.dtype

    def __getattr__(self, name: str):
        return getattr(self.hf_model_for_lmeval, name)

    # add this to make python think we have implemented the abstract methods
    def eot_token_id(self):
        return self.hf_model_for_lmeval.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.hf_model_for_lmeval.tok_encode(string, **kwargs)

    def generate_until(self, *args, **kwargs):
        return self.hf_model_for_lmeval.generate_until(*args, **kwargs)

    # for perplexity
    def loglikelihood_rolling(self, *args, **kwargs):
        self.hf_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        return self.hf_model_for_lmeval.loglikelihood_rolling(*args, **kwargs)

    @property
    def tokenizer_name(self):
        return self.hf_model_for_lmeval.tokenizer_name

    @property
    def tokenizer(self):
        return self.hf_model_for_lmeval.tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.hf_model_for_lmeval.apply_chat_template(*args, **kwargs)

    # for discriminative tasks
    def _loglikelihood_tokens(self, *args, **kwargs):
        self.hf_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        return self.hf_model_for_lmeval._loglikelihood_tokens(*args, **kwargs)


if __name__ == "__main__":
    from lm_eval.evaluator import simple_evaluate
    from transformers import PreTrainedTokenizer

    model_name = "epfl-dlab/zip2zip-Llama-3.2-3B-Instruct-v0.1"

    model = Zip2ZipModel.from_pretrained(
        model_name, device_map="cuda", torch_dtype=torch.float16
    )
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_name)

    # assert isinstance(tokenizer, PreTrainedTokenizer)

    model = Zip2ZipForLMEval(
        model,
        tokenizer,
    )

    results = simple_evaluate(
        model,
        tasks="arc_easy",
        limit=10,
        batch_size=2,
        num_fewshot=2,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        confirm_run_unsafe_code=True,
    )

    if results is not None:
        print(make_table(results))
