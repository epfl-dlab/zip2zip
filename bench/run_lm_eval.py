from datetime import datetime
import os
from typing import Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm
from lm_eval.api.model import TemplateLM
from typing import List, Dict, Tuple, Any
from lm_eval.api.instance import Instance
from huggingface_hub import snapshot_download
from torchao.float8 import convert_to_float8_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table
from lm_eval.models.huggingface import HFLM
from zip2zip.config import Zip2ZipConfig
from zip2zip.logging_utils import configure_logging
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.model import Zip2ZipModel


# create a type alias for Tuple[List[int]] called Batched

Pair = Tuple[List[int], List[int]]


class Zip2ZipForLMEval(TemplateLM):
    def __init__(
        self,
        zip2zip_model: Zip2ZipModel,
        zip2zip_tokenizer: Zip2ZipTokenizer,
        **model_kwargs,
    ):
        self.hf_model_for_lmeval = HFLM(
            pretrained=zip2zip_model, tokenizer=zip2zip_tokenizer, **model_kwargs
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            zip2zip_model.zip2zip_config.base_model_name_or_path
        )

    def __getattr__(self, name: str):
        return getattr(self.hf_model_for_lmeval, name)

    # add this to make python think we have implemented the abstract methods
    @property
    def eot_token_id(self):
        return self.hf_model_for_lmeval.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.hf_model_for_lmeval.tok_encode(string, **kwargs)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.hf_model_for_lmeval.tok_decode(tokens, skip_special_tokens)

    @torch.no_grad()
    def generate_until(self, *args, **kwargs):
        return self.hf_model_for_lmeval.generate_until(*args, **kwargs)

    # for perplexity
    @torch.no_grad()
    def loglikelihood_rolling(self, *args, **kwargs):
        self.hf_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        return self.hf_model_for_lmeval.loglikelihood_rolling(*args, **kwargs)

    # for discriminative tasks
    @torch.no_grad()
    def _loglikelihood_tokens(self, *args, **kwargs):
        self.hf_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        return self.hf_model_for_lmeval._loglikelihood_tokens(*args, **kwargs)

    @property
    def tokenizer_name(self):
        return self.hf_model_for_lmeval.tokenizer_name

    @property
    def tokenizer(self):
        return self.hf_model_for_lmeval.tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.hf_model_for_lmeval.apply_chat_template(*args, **kwargs)


def save_lm_eval_results_to_yaml(results: Dict[str, Any], filepath: str) -> None:
    """
    Safely save a nested results dictionary into a YAML file.

    Args:
        results (dict): The results dictionary you want to save.
        filepath (str): Path to the output YAML file.
    """
    # Optional: configure nice YAML formatting
    yaml_dump_settings = {
        "allow_unicode": True,  # allow non-ASCII (French accents etc.)
        "default_flow_style": False,  # use block style (prettier)
        "sort_keys": False,  # keep dictionary key order
    }
    import yaml

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(results, f, **yaml_dump_settings)
        print(f"Successfully saved results to {filepath}")
    except Exception as e:
        print(f"Error saving YAML: {e}")


if __name__ == "__main__":
    from lm_eval.evaluator import simple_evaluate

    model_name = "epfl-dlab/zip2zip-Llama-3.2-3B-Instruct-v0.1"
    # model_name = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"

    model = Zip2ZipModel.from_pretrained(
        model_name, device_map="cuda", torch_dtype=torch.float16
    )
    tokenizer = Zip2ZipTokenizer.from_pretrained(model_name)

    model = Zip2ZipForLMEval(
        model,
        tokenizer,
        # max_length=1024,
        batch_size=5,
    )

    # disable all parameters in the model
    for param in model.model.parameters():
        param.requires_grad = False

    results = simple_evaluate(
        model,
        # tasks="arc_easy",
        # ai2_arc,openbookqa,piqa,winogrande,commonsense_qa,lambada,mathqa,hellaswag
        # tasks=["openbookqa", "piqa", "winogrande", "commonsense_qa", "lambada", "mathqa", "hellaswag"],
        tasks=["wmt14-fr-en", "wmt14-en-fr"],
        limit=10,
        num_fewshot=2,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        confirm_run_unsafe_code=True,
    )

    if results is not None:
        print(make_table(results))

    if True:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = os.path.join("eleutherai_eval", current_time)
        os.makedirs(session_dir, exist_ok=False)
        samples = results["samples"]
        metadata = {k: v for k, v in results.items() if k != "samples"}

        for task_name, task_results in samples.items():
            save_lm_eval_results_to_yaml(
                results=task_results,
                filepath=os.path.join(session_dir, f"{task_name}.yaml"),
            )

        # save the metadata
        save_lm_eval_results_to_yaml(
            results=metadata,
            filepath=os.path.join(session_dir, f"metadata.yaml"),
        )
