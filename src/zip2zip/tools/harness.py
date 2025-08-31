from datetime import datetime
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import lm_eval
from lm_eval.api.model import TemplateLM
from typing import List, Dict, Any
from lm_eval.api.instance import Instance
from transformers import AutoTokenizer
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window, make_table
from lm_eval.models.huggingface import HFLM
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.model import Zip2ZipModel

# Global debug flag - set to True to enable debug prints
DEBUG = False


def debug_print(*args, **kwargs):
    """Print debug messages only when DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)


# verify that we are using the zip2zip fork of lm-evaluation-harness
if getattr(lm_eval, "__fork__", None) != "epfl-dlab/zip2zip_lm_eval":
    raise ValueError(
        "You are not using the zip2zip fork of lm-evaluation-harness. Please uninstall it using `pip uninstall lm-eval` and install it using `pip install git+https://github.com/epfl-dlab/zip2zip_lm_eval.git`"
    )


class Zip2ZipForLMEval(TemplateLM):
    def __init__(
        self,
        zip2zip_model: Zip2ZipModel,
        zip2zip_tokenizer: Zip2ZipTokenizer,
        **model_kwargs,
    ):
        self.zip2zip_model_for_lmeval = HFLM(
            pretrained=zip2zip_model, tokenizer=zip2zip_tokenizer, **model_kwargs
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            zip2zip_model.zip2zip_config.base_model_name_or_path
        )

    def __getattr__(self, name: str):
        return getattr(self.zip2zip_model_for_lmeval, name)

    # add this to make python think we have implemented the abstract methods
    @property
    def eot_token_id(self):
        return self.zip2zip_model_for_lmeval.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.zip2zip_model_for_lmeval.tok_encode(string, **kwargs)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.zip2zip_model_for_lmeval.tok_decode(tokens, skip_special_tokens)

    @torch.no_grad()
    def generate_until(self, *args, **kwargs):
        return self.zip2zip_model_for_lmeval.generate_until(*args, **kwargs)

    # for discriminative tasks
    @torch.no_grad()
    def _loglikelihood_tokens(self, *args, **kwargs):
        self.zip2zip_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        return self.zip2zip_model_for_lmeval._loglikelihood_tokens(*args, **kwargs)

    @property
    def tokenizer_name(self):
        return self.zip2zip_model_for_lmeval.tokenizer_name

    @property
    def tokenizer(self):
        return self.zip2zip_model_for_lmeval.tokenizer

    def apply_chat_template(self, *args, **kwargs):
        return self.zip2zip_model_for_lmeval.apply_chat_template(*args, **kwargs)

    # for perplexity
    @torch.no_grad()
    def loglikelihood_rolling(
        self, requests: List[Instance], _: bool = False
    ) -> List[float]:
        self.zip2zip_model_for_lmeval.model.clear_zip2zip_cache_after_forward = True
        outputs = []

        for request in tqdm(requests, desc="Running loglikelihood requests"):
            context = request.args[0]
            context_enc = self._original_tokenizer.encode(context)

            if not context_enc:
                outputs.append(0.0)
                continue

            total_log_prob = 0.0
            for prefix_tokens, pred_tokens in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    context_enc,
                    prefix_token=self.tokenizer.eos_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            ):
                prefix_tokens = [
                    self.tokenizer.eos_token_id
                ]  # TODO, we are throwing away the prefix token, which would cause different from original HF model
                lzw_token_ids, attention_mask, codebook = self.tokenizer._lzw_encode(
                    [prefix_tokens + pred_tokens], padding=False
                )[0]

                lzw_prefix_length = len(
                    self.tokenizer._lzw_encode([prefix_tokens], padding=False)[0][0]
                )

                input_ids = torch.tensor(lzw_token_ids, device=self.device).unsqueeze(0)
                # attention_mask = torch.tensor(attention_mask, device=self.device).unsqueeze(0)
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

                # logits: (B, S, V + V_E + V_reserved)
                out = self.zip2zip_model_for_lmeval.model.forward(
                    input_ids, attention_mask=attention_mask
                )
                log_probs = F.log_softmax(out.logits, dim=-1)

                # drop the last token
                log_probs = log_probs[:, lzw_prefix_length - 1 : -1, :]
                # drop the first token from the input_ids
                labels = input_ids[:, lzw_prefix_length:]

                chunk_log_probs = torch.gather(
                    log_probs, 2, labels.unsqueeze(-1)
                ).squeeze(-1)

                total_log_prob += chunk_log_probs.sum().item()
                debug_print(f"chunk_log_probs: {chunk_log_probs.sum().item()}")

            outputs.append(total_log_prob)

        return outputs


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
