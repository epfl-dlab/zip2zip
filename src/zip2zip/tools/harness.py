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

DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def _patch_lm_eval():
    """Monkey-patch lm-eval to support Zip2ZipTokenizer and LZW-compressed sequences.

    Replaces 3 behaviors from the epfl-dlab/zip2zip_lm_eval fork:
    1. Skip tokenizer isinstance check in HFLM (Zip2ZipTokenizer is not a PreTrainedTokenizer)
    2. Fix generate_until: decode full output and strip input text instead of token slicing
       (LZW compression shifts token boundaries so slicing by token count is wrong)
    3. Fix MultiTokenEOSCriteria: same decode-and-strip approach for stop sequence detection
    """
    import lm_eval.models.huggingface as hf_module
    import lm_eval.models.utils as utils_module

    # 1. Patch HFLM._create_tokenizer to skip tokenizer type check
    _orig_create_tokenizer = HFLM._create_tokenizer

    def _patched_create_tokenizer(self, pretrained, tokenizer, *args, **kwargs):
        if isinstance(tokenizer, Zip2ZipTokenizer):
            self.tokenizer = tokenizer
            return None
        return _orig_create_tokenizer(self, pretrained, tokenizer, *args, **kwargs)

    HFLM._create_tokenizer = _patched_create_tokenizer

    # 2. Patch generate_until to use text-based prompt stripping
    _orig_generate_until = HFLM.generate_until

    def _patched_generate_until(self, requests, disable_tqdm=False):
        import copy
        if not isinstance(self.tokenizer, Zip2ZipTokenizer):
            return _orig_generate_until(self, requests, disable_tqdm=disable_tqdm)

        # For Zip2ZipTokenizer, we need to intercept the decoding step.
        # The simplest approach: call original but patch tok_decode to handle full sequence
        return _orig_generate_until(self, requests, disable_tqdm=disable_tqdm)

    # Actually, the cleaner patch is on the _model_generate + decode path.
    # Let's patch the specific decode logic in generate_until's post-processing.
    # The issue is in generate_until where it does cont_toks[context_enc.shape[1]:]
    # We override _model_generate to store input lengths, then fix decoding.

    # Simpler: just override generate_until entirely for Zip2ZipTokenizer
    def _zip2zip_generate_until(self, requests, disable_tqdm=False):
        if not isinstance(getattr(self, 'tokenizer', None), Zip2ZipTokenizer):
            return _orig_generate_until(self, requests, disable_tqdm=disable_tqdm)

        res = []
        for req in tqdm(requests, desc="generate_until", disable=disable_tqdm or len(requests) < 4):
            context = req.args[0]
            until = req.args[1].get("until", []) if len(req.args) > 1 and req.args[1] else []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int((req.args[1] or {}).get("max_gen_toks", self.max_gen_toks)) if len(req.args) > 1 else self.max_gen_toks
            do_sample = bool((req.args[1] or {}).get("do_sample", False)) if len(req.args) > 1 else False

            context_enc = self.tok_encode(context)
            if len(context_enc) > self.max_length - max_gen_toks:
                context_enc = context_enc[-(self.max_length - max_gen_toks):]

            input_ids = torch.tensor([context_enc], device=self.device)
            input_text = self.tok_decode(context_enc)

            gen_kwargs = {"do_sample": do_sample}
            max_length = len(context_enc) + max_gen_toks

            cont = self._model_generate(input_ids, max_length=max_length, stop=until, **gen_kwargs)

            full_text = self.tok_decode(cont[0].tolist())
            if full_text.startswith(input_text):
                s = full_text[len(input_text):]
            else:
                s = self.tok_decode(cont[0].tolist()[len(context_enc):])

            for term in until:
                if term in s:
                    s = s[:s.index(term)]
                    break

            res.append(s)
        return res

    HFLM.generate_until = _zip2zip_generate_until

    # 3. Patch MultiTokenEOSCriteria to use text-based comparison
    try:
        _OrigEOS = utils_module.MultiTokenEOSCriteria
    except AttributeError:
        import lm_eval.models.utils_hf as utils_hf_module
        _OrigEOS = utils_hf_module.MultiTokenEOSCriteria
    _orig_eos_call = _OrigEOS.__call__

    def _patched_eos_call(self, input_ids, scores, **kwargs):
        initial_text = self.tokenizer.batch_decode(
            input_ids[:, :self.initial_decoder_input_length]
        )
        full_text = self.tokenizer.batch_decode(input_ids)

        generated_text = []
        for init, full in zip(initial_text, full_text):
            if full.startswith(init):
                generated_text.append(full[len(init):])
            else:
                generated_text.append(full)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in generated_text[i]
        return False not in self.done_tracker

    _OrigEOS.__call__ = _patched_eos_call


_patch_lm_eval()


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
                prefix_tokens = [self.tokenizer.eos_token_id]
                lzw_token_ids, attention_mask, codebook = self.tokenizer._lzw_encode(
                    [prefix_tokens + pred_tokens], padding=False
                )[0]

                lzw_prefix_length = len(
                    self.tokenizer._lzw_encode([prefix_tokens], padding=False)[0][0]
                )

                input_ids = torch.tensor(lzw_token_ids, device=self.device).unsqueeze(0)
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

                out = self.zip2zip_model_for_lmeval.model.forward(
                    input_ids, attention_mask=attention_mask
                )
                log_probs = F.log_softmax(out.logits, dim=-1)

                log_probs = log_probs[:, lzw_prefix_length - 1 : -1, :]
                labels = input_ids[:, lzw_prefix_length:]

                chunk_log_probs = torch.gather(
                    log_probs, 2, labels.unsqueeze(-1)
                ).squeeze(-1)

                total_log_prob += chunk_log_probs.sum().item()
                debug_print(f"chunk_log_probs: {chunk_log_probs.sum().item()}")

            outputs.append(total_log_prob)

        return outputs


def save_lm_eval_results_to_yaml(results: Dict[str, Any], filepath: str) -> None:
    yaml_dump_settings = {
        "allow_unicode": True,
        "default_flow_style": False,
        "sort_keys": False,
    }
    import yaml

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(results, f, **yaml_dump_settings)
        print(f"Successfully saved results to {filepath}")
    except Exception as e:
        print(f"Error saving YAML: {e}")
