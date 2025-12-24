"""
A few notations:


We use
- V to represent the (initial) vocabulary size,
- V_E to represent the extra vocabulary size. The total vocabulary size is V + V_E.
- M to represent the maximum number of subtokens in the extra vocabulary.
- Codebook_dict refers to the dictionary that maps subtokens to their indices in the extra vocabulary.
- Codebook_list refers to the list of list of subtokens in the extra vocabulary. The lists are padded to the shape of (V_E, M).
- Codebook_Tensor refers to the tensor that represents the codebook as a padded tensor in the shape of (B, V_E, M), where codebook_tensor[b, i] is the composition of i-th hyper token in the extra vocabulary for the b-th batch.

For the generation process:
- base_token_ids refers to the token ids that are only base tokens. This can refer to the subset of `lzw_token_ids` that are base tokens.
- hyper_token_ids refers to the token ids that are only hyper tokens. This can refer to the subset of `lzw_token_ids` that are hyper tokens.
- lzw_token_ids refers to the token ids that are compressed using LZW, which is a mixture of base tokens and hyper tokens.
- normal_token_ids refers to the token ids that are not compressed using LZW, which is only base tokens. This only refer to the input token ids without lzw or decompressed `lzw_token_ids`.
"""

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
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


from configs import Config
from utils import (
    adapt_model,
    adjust_kv_cache,
    dataclass_from_dict,
    dequantize_float8_training,
    get_device,
    pad_codebook,
    support_float8,
    unflatten,
)
from nn.unembedding import HyperUnembedding
from nn.embedding import HyperEmbedding
from nn.encoders import EmbeddingEncoder
from fast_compression import batch_lzw_compress, lzw_compress


class OnlineZZModel(nn.Module, TemplateLM):
    def __init__(self, config: Config, device: Union[str, torch.device]):
        nn.Module.__init__(self)
        TemplateLM.__init__(self)
        self.config = config
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            torch_dtype=config.dtype,
            device_map=device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_tokenizer_name_or_path
        )
        # set tokenizer pad token to eos token if it exists, e.g. GPT2
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
            if self.tokenizer.eos_token
            else self.tokenizer.pad_token
        )
        adapt_model(self.model, config)

        embedding: nn.Embedding = self.model.get_input_embeddings()
        self.pad_token_id = (
            embedding.padding_idx
            if embedding.padding_idx is not None
            else self.tokenizer.pad_token_id
        )

        if config.embedding_encoder.tie_embedding_encoder:
            hyper_encoder = EmbeddingEncoder.init(
                config.embedding_encoder.embedding_encoder_name, config=config
            )
            # set the cls token vector to the average of the base token embeddings
            hyper_encoder.cls_token_vector = nn.Parameter(
                torch.mean(
                    embedding.weight[: config.initial_vocab_size], dim=0
                ).detach()
            )

            embed_hyper_encoder = hyper_encoder
            unembed_hyper_encoder = hyper_encoder
        else:
            embed_hyper_encoder = EmbeddingEncoder.init(
                config.embedding_encoder.embedding_encoder_name, config=config
            )
            unembed_hyper_encoder = EmbeddingEncoder.init(
                config.embedding_encoder.embedding_encoder_name, config=config
            )

        self.hyper_embedding = (
            HyperEmbedding(
                initial_vocab_size=config.initial_vocab_size,
                extra_vocab_size=config.extra_vocab_size,
                embedding_dim=embedding.embedding_dim,
                weight=embedding.weight,
                embedding_encoder=embed_hyper_encoder,
                pad_token_id=self.pad_token_id,  # type: ignore
            )
            .to(device)
            .to(config.dtype)
        )

        self.hyper_lm_head = (
            HyperUnembedding.from_pretrained(
                pretrained_linear=self.model.lm_head,
                initial_vocab_size=config.initial_vocab_size,
                embedding_encoder=unembed_hyper_encoder,
                bias=self.model.lm_head.bias is not None,
                pad_token_id=self.pad_token_id,  # type: ignore
            )
            .to(device)
            .to(config.dtype)
        )

    def convert_to_float8_training(self, raise_error_if_not_supported: bool = False):
        """Convert model to float8 if configured and supported."""
        if support_float8():
            convert_to_float8_training(self.model)
        else:
            if raise_error_if_not_supported:
                raise ValueError("Float8 is not supported on this device.")

    @classmethod
    def load_pretrained(
        cls,
        adapter_path: str,
        hub_adapter: Optional[str] = None,
        extra_vocab_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "OnlineZZModel":
        device = get_device() if device is None else device
        if hub_adapter:
            local_folder_path = snapshot_download(
                repo_id=hub_adapter,
                allow_patterns=[adapter_path],
            )
            adapter_path = os.path.join(local_folder_path, adapter_path)

        # load config from checkpoint
        metadata = {}
        with safe_open(adapter_path, framework="pt", device=device) as f:
            for k, v in f.metadata().items():
                metadata[k] = v
        config = dataclass_from_dict(Config, unflatten(metadata)["config"])

        if extra_vocab_size is not None:
            if extra_vocab_size != config.extra_vocab_size:
                print(
                    f"\033[91mOverriding extra_vocab_size from pretrained config: {config.extra_vocab_size} to {extra_vocab_size}\033[0m"
                )
            config.extra_vocab_size = extra_vocab_size

        model = cls(config, device)

        ckp_config = model._load_checkpoint(adapter_path)
        return model

    def _load_checkpoint(self, adapter_path: str) -> Config:
        metadata = {}
        adapter_state_dict = {}
        with safe_open(adapter_path, framework="pt", device=self.device) as f:
            for k in f.keys():
                adapter_state_dict[k.replace("_orig_mod.", "")] = f.get_tensor(k)

            for k, v in f.metadata().items():
                metadata[k] = v
        ckp_config = dataclass_from_dict(Config, unflatten(metadata)["config"])

        self.load_state_dict(adapter_state_dict, strict=False)

        # in case the model was trained with float8, convert it back to float16 if needed
        if not (
            torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        ):
            print("Dequantizing float8 model if needed")
            self.model = dequantize_float8_training(
                self.model, dtype=ckp_config.dtype, device=self.device
            )
        self.to(self.config.dtype)

        return ckp_config

    def forward(
        self,
        input_ids: torch.Tensor,
        codebook_tensor: torch.Tensor,
        metadata: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, dict]:
        """
        input_ids: (B, S)
        codebook_tensor: (B, V_E, M)
        """
        B, S = input_ids.shape
        V_E, M = codebook_tensor.shape[1:] if codebook_tensor.numel() > 0 else (0, 0)
        V = self.config.initial_vocab_size

        if metadata.get("kv_cache", None) is None:
            kv_cache = {
                "past_input_ids": None,
                "past_key_values": None,
            }
        else:
            kv_cache = metadata["kv_cache"]

        new_input_ids, valid_kv_cache = adjust_kv_cache(
            kv_cache["past_key_values"], input_ids, kv_cache["past_input_ids"]
        )

        inputs_embeds, embedding_metadata = self.hyper_embedding(
            new_input_ids,
            codebook_tensor,
            metadata.get("embedding_metadata", {}),
        )

        # Determine whether to use 'model' or 'transformer'
        if hasattr(self.model, "model"):
            base_model = self.model.model  # Newer models
        elif hasattr(self.model, "transformer"):
            base_model = self.model.transformer  # Older models like GPT-2
        else:
            raise ValueError(
                "Unknown model structure. Neither 'model' nor 'transformer' found."
            )

        transformer_output = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=(input_ids != self.pad_token_id).to(self.config.dtype),
            past_key_values=valid_kv_cache,
            use_cache=True,
        )
        last_hidden_states = transformer_output.last_hidden_state

        kv_cache = {
            "past_input_ids": input_ids,
            "past_key_values": transformer_output.past_key_values,
        }

        raw_logits, lm_head_metadata = self.hyper_lm_head(
            last_hidden_states,
            codebook_tensor,
            metadata.get("lm_head_metadata", {}),
        )  # (B, S, V + V_E+ V_reserved)

        # if extra vocab is used, mask out the unused hypertokens from softmax
        if V_E > 0:
            V_output = raw_logits.shape[-1]

            unused_hypertoken_mask = torch.zeros((B, V_output), device=self.device)
            unused_hypertoken_mask[:, V : V + V_E] = (
                codebook_tensor == self.pad_token_id
            ).all(dim=-1)

            logits = raw_logits + unused_hypertoken_mask.unsqueeze(1) * -1e9
        else:
            logits = raw_logits

        logits = raw_logits

        metadata = {
            "embedding_metadata": embedding_metadata,
            "lm_head_metadata": lm_head_metadata,
            "kv_cache": kv_cache,
        }

        return logits, metadata

    def lzw_compress(self, input_ids: List[List[int]]) -> Tuple[torch.Tensor, dict]:
        if isinstance(input_ids[0], int):
            input_ids = [input_ids]
        # max_codebook_size = max(len(lzw_merge_dict) for _, lzw_merge_dict in codebook_pairs) TODO, see TODO below
        codebook_pairs: List[Tuple[List[int], dict]] = batch_lzw_compress(
            ids=input_ids,
            initial_vocab_size=self.config.initial_vocab_size,
            extra_vocab_size=self.config.extra_vocab_size,
            max_out_seq_length=self.config.seq_length,
            max_subtokens=self.config.compression.max_subtokens,
        )
        max_seq_length = max(
            len(compressed_ids) for compressed_ids, _ in codebook_pairs
        )

        batched_compressed_ids: List[List[int]] = []
        batched_codebooks: List[List[List[int]]] = []

        for compressed_ids, codebook_dict in codebook_pairs:
            padded_compressed_ids: List[int] = compressed_ids + [
                self.tokenizer.pad_token_id
            ] * (
                max_seq_length - len(compressed_ids)
            )  # +1 to account for the shift in the labels
            batched_compressed_ids.append(padded_compressed_ids)
            codebook_list, codebook_size, mean_hypertoken_size = pad_codebook(
                codebook_dict,
                self.config.initial_vocab_size,
                self.config.extra_vocab_size,  # also change this to len(lzw_merge_dict) TODO, together with TODO in Embedding.py
                self.config.compression.max_subtokens,
                self.tokenizer.pad_token_id,
            )
            batched_codebooks.append(codebook_list)
        input_ids = torch.tensor(batched_compressed_ids).to(self.device)  # (B, S)
        codebook_tensor = torch.tensor(batched_codebooks).to(
            self.device
        )  # (B, V_E, M) where V_E is the extra vocab size and M is the max subtokens size

        return input_ids, codebook_tensor

    def compute_all_hypertoken_embeddings(
        self, codebook_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        codebook_tensor: (B, V_E, M)
        """
        B, V_E, M = codebook_tensor.shape
        all_hypertoken_ids = torch.tensor(
            [
                self.config.initial_vocab_size + i
                for i in range(self.config.extra_vocab_size)
            ],
            device=self.device,
        ).unsqueeze(0)
        all_hypertoken_embeddings, metadata = self.hyper_embedding.forward(
            all_hypertoken_ids, codebook_tensor
        )  # (B, V_E, D)
        return all_hypertoken_embeddings, metadata

    def compute_codebook_embeddings(
        self, codebook_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        codebook_tensor: (B, V_E, M)
        """
        # Determine whether to use 'model' or 'transformer'
        if hasattr(self.model, "model"):
            base_model = self.model.model  # Newer models
        elif hasattr(self.model, "transformer"):
            base_model = self.model.transformer  # Older models like GPT-2
        else:
            raise ValueError(
                "Unknown model structure. Neither 'model' nor 'transformer' found."
            )
        return base_model.embed_tokens(codebook_tensor)  # shape (B, V_E, M, D)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, **kwargs)

    @torch.no_grad()
    def _loglikelihood_tokens(
        self, requests: List[Tuple[Tuple[str, str], List[int], List[int]]], **kwargs
    ) -> List[Tuple[float, bool]]:
        outputs = []

        for _, context_enc, continuation_enc in requests:
            lzw_token_ids, codebook_dict = lzw_compress(
                ids=context_enc + continuation_enc,
                initial_vocab_size=self.config.initial_vocab_size,
                extra_vocab_size=self.config.extra_vocab_size,
                max_out_seq_length=max(
                    len(context_enc + continuation_enc) - 1, self.config.seq_length
                ),
                max_subtokens=self.config.compression.max_subtokens,
            )[0]

            lzw_token_ids_tensor = torch.tensor(
                lzw_token_ids, device=self.device
            ).unsqueeze(0)

            context_lzw_token_ids, _ = lzw_compress(
                ids=context_enc,
                initial_vocab_size=self.config.initial_vocab_size,
                extra_vocab_size=self.config.extra_vocab_size,
                max_out_seq_length=max(len(context_enc), self.config.seq_length),
                max_subtokens=self.config.compression.max_subtokens,
            )[0]

            codebook_list, _, _ = pad_codebook(
                codebook_dict=codebook_dict,
                initial_vocab_size=self.config.initial_vocab_size,
                extra_vocab_size=self.config.extra_vocab_size,
                max_subtokens=self.config.compression.max_subtokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            continuation_enc_tensor = torch.tensor(
                lzw_token_ids[len(context_lzw_token_ids) :], device=self.device
            ).unsqueeze(0)

            codebook_tensor = torch.tensor(codebook_list, device=self.device).unsqueeze(
                0
            )

            logits, _ = self.forward(lzw_token_ids_tensor, codebook_tensor)
            log_probs = F.log_softmax(
                logits[:, len(context_lzw_token_ids) - 1 : -1, :], dim=-1
            )  # drop the last token
            log_likelihood = torch.gather(
                log_probs, 2, continuation_enc_tensor.unsqueeze(-1)
            ).squeeze(-1)

            greedy_tokens = torch.argmax(log_probs, dim=-1)
            is_greedy = torch.all(greedy_tokens == continuation_enc_tensor, dim=-1)

            outputs.append((log_likelihood.sum().item(), is_greedy.item()))

        return outputs

    @torch.no_grad()
    def loglikelihood_rolling(
        self, requests: List[Instance], _: bool = False
    ) -> List[float]:
        outputs = []

        for request in requests:
            context = request.args[0]
            context_enc = self.tok_encode(context)

            if not context_enc:
                outputs.append(0.0)
                continue

            total_log_prob = 0.0
            for prefix_tokens, pred_tokens in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    context_enc,
                    prefix_token=self.tokenizer.eos_token_id,
                    max_seq_len=self.config.seq_length,
                    context_len=1,
                ),
            ):
                # pred_tokens of shape (S, )
                lzw_token_ids, codebook_dict = lzw_compress(
                    ids=pred_tokens,
                    initial_vocab_size=self.config.initial_vocab_size,
                    extra_vocab_size=self.config.extra_vocab_size,
                    max_out_seq_length=max(len(pred_tokens), self.config.seq_length),
                    max_subtokens=self.config.compression.max_subtokens,
                )[0]

                codebook_list, _, _ = pad_codebook(
                    codebook_dict=codebook_dict,
                    initial_vocab_size=self.config.initial_vocab_size,
                    extra_vocab_size=self.config.extra_vocab_size,
                    max_subtokens=self.config.compression.max_subtokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                input_ids = torch.tensor(
                    prefix_tokens + lzw_token_ids, device=self.device
                ).unsqueeze(0)
                codebook_tensor = torch.tensor(
                    codebook_list, device=self.device
                ).unsqueeze(0)

                # logits: (B, S, V + V_E + V_reserved)
                logits, _ = self.forward(input_ids, codebook_tensor)
                log_probs = F.log_softmax(logits, dim=-1)

                # drop the prefix tokens, a single EOS token
                log_probs = log_probs[:, len(prefix_tokens) - 1 :, :]
                input_ids = input_ids[:, len(prefix_tokens) :]
                # drop the last token
                log_probs = log_probs[:, :-1, :]

                chunk_log_probs = torch.gather(
                    log_probs, 2, input_ids.unsqueeze(-1)
                ).squeeze(-1)

                total_log_prob += chunk_log_probs.sum().item()

            outputs.append(total_log_prob)

        return outputs

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], _: bool = False) -> List[str]:
        from generate import z2z_generate, GenerateConfig

        completions = []

        for request in requests:
            context, gen_kwargs = request.args

            gen_kwargs["until"] = ["\\n", "<|end|>", "\n"]

            # remap some arg names
            if gen_kwargs.get("max_gen_toks", None) is not None:
                gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_gen_toks")

            (
                full_text,
                full_lzw_token_ids,
                out_lzw_token_ids,
                codebook_dict,
                first_token_time,
            ) = z2z_generate(context, self, GenerateConfig(**gen_kwargs))
            completion = full_text  # strictly speaking, this is not the completion, but the full text
            completions.append(completion)

        return completions

    @property
    def tokenizer_name(self):
        return self._tokenizer_name

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated

    def torch_compile(self):
        self.model = torch.compile(self.model)
        self.hyper_embedding = torch.compile(self.hyper_embedding)
        self.hyper_lm_head = torch.compile(self.hyper_lm_head)
        return self
