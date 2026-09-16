from __future__ import annotations

import torch
import logging
from typing import List, Optional
from transformers import AutoTokenizer
from zip2zip_compression import CompressionConfig
from zip2zip_compression import Codebook, CodebookManager as RustCodebookManager

from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.encoders.base import EncoderFn

logger = logging.getLogger(__name__)


class CodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        embedding_dim: int,
        pad_token_id: int,
        disabled_ids: set[int] = set(),
    ):
        self.pad_token_id = pad_token_id
        self.max_subtokens = max_subtokens
        self.embedding_dim = embedding_dim
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size

        self.internal_codebook_manager = RustCodebookManager(
            config=CompressionConfig(
                initial_vocab_size=initial_vocab_size,
                max_codebook_size=max_codebook_size,
                max_subtokens=max_subtokens,
                pad_token_id=pad_token_id,
                disabled_ids=disabled_ids,
            )
        )

        self.updates = None
        self.updates_indices = None

        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None

        self.runtime_batch_size = None
        self.hyper_token_spans = None
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False

    def prepare_input_ids(
        self,
        ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        """Install codebook updates and compute zip2zip++ RoPE positions.

        The model-side codebook can be reconstructed from the compressed token
        stream. Keeping span tracking here makes prompt prefill and cached
        generation use the same state transitions as hyper-embedding creation.
        """
        if ids.ndim != 2:
            raise ValueError(f"input_ids must be rank 2, got shape {tuple(ids.shape)}")

        batch_size, _ = ids.shape
        if self.runtime_batch_size is not None and self.runtime_batch_size != batch_size:
            raise ValueError(
                f"codebook state has batch size {self.runtime_batch_size}, received "
                f"{batch_size}; reset the model between unrelated batches"
            )
        self.runtime_batch_size = batch_size

        updates, updates_indices = self.internal_codebook_manager.update_codebooks(
            ids.tolist()
        )
        self.updates = torch.tensor(
            updates, device=ids.device, dtype=torch.long
        ).view(batch_size, -1, self.max_subtokens)
        self.updates_indices = updates_indices

        if self.hyper_token_spans is None:
            self.hyper_token_spans = torch.zeros(
                batch_size,
                self.max_codebook_size,
                device=ids.device,
                dtype=torch.long,
            )
        else:
            self.hyper_token_spans = self.hyper_token_spans.to(ids.device)

        for batch_idx, indices in enumerate(updates_indices):
            if indices:
                rows = self.updates[batch_idx, : len(indices)]
                spans = (rows != self.pad_token_id).sum(dim=-1).clamp_min(1)
                self.hyper_token_spans[batch_idx, indices] = spans

        is_hyper = ids >= self.initial_vocab_size
        spans = torch.ones_like(ids)
        if is_hyper.any():
            entry_ids = (ids - self.initial_vocab_size).clamp_min(0)
            if int(entry_ids[is_hyper].max()) >= self.max_codebook_size:
                raise ValueError("input_ids contain a hypertoken outside the codebook")
            hyper_spans = self.hyper_token_spans.gather(1, entry_ids)
            if (hyper_spans[is_hyper] == 0).any():
                raise ValueError("input_ids reference a hypertoken before it is installed")
            spans = torch.where(is_hyper, hyper_spans, spans)

        if attention_mask is not None:
            if attention_mask.shape != ids.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match "
                    f"new input_ids shape {tuple(ids.shape)}"
                )
            valid = attention_mask.to(device=ids.device, dtype=torch.bool)
            spans = torch.where(valid, spans, torch.zeros_like(spans))
        else:
            valid = torch.ones_like(ids, dtype=torch.bool)

        if self.base_position_offset is None:
            self.base_position_offset = torch.zeros(
                batch_size, 1, device=ids.device, dtype=torch.long
            )
        else:
            self.base_position_offset = self.base_position_offset.to(ids.device)

        positions = self.base_position_offset + spans.cumsum(dim=-1) - 1
        positions = torch.where(valid, positions, torch.zeros_like(positions))
        self.base_position_offset = self.base_position_offset + spans.sum(
            dim=-1, keepdim=True
        )
        self.position_ids = positions
        self._prepared_for_embedding = True
        return positions


    def init_codebooks_and_hyper_weight_cache(
        self, batch_size: int, codebooks: Optional[List[Codebook]] = None
    ) -> None:
        if codebooks is not None:
            self.internal_codebook_manager.set_codebooks(codebooks)

    def get_hyper_embedding_weights(
        self,
        ids: torch.LongTensor,
        base_weight: torch.Tensor,
        encoder_fn: EncoderFn,
    ) -> torch.Tensor:
        curr_device = base_weight.device
        dtype = base_weight.dtype
        if self.hyper_embedding_weight_cache is None:
            self.runtime_batch_size = ids.shape[0]
            self.hyper_embedding_weight_cache = torch.zeros(
                self.runtime_batch_size,
                self.max_codebook_size,
                self.embedding_dim,
                dtype=dtype,
                device=curr_device,
            )
        else:
            self.hyper_embedding_weight_cache = self.hyper_embedding_weight_cache.to(
                curr_device
            ).to(dtype)

        if not self._prepared_for_embedding:
            self.prepare_input_ids(ids)
        self.updates = self.updates.to(curr_device)
        logger.debug(f"\n codebook update indices: {self.updates_indices}")

        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.hyper_embedding_weight_cache[i, ui] = new_weights[i, : len(ui)]
        self._prepared_for_embedding = False
        return self.hyper_embedding_weight_cache

    def get_hyper_linear_weights(
        self, base_weight: torch.Tensor, encoder_fn: EncoderFn
    ) -> torch.Tensor:
        curr_device = base_weight.device
        dtype = base_weight.dtype
        if self.hyper_linear_weight_cache is None:
            assert self.runtime_batch_size is not None, "Runtime batch size is not set"
            self.hyper_linear_weight_cache = torch.zeros(
                self.runtime_batch_size,
                self.max_codebook_size,
                self.embedding_dim,
                dtype=dtype,
                device=curr_device,
            )
        else:
            self.hyper_linear_weight_cache = self.hyper_linear_weight_cache.to(
                curr_device
            ).to(dtype)

        # move to the correct device if needed
        self.updates = self.updates.to(curr_device)

        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.hyper_linear_weight_cache[i, ui] = new_weights[i, : len(ui)]

        return self.hyper_linear_weight_cache

    def reset(self) -> None:
        self.updates = None
        self.updates_indices = None

        self.hyper_embedding_weight_cache = None
        self.hyper_linear_weight_cache = None
        self.runtime_batch_size = None
        self.hyper_token_spans = None
        self.base_position_offset = None
        self.position_ids = None
        self._prepared_for_embedding = False

        self.internal_codebook_manager.reset()

    @classmethod
    def from_config(
        cls,
        config: Zip2ZipConfig,
        tokenizer_kwargs: Optional[dict] = None,
    ) -> CodebookManager:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, **(tokenizer_kwargs or {})
        )
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        return cls(
            initial_vocab_size=config.compression.initial_vocab_size,
            max_codebook_size=config.compression.max_codebook_size,
            max_subtokens=config.compression.max_subtokens,
            embedding_dim=getattr(config.encoder, "model_hidden_size", None) or config.encoder.hidden_size,
            pad_token_id=pad_token_id,
            disabled_ids=config.compression.disabled_ids,
        )
