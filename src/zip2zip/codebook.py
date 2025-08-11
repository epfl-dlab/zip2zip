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

        updates, updates_indices = self.internal_codebook_manager.update_codebooks(
            ids.tolist()
        )
        logger.debug(f"\n updates to codebooks: {updates}")

        self.updates = torch.tensor(
            updates,
            device=curr_device,
            dtype=torch.long,
        ).view(ids.shape[0], -1, self.max_subtokens)
        self.updates_indices = updates_indices

        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = encoder_fn(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.hyper_embedding_weight_cache[i, ui] = new_weights[i, : len(ui)]
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

        self.internal_codebook_manager.reset()

    @classmethod
    def from_config(
        cls,
        config: Zip2ZipConfig,
    ) -> CodebookManager:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        return cls(
            initial_vocab_size=config.compression.initial_vocab_size,
            max_codebook_size=config.compression.max_codebook_size,
            max_subtokens=config.compression.max_subtokens,
            embedding_dim=config.encoder.hidden_size,
            pad_token_id=pad_token_id,
            disabled_ids=config.compression.disabled_ids,
        )
