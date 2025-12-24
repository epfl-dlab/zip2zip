from __future__ import annotations

import torch
from typing import List, Optional
from transformers import AutoTokenizer
from compression import Codebook, CodebookManager as FastCodebookManager

from zip2zip_batch.config import Zip2ZipConfig
from zip2zip_batch.nn.encoders.base import BaseEncoder


class CodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        embedding_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        pad_token_id: int,
    ):
        self.dtype = dtype
        self.device = device
        self.pad_token_id = pad_token_id
        self.max_subtokens = max_subtokens
        self.embedding_dim = embedding_dim
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size

        self.codebook_manager = FastCodebookManager()

        self.updates = None
        self.updates_indices = None

        self.embedding_weights = None
        self.linear_weights = None

    def set_codebooks(self, codebooks: List[Codebook]) -> None:
        batch_size = len(codebooks)
        self.codebook_manager.set_codebooks(codebooks)

        self.embedding_weights = torch.zeros(
            batch_size,
            self.max_codebook_size,
            self.embedding_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.linear_weights = torch.zeros(
            batch_size,
            self.max_codebook_size,
            self.embedding_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def get_embedding_weights(
        self,
        ids: torch.LongTensor,
        base_weight: torch.Tensor,
        encoder: BaseEncoder,
    ) -> torch.Tensor:
        updates, updates_indices = self.codebook_manager.update_codebooks(ids.tolist())
        self.updates = torch.tensor(
            updates,
            device=self.device,
            dtype=torch.long,
        ).view(ids.shape[0], -1, self.max_subtokens)
        self.updates_indices = updates_indices

        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = encoder(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.embedding_weights[i, ui] = new_weights[i, :len(ui)]

        return self.embedding_weights

    def get_linear_weights(
        self, base_weight: torch.Tensor, encoder: BaseEncoder
    ) -> torch.Tensor:
        if any(len(ui) > 0 for ui in self.updates_indices):
            new_weights = encoder(self.updates, base_weight, self.pad_token_id)

            for i, ui in enumerate(self.updates_indices):
                self.linear_weights[i, ui] = new_weights[i, :len(ui)]

        return self.linear_weights

    def reset(self) -> None:
        self.updates = None
        self.updates_indices = None

        self.embedding_weights = None
        self.linear_weights = None

        self.codebook_manager.reset()

    def to(self, *args, **kwargs):
        self.embedding_weights = self.embedding_weights.to(*args, **kwargs)
        self.linear_weights = self.linear_weights.to(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Zip2ZipConfig,
        dtype: torch.dtype,
        device: torch.device,
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
            dtype=dtype,
            device=device,
            pad_token_id=pad_token_id,
        )
