from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from zip2zip.config import Zip2ZipConfig
from zip2zip.codebook import CodebookManager
from zip2zip.nn.encoders.base import BaseEncoder


class HyperEmbedding(nn.Embedding):
    def __init__(
        self,
        config: Zip2ZipConfig,
        encoder: BaseEncoder,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_vocab_size: int,
        codebook_manager: CodebookManager,
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, padding_idx, device=device, dtype=dtype
        )
        self.config = config
        self.encoder = encoder
        self.initial_vocab_size = initial_vocab_size
        self.codebook_manager = codebook_manager

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_token_mask = input < self.initial_vocab_size
        hyper_token_mask = ~base_token_mask
        base_input_ids = input * base_token_mask.long()
        hyper_input_ids = (input - self.initial_vocab_size) * hyper_token_mask.long()

        hyper_embedding_weights = self.codebook_manager.get_hyper_embedding_weights(
            input, self.weight, self.encoder
        )

        batch_offsets = torch.arange(
            input.size(0), device=input.device, dtype=torch.long
        ).unsqueeze(-1).expand_as(input) * hyper_embedding_weights.size(1)

        hyper_input_ids += batch_offsets
        base_embedding = super().forward(base_input_ids) * base_token_mask.unsqueeze(-1)
        hyper_embedding = F.embedding(
            hyper_input_ids, hyper_embedding_weights.view(-1, self.embedding_dim)
        ) * hyper_token_mask.unsqueeze(-1)

        return base_embedding + hyper_embedding

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        config: Zip2ZipConfig,
        encoder: BaseEncoder,
        codebook_manager: CodebookManager,
    ) -> HyperEmbedding:
        with torch.device("meta"):
            hyper_embedding = cls(
                config,
                encoder,
                embedding.num_embeddings,
                embedding.embedding_dim,
                embedding.padding_idx,
                embedding.weight.device,
                embedding.weight.dtype,
                config.compression.initial_vocab_size,
                codebook_manager,
            )
        hyper_embedding.to_empty(device=embedding.weight.device)
        hyper_embedding.weight = embedding.weight
        return hyper_embedding
