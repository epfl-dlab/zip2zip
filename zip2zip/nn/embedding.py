import torch
from torch import nn
import torch.nn.functional as F

from zip2zip.codebook import CodebookManager
from zip2zip.nn.encoders.base import CodebookEmbeddingFn
from zip2zip.config import Zip2ZipConfig


class PreInitializedEmbedding(nn.Module):
    def __init__(
        self,
        initial_vocab_size: int,
        embedding_dim: int,
        weight: nn.Parameter,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.initial_vocab_size = initial_vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.pad_token_id = pad_token_id

    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        return F.embedding(input, self.weight, self.pad_token_id)


class HyperEmbedding(PreInitializedEmbedding):
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        embedding_dim: int,
        weight: nn.Parameter,
        callable_encoder: CodebookEmbeddingFn,
        codebook_manager: CodebookManager,
        pad_token_id: int,
    ) -> None:
        super().__init__(initial_vocab_size, embedding_dim, weight, pad_token_id)
        self.max_codebook_size = max_codebook_size

        self.callable_encoder = callable_encoder
        self.codebook_manager = codebook_manager

    def forward(self, input: torch.LongTensor) -> torch.Tensor:

        # TODO: we don't support batching for now, input is (B, S)
        hyper_embedding_weight = self.codebook_manager.fill_hyper_embedding_weight(
            input[0], self.weight, self.callable_encoder
        )  # hyper_embedding_weight is (C, d)

        base_token_mask = input < self.initial_vocab_size  # (B, S)
        hyper_token_mask = ~base_token_mask  # (B, S)
        base_input_ids = input * base_token_mask.long()  # (B, S)
        hyper_input_ids = (
            input - self.initial_vocab_size
        ) * hyper_token_mask.long()  # (B, S)

        base_embedding = super().forward(base_input_ids) * base_token_mask.unsqueeze(
            -1
        )  # (B, S, d)
        hyper_embedding = F.embedding(
            hyper_input_ids, hyper_embedding_weight
        ) * hyper_token_mask.unsqueeze(
            -1
        )  # (B, S, d)
        return base_embedding + hyper_embedding

    @staticmethod
    def from_embedding(
        embedding: nn.Embedding,
        initial_vocab_size: int,
        max_codebook_size: int,
        codebook_embedding_fn: CodebookEmbeddingFn,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> "HyperEmbedding":
        return HyperEmbedding(
            initial_vocab_size,
            max_codebook_size,
            embedding.weight.shape[1],
            embedding.weight,
            codebook_embedding_fn,
            codebook_manager,
            pad_token_id,
        )

    @classmethod
    def from_config(
        cls,
        embedding: nn.Embedding,
        config: Zip2ZipConfig,
        codebook_embedding_fn: CodebookEmbeddingFn,
        codebook_manager: CodebookManager,
    ) -> "HyperEmbedding":
        return cls.from_embedding(
            embedding,
            config.compression.initial_vocab_size,
            config.compression.max_codebook_size,
            codebook_embedding_fn,
            codebook_manager.pad_token_id,
            codebook_manager,
        )
