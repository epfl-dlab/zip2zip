import torch
from torch import nn
import torch.nn.functional as F

from zip2zip.codebook import CodebookManager
from zip2zip.nn.encoders.base import CodebookEmbeddingFn
from zip2zip.config import Zip2ZipConfig


class HyperUnembedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_vocab_size: int,
        codebook_embedding_fn: CodebookEmbeddingFn,
        bias: bool,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> None:
        super().__init__()
        self.base_proj = nn.Linear(in_features, out_features, bias)

        self.pad_token_id = pad_token_id
        self.codebook_embedding_fn = codebook_embedding_fn
        self.codebook_manager = codebook_manager
        self.initial_vocab_size = initial_vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_proj(x)

        hyper_unembedding_weight = self.codebook_manager.fill_hyper_unembedding_weight(
            self.base_proj.weight, self.codebook_embedding_fn
        )

        hyper_logits = F.linear(x, hyper_unembedding_weight)
        hyper_mask = (
            torch.arange(end=hyper_logits.shape[-1], device=x.device)
            < self.codebook_manager.get_working_index()
        )

        return torch.cat(
            [
                base_logits[..., : self.initial_vocab_size],
                hyper_logits * hyper_mask,
                base_logits[..., self.initial_vocab_size :],
            ],
            dim=-1,
        )

    @staticmethod
    def from_unembedding(
        linear: nn.Linear,
        initial_vocab_size: int,
        codebook_embedding_fn: CodebookEmbeddingFn,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> "HyperUnembedding":
        with torch.device("meta"):
            hyper_unembedding = HyperUnembedding(
                linear.weight.shape[1],
                linear.weight.shape[0],
                initial_vocab_size,
                codebook_embedding_fn,
                linear.bias is not None,
                pad_token_id,
                codebook_manager,
            )
        hyper_unembedding.to_empty(device=linear.weight.device)

        hyper_unembedding.base_proj.weight = linear.weight
        if linear.bias is not None:
            hyper_unembedding.base_proj.bias = linear.bias

        return hyper_unembedding

    @classmethod
    def from_config(
        cls,
        linear: nn.Linear,
        config: Zip2ZipConfig,
        codebook_embedding_fn: CodebookEmbeddingFn,
        codebook_manager: CodebookManager,
    ) -> "HyperUnembedding":
        return cls.from_unembedding(
            linear,
            config.compression.initial_vocab_size,
            codebook_embedding_fn,
            codebook_manager.pad_token_id,
            codebook_manager,
        )
