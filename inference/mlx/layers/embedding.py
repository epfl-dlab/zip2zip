import mlx.core as mx
import mlx.nn as nn

from inference.mlx.layers.encoder import Encoder
from inference.mlx.codebook import CodebookManager


class HyperEmbedding(nn.Module):
    def __init__(
        self,
        initial_vocab_size: int,
        embedding_dim: int,
        encoder: Encoder,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> None:
        super().__init__()
        self.weight = mx.random.normal(
            shape=(initial_vocab_size, embedding_dim),
            scale=embedding_dim**-0.5,
        )
        self.encoder = encoder
        self.pad_token_id = pad_token_id
        self.codebook_manager = codebook_manager
        self.initial_vocab_size = initial_vocab_size

    def __call__(self, x: mx.array) -> mx.array:
        hyper_weight = self.codebook_manager.get_hyper_embedding(
            x[0], self.weight, self.encoder
        )

        mask = x < self.initial_vocab_size

        bx = x * mask
        hx = (x - self.initial_vocab_size) * ~mask

        return self.weight[bx] * mask[:, :, None] + hyper_weight[hx] * ~mask[:, :, None]

    def as_linear(self, x: mx.array) -> mx.array:
        full_weight = mx.concatenate(
            [self.weight, self.codebook_manager.hyper_embedding]
        )
        return x @ full_weight.T

    @staticmethod
    def from_embedding(
        embedding: nn.Embedding,
        encoder: Encoder,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> "HyperEmbedding":
        hyper_embedding = HyperEmbedding(
            codebook_manager.initial_vocab_size,
            embedding.weight.shape[1],
            encoder,
            pad_token_id,
            codebook_manager,
        )
        hyper_embedding.weight = embedding.weight
        return hyper_embedding
