from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


from zip2zip_batch.config import CompressionConfig
from zip2zip_batch.nn.encoders.base import BaseEncoder
from zip2zip_batch.nn.encoders.attention import SelfAttention
from zip2zip_batch.nn.encoders.config import TransformerEncoderConfig


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int
    ) -> None:
        super().__init__()

        self.mlp = MLP(hidden_size, intermediate_size)
        self.attention = SelfAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.post_mlp_layernorm = nn.LayerNorm(hidden_size)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.post_attention_layernorm(x + self.attention(x, mask))
        x = self.post_mlp_layernorm(x + self.mlp(x))
        return x


class TransformerEncoder(BaseEncoder[TransformerEncoderConfig]):
    def __init__(
        self,
        encoder_config: TransformerEncoderConfig,
        compression_config: CompressionConfig,
    ) -> None:
        super().__init__(encoder_config, compression_config)

        self.num_heads = self.encoder_config.num_heads
        self.hidden_size = self.encoder_config.hidden_size
        self.intermediate_size = self.encoder_config.intermediate_size
        self.num_hidden_layers = self.encoder_config.num_hidden_layers

        self.position_embeddings = nn.Parameter(
            torch.randn(self.compression_config.max_subtokens, self.hidden_size)
            * self.hidden_size**-0.5
        )

        self.layers = nn.ModuleList(
            [
                Layer(self.hidden_size, self.intermediate_size, self.num_heads)
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(
        self, codebook: torch.Tensor, embeddings: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        B, H, S = codebook.size()
        codebook = codebook.view(-1, S)
        x = (
            F.embedding(codebook, embeddings, padding_idx=pad_token_id)
            + self.position_embeddings
        )

        mask = codebook != pad_token_id
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        for layer in self.layers:
            x = layer(x, mask)

        return x.mean(dim=1).view(B, H, -1)
