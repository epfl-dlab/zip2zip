import torch
from torch import nn
from typing import Callable, Optional
import torch.nn.functional as F

from configs import Config

from zip2zip.config import CompressionConfig
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.nn.encoders.attention import MultiHeadAttention
from zip2zip.nn.encoders.config import TransformerEncoderConfig


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
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.post_mlp_layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_output = self.attention(hidden_states, attn_mask)
        hidden_states = self.post_attention_layernorm(hidden_states + attn_output)
        mlp_output = self.mlp(hidden_states)
        return self.post_mlp_layernorm(hidden_states + mlp_output)


class TransformerEncoder(BaseEncoder[TransformerEncoderConfig]):
    def __init__(
        self,
        encoder_config: TransformerEncoderConfig,
        compression_config: CompressionConfig,
    ) -> None:
        super().__init__(encoder_config, compression_config)

        self.hidden_size = encoder_config.hidden_size
        self.num_hidden_layers = encoder_config.num_hidden_layers
        self.max_subtokens = compression_config.max_subtokens
        self.intermediate_size = (
            encoder_config.intermediate_size
            if encoder_config.intermediate_size is not None
            else 4 * encoder_config.hidden_size
        )
        self.num_heads = encoder_config.num_heads
        self.position_encoding = encoder_config.position_encoding

        if self.position_encoding == "learnable":
            self.pos_embed = nn.Embedding(self.max_subtokens, self.hidden_size)
        elif self.position_encoding is None:
            self.pos_embed = None
        else:
            raise ValueError(f"Invalid position encoding: {self.position_encoding}")

        self.layers = nn.ModuleList(
            [
                Layer(self.hidden_size, self.intermediate_size, self.num_heads)
                for _ in range(self.num_hidden_layers)
            ]
        )

    def forward(
        self, codebook: torch.Tensor, embeddings: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        extra_vocab_size, max_subtokens = codebook.size()
        if self.pos_embed is not None:
            x = embeddings[codebook] + self.pos_embed(
                torch.arange(max_subtokens, device=embeddings.device)
            )
        else:
            x = embeddings[codebook]

        mask = codebook != pad_token_id
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)

        for layer in self.layers:
            x = layer(x, mask)
        return x.mean(dim=1)

    @classmethod
    def from_old_config(cls, old_config: Config) -> "TransformerEncoder":
        return cls(
            old_config.embedding_encoder["hidden_size"],
            old_config.embedding_encoder.unsafe_config["num_hidden_layers"],
            old_config.compression.max_subtokens,
            old_config.embedding_encoder.unsafe_config.get("num_heads", None),
            old_config.embedding_encoder.unsafe_config.get("intermediate_size", None),
            old_config.embedding_encoder.get("position_encoding", None),
        )
