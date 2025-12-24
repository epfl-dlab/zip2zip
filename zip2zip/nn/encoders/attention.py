import torch
from torch import nn
import torch.nn.functional as F

from configs import Config

from zip2zip.config import CompressionConfig
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.nn.encoders.config import AttentionEncoderConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self, input_embeddings: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        B, M, _ = input_embeddings.size()

        queries = self.q_proj(input_embeddings)
        keys = self.k_proj(input_embeddings)
        values = self.v_proj(input_embeddings)
        output = F.scaled_dot_product_attention(
            queries.view(B, M, self.num_heads, -1).transpose(1, 2).contiguous(),
            keys.view(B, M, self.num_heads, -1).transpose(1, 2).contiguous(),
            values.view(B, M, self.num_heads, -1).transpose(1, 2).contiguous(),
            attn_mask=attn_mask.unsqueeze(1),
        )
        output = output.transpose(1, 2).contiguous().view(B, M, -1)
        return self.o_proj(output)


class AttentionEncoder(BaseEncoder[AttentionEncoderConfig]):
    def __init__(self, hidden_size: int, num_heads: int, max_subtokens: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.pos_embed = nn.Embedding(max_subtokens, self.hidden_size)
        self.attention = MultiHeadAttention(self.hidden_size, self.num_heads)

    def forward(
        self, codebook: torch.Tensor, embeddings: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        H, S = codebook.shape

        codebook_embeddings = F.embedding(
            codebook, embeddings, padding_idx=pad_token_id
        ) + self.pos_embed(torch.arange(S, device=embeddings.device))

        mask = codebook != pad_token_id
        mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        output = self.attention(codebook_embeddings, mask)
        return output.mean(dim=1)

    @classmethod
    def from_old_config(cls, old_config: Config) -> "AttentionEncoder":
        return cls(
            old_config.embedding_encoder.unsafe_config.get("hidden_size", 768),
            old_config.embedding_encoder.unsafe_config.get("num_heads", 12),
            old_config.compression.max_subtokens,
        )

    @classmethod
    def from_config(
        cls, config: AttentionEncoderConfig, compression_config: CompressionConfig
    ) -> "AttentionEncoder":
        return cls(
            config.hidden_size,
            config.num_heads,
            compression_config.max_subtokens,
        )
