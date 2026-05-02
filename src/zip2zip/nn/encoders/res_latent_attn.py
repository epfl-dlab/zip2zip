from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zip2zip.config import CompressionConfig
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.nn.encoders.config import ResLatentAttnConfig


class ResLatentAttnLayer(nn.Module):
    """Pre-norm transformer layer with weight keys matching zip2zip-core exactly.

    Key layout: wq/wk/wv/wo (attention), w1/w2 (FFN), norm1/norm2 (pre-norm LN).
    This allows direct state_dict transfer from a zip2zip-core checkpoint.
    """

    def __init__(self, dim: int, intermediate_size: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, S, D = x.shape

        # Pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)
        q = self.wq(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = mask.unsqueeze(1).unsqueeze(2).float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        if causal:
            causal_mask = (
                torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attn_mask = attn_mask + causal_mask

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.wo(attn_out)

        # Pre-norm FFN
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.w2(F.gelu(self.w1(x_norm)))

        return x


class ResLatentAttnEncoder(BaseEncoder[ResLatentAttnConfig]):
    """Residual latent-space attention encoder matching zip2zip-core's HyperEncoder layout.

    Differences vs TransformerEncoder:
    - Pre-LayerNorm (zip2zip-core style) instead of post-norm
    - Final layer norm (self.norm) after all transformer layers
    - Optional proj_in / proj_out when hidden_size != model_hidden_size (latent space)
    - Optional first-token residual: output = first_token_embed + encoder_delta
      (proj_out is zero-inited so the model starts as identity on the first token)
    - Causal (last-token) pooling option in addition to mean pooling

    Weight keys mirror zip2zip-core exactly so a checkpoint produced by zip2zip-core
    can be loaded with a direct state_dict copy (no key remapping needed).
    """

    def __init__(
        self,
        encoder_config: ResLatentAttnConfig,
        compression_config: CompressionConfig,
    ) -> None:
        super().__init__(encoder_config, compression_config)

        self.hidden_size = encoder_config.hidden_size
        model_dim = encoder_config.model_hidden_size or self.hidden_size
        self.model_hidden_size = model_dim
        self.causal = encoder_config.causal
        self.residual = encoder_config.residual

        intermediate_size = (
            encoder_config.intermediate_size
            if encoder_config.intermediate_size is not None
            else 4 * self.hidden_size
        )

        if model_dim != self.hidden_size:
            self.proj_in = nn.Linear(model_dim, self.hidden_size, bias=False)
            self.proj_out = nn.Linear(self.hidden_size, model_dim, bias=False)
        else:
            self.proj_in = None
            self.proj_out = None

        self.pos_embed = nn.Embedding(compression_config.max_subtokens, self.hidden_size)
        self.layers = nn.ModuleList([
            ResLatentAttnLayer(self.hidden_size, intermediate_size, encoder_config.num_heads)
            for _ in range(encoder_config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(self.hidden_size)

        # Zero-init proj_out so the encoder starts as a near-identity (residual warmup).
        # Matches zip2zip-core's init_weights() behaviour.
        if self.residual and self.proj_out is not None:
            nn.init.zeros_(self.proj_out.weight)

    def forward(
        self,
        codebook: torch.Tensor,
        embeddings: torch.Tensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Args:
            codebook:   (B, H, S) base-token IDs for each hypertoken entry
            embeddings: (vocab_size, model_dim) embedding weight matrix
            pad_token_id: padding token ID used to build the valid-token mask
        Returns:
            (B, H, model_dim) encoded hypertoken embeddings
        """
        B, H, S = codebook.size()
        cb = codebook.view(-1, S)  # (B*H, S)

        x = F.embedding(cb, embeddings, padding_idx=pad_token_id)  # (B*H, S, model_dim)

        if self.residual:
            first_tok = x[:, 0, :].clone()  # (B*H, model_dim) — saved before projection

        if self.proj_in is not None:
            x = self.proj_in(x)

        pos = torch.arange(S, device=x.device)
        x = x + self.pos_embed(pos)

        mask = cb != pad_token_id  # (B*H, S)

        for layer in self.layers:
            x = layer(x, mask, causal=self.causal)

        x = self.norm(x)

        if self.causal:
            lengths = mask.sum(-1).clamp(min=1)
            last_idx = (lengths - 1).long()
            result = x[torch.arange(x.size(0), device=x.device), last_idx]
        else:
            masked_x = x * mask.unsqueeze(-1)
            count = mask.sum(-1, keepdim=True).clamp(min=1)
            result = masked_x.sum(1) / count  # (B*H, hidden_size)

        if self.proj_out is not None:
            result = self.proj_out(result)  # (B*H, model_dim)

        if self.residual:
            result = first_tok + result

        return result.view(B, H, -1)
