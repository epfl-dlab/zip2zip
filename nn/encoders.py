import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from nn.attention import (
    SinusoidalPositionalEncoding,
    py_scaled_dot_product_attention,
)

devnull = open(os.devnull, "w")

if TYPE_CHECKING:
    from configs import Config


class EmbeddingEncoder(nn.Module, ABC):
    def __init__(self, config: "Config") -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        codebook_tensor: torch.Tensor,
        base_embeddings: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create the embedding vectors for the vocabulary given the embeddings of the subtokens.

        Args:
            vocab (torch.Tensor): The vocabulary to encode. Shape: (batch_size, config.compression.extra_vocab_size, config.compression.max_subtokens)
            embeddings (torch.Tensor): The embeddings of the subtokens. Shape: (batch_size, config.initial_vocab_size, embedding_dim)
            pad_token_id (int): The padding index.

        Returns:
            torch.Tensor: The embedding vectors for the vocabulary. Shape: (batch_size, config.compression.extra_vocab_size, embedding_dim)
        """
        raise NotImplementedError

    @staticmethod
    def init(name: str, config: "Config") -> "EmbeddingEncoder":
        mapping = {
            "average": AverageEmbeddingEncoder,
            "attention": AttentionEmbeddingEncoder,
            "block_masked_attention": AttentionEmbeddingEncoder,  # for backward compatibility, the old BlockMaskedAttention is now named to AttentionEmbeddingEncoder
            "transformer": TransformerEmbeddingEncoder,
        }

        if name not in mapping:
            raise ValueError(f"Invalid embedding encoder name: {name}")

        return mapping[name](config)


class AverageEmbeddingEncoder(EmbeddingEncoder):
    def forward(
        self,
        codebook_tensor: torch.Tensor,
        base_embeddings: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        vocab_embeds = F.embedding(codebook_tensor, base_embeddings)
        mask = codebook_tensor != pad_token_id

        masked_sum = (vocab_embeds * mask.unsqueeze(-1)).sum(dim=2)
        masked_count = mask.sum(dim=2, keepdim=True).clamp(min=1)
        metadata = {}
        return masked_sum / masked_count, metadata


class Attention(nn.Module):
    def __init__(self, config: "Config") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )

        self.num_heads = int(
            config.embedding_encoder.unsafe_config.get("num_heads", 12)
        )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            config.dtype
        )  # TODO, init this correctly with kaiming
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            config.dtype
        )
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            config.dtype
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
            config.dtype
        )

    def _init_weights(self):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            torch.nn.init.kaiming_uniform_(
                layer.weight, a=math.sqrt(5)
            )  # Default Kaiming for Linear layers

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, _ = x.size()
        query_state = self.q_proj(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        key_state = self.k_proj(x).view(B, S, self.num_heads, -1).transpose(1, 2)
        value_state = self.v_proj(x).view(B, S, self.num_heads, -1).transpose(1, 2)

        _attn_scale = self.config.embedding_encoder.unsafe_config.get(
            "attn_scale", None
        )
        attn_scale = float(_attn_scale) if _attn_scale is not None else None
        if (
            self.config.embedding_encoder.unsafe_config["attn_implementation"]
            == "debug"
        ):
            # for debugging purposes and check the attention map
            out_x, attn_weight = py_scaled_dot_product_attention(
                query_state,
                key_state,
                value_state,
                attn_mask=attn_mask,
                scale=attn_scale,
            )
        else:
            out_x = F.scaled_dot_product_attention(
                query_state,
                key_state,
                value_state,
                attn_mask=attn_mask,
                scale=attn_scale,
            )  # (B, S, D) just like x
            attn_weight = None
        out_x = out_x.transpose(1, 2).contiguous().view_as(x)
        return self.o_proj(out_x), attn_weight


class AttentionEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, config: "Config") -> None:
        super().__init__(config)
        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )

        self.attention = Attention(config)

        if config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
            self.decoder_vXtU = Attention(config)
            # Add a language model head here
            self.lm_head_vXtU = nn.Linear(
                self.hidden_size, config.initial_vocab_size, bias=False
            )

        #  Define `[CLS]` token as a learnable parameter
        self.cls_token_vector = nn.Parameter(torch.zeros(self.hidden_size))
        # Kaiming initialization works well for relu activations
        nn.init.kaiming_uniform_(self.cls_token_vector.unsqueeze(0))

        if self.config.embedding_encoder.position_encoding == "rotary_sinusoidal":
            self.pos_embed = SinusoidalPositionalEncoding(
                max_seq_len=self.config.compression.max_subtokens,
                hidden_size=self.hidden_size,
            )
        elif self.config.embedding_encoder.position_encoding == "learnable":
            self.pos_embed = nn.Embedding(
                config.compression.max_subtokens, self.hidden_size
            )
            nn.init.kaiming_uniform_(self.pos_embed.weight)
        else:
            raise ValueError(
                f"Invalid position encoding: {self.config.embedding_encoder.position_encoding}"
            )

        self.down = (
            nn.Linear(
                config.embedding_encoder.embedding_size, self.hidden_size, bias=False
            )
            if self.hidden_size != config.embedding_encoder.embedding_size
            else nn.Identity()
        )
        self.up = (
            nn.Linear(
                self.hidden_size, config.embedding_encoder.embedding_size, bias=False
            )
            if self.hidden_size != config.embedding_encoder.embedding_size
            else nn.Identity()
        )

    def forward(
        self,
        codebook_tensor: torch.Tensor,
        base_embeddings: torch.Tensor,
        pad_token_id: int,
        cls_token_id: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, extra_vocab_size, max_subtokens = codebook_tensor.size()
        codebook_tensor = codebook_tensor.reshape(
            batch_size * extra_vocab_size, max_subtokens
        )

        if self.config.embedding_encoder.position_encoding == "learnable":
            pos_embed = self.pos_embed(
                torch.arange(
                    0, max_subtokens, device=codebook_tensor.device, dtype=torch.long
                )
            )
        elif self.config.embedding_encoder.position_encoding == "rotary_sinusoidal":
            pos_embed = self.pos_embed(max_subtokens)
        else:
            pos_embed = 0

        vocab_embeds = self.down(
            F.embedding(codebook_tensor, base_embeddings)
        )  # (B*V_E, S, D)

        fused_embed = vocab_embeds + pos_embed
        if self.config.embedding_encoder.unsafe_config.get("use_cls_token", False):
            # prepend the `[CLS]` token to the fused_embed
            fused_embed = torch.cat(
                [
                    self.cls_token_vector[None, None, :].expand(
                        batch_size * extra_vocab_size, -1, -1
                    ),
                    fused_embed,
                ],
                dim=1,
            )

            # concatenate the cls_token_id to the codebook_tensor
            cls_token_id_col = (
                torch.tensor([[cls_token_id]])
                .expand(batch_size * extra_vocab_size, 1)
                .to(codebook_tensor.device)
            )
            codebook_tensor = torch.cat([cls_token_id_col, codebook_tensor], dim=1)
        _self_dtype = next(self.parameters()).dtype
        attn_mask = codebook_tensor.eq(pad_token_id).logical_not()
        attn_mask = attn_mask.unsqueeze(-1).mul(attn_mask.unsqueeze(-2)).unsqueeze(1)
        # Here the attn_mask is in a top-left block
        """
        tensor([
        [ True,  True,  True, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]])
        """
        output, attn_weight = self.attention(fused_embed, attn_mask)
        if self.config.embedding_encoder.unsafe_config.get("use_cls_token", False):
            output = output[:, 0, :]
        else:
            output = torch.mean(output, dim=1)
        reshaped_output = output.reshape(batch_size, extra_vocab_size, -1)

        up_output = self.up(reshaped_output)

        metadata = {
            "attn_weight": attn_weight,
            "attn_mask": attn_mask,
        }

        if self.config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
            decoder_input = torch.cat([output[:, None, :], vocab_embeds], dim=1)
            causal_mask = torch.ones(
                decoder_input.size(1),
                decoder_input.size(1),
                dtype=torch.bool,
                device=decoder_input.device,
            ).tril()[None, None, :, :]
            decoder_output, _ = self.decoder_vXtU(decoder_input, causal_mask)

            decoder_logits = self.lm_head_vXtU(decoder_output)
            # Optionally, apply softmax to get probabilities
            # probabilities = F.softmax(logits, dim=-1)

            metadata = {
                "AE_logits": decoder_logits,
                **metadata,
            }
        return up_output, metadata


class MLP(nn.Module):
    def __init__(self, config: "Config") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )
        self.intermediate_size = int(
            config.embedding_encoder.unsafe_config.get(
                "intermediate_size", self.hidden_size * 4
            )
        )

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            torch.nn.init.kaiming_uniform_(layer.weight)


class AttentionBlockLayer(nn.Module):
    def __init__(self, config: "Config") -> None:
        super().__init__()
        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )

        self.mlp = MLP(config)
        self.attention = Attention(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_mlp_layernorm = nn.LayerNorm(self.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output, attn_weight = self.attention(hidden_states, attn_mask)
        hidden_states = self.post_attention_layernorm(hidden_states + attn_output)
        mlp_output = self.mlp(hidden_states)
        return self.post_mlp_layernorm(hidden_states + mlp_output), attn_weight


class TransformerEmbeddingEncoder(EmbeddingEncoder):
    def __init__(self, config: "Config") -> None:
        super().__init__(config)
        self.embedding_size = config.embedding_encoder.embedding_size
        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )
        #  Define `[CLS]` token as a learnable parameter
        self.cls_token_vector = nn.Parameter(torch.zeros(self.hidden_size))
        # Kaiming initialization works well for relu activations
        nn.init.kaiming_uniform_(self.cls_token_vector.unsqueeze(0))

        if self.config.embedding_encoder.position_encoding == "learnable":
            self.pos_embed = nn.Embedding(
                config.compression.max_subtokens, self.hidden_size
            )
            nn.init.kaiming_uniform_(self.pos_embed.weight)
        else:
            self.pos_embed = None

        self.layers = nn.ModuleList(
            [
                AttentionBlockLayer(config)
                for _ in range(
                    int(
                        config.embedding_encoder.unsafe_config.get(
                            "num_hidden_layers", 4
                        )
                    )
                )
            ]
        )

        if config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
            self.decoder_vXtU = nn.ModuleList(
                [
                    AttentionBlockLayer(config)
                    for _ in range(
                        int(
                            config.embedding_encoder.unsafe_config.get(
                                "num_hidden_layers", 4
                            )
                        )
                    )
                ]
            )
            # Add a language model head here
            self.lm_head_vXtU = nn.Linear(
                self.hidden_size, config.initial_vocab_size, bias=False
            )

    def forward(
        self,
        codebook_tensor: torch.Tensor,
        base_embeddings: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, extra_vocab_size, max_subtokens = codebook_tensor.size()
        codebook_tensor = codebook_tensor.reshape(
            batch_size * extra_vocab_size, max_subtokens
        )

        if self.config.embedding_encoder.position_encoding == "learnable":
            pos_embed = self.pos_embed(
                torch.arange(
                    0, max_subtokens, device=codebook_tensor.device, dtype=torch.long
                )
            )
        else:
            pos_embed = 0

        vocab_embeds = F.embedding(codebook_tensor, base_embeddings)

        output = vocab_embeds + pos_embed
        attn_mask = codebook_tensor != pad_token_id
        attn_mask = attn_mask.unsqueeze(-1).mul(attn_mask.unsqueeze(-2)).unsqueeze(1)

        multi_layer_attn_weights = []
        for layer in self.layers:
            output, attn_weight = layer(output, attn_mask=attn_mask)
            multi_layer_attn_weights.append(attn_weight)

        if self.config.embedding_encoder.unsafe_config.get("use_cls_token", False):
            output = output[:, 0, :]
        else:
            output = torch.mean(output, dim=1)
        up_output = output.reshape(batch_size, extra_vocab_size, -1)

        metadata = {
            "attn_weight": multi_layer_attn_weights,
        }

        if self.config.embedding_encoder.auto_encoder_loss_alpha > 0.0:
            decoder_input = torch.cat([output[:, None, :], vocab_embeds], dim=1)
            causal_mask = torch.ones(
                decoder_input.size(1),
                decoder_input.size(1),
                dtype=torch.bool,
                device=decoder_input.device,
            ).tril()[None, None, :, :]
            for layer in self.decoder_vXtU:
                decoder_output, _ = layer(decoder_input, causal_mask)
            # decoder_output, _ = self.decoder_vXtU(decoder_input, causal_mask)

            decoder_logits = self.lm_head_vXtU(decoder_output)

            metadata = {
                "AE_logits": decoder_logits,
                **metadata,
            }
        return up_output, metadata
