import mlx.core as mx
import mlx.nn as nn

from configs import Config


class Encoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def __call__(
        self, codebook: mx.array, weight: mx.array, pad_token_id: int
    ) -> mx.array:
        pass


class AttentionEncoder(Encoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.num_heads = 32
        self.hidden_size = 3072
        self.position_embeddings = mx.random.normal(
            shape=(self.config.compression.max_subtokens, self.hidden_size),
            scale=self.hidden_size**-0.5,
        )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def __call__(
        self, codebook: mx.array, weight: mx.array, pad_token_id: int
    ) -> mx.array:
        H, S = codebook.shape

        codebook_embeddings = weight[codebook] + self.position_embeddings
        queries = self.q_proj(codebook_embeddings)
        keys = self.k_proj(codebook_embeddings)
        values = self.v_proj(codebook_embeddings)

        mask = codebook != pad_token_id
        mask = mask[..., None] * mask[..., None, :]
        output = mx.fast.scaled_dot_product_attention(
            queries.reshape(H, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            keys.reshape(H, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            values.reshape(H, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            scale=self.num_heads**-0.5,
            mask=mask[:, None, :, :],
        )
        output = output.transpose(0, 2, 1, 3).reshape(H, S, -1)
        return self.o_proj(output).mean(axis=1)


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

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

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )
        self.num_heads = int(
            config.embedding_encoder.unsafe_config.get("num_heads", 12)
        )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        B, S, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        output = mx.fast.scaled_dot_product_attention(
            queries.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            keys.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            values.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3),
            scale=self.num_heads**-0.5,
            mask=attn_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return self.o_proj(output)


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )

        self.mlp = MLP(config)
        self.attention = Attention(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_mlp_layernorm = nn.LayerNorm(self.hidden_size)

    def __call__(self, x: mx.array, attn_mask: mx.array) -> mx.array:
        x = self.post_attention_layernorm(x + self.attention(x, attn_mask))
        x = self.post_mlp_layernorm(x + self.mlp(x))
        return x


class TransformerEncoder(Encoder):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.hidden_size = int(
            config.embedding_encoder.unsafe_config.get("hidden_size", 768)
        )

        self.position_embeddings = mx.random.normal(
            shape=(self.config.compression.max_subtokens, self.hidden_size),
            scale=self.hidden_size**-0.5,
        )

        self.layers = [
            Layer(config)
            for _ in range(
                int(config.embedding_encoder.unsafe_config.get("num_hidden_layers", 4))
            )
        ]

    def __call__(
        self, codebook: mx.array, weight: mx.array, pad_token_id: int
    ) -> mx.array:
        x = weight[codebook] + self.position_embeddings

        attn_mask = codebook != pad_token_id
        attn_mask = attn_mask[..., None] * attn_mask[..., None, :]

        for layer in self.layers:
            x = layer(x, attn_mask[:, None, :, :])

        return x.mean(axis=1)
