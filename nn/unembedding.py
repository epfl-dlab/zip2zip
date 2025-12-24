import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any


from nn.encoders import EmbeddingEncoder


class HyperUnembedding(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_vocab_size: int,
        embedding_encoder: EmbeddingEncoder,
        bias: bool,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.initial_vocab_size = initial_vocab_size
        self.pad_token_id = pad_token_id
        self.embedding_encoder = embedding_encoder

    def forward(
        self,
        x: torch.Tensor,
        codebook_tensor: Optional[torch.Tensor] = None,
        metadata: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # x (B, S, D)
        output = self.linear(x)  # (B, S, V)
        if codebook_tensor is None or codebook_tensor.numel() == 0:
            return output, metadata

        hyper_weight, encoder_metadata = self.embedding_encoder(
            codebook_tensor, self.linear.weight, self.pad_token_id
        )  # (B, V_E, D)
        metadata["hyper_weight"] = hyper_weight

        hyper_output = torch.bmm(
            x, hyper_weight.transpose(-2, -1)
        )  # (B, S, V_E) where the [:,:,V_E_used:] are zeros
        """hyper_output[0]:
        tensor([[-0.1182,  0.0226,  0.1094,  ...,  0.0000,  0.0000,  0.0000],
        [-0.2471,  0.0009,  0.2041,  ...,  0.0000,  0.0000,  0.0000],
        [-0.2852,  0.0160,  0.2344,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.1357,  0.0835,  0.1689,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.1377,  0.1069,  0.1963,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.1338,  0.0564,  0.2432,  ...,  0.0000,  0.0000,  0.0000]],
        """
        # TODO, maybe we need to mask out the empty slots

        for key, value in encoder_metadata.items():
            metadata[key] = value

        return (
            torch.cat(
                [
                    output[..., : self.initial_vocab_size],
                    hyper_output,
                    output[..., self.initial_vocab_size :],
                ],
                dim=-1,
            ),
            metadata,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_linear: nn.Linear,
        initial_vocab_size: int,
        embedding_encoder: EmbeddingEncoder,
        bias: bool,
        pad_token_id: Optional[int] = None,
    ) -> "HyperUnembedding":
        hyper_linear = cls(
            pretrained_linear.in_features,
            pretrained_linear.out_features,
            initial_vocab_size,
            embedding_encoder,
            bias,
            pad_token_id,
        ).to(device=pretrained_linear.weight.device)

        hyper_linear.linear.weight = pretrained_linear.weight
        if pretrained_linear.bias is not None:
            hyper_linear.linear.bias = pretrained_linear.bias

        return hyper_linear
