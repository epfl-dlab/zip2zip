from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from zip2zip.config import Zip2ZipConfig
from zip2zip.codebook import CodebookManager
from zip2zip.nn.encoders.base import BaseEncoder


class HyperLinear(nn.Linear):
    def __init__(
        self,
        config: Zip2ZipConfig,
        encoder: BaseEncoder,
        in_features: int,
        out_features: int,
        bias: bool,
        device: torch.device,
        dtype: torch.dtype,
        initial_vocab_size: int,
        codebook_manager: CodebookManager,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.config = config
        self.encoder = encoder
        self.initial_vocab_size = initial_vocab_size
        self.codebook_manager = codebook_manager

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_logits = super().forward(x)

        hyper_linear_weights = self.codebook_manager.get_hyper_linear_weights(
            self.weight, self.encoder
        )

        hyper_logits = torch.bmm(x, hyper_linear_weights.transpose(-2, -1))

        return torch.cat(
            [
                base_logits[..., : self.initial_vocab_size],
                hyper_logits,
                base_logits[..., self.initial_vocab_size :],
            ],
            dim=-1,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Zip2ZipConfig,
        encoder: BaseEncoder,
        codebook_manager: CodebookManager,
    ) -> HyperLinear:
        with torch.device("meta"):
            hyper_linear = cls(
                config,
                encoder,
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
                linear.weight.device,
                linear.weight.dtype,
                config.compression.initial_vocab_size,
                codebook_manager,
            )
        hyper_linear.to_empty(device=linear.weight.device)
        hyper_linear.weight = linear.weight
        if linear.bias is not None:
            hyper_linear.bias = linear.bias
        return hyper_linear
