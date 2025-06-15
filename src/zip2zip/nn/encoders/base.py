from __future__ import annotations

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Generic, Callable


from zip2zip.config import EncoderConfigType, CompressionConfig

EncoderFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


class BaseEncoder(nn.Module, ABC, Generic[EncoderConfigType]):
    def __init__(
        self, encoder_config: EncoderConfigType, compression_config: CompressionConfig
    ) -> None:
        super().__init__()
        self.encoder_config = encoder_config
        self.compression_config = compression_config

    @abstractmethod
    def forward(
        self, codebook: torch.Tensor, embeddings: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        ...

    @classmethod
    def from_config(
        cls,
        encoder_config: EncoderConfigType,
        compression_config: CompressionConfig,
    ) -> BaseEncoder:
        from zip2zip.nn.encoders.attention import AttentionEncoder
        from zip2zip.nn.encoders.transformer import TransformerEncoder

        from zip2zip.nn.encoders.config import (
            AttentionEncoderConfig,
            TransformerEncoderConfig,
        )

        config2encoder_mapping = {
            AttentionEncoderConfig: AttentionEncoder,
            TransformerEncoderConfig: TransformerEncoder,
        }

        encoder_class = config2encoder_mapping[type(encoder_config)]

        return encoder_class(encoder_config, compression_config)

    def get_encoder_fn(self) -> EncoderFn:
        return lambda codebook, embeddings, pad_token_id: self(
            codebook, embeddings, pad_token_id
        )
