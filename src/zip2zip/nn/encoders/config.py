from __future__ import annotations

from enum import Enum
from typing import Optional, TypeVar
from dataclasses import dataclass, field


class EncoderType(str, Enum):
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    HYPER = "res_latent_attn"


@dataclass
class EncoderConfig:
    hidden_size: int = field(
        default=None, metadata={"help": "The hidden size of the model"}
    )
    tie_encoders: bool = field(
        default=False, metadata={"help": "Whether to tie the input and output encoders"}
    )

    position_encoding: Optional[str] = field(
        default=None, metadata={"help": "The position encoding to use"}
    )


EncoderConfigType = TypeVar("EncoderConfigType", bound=EncoderConfig)


@dataclass
class AttentionEncoderConfig(EncoderConfig):
    num_heads: int = field(
        default=None, metadata={"help": "The number of attention heads"}
    )


@dataclass
class TransformerEncoderConfig(EncoderConfig):
    num_hidden_layers: int = field(
        default=None,
        metadata={"help": "The number of layers in the transformer encoder"},
    )
    intermediate_size: int = field(
        default=None,
        metadata={"help": "The intermediate size of the MLP"},
    )
    num_heads: int = field(
        default=None,
        metadata={"help": "The number of attention heads"},
    )


@dataclass
class ResLatentAttnConfig(EncoderConfig):
    model_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Model embedding dim. If set and != hidden_size, proj_in/proj_out are added."},
    )
    num_hidden_layers: int = field(
        default=None,
        metadata={"help": "Number of transformer layers"},
    )
    intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "FFN intermediate dim (default: 4 * hidden_size)"},
    )
    num_heads: int = field(
        default=None,
        metadata={"help": "Number of attention heads"},
    )
    causal: bool = field(
        default=False,
        metadata={"help": "If True, use causal attention + last-token pooling; else bidirectional + mean pooling"},
    )
    residual: bool = field(
        default=True,
        metadata={"help": "If True, add first base-token embedding to encoder output (residual warmup)"},
    )


ENCODER_CONFIG_MAPPING = {
    "attention": AttentionEncoderConfig,
    "transformer": TransformerEncoderConfig,
    "res_latent_attn": ResLatentAttnConfig,
}
