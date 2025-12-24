from __future__ import annotations

from enum import Enum
from typing import Optional, TypeVar
from dataclasses import dataclass, field


class EncoderType(str, Enum):
    ATTENTION = "attention"
    TRANSFORMER = "transformer"


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


ENCODER_CONFIG_MAPPING = {
    "attention": AttentionEncoderConfig,
    "transformer": TransformerEncoderConfig,
}
