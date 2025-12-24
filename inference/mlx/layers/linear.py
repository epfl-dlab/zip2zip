import mlx.core as mx
import mlx.nn as nn

from inference.mlx.layers.encoder import Encoder
from inference.mlx.codebook import CodebookManager


class HyperLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        initial_vocab_size: int,
        encoder: Encoder,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> None:
        super().__init__()
        self.weight = mx.random.uniform(
            low=-(input_dims**-0.5),
            high=input_dims**-0.5,
            shape=(input_dims, initial_vocab_size),
        )
        self.encoder = encoder
        self.pad_token_id = pad_token_id
        self.codebook_manager = codebook_manager
        self.initial_vocab_size = initial_vocab_size

    def __call__(self, x: mx.array) -> mx.array:
        output = x @ self.weight.T

        hyper_weight = self.codebook_manager.get_hyper_linear(self.weight, self.encoder)
        hyper_output = x @ hyper_weight.T

        return mx.concatenate(
            [
                output[..., : self.initial_vocab_size],
                hyper_output,
            ],
            axis=-1,
        )

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        encoder: Encoder,
        pad_token_id: int,
        codebook_manager: CodebookManager,
    ) -> "HyperLinear":
        hyper_linear = HyperLinear(
            linear.weight.shape[1],
            codebook_manager.initial_vocab_size,
            encoder,
            pad_token_id,
            codebook_manager,
        )
        hyper_linear.weight = linear.weight
        return hyper_linear
