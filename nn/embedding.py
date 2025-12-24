import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from nn.encoders import EmbeddingEncoder


class PreInitializedEmbedding(nn.Module):
    def __init__(
        self,
        initial_vocab_size: int,
        embedding_dim: int,
        weight: nn.Parameter,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.initial_vocab_size = initial_vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight
        self.pad_token_id = pad_token_id

    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        return F.embedding(input, self.weight, self.pad_token_id)


class HyperEmbedding(PreInitializedEmbedding):
    def __init__(
        self,
        initial_vocab_size: int,
        extra_vocab_size: int,
        embedding_dim: int,
        weight: nn.Parameter,
        embedding_encoder: EmbeddingEncoder,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__(initial_vocab_size, embedding_dim, weight, pad_token_id)
        self.extra_vocab_size = extra_vocab_size

        self.embedding_encoder = embedding_encoder

    def forward(
        self,
        input: torch.LongTensor,
        extra_vocab: Optional[torch.Tensor] = None,
        metadata: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """
        Processes input tokens using base and hyper embeddings.

        Args:
            input (torch.LongTensor): Input token IDs of shape (B, S), where B is the batch size and S is the sequence length.
            extra_vocab (torch.Tensor): Extra vocabulary tensor of shape (M, D), where M is the number of extra tokens and D is the max number of subtokens.

        Returns:
            torch.Tensor: Combined embedding tensor of shape (B, S, embedding_dim).

        Example:
            # Input tokens
            input = torch.tensor([
                [10, 11, 1, 2],  # Batch 0
                [12, 1, 13, 3],  # Batch 1
                [14, 4, 15, 5]   # Batch 2
            ])

            # With initial_vocab_size=10, base tokens are 0-9 and hyper tokens are 10+.
            # Masks:
            base_token_mask = [
                [False, False, True, True],   # Batch 0
                [False, True, False, True],   # Batch 1
                [False, True, False, True]    # Batch 2
            ]
            hyper_token_mask = [
                [True, True, False, False],   # Batch 0
                [True, False, True, False],   # Batch 1
                [True, False, True, False]    # Batch 2
            ]
            base_input_ids = [
                [0, 0, 1, 2],   # Batch 0
                [0, 1, 0, 3],   # Batch 1
                [0, 4, 0, 5]    # Batch 2
            ]
            hyper_input_ids = [
                [0, 1, 0, 0],   # Batch 0
                [2, 0, 3, 0],   # Batch 1
                [4, 0, 5, 0]    # Batch 2
            ]
            batch_offsets = [
                [0, 0, 0, 0],   # Batch 0
                [8, 8, 8, 8],   # Batch 1
                [16, 16, 16, 16] # Batch 2
            ]
            hyper_input = [
                [0, 1, 0, 0],   # Batch 0
                [10, 8, 11, 8], # Batch 1
                [20, 16, 21, 16] # Batch 2
            ]
        """
        if extra_vocab is None or extra_vocab.numel() == 0:
            output = super().forward(input)
            return output, metadata

        # For extra_vocab, the shape  is (B, S, 8) if B>1, S is always 256 (extra_vocab_size) ; if B=1, S can be any number
        base_token_mask = input < self.initial_vocab_size  # (B, S)
        hyper_token_mask = ~base_token_mask  # (B, S)
        base_input_ids = input * base_token_mask.long()  # (B, S)
        hyper_input_ids = (
            input - self.initial_vocab_size
        ) * hyper_token_mask.long()  # (B, S)

        hyper_embedding_weight, encoder_metadata = self.embedding_encoder(
            extra_vocab, self.weight, self.pad_token_id
        )
        metadata["hyper_embedding_weight"] = hyper_embedding_weight

        stacked_hyper_embedding_weight = hyper_embedding_weight.view(
            -1, self.embedding_dim
        )  # (B, S, 8) -> (B*S, 8)

        batch_offsets = torch.arange(
            input.size(0), device=input.device, dtype=torch.long
        ).unsqueeze(-1).expand_as(input) * extra_vocab.size(
            1
        )  #  (B, S)

        hyper_input_ids += batch_offsets

        base_embedding = super().forward(base_input_ids) * base_token_mask.unsqueeze(
            -1
        )  # (B, S, D)
        hyper_embedding = F.embedding(
            hyper_input_ids, stacked_hyper_embedding_weight
        ) * hyper_token_mask.unsqueeze(
            -1
        )  # (B, S, D)

        for key, value in encoder_metadata.items():
            metadata[key] = value
        return base_embedding + hyper_embedding, metadata
