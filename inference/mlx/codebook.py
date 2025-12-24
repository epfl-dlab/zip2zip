import mlx.core as mx
from typing import List, Optional
from fast_compression import CodebookManager as FastCodebookManager

from inference.mlx.layers.encoder import Encoder


class CodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:
        self.initial_vocab_size = initial_vocab_size
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.pad_token_id = pad_token_id
        self.disabled_ids = disabled_ids

        self.codebook_manager = FastCodebookManager(
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            disabled_ids,
        )

        self.hyper_embedding = None
        self.hyper_linear = None
        self.updates = None
        self.num_updates = 0

    def get_subtokens(self, id: int) -> List[int]:
        return self.codebook_manager.get_subtokens(id)

    def get_hyper_embedding(
        self, ids: mx.array, weight: mx.array, encoder: Encoder
    ) -> mx.array:
        updates, num_updates = self.codebook_manager.update_codebook(
            ids.tolist(), False
        )
        self.updates = mx.array(updates)
        self.num_updates = num_updates

        if self.num_updates > 0:
            new_hyper_embedding = encoder(self.updates, weight, self.pad_token_id)

            if self.hyper_embedding is None:
                self.hyper_embedding = new_hyper_embedding
            else:
                self.hyper_embedding = mx.concatenate(
                    [self.hyper_embedding, new_hyper_embedding]
                )

        return self.hyper_embedding

    def get_hyper_linear(self, weight: mx.array, encoder: Encoder) -> mx.array:
        if self.num_updates > 0:
            new_hyper_linear = encoder(self.updates, weight, self.pad_token_id)

            if self.hyper_linear is None:
                self.hyper_linear = new_hyper_linear
            else:
                self.hyper_linear = mx.concatenate(
                    [self.hyper_linear, new_hyper_linear]
                )

        return self.hyper_linear

    def reset(self) -> None:
        self.hyper_embedding = None
        self.hyper_linear = None
        self.updates = None
        self.num_updates = 0
        self.codebook_manager.reset()
