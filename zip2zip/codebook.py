import torch
from typing import List, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from zip2zip_compression import CodebookManager as FastCodebookManager

from zip2zip.nn.encoders.base import CodebookEmbeddingFn
from utils import get_base_vocab_size
from zip2zip.config import Zip2ZipConfig


class CodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        embedding_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.pad_token_id = pad_token_id
        self.max_subtokens = max_subtokens
        self.embedding_dim = embedding_dim
        self.max_codebook_size = max_codebook_size
        self.initial_vocab_size = initial_vocab_size

        self.codebook_manager = FastCodebookManager(
            initial_vocab_size,
            max_codebook_size,
            max_subtokens,
            pad_token_id,
            disabled_ids,
        )

        self.working_index = 0
        self.hyper_embedding = torch.zeros(
            self.max_codebook_size, embedding_dim, dtype=dtype, device=self.device
        )
        self.hyper_unembedding = torch.zeros(
            self.max_codebook_size, embedding_dim, dtype=dtype, device=self.device
        )

        self.updates = None
        self.num_updates = 0

    @classmethod
    def from_config(
        cls, config: Zip2ZipConfig, dtype: torch.dtype, device: torch.device
    ) -> "CodebookManager":
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        return cls(
            initial_vocab_size=config.compression.initial_vocab_size,
            max_codebook_size=config.compression.max_codebook_size,
            max_subtokens=config.compression.max_subtokens,
            embedding_dim=config.encoder.hidden_size,
            dtype=dtype,
            device=device,
            pad_token_id=pad_token_id,
            disabled_ids=config.compression.disabled_ids,
        )

    def to(self, *args, **kwargs):
        self.hyper_embedding = self.hyper_embedding.to(*args, **kwargs)
        self.hyper_unembedding = self.hyper_unembedding.to(*args, **kwargs)

    def fill_hyper_embedding_weight(
        self,
        ids: torch.LongTensor,
        base_weight: torch.Tensor,
        callable_encoder: CodebookEmbeddingFn,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            ids_list = ids.tolist()
            updates, num_updates = self.codebook_manager.update_codebook(
                ids_list, len(ids_list) > 1
            )
            self.updates = torch.tensor(
                updates,
                device=self.device,
                dtype=torch.long,
            )
            self.num_updates = num_updates

            if self.num_updates > 0:
                self.hyper_embedding[
                    self.working_index : self.working_index + self.num_updates
                ] = callable_encoder(self.updates, base_weight, self.pad_token_id)[
                    : self.num_updates
                ]

        active_hyper_embedding = self.hyper_embedding[
            : self.working_index + self.num_updates
        ]
        return active_hyper_embedding

    def fill_hyper_unembedding_weight(
        self, base_weight: torch.Tensor, callable_encoder: CodebookEmbeddingFn
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            if self.num_updates > 0:
                self.hyper_unembedding[
                    self.working_index : self.working_index + self.num_updates
                ] = callable_encoder(self.updates, base_weight, self.pad_token_id)[
                    : self.num_updates
                ]
                self.set_working_index(self.working_index + self.num_updates)

        active_hyper_unembedding = self.hyper_unembedding[: self.working_index]
        return active_hyper_unembedding

    def get_working_index(self) -> int:
        return self.working_index

    def set_working_index(self, value: int) -> None:
        if value > self.max_codebook_size:
            raise ValueError(
                f"Index {value} exceeds maximum codebook size {self.max_codebook_size}"
            )
        self.working_index = value

    def reset(self) -> None:
        self.codebook_manager.reset()
        self.working_index = 0
        self.hyper_embedding = torch.zeros(
            self.max_codebook_size,
            self.embedding_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.hyper_unembedding = torch.zeros(
            self.max_codebook_size,
            self.embedding_dim,
            dtype=self.dtype,
            device=self.device,
        )
