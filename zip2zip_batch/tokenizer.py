from __future__ import annotations

import torch
import numpy as np
from typing import Optional
from utils import get_base_vocab_size
from compression import LZWCompressor
from typing import List, Union, Optional
from transformers.utils import PushToHubMixin
from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding
from visual import ColoredToken, colorise_lzwtokens, ColorfulTokenizer
from compression import Codebook
from zip2zip_batch.config import Zip2ZipConfig


class Zip2ZipTokenizer(PushToHubMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Zip2ZipConfig,
    ) -> None:
        self.config = config
        self.compression_config = config.compression
        self.initial_vocab_size = get_base_vocab_size(tokenizer)
        self.max_codebook_size = self.compression_config.max_codebook_size
        self.max_subtokens = self.compression_config.max_subtokens
        self.disabled_ids = self.compression_config.disabled_ids

        self.old_batch_encode_plus = tokenizer._batch_encode_plus
        tokenizer._batch_encode_plus = self._batch_encode_plus

        self.old_decode = tokenizer._decode
        tokenizer._decode = self._decode

        # self.tokenizer = tokenizer
        self.tokenizer = ColorfulTokenizer(tokenizer)
        self.compressor = LZWCompressor(
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            pad_token_id=self.tokenizer.pad_token_id,
            disabled_ids=self.disabled_ids,
        )

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # TODO: we don't support padding here
        return_tensors = kwargs.pop("return_tensors", None)
        padding = kwargs.pop("padding_strategy").value
        truncation = kwargs.pop("truncation_strategy").value
        max_length = kwargs.pop("max_length", None)

        encoding = self.old_batch_encode_plus(*args, **kwargs)

        (
            encoding["input_ids"],
            encoding["attention_mask"],
            codebooks,
        ) = self.compressor.batch_encode(
            encoding["input_ids"],
            padding=padding,
            truncation=truncation != "do_not_truncate",
            max_length=max_length,
        )

        if return_tensors:
            encoding = encoding.convert_to_tensors(return_tensors)

        encoding["codebooks"] = codebooks
        return encoding

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], np.ndarray, torch.Tensor],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> List[str]:
        codebooks = kwargs.pop("codebooks", [None] * len(sequences))

        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                codebook=codebook,
                **kwargs,
            )
            for (seq, codebook) in zip(sequences, codebooks)
        ]

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        token_ids = self.compressor.decode(token_ids, kwargs.get("codebook", None))

        return self.old_decode(
            token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )

    @classmethod
    def from_config(
        cls,
        config: Zip2ZipConfig,
        *args,
        **kwargs,
    ) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path, *args, **kwargs
        )

        return cls(tokenizer, config)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.config.save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> Zip2ZipTokenizer:
        config = Zip2ZipConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )

        return cls.from_config(config)

    def color_decode(
        self,
        sequences: Union[List[int], List[List[int]], np.ndarray, torch.Tensor],
        codebook: Union[Codebook, List[Codebook]],
        color_scheme: str = "finegrained",
    ) -> List[str]:
        if isinstance(codebook, Codebook):
            decompress_maps = [codebook.to_decoding_dict()]
            sequences = [sequences]
        else:
            decompress_maps = [codebook.to_decoding_dict() for codebook in codebook]

        # convert tensor to list
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()

        out = []

        for seq, decompress_map in zip(sequences, decompress_maps):
            special_token_ids = set(self.tokenizer.get_added_vocab().values())
            colored_tokens = colorise_lzwtokens(
                seq, decompress_map, color_scheme, special_token_ids
            )
            out.append(self.tokenizer.decode_colored_token(colored_tokens))
        return out


if __name__ == "__main__":
    config = Zip2ZipConfig.from_pretrained(
        "Saibo-creator/zip2zip-Phi-3.5-mini-instruct-v0.1"
    )
    tokenizer = Zip2ZipTokenizer.from_config(config)
    tokenizer.tokenizer = ColorfulTokenizer(tokenizer.tokenizer)
    # Read this script's own source code
    with open(__file__, "r") as f:
        text = f.read()
    compressed_ids = tokenizer.encode(text)
    assert tokenizer.decode(compressed_ids) == text

    red_token = ColoredToken(token_ids=[100], color="\033[31m")
    print(tokenizer.tokenizer.decode_colored_token(red_token))
