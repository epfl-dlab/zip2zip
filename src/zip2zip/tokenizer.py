from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple
from zip2zip.utils import get_base_vocab_size
from zip2zip_compression import LZWCompressor
from typing import List, Union, Optional
from transformers.utils import PushToHubMixin
from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding
from zip2zip.visual import ColoredToken, colorise_lzwtokens, ColorfulTokenizer
from zip2zip_compression import Codebook
from zip2zip.config import Zip2ZipConfig


class Zip2ZipTokenizer(PushToHubMixin):
    def __init__(
        self,
        config: Zip2ZipConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.zip2zip_config = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        set_pad_token_if_none(tokenizer)

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

    def _lzw_encode(
        self, *args, **kwargs
    ) -> List[Tuple[List[int], torch.Tensor, Codebook]]:
        encodings, attention_masks, codebooks = self.compressor.batch_encode(
            *args, **kwargs
        )
        return [
            (encoding, attention_mask, codebook)
            for encoding, attention_mask, codebook in zip(
                encodings, attention_masks, codebooks
            )
        ]

    def _lzw_decode(self, *args, **kwargs) -> List[Tuple[List[int], Codebook]]:
        return self.compressor.batch_decode(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        return_tensors = kwargs.pop("return_tensors", None)
        padding = kwargs.pop("padding_strategy").value
        truncation = kwargs.pop("truncation_strategy").value
        max_length = kwargs.pop("max_length", None)
        return_codebook = kwargs.pop("return_codebook", False)

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

        if return_codebook:
            encoding["codebooks"] = codebooks

        return encoding

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> Union[str, Tuple[str, Codebook]]:
        # we add a dimension to the token_ids to make it a list of lists, which is required by _lzw_decode
        if isinstance(token_ids, int):
            token_ids = [[token_ids]]
        else:
            token_ids = [token_ids]

        return_codebook = kwargs.pop("return_codebook", False)

        base_token_ids, codebook = self._lzw_decode(token_ids)[0]

        text = self.old_decode(
            base_token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )
        if return_codebook:
            return text, codebook
        else:
            return text

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        self.zip2zip_config.save_pretrained(save_directory, **kwargs)

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

        return cls(config)

    def color_decode(
        self,
        sequences: Union[List[int], List[List[int]], np.ndarray, torch.Tensor],
        codebooks: Optional[Union[Codebook, List[Codebook]]] = None,
        color_scheme: str = "finegrained",
    ) -> List[str]:
        # convert tensor to list
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()

        if codebooks is None:
            token_ids_codebook_pairs = self._lzw_decode(sequences)
            codebooks = [codebook for _, codebook in token_ids_codebook_pairs]

        if isinstance(codebooks, Codebook):
            codebooks = [codebooks]

        codebook_maps = [codebook.to_dict() for codebook in codebooks]

        out = []

        for seq, codebook_map in zip(sequences, codebook_maps):
            special_token_ids = set(self.tokenizer.get_added_vocab().values())
            colored_tokens = colorise_lzwtokens(
                seq, codebook_map, color_scheme, special_token_ids
            )
            out.append(self.tokenizer.decode_colored_token(colored_tokens))
        return out


def set_pad_token_if_none(
    tokenizer: PreTrainedTokenizerBase, pad_token_id: Optional[int] = None
) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
        )


if __name__ == "__main__":
    config = Zip2ZipConfig.from_pretrained(
        "Saibo-creator/zip2zip-Phi-3.5-mini-instruct-v0.1"
    )
    tokenizer = Zip2ZipTokenizer(config)
    tokenizer.tokenizer = ColorfulTokenizer(tokenizer.tokenizer)
    # Read this script's own source code
    with open(__file__, "r") as f:
        text = f.read()
    compressed_ids = tokenizer.encode(text)
    assert tokenizer.decode(compressed_ids) == text

    red_token = ColoredToken(token_ids=[100], color="\033[31m")
    print(tokenizer.tokenizer.decode_colored_token(red_token))
