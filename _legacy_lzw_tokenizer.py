from typing import Dict, List, Optional, Tuple, Union
from py_fast_compression import lzw_compress, batched_lzw_compress
from sympy import use
import torch
from transformers import PreTrainedTokenizerBase
from utils import get_base_vocab_size
from custom_types import Codebook, BatchedLZWTokenization, LZWTokenization
import copy


class Legacy_LZW_Tokenizer:
    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizerBase,
        disable_whitespace: bool = False,
        disable_digits: bool = False,
    ):
        self.hf_tokenizer = hf_tokenizer
        self.hf_tokenizer.pad_token = (
            self.hf_tokenizer.eos_token
            if self.hf_tokenizer.eos_token is not None
            else self.hf_tokenizer.pad_token
        )
        self.base_vocab_size = get_base_vocab_size(hf_tokenizer)
        self.disabled_ids = []
        if disable_whitespace:
            self.disabled_ids.extend(
                Legacy_LZW_Tokenizer.get_whitespace_token_ids(hf_tokenizer)
            )
        if disable_digits:
            self.disabled_ids.extend(
                Legacy_LZW_Tokenizer.get_digits_token_ids(hf_tokenizer)
            )

    def batch_encode(
        self,
        texts: List[str],
        extra_vocab_size: Optional[int] = None,
        max_subtoken: int = 4,
        chunk_size: Optional[int] = None,
    ) -> BatchedLZWTokenization:
        base_token_ids: List[List[int]] = self.hf_tokenizer.batch_encode_plus(texts)[
            "input_ids"
        ]
        max_base_token_length = max(len(ids) for ids in base_token_ids)
        batched_lzw_tokenization: BatchedLZWTokenization = batched_lzw_compress(
            ids=base_token_ids,
            initial_vocab_size=self.base_vocab_size,
            extra_vocab_size=max_base_token_length
            if extra_vocab_size is None
            else extra_vocab_size,
            max_out_seq_length=max_base_token_length
            if chunk_size is None
            else chunk_size,
            max_subtokens=max_subtoken,
            pad_token_id=self.hf_tokenizer.pad_token_id,
            disabled_ids=self.disabled_ids,
        )
        return batched_lzw_tokenization

    def encode(
        self, text: str, extra_vocab_size: Optional[int] = None, max_subtoken: int = 4
    ) -> LZWTokenization:
        """
        Encode a text with LZW compression.
        This doesn't apply any truncation or padding.
        It returns a single LZWTokenization object.
        """
        batched_lzw_tokenization: BatchedLZWTokenization = self.batch_encode(
            [text], extra_vocab_size, max_subtoken
        )
        assert (
            len(batched_lzw_tokenization) == 1
        ), f"Only one lzw tokenization is expected, got {len(batched_lzw_tokenization)}"
        return batched_lzw_tokenization[0]

    def decode(self, compressed_ids: List[int]) -> str:
        raise NotImplementedError

    def hypertoken_vocab(self, codebook: Codebook) -> Dict[str, int]:
        vocab = {}

        for code in codebook.codes:
            idx = self.base_vocab_size + len(vocab)
            subtokens: List[str] = self.hf_tokenizer.convert_ids_to_tokens(code)
            combined_token: str = self.hf_tokenizer.convert_tokens_to_string(subtokens)
            if subtokens[0].startswith(chr(9601)):
                combined_token = chr(9601) + combined_token
            vocab[combined_token] = idx
        return vocab

    def export_to_hf_tokenizer(self, codebook: Codebook) -> PreTrainedTokenizerBase:
        new_tokenizer = copy.deepcopy(self.hf_tokenizer)
        new_tokenizer.add_tokens(list(self.hypertoken_vocab(codebook).keys()))
        return new_tokenizer

    @staticmethod
    def get_whitespace_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
        return [i for i in range(tokenizer.vocab_size) if tokenizer.decode(i).isspace()]

    @staticmethod
    def get_digits_token_ids(tokenizer: PreTrainedTokenizerBase) -> List[int]:
        return [i for i in range(tokenizer.vocab_size) if tokenizer.decode(i).isdigit()]


if __name__ == "__main__":

    from transformers import AutoTokenizer

    phi_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", use_fast=True
    )
    lzw_tokenizer = Legacy_LZW_Tokenizer(phi_tokenizer)

    print(Legacy_LZW_Tokenizer.get_digits_token_ids(phi_tokenizer))

    text = "Hello, my dog is cute; Hello, my dog is cute; Hello, my dog is cute;"
    lzw_tokenization = lzw_tokenizer.encode(text)
    print(lzw_tokenization.token_ids)
    print(lzw_tokenization.codebook)
    print("uncompressed token ids: ", lzw_tokenization.get_token_ids_decompressed())
    print(
        "uncompressed token ids length: ",
        lzw_tokenization.get_num_tokens_decompressed(),
    )

    x: torch.tensor = lzw_tokenization.codebook.pad()
    print(x)
    print(lzw_tokenizer.hypertoken_vocab(lzw_tokenization.codebook))

    new_tokenizer = lzw_tokenizer.export_to_hf_tokenizer(lzw_tokenization.codebook)

    print(new_tokenizer.decode(lzw_tokenization.token_ids))

    texts = [
        "Hello, my dog is cute; Hello, my dog is cute; Hello, my dog is cute;",
        "Hello, my dog is cute; Hello, my dog is cute; Hello, my dog is cute;",
    ]
    batched_lzw_tokenization = lzw_tokenizer.batch_encode(texts)
    batch_token_ids, batch_codebooks = batched_lzw_tokenization.to_tensor()
    print(batch_token_ids.shape)
    print(batch_codebooks.shape)

    print(lzw_tokenization.stats)

    print(batched_lzw_tokenization.stats)
