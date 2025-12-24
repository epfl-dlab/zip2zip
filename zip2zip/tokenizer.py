from __future__ import annotations

from typing import Optional
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import Dict, List, Union, Optional
from zip2zip_compression import batch_encode, decode
from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding
from transformers.utils import PushToHubMixin
from visual import ColoredToken, colorise_lzwtokens, ColorfulTokenizer
from utils import get_base_vocab_size


from zip2zip.config import Zip2ZipConfig


class Zip2ZipTokenizer(PushToHubMixin):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: Zip2ZipConfig,
    ) -> None:
        self.config = config
        self.compression_config = config.compression
        self.tokenizer = ColorfulTokenizer(tokenizer)
        self.initial_vocab_size = get_base_vocab_size(tokenizer)
        self.max_codebook_size = self.compression_config.max_codebook_size
        self.max_subtokens = self.compression_config.max_subtokens
        self.disabled_ids = self.compression_config.disabled_ids

        self.old_batch_encode_plus = self.tokenizer._batch_encode_plus
        self.tokenizer._batch_encode_plus = self._batch_encode_plus

        self.old_decode = self.tokenizer._decode
        self.tokenizer._decode = self._decode

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        return self.tokenizer(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # TODO: we don't support padding here
        return_tensors = kwargs.pop("return_tensors", None)

        encoding = self.old_batch_encode_plus(*args, **kwargs)

        encoding["input_ids"], encoding["attention_mask"] = batch_encode(
            encoding["input_ids"],
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            disabled_ids=self.disabled_ids,
        )

        if return_tensors:
            encoding = encoding.convert_to_tensors(return_tensors)

        return encoding

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        token_ids = decode(
            token_ids,
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            disabled_ids=self.disabled_ids,
        )

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

    def set_disabled_ids(self, disabled_ids: List[int]):
        self.disabled_ids = disabled_ids

    def color_decode(
        self,
        lzw_token_ids: List[int],
        codebook: Dict[int, List[int]],
        color_scheme: str = "finegrained",
    ) -> None:
        special_token_ids = set(self.tokenizer.get_added_vocab().values())
        colored_token_groups = colorise_lzwtokens(
            lzw_token_ids, codebook, color_scheme, special_token_ids
        )
        print(self.tokenizer.decode_colored_token(colored_token_groups))

    def colorprint_lzw_tokens_by_ppl(
        self, lzw_token_ids: List[int], ppl: List[float], codebook: Dict[int, List[int]]
    ) -> None:
        from visual import colorise_lzw_tokens_by_ppl

        colored_token_groups = colorise_lzw_tokens_by_ppl(lzw_token_ids, ppl, codebook)
        print(self.tokenizer.decode_colored_token(colored_token_groups))

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


if __name__ == "__main__":
    tokenizer = Zip2ZipTokenizer.from_config(
        Zip2ZipConfig.from_pretrained(
            "Saibo-creator/zip2zip-Phi-3.5-mini-instruct-v0.1"
        ),
    )
    # Read this script's own source code
    with open(__file__, "r") as f:
        text = f.read()
    compressed_ids = tokenizer.encode(text)
    assert tokenizer.decode(compressed_ids) == text

    red_token = ColoredToken(token_ids=[100], color="\033[31m")
    print(tokenizer.tokenizer.decode_colored_token(red_token))
