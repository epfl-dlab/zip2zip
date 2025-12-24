from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union
from itertools import cycle
from functools import lru_cache

from transformers import AutoTokenizer
from _legacy_lzw_tokenizer import Legacy_LZW_Tokenizer
from custom_types import Codebook

color_codes = {
    "bright-blue": "\u001b[104m",
    "bright-yellow": "\u001b[103m",
    "cyan": "\u001b[46m",
    "white": "\u001b[47m",
    "bright-green": "\u001b[102m",
    "green": "\u001b[42m",
    "bright-red": "\u001b[101m",
    "bright-magenta": "\u001b[105m",
    "bright-cyan": "\u001b[106m",
}
reset = "\033[0m"

# abstract class for tokenizers inheriting from ABC
class TokenizerInterface(ABC):

    NOT_COMPLETE_SYMBOL_ORD = None

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, text: List[int]) -> str:
        raise NotImplementedError

    @abstractproperty
    def pretty_name(self) -> str:
        raise NotImplementedError

    @classmethod
    def format_color(cls, text, color):
        """
        Prints the specified text in the specified color.
        """

        if color not in color_codes:
            raise ValueError("Invalid color: {}".format(color))
        return color_codes[color] + text + reset

    def print_pretty_tokens(
        self, tokens: List[int], print_total=False, print_newline_in_color=False
    ):

        token_words = [self.decode([t]) for t in tokens]
        # colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]
        colors = list(color_codes.keys())

        for t, w, c in zip(tokens, token_words, cycle(colors)):
            if w == "\n":
                print(self.format_color(str(t), c), end="")
            else:
                print(self.format_color(str(t), c), end="")

        print("\n\n")

        for t, w, c in zip(tokens, token_words, cycle(colors)):
            if w.isspace():
                if print_newline_in_color:
                    print(self.format_color(str(w), c), end="")
                else:
                    print(str(w), end="")
            else:
                print(self.format_color(str(w), c), end="")
        print("")

        if print_total:
            print(f"Total {len(tokens)} tokens")

    def print_pretty_text(self, text: str, print_total=False):
        tokens = self.encode(text)
        self.print_pretty_tokens(tokens, print_total)

    def print_pretty(self, test_or_tokens: Union[str, List[int]], print_total=False):
        if isinstance(test_or_tokens, str):
            self.print_pretty_text(test_or_tokens, print_total=print_total)
        elif isinstance(test_or_tokens, list):
            self.print_pretty_tokens(test_or_tokens, print_total=print_total)
        else:
            raise ValueError(
                f"Invalid input type for print_pretty. Must be str or list of ints. Found {type(test_or_tokens)}"
            )

    def align_tokens_to_text(self, tokens, reverse=False):
        processed_tokens = []
        processed_strs = []

        pred = []
        for t in tokens:
            unicode_error = False
            dec = ""

            curr = pred + [t]

            try:
                dec = self.decode(curr)
            except UnicodeDecodeError:
                unicode_error = True

            if (
                (len(dec) > 1)
                or (len(dec) == 1 and ord(dec) != self.NOT_COMPLETE_SYMBOL_ORD)
                or unicode_error
            ):
                processed_tokens.append(tuple(curr))
                processed_strs.append(dec)
                pred = []
            else:
                pred.append(t)

        if reverse:
            processed_tokens = processed_tokens[::-1]
            processed_strs = processed_strs[::-1]

        return processed_tokens, processed_strs

    @abstractmethod
    def count_unknown(self, text: str) -> int:
        raise NotImplementedError


class HuggingFaceTokenizer(TokenizerInterface):

    NOT_COMPLETE_SYMBOL_ORD = 65533
    init_kwargs = {"use_fast": False}

    def __init__(self, tokenizer_name: str):
        self.tokenizer_name = tokenizer_name
        self.encoder = AutoTokenizer.from_pretrained(tokenizer_name, **self.init_kwargs)

    def encode(self, text: str) -> List[int]:
        return self.encoder.convert_tokens_to_ids(self.encoder.tokenize(text))

    def decode(self, tokens: List[int]) -> str:
        raw_tokens = self.encoder.convert_ids_to_tokens(tokens)

        # we want to keep the begnning space of the first token; the space of the following tokens
        # is handled by the sp.model
        if raw_tokens[0].startswith(chr(9601)):
            string = " " + self.encoder.decode(tokens)
        else:
            string = self.encoder.decode(tokens)
        return string

    @property
    def pretty_name(self) -> str:
        return self.tokenizer_name

    def count_unknown(self, text: str) -> int:
        unknown_token = self.encoder.convert_tokens_to_ids([self.encoder.unk_token])[0]
        tokens = self.encode(text)
        tokens_wo_unk = [t for t in tokens if t != unknown_token]
        text_wo_unk = self.decode(tokens_wo_unk)
        return max(0, int(len(tokens) * (len(text) - len(text_wo_unk)) / len(text)))


class ColorLZW_Tokenizer(TokenizerInterface):
    def __init__(self, tokenizer: Legacy_LZW_Tokenizer):
        self.lzw_tokenizer = tokenizer
        self.codebook = None
        self._last_codebook_hash = None
        self._cached_hf_tokenizer = None

    def get_hf_tokenizer(self, codebook: Codebook):
        codebook_hash = codebook.get_hash()
        if codebook_hash != self._last_codebook_hash:
            self._cached_hf_tokenizer = self.lzw_tokenizer.export_to_hf_tokenizer(
                codebook
            )
            self._last_codebook_hash = codebook_hash
        return self._cached_hf_tokenizer

    def encode(self, text: str) -> List[int]:
        lzw_tokenization = self.lzw_tokenizer.encode(text)
        self.codebook = lzw_tokenization.codebook
        return lzw_tokenization.token_ids

    def decode(self, tokens: List[int]) -> str:
        hf_tokenizer = self.get_hf_tokenizer(self.codebook)
        raw_tokens = hf_tokenizer.convert_ids_to_tokens(tokens)
        if raw_tokens[0].startswith(chr(9601)):
            text = " " + hf_tokenizer.decode(tokens)
        else:
            text = hf_tokenizer.decode(tokens)
        return text

    def pretty_name(self) -> str:
        return "Color LZW Tokenizer"

    def count_unknown(self, text: str) -> int:
        return 0
