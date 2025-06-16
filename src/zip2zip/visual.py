from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
import dataclasses

from typing import Dict, List, Set, Union


@dataclasses.dataclass
class ColoredToken:
    # colored token for printing
    token_ids: List[int]
    color: str = "\033[0m"  # default to no color
    END_COLOR: str = "\033[0m"  # Reset color

    def __str__(self):

        return f"{self.color}{self.token_ids}{self.END_COLOR}"


def colorise_lzwtokens(
    lzw_token_ids: List[int],
    codebook: Dict[int, List[int]],
    color_scheme: str = "finegrained",
    special_token_ids: Set[int] = None,
) -> List[ColoredToken]:
    """
    Given a sequence of token IDs and a codebook, prints the tokens in a colored format with base tokens in one color and hypertokens in another.
    Special tokens are printed in black.
    """
    basetoken_color = BLUE = "\033[34m"  # blue
    hypertoken_color = YELLOW = "\033[33m"  # yellow
    size3_hypertoken_color = ORANGE = "\033[38;5;208m"  # orange
    size4_hypertoken_color = RED = "\033[31m"  # red
    size5_hypertoken_color = DARK_RED = "\033[38;5;88m"  # dark red
    size_n_hypertoken_color = BROWN = "\033[38;5;130m"
    special_token_color = BLACK = "\033[30m"

    token_color_scheme = {
        1: basetoken_color,
        2: hypertoken_color,
    }  # map from token size to color

    finegrained_token_color_scheme = {
        1: basetoken_color,
        2: hypertoken_color,
        3: size3_hypertoken_color,
        4: size4_hypertoken_color,
        5: size5_hypertoken_color,
    }  # map from token size to color

    more_finegrained_token_color_scheme = {
        1: {  # blue
            0: basetoken_color,
        },
        2: {0: hypertoken_color, 1: size3_hypertoken_color},  # yellow
        3: {  # orange, red
            0: hypertoken_color,
            1: size3_hypertoken_color,
            2: size4_hypertoken_color,
        },  # brown
        4: {
            0: hypertoken_color,
            1: size3_hypertoken_color,
            2: size4_hypertoken_color,
            3: size5_hypertoken_color,
        },
    }  # map from token size to color

    colored_token_groups = []

    for token_id in lzw_token_ids:
        if token_id in codebook:
            base_ids = codebook[token_id]
        else:
            base_ids = [token_id]
        token_size = len(base_ids)
        if color_scheme == "basic":
            color = token_color_scheme.get(token_size, hypertoken_color)
        elif color_scheme == "finegrained":
            color = finegrained_token_color_scheme.get(
                token_size, size_n_hypertoken_color
            )
        elif color_scheme == "more_finegrained":
            color = more_finegrained_token_color_scheme[token_size]
        else:
            raise ValueError(f"Invalid color scheme: {color_scheme}")

        # check if the token is a special token
        if token_id in special_token_ids:
            color = special_token_color

        colored_token_groups.append(ColoredToken(token_ids=base_ids, color=color))

    return colored_token_groups


def colorise_lzw_tokens_by_ppl(
    lzw_token_ids: List[int], norm_ppl: List[float], codebook: Dict[int, List[int]]
) -> List[ColoredToken]:
    """
    Given a sequence of token IDs and a list of ppl values, prints the tokens in a gradient of green
    """
    # normalize the ppl to be between 0 and 1
    norm_ppl = [float(e) / max(norm_ppl) for e in norm_ppl]

    ppl_to_intensity = lambda ppl: int(255 * (1 - ppl))

    # map the ppl to a color
    color = (
        lambda ppl: f"\033[38;2;{ppl_to_intensity(ppl)};{ppl_to_intensity(ppl)};255m"
    )

    colored_token_groups = []

    for i, token_id in enumerate(lzw_token_ids):
        if token_id in codebook:
            base_ids = codebook[token_id]
        else:
            base_ids = [token_id]
        colored_token_groups.append(
            ColoredToken(token_ids=base_ids, color=color(norm_ppl[i]))
        )

    return colored_token_groups


def colorise_lzw_tokens_random(
    lzw_token_ids: List[int], codebook: Dict[int, List[int]]
) -> List[ColoredToken]:
    """
    Given a sequence of token IDs and a codebook, prints the tokens in a random color
    """
    import random

    color = lambda e: f"\033[38;5;{random.randint(0, 255)}m"
    colored_token_groups = []
    for token_id in lzw_token_ids:
        if token_id in codebook:
            base_ids = codebook[token_id]
        else:
            base_ids = [token_id]
        colored_token_groups.append(ColoredToken(token_ids=base_ids, color=color(i)))
    return colored_token_groups


def legacy_contrast_colorprint_tokens(
    token_ids: List[int],
    codebook: Dict[int, List[int]],
    tokenizer: PreTrainedTokenizer,
    color_scheme: str = "finegrained",  # "finegrained", "more_finegrained"
) -> None:
    """
    Given a sequence of token IDs and a codebook, prints the tokens in a colored format with base tokens in one color and hypertokens in another.
    Special tokens are printed in black.
    """
    basetoken_color = BLUE = "\033[34m"  # blue
    hypertoken_color = YELLOW = "\033[33m"  # yellow
    size3_hypertoken_color = ORANGE = "\033[38;5;208m"  # orange
    size4_hypertoken_color = RED = "\033[31m"  # red
    size5_hypertoken_color = DARK_RED = "\033[38;5;88m"  # dark red
    size_n_hypertoken_color = BROWN = "\033[38;5;130m"
    special_token_color = BLACK = "\033[30m"
    RESET = "\033[0m"

    token_color_scheme = {
        1: basetoken_color,
        2: hypertoken_color,
    }  # map from token size to color

    finegrained_token_color_scheme = {
        1: basetoken_color,
        2: hypertoken_color,
        3: size3_hypertoken_color,
        4: size4_hypertoken_color,
        5: size5_hypertoken_color,
    }  # map from token size to color

    more_finegrained_token_color_scheme = {
        1: {  # blue
            0: basetoken_color,
        },
        2: {0: hypertoken_color, 1: size3_hypertoken_color},  # yellow
        3: {  # orange, red
            0: hypertoken_color,
            1: size3_hypertoken_color,
            2: size4_hypertoken_color,
        },  # brown
        4: {
            0: hypertoken_color,
            1: size3_hypertoken_color,
            2: size4_hypertoken_color,
            3: size5_hypertoken_color,
        },
    }  # map from token size to color

    colored_string = ""

    def decode(tokenizer, tokens: List[int]) -> str:
        raw_tokens = tokenizer.convert_ids_to_tokens(tokens)

        # we want to keep the begnning space of the first token; the space of the following tokens
        # is handled by the sp.model
        if raw_tokens[0].startswith(chr(9601)):
            string = " " + tokenizer.decode(tokens)
        else:
            string = tokenizer.decode(tokens)
        return string

    for token_id in token_ids:
        if token_id in codebook:
            # if fine_grained_colors:
            base_ids = codebook[token_id]
            # color = hypertoken_color
        else:
            base_ids = [token_id]
            # color = basetoken_color
        # use the color of the hypertoken of a given size
        token_size = len(base_ids)

        if color_scheme == "basic":
            color = token_color_scheme.get(token_size, hypertoken_color)
            colored_string += f"{color}{decode(tokenizer, base_ids)}{RESET}"
        elif color_scheme == "finegrained":
            color = finegrained_token_color_scheme.get(
                token_size, size_n_hypertoken_color
            )
            colored_string += f"{color}{decode(tokenizer, base_ids)}{RESET}"
        elif color_scheme == "more_finegrained":
            sub_color_scheme = more_finegrained_token_color_scheme[token_size]
            for i, base_id in enumerate(base_ids):
                color = sub_color_scheme[i]
                colored_string += f"{color}{decode(tokenizer, [base_id])}{RESET}"
        else:
            raise ValueError(f"Invalid color scheme: {color_scheme}")

    if "<|end|>" in colored_string:
        # make the conversion more readable
        colored_string = colored_string.replace("<|end|>", "<|end|>\n")

    # make the special tokens in black
    colored_string = colored_string.replace(
        "<|user|>", f"{special_token_color}<|user|>{RESET}"
    )
    colored_string = colored_string.replace(
        "<|assistant|>", f"{special_token_color}<|assistant|>{RESET}"
    )
    colored_string = colored_string.replace(
        "<|end|>", f"{special_token_color}<|end|>{RESET}"
    )

    print(colored_string)


class ColorfulTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __getattr__(self, attr):
        return getattr(self.tokenizer, attr)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def decode_preserving_leading_space(self, tokens: List[int]) -> str:
        raw_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        if any(token is None for token in raw_tokens):
            raise ValueError(f"Encountered tokens non-decodable: {tokens}")

        # we want to keep the begnning space of the first token; the space of the following tokens
        # is handled by the sp.model
        if raw_tokens[0].startswith(chr(9601)):
            string = " " + self.tokenizer.decode(tokens)
        else:
            string = self.tokenizer.decode(tokens)
        return string

    def decode_colored_token(
        self, colored_tokens: Union[ColoredToken, List[ColoredToken]]
    ) -> str:
        if isinstance(colored_tokens, ColoredToken):
            colored_tokens = [colored_tokens]
        out = ""
        for colored_token in colored_tokens:
            out += f"{colored_token.color}{self.decode_preserving_leading_space(colored_token.token_ids)}{colored_token.END_COLOR}"
        return out

    @staticmethod
    def random_colorise_tokens(tokens: List[int]) -> List[ColoredToken]:
        """
        Given a sequence of token IDs, prints the tokens in a random color
        """
        import random

        color = lambda e: f"\033[38;5;{random.randint(0, 255)}m"
        return [
            ColoredToken(token_ids=[token_id], color=color(i))
            for i, token_id in enumerate(tokens)
        ]

    def random_color_decode(self, token_ids: List[int]) -> str:
        """
        Given a sequence of token IDs, prints the tokens in a random color
        """
        colored_token_groups = self.random_colorise_tokens(token_ids)
        return self.decode_colored_token(colored_token_groups)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = ColorfulTokenizer(
        AutoTokenizer.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
        )
    )
    # Read this script's own source code
    with open(__file__, "r") as f:
        text = f.read()
    token_ids = tokenizer.encode(text)

    print(tokenizer.random_color_decode(token_ids))
