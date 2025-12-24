from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import SPMStreamingDetokenizer, TokenizerWrapper

from inference.mlx.codebook import CodebookManager


class LZWSPMStreamingDetokenizer(SPMStreamingDetokenizer):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        initial_vocab_size: int,
        codebook_manager: CodebookManager,
        trim_space: bool = True,
    ) -> None:
        super().__init__(tokenizer, trim_space)
        self.codebook_manager = codebook_manager
        self.initial_vocab_size = initial_vocab_size

    def add_token(self, token: int) -> None:
        tokens = []
        if token >= self.initial_vocab_size:
            tokens = self.codebook_manager.get_subtokens(token)
        else:
            tokens.append(token)

        for token in tokens:
            super().add_token(token)

    @staticmethod
    def from_tokenizer(
        tokenizer: TokenizerWrapper,
        codebook_manager: CodebookManager,
        initial_vocab_size: int,
    ) -> "LZWSPMStreamingDetokenizer":
        return LZWSPMStreamingDetokenizer(
            tokenizer._tokenizer,
            initial_vocab_size,
            codebook_manager,
            tokenizer._detokenizer.trim_space,
        )
