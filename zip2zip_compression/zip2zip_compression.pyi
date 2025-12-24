from typing import List, Optional, Tuple, Dict

def bounded_lzw_in_chunks(
    ids: List[int],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[Tuple[List[int], Dict[str, int]]]: ...
def batch_bounded_lzw_in_chunks(
    ids: List[List[int]],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[Tuple[List[int], Dict[str, int]]]: ...
def encode(
    ids: List[int],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]: ...
def batch_encode(
    ids: List[List[int]],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> Tuple[List[List[int]], List[List[int]]]: ...
def decode(
    compressed_ids: List[int],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[int]: ...
def batch_decode(
    compressed_ids: List[List[int]],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[List[int]]: ...

class CodebookManager:
    def __init__(
        self,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        pad_token_id: int,
        disabled_ids: Optional[List[int]] = None,
    ): ...
    @property
    def codebook(self) -> Dict[int, List[int]]: ...
    def get_subtokens(self, id: int) -> List[int]: ...
    def update_codebook(
        self, ids: List[int], return_all_entries: bool
    ) -> Tuple[List[List[int]], int]: ...
    def reset(self): ...
