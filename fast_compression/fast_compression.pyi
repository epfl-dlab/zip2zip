from typing import List, Optional, Tuple, Dict

def lzw_compress(
    ids: List[int],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
) -> List[Tuple[List[int], Dict[str, int]]]: ...

def batch_lzw_compress(
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
    def __init__(self, initial_vocab_size: int, max_codebook_size: int, max_subtokens: int, pad_token_id: int, disabled_ids: Optional[List[int]] = None): ...

    def get_subtokens(self, id: int) -> List[int]: ...

    def update_codebook(self, ids: List[int], prefill: bool) -> Tuple[List[List[int]], int]: ...

    def reset(self): ...
