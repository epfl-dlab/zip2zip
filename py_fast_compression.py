from typing import List, Optional, Tuple
from custom_types import Codebook, BatchedLZWTokenization, LZWTokenization
import fast_compression  # <-- Your compiled module


def batched_lzw_compress(
    ids: List[List[int]],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
    pad_token_id: int = 0,
) -> BatchedLZWTokenization:
    if disabled_ids is None:
        disabled_ids = []
    if pad_token_id is None:
        raise ValueError("pad_token_id is required")
    raw_results = fast_compression.batch_lzw_compress(
        ids,
        initial_vocab_size,
        extra_vocab_size,
        max_out_seq_length,
        max_subtokens,
        disabled_ids,
    )
    token_ids = []
    codebooks = []
    for compressed_ids, token_map in raw_results:
        codebook = Codebook.from_token_map(
            token_map, initial_vocab_size, extra_vocab_size, max_subtokens, pad_token_id
        )
        token_ids.append(compressed_ids)
        codebooks.append(codebook)
    return BatchedLZWTokenization(token_ids, codebooks)


def lzw_compress(
    ids: List[int],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    disabled_ids: Optional[List[int]] = None,
    pad_token_id: int = 0,
) -> LZWTokenization:
    return batched_lzw_compress(
        [ids],
        initial_vocab_size,
        extra_vocab_size,
        max_out_seq_length,
        max_subtokens,
        disabled_ids,
        pad_token_id,
    )[0]
