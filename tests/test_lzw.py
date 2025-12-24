import sys
import os
import pytest
from fast_compression import lzw_compress

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import lzw_decomp


# the target is from https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch#Example
# original implementation starts from 1, but we start from 0, so we need to minus 1
wikipedia_example = {
    "text": "TOBEORNOTTOBEORTOBEORNOT",
    "compressed_ids": [19, 14, 1, 4, 14, 17, 13, 14, 19, 26, 28, 30, 35, 29, 31, 33],
}
# map A-Z to 1-26 using ord
wikipedia_example["ids"] = [ord(c) - 65 for c in wikipedia_example["text"]]


def test_lzw_compress():
    base_ids = wikipedia_example["ids"]

    target_compressed_ids = wikipedia_example["compressed_ids"]

    initial_vocab_size = 26
    extra_vocab_size = 1024  # a large enough number
    out_seq_length = 1024  # a large enough number
    max_subtokens = 6

    # Call the function
    compressed_chunks = lzw_compress(
        ids=base_ids,
        initial_vocab_size=initial_vocab_size,
        extra_vocab_size=extra_vocab_size,
        max_out_seq_length=out_seq_length,
        max_subtokens=max_subtokens,
    )
    compressed_ids, codebook_dict = compressed_chunks[0]

    # order the merges by the values
    codebook_dict = dict(sorted(codebook_dict.items(), key=lambda item: item[1]))
    print(base_ids)
    print(codebook_dict)
    print(compressed_ids)

    assert compressed_ids == target_compressed_ids, "The compressed IDs are incorrect"
    assert len(codebook_dict) + 26 == 41, "The number of merges is incorrect"

    decompressed_ids, reconstructed_codebook_dict = lzw_decomp(
        compressed_ids, initial_vocab_size
    )

    assert (
        decompressed_ids == base_ids
    ), "The decompressed IDs are not identical to the original IDs"
    assert (
        codebook_dict == reconstructed_codebook_dict
    ), "The reconstructed codebook in decompression is not identical to the codebook built in compression"


def test_lzw_compress_with_zero_vocab_size():
    base_ids = wikipedia_example["ids"]

    target_compressed_ids = wikipedia_example["compressed_ids"]

    initial_vocab_size = 26
    extra_vocab_size = 0
    out_seq_length = 1024  # a large enough number
    max_subtokens = 6

    # Call the function
    compressed_chunks = lzw_compress(
        ids=base_ids,
        initial_vocab_size=initial_vocab_size,
        extra_vocab_size=extra_vocab_size,
        max_out_seq_length=out_seq_length,
        max_subtokens=max_subtokens,
    )
    compressed_ids, codebook_dict = compressed_chunks[0]

    assert compressed_ids == base_ids, "The compressed IDs are incorrect"
    assert type(codebook_dict) == dict, "The codebook_dict should be a dictionary"
    assert len(codebook_dict) == 0, "The codebook_dict should be empty"

    decompressed_ids, reconstructed_codebook_dict = lzw_decomp(
        compressed_ids, initial_vocab_size, extra_vocab_size=extra_vocab_size
    )

    assert (
        decompressed_ids == base_ids
    ), "The decompressed IDs are not identical to the original IDs"
    assert (
        codebook_dict == reconstructed_codebook_dict
    ), "The reconstructed codebook in decompression is not identical to the codebook built in compression"


def test_lzw_chunk_compress():
    base_ids = wikipedia_example["ids"] * 3

    initial_vocab_size = 26
    extra_vocab_size = 1024
    out_seq_length = len(wikipedia_example["compressed_ids"])
    max_subtokens = 6

    compressed_chunks = lzw_compress(
        ids=base_ids,
        initial_vocab_size=initial_vocab_size,
        extra_vocab_size=extra_vocab_size,
        max_out_seq_length=out_seq_length,
        max_subtokens=max_subtokens,
    )

    for compressed_ids, codebook_dict in compressed_chunks[:-1]:
        assert (
            len(compressed_ids) == out_seq_length
        ), "The length of the compressed chunk is incorrect"


test_lzw_compress_with_zero_vocab_size()
