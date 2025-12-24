# zip2zip_compression



**zip2zip_compression** is a Rust-based LZW compression module for use with zip2zip projects.
It leverages the power of Rust for computational efficiency, multi-threading via Rayon and Python bindings via pyo3.

## Installation

You can build and install the package using maturin:

```bash
pip install maturin
maturin develop  # or `maturin build` to create a wheel
```
Or install directly into your Python environment:

```bash
pip install .
```

Note: Ensure you have Rust installed.

## Usage

After installing, you can import and use the module in Python:

```python
import zip2zip_compression

# Example usage (fill in with real API once available)
result = zip2zip_compression.encode(...)
```


## Core Functions

- `encode`

The `encode` function compresses a sequence of token IDs using the LZW algorithm. You can control the compression behavior with `initial_vocab_size`, `max_codebook_size`, and `max_subtokens`.

For example, given `ids = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]` and an `initial_vocab_size` of 6, you can compress it like this:

```python
import zip2zip_compression
ids = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
initial_vocab_size = 6
max_codebook_size = 1000  # A high number to avoid limiting the codebook size
max_subtokens = 4       # Don't merge more than 4 tokens into a new token
compressed_ids, _ = zip2zip_compression.encode(ids, initial_vocab_size, max_codebook_size, max_subtokens)
print(compressed_ids)
# Output: [0, 1, 2, 3, 4, 5, 6, 8, 10]
```

You can also prevent specific token IDs from being merged by using the `disabled_ids` parameter:

```python
disabled_ids = [0]
compressed_ids, _ = zip2zip_compression.encode(ids, initial_vocab_size, max_codebook_size, max_subtokens, disabled_ids)
print(compressed_ids)
# Output: [0, 1, 2, 3, 4, 5, 0, 6, 8, 5]
```
The output differs because token ID 0 is not merged and remains in its original position.

- `decode`

The `decode` function performs LZW decompression. To decompress the `compressed_ids` from the previous example:

```python
decoded_ids = zip2zip_compression.decode(compressed_ids, initial_vocab_size, max_codebook_size, max_subtokens)
print(decoded_ids)
# Output: [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
```

-  `batch_encode` and `batch_decode`

These functions provide the same `encode` and `decode` functionality but operate on a list of sequences, making them efficient for batch processing.

```python
batch_ids = [ids, ids, ids]
batch_encoded_ids, _ = zip2zip_compression.batch_encode(batch_ids, initial_vocab_size, max_codebook_size, max_subtokens)
print(batch_encoded_ids)
# Output: [[0, 1, 2, 3, 4, 5, 6, 8, 10], [0, 1, 2, 3, 4, 5, 6, 8, 10], [0, 1, 2, 3, 4, 5, 6, 8, 10]]

decoded_ids_list = zip2zip_compression.batch_decode(batch_encoded_ids, initial_vocab_size, max_codebook_size, max_subtokens)
print(decoded_ids_list)
# Output: [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]]
```

-  `bounded_lzw_in_chunks`

This function is a bounded version of LZW compression, designed for scenarios where you need to process sequences in fixed-size chunks, such as preparing training data for LLMs. It iteratively compresses the input sequence into chunks of `max_out_seq_length`.

```python
max_out_seq_length = 6
lzw_chunks = zip2zip_compression.bounded_lzw_in_chunks(ids, initial_vocab_size, max_codebook_size, max_out_seq_length, max_subtokens)
print(lzw_chunks)
# Output: [([0, 1, 2, 3, 4, 5], {'0,1': 6, '4,5': 10, '3,4': 9, '1,2': 7, '2,3': 8}), ([0, 1, 2, 3, 4, 5], {'4,5': 10, '3,4': 9, '1,2': 7, '2,3': 8, '0,1': 6})]
```
The output is a list of tuples, with each tuple containing the compressed token IDs and their corresponding codebook.

- `batch_bounded_lzw_in_chunks`

Similar to `bounded_lzw_in_chunks`, this function processes a list of input sequences, returning a concatenated list of all compressed chunks.

```python
batch_ids = [ids, ids]
batch_lzw_chunks = zip2zip_compression.batch_bounded_lzw_in_chunks(batch_ids, initial_vocab_size, max_codebook_size, max_out_seq_length, max_subtokens)
print(batch_lzw_chunks)
# Output: [([0, 1, 2, 3, 4, 5], {'3,4': 9, '2,3': 8, '0,1': 6, '4,5': 10, '1,2': 7}), ([0, 1, 2, 3, 4, 5], {'1,2': 7, '0,1': 6, '3,4': 9, '4,5': 10, '2,3': 8}), ([0, 1, 2, 3, 4, 5], {'2,3': 8, '1,2': 7, '0,1': 6, '3,4': 9, '4,5': 10}), ([0, 1, 2, 3, 4, 5], {'2,3': 8, '1,2': 7, '4,5': 10, '0,1': 6, '3,4': 9})]
```

## `CodebookManager` Class

The `CodebookManager` is a runtime utility that manages the LZW codebook, which maps hyper-tokens to their corresponding sub-tokens. This is a crucial component in the zip2zip architecture as it:
- Defines dynamic embeddings for hyper-tokens
- Manages incremental codebook updates
- Handles efficient retrieval of original sub-tokens

#### Key Concepts
- **Initial Vocabulary**: The base set of tokens (e.g., 0-5 in the example)
- **Codebook**: A dictionary mapping hyper-tokens to sequences of sub-tokens
- **Merges**: New token combinations that get added to the codebook
- **Max Merge Length**: Maximum number of tokens that can be combined (e.g., 4 in the example)

#### Example Usage

Following the same example as before, one has a prompt = `[0, 1, 2, 3, 4, 5]` and the expected output is `[0, 1, 2, 3, 4, 5]`, which in the case of LZW compression would be `[0, 1, 2, 3, 4, 5, 6, 8, 10]`.

1. **Initialize the `CodebookManager`**

```python
initial_vocab_size = 6 # tokens from 0 to 5
max_codebook_size = 1000 # maximum number of merges in the codebook
max_subtokens = 4 # maximum number of tokens that can be combined into a new token
pad_token_id = 0 # padding token id

codebook_manager = CodebookManager(initial_vocab_size, max_codebook_size, max_subtokens, pad_token_id)
```

2. **Prompt Prefill**
```python
prompt = [0, 1, 2, 3, 4, 5]
updates = codebook_manager.update_codebook(prompt, return_all_entries=False)
print(updates)
# Output: ([[0, 1, 0, 0], [1, 2, 0, 0], [2, 3, 0, 0], [3, 4, 0, 0], [4, 5, 0, 0]], 5)
# This creates 5 new merges in the codebook
# Each new merge gets a unique ID starting from initial_vocab_size
```

The codebook now looks like this:

```python
# Inspect the codebook
print(codebook_manager.codebook)
# Output: {6: [0, 1], 7: [1, 2], 8: [2, 3], 9: [3, 4], 10: [4, 5]}
# Each new merge gets a unique ID starting from initial_vocab_size
```

3. **Incremental Updates**
```python
# When model generates next token (6)
next_token = 6
updates = codebook_manager.update_codebook([next_token], return_all_entries=False)
print(updates)
# ([[5, 0, 0, 0]], 1)
```
This adds the new merge `[5, 0]` to the codebook.

Let's inspect the codebook again

```python
print(codebook_manager.codebook)
# {6: [0, 1], 7: [1, 2], 8: [2, 3], 9: [3, 4],10: [4, 5], 11: [5, 0]}
```

4. **Processing Multiple Tokens**

Everything is working as expected, we have an incremental codebook that is updated with each new token.

Let's fast forward and assume that the model generates all the following tokens: `[8, 10]`

```python
next_tokens = [8, 10]
updates = codebook_manager.update_codebook(next_tokens, return_all_entries=False)
print(updates)
# Output: ([[0, 1, 2, 0], [2, 3, 4, 0]], 2)
# Creates two new merges in the codebook
```
