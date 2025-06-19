# Tokenizer

`zip2zip` tokenizer is a wrapper around a Hugging Face tokenizer that routes all the calls to the underlying tokenizer except for the `encoding` and `decoding` methods where it applies the LZW compression on top of bpe-segmented tokens.

The following diagram illustrates the tokenization process:
```
[text]  --> HF Tokenizer.encode()  --> [base token ids (BPE)] --> lzw_compress() --> [hyper-token ids]
    |                                                                                         |
    |----------------------------------> zip2zip.encode() ----------------------------------->|
```

The `zip2zip` tokenizer also supports decoding the hyper-token ids back to the base token ids.
```
[hyper-token ids]  --> lzw_decompress()  --> [base token ids (BPE)] --> HF Tokenizer.decode() --> [text]
       |                                                                                            |
       |-----------------------------> zip2zip.decode() ------------------------------------------->|
```

## Codebook

Optionally, one can let the tokenizer to return the codebook along with the encoded/decoded tokens. This is useful for debugging and inspecting the compression.

For this, simply pass `return_codebook=True` to the corresponding encoding/decoding methods.

For example,

```python
from zip2zip.tokenizer import Zip2ZipTokenizer

tokenizer = Zip2ZipTokenizer.from_pretrained("epfl-dlab/zip2zip-Llama-3.1-8B-Instruct-v0.1")

encoding = tokenizer("Hello, world!", return_codebook=True)

print(encoding["input_ids"])
print(encoding["codebooks"])

decoding, decoding_codebook = tokenizer.decode(encoding["input_ids"], return_codebook=True)

print(decoding)
print(decoding_codebook)
```
