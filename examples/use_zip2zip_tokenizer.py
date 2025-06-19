from zip2zip.tokenizer import Zip2ZipTokenizer


def tokenizer():
    model_name = "Saibo-creator/zip2zip-Phi-3.5-mini-instruct-v0.1"
    return Zip2ZipTokenizer.from_pretrained(model_name)


def test_encode_decode_with_codebook(tokenizer):
    text = "Hello, world! " * 3

    # Encode with return_codebook=True
    encoding = tokenizer(text, return_codebook=True, padding=True)
    assert "input_ids" in encoding
    assert "codebooks" in encoding

    # Check codebooks type
    encoding_codebook = encoding["codebooks"][0]

    # Decode with return_codebook=True
    decoded_text, decoding_codebook = tokenizer.decode(
        encoding["input_ids"][0], return_codebook=True
    )

    assert isinstance(decoded_text, str)
    assert text == decoded_text

    assert encoding_codebook.to_dict() == decoding_codebook.to_dict()


if __name__ == "__main__":
    tokenizer = tokenizer()
    test_encode_decode_with_codebook(tokenizer)
