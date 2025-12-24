"""
# model_name= "microsoft/Phi-3.5-mini-instruct"
bpe_vocab_size: 32000
embed_vocab_size: 32064
trained_vocab_size: 32011

# model_name= "microsoft/Phi-4-mini-instruct"
bpe_vocab_size: 200019
embed_vocab_size: 200064
trained_vocab_size: 200029

# model_name= "meta-llama/Llama-3.2-1B"
bpe_vocab_size: 128000
embed_vocab_size: 128256
trained_vocab_size: 128256

# model_name= "meta-llama/Llama-2-7b-chat-hf
bpe_vocab_size: 32000
embed_vocab_size: 32000
trained_vocab_size: 32000
"""

import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoConfig


def main():
    parser = argparse.ArgumentParser(
        description="Get initial vocabulary sizes for a language model."
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        required=True,
        help="The name of the tokenizer to use, e.g., 'microsoft/Phi-4-mini-instruct'.",
    )
    args = parser.parse_args()

    model_name = args.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    bpe_vocab_size = tokenizer.vocab_size
    print(f"bpe_vocab_size: {bpe_vocab_size}")
    # 32000

    embed_vocab_size = config.vocab_size
    # this is the size that model embeddings are built for
    print(f"embed_vocab_size: {embed_vocab_size}")
    # 32064

    trained_vocab_size = len(tokenizer.vocab)
    print(
        f"trained_vocab_size: {trained_vocab_size} -> go to config.yaml as 'initial_vocab_size'"
    )
    # 32011
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_layers = config.num_hidden_layers

    print(f"hidden_size: {hidden_size}")
    print(f"num_heads: {num_heads}")
    print(f"num_layers: {num_layers}")


if __name__ == "__main__":
    main()
