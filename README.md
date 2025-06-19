# zip2zip (Beta)

zip2zip is a framework for inference-time adaptive token vocabularies for LLMs. It enables dynamic vocabulary adaptation during inference, allowing for more efficient and flexible language model usage.

## Features

- Dynamic vocabulary adaptation during inference
- LZW-based token compression
- Support for various encoder configurations
- Integration with Hugging Face's transformers library
- Compatible with PEFT (Parameter-Efficient Fine-Tuning) models

## Installation

You can install zip2zip using pip:

```bash
pip install git+https://github.com/epfl-dlab/zip2zip-internal-testing.git
```

## Usage

### Generate text with a pretrained model

```python
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer, Zip2ZipConfig

pretrained_model_url = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"

# Initialize tokenizer
tokenizer = Zip2ZipTokenizer.from_pretrained(pretrained_model_url)

# Initialize model
model = Zip2ZipModel.from_pretrained(pretrained_model_url, device_map="auto")

# Generate text
inputs = tokenizer("Write a MultiHeadAttention layer in PyTorch", return_tensors="pt")
outputs = model.generate(**inputs)
generated_text = tokenizer.batch_decode(outputs)
```

You can apply quantization to the model to reduce the memory usage just as you would do with HF models.

```python
model = Zip2ZipModel.from_pretrained(pretrained_model_url, device_map="auto", load_in_8bit=True)
```


### Examples

We provide some examples in the `examples` folder.

### Advanced Usage

The framework supports various configurations for compression and encoding:

```python
from zip2zip.config import CompressionConfig, EncoderConfig

# Custom compression configuration
compression_config = CompressionConfig(
    max_codebook_size=8192,
    max_subtokens=4,
    disabled_ids=set()
)

# Custom encoder configuration
encoder_config = EncoderConfig(
    # Add your encoder-specific configuration here
)

# Create custom config
config = Zip2ZipConfig(
    base_model_name_or_path="your-base-model",
    compression=compression_config,
    encoder=encoder_config
)
```

## Pretrained models

We provide pretrained models for zip2zip at [Hugging Face](https://huggingface.co/collections/epfl-dlab/zip2zip-models-6852ec90f3dacc02aa6a0dca), including:

- `epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1`: Phi-3.5-mini-instruct-v0.1
- `epfl-dlab/zip2zip-Llama-3.1-8B-Instruct-v0.1`: Llama-3.1-8B-Instruct-v0.1


## Citation

[Add citation information here]
