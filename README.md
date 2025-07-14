# zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01084-b31b1b.svg)](https://arxiv.org/abs/2506.01084)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Zip2Zip-yellow.svg)](https://huggingface.co/collections/epfl-dlab/zip2zip-models-6852ec90f3dacc02aa6a0dca)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

zip2zip enables inference-time adaptive token vocabularies for large language models (LLMs). It allows vocabularies to be dynamically augmented at inference time, leading to reduced decoding steps and faster inference.

<p align = 'center'>
  <img alt="zip2zip decoding" src='assets/zip2zip-decoding.gif' width='75%'/>
</p>

## Features

- Dynamic vocabulary adaptation during inference
- LZW-based token compression
- Support for various encoder configurations
- Integration with Hugging Face's transformers library
- Compatible with PEFT (Parameter-Efficient Fine-Tuning) models

## Installation

You can install zip2zip using pip:

```bash
pip install zip2zip
```

## Usage

### Same API as Hugging Face

| zip2zip | Corresponding HF class |
|---------|-------------------------|
| Zip2ZipModel | AutoModelForCausalLM |
| Zip2ZipTokenizer | AutoTokenizer |
| Zip2ZipConfig | AutoConfig |
| Zip2ZipModel.from_pretrained | AutoModelForCausalLM.from_pretrained |
| Zip2ZipTokenizer.from_pretrained | AutoTokenizer.from_pretrained |
| Zip2ZipConfig.from_pretrained | AutoConfig.from_pretrained |



### Pretrained model weights

| Size | Model | HF Hub |
|------|-------|--------|
| 3.8B | Phi-3.5-mini-instruct-v0.1 | [epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1](https://huggingface.co/epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1) |
| 14B | Llama-3.1-8B-Instruct-v0.1 | [epfl-dlab/zip2zip-Phi-3-medium-instruct-v0.1](https://huggingface.co/epfl-dlab/zip2zip-Phi-3-medium-instruct-v0.1) |
| ... | ... | [epfl-dlab/zip2zip-models](https://huggingface.co/collections/epfl-dlab/zip2zip-models-6852ec90f3dacc02aa6a0dca) |




### Run a pretrained model

```python
import torch
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer

pretrained_model_url = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize tokenizer
tokenizer = Zip2ZipTokenizer.from_pretrained(pretrained_model_url)

# Initialize model
model = Zip2ZipModel.from_pretrained(pretrained_model_url, device_map=device)

# Generate text
inputs = tokenizer("Write a MultiHeadAttention layer in PyTorch", return_tensors="pt").to(device)
outputs = model.generate(**inputs)

# Print the coloried
generated_text = tokenizer.color_decode(outputs)
```

You can apply quantization to the model to reduce the memory usage just as you would do with HF models.

```python
model = Zip2ZipModel.from_pretrained(pretrained_model_url, device_map="auto", load_in_8bit=True)
```


### Examples

We provide some examples in the `examples` folder.


## Evaluation

We provide a script to evaluate the performance of the model, compatible with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

To run the evaluation, you need to install the zip2zip fork of lm-evaluation-harness (the original one is not compatible with zip2zip).

```bash
pip install git+https://github.com/epfl-dlab/zip2zip_lm_eval.git
```

Then, you can run the evaluation:

```bash
python bench/run_lm_eval.py
```



## Citation

```bibtex
@misc{geng2025zip2zipinferencetimeadaptivevocabularies,
      title={zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression},
      author={Saibo Geng and Nathan Ranchin and Yunzhen yao and Maxime Peyrard and Chris Wendler and Michael Gastpar and Robert West},
      year={2025},
      eprint={2506.01084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01084},
}
```
