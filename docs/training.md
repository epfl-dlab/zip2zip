# Training a `zip2zip` model

## Finetuning using [TRL](https://github.com/huggingface/trl)

```python
from zip2zip import (
    Zip2ZipModel,
    Zip2ZipTokenizer,
    Zip2ZipConfig,
    EncoderType,
    TransformerEncoderConfig,
    CompressionConfig,
)

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from peft import LoraConfig, TaskType

dataset = load_dataset("epfl-dlab/zip2zip-1B", name="code", split="validation")

config = Zip2ZipConfig(
    "microsoft/Phi-3.5-mini-instruct",
    encoder_type=EncoderType.TRANSFORMER,
    encoder=TransformerEncoderConfig(
        hidden_size=3072,
        tie_encoders=False,
        num_hidden_layers=2,
        intermediate_size=12288,
        num_heads=32,
    ),
    compression=CompressionConfig(
        initial_vocab_size=32011,
        max_codebook_size=2048,
        max_subtokens=4,
    ),
)

model = Zip2ZipModel(config, device_map="cuda")
tokenizer = Zip2ZipTokenizer(config)

trainer_args = SFTConfig(
    max_length=256,
    output_dir="zip2zip-train-debug/",
    torch_compile=True,
)

trainer = SFTTrainer(
    model,
    args=trainer_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["qkv_proj"],
    ),
)

trainer.train()

```
