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

import torch
from datasets import load_dataset
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

current_device = Accelerator().process_index

train_dataset = load_dataset(
    "epfl-dlab/zip2zip-1B", name="default", split="train"
).take(300_000)
eval_dataset = load_dataset(
    "epfl-dlab/zip2zip-1B", name="default", split="validation"
).take(250)

config = Zip2ZipConfig(
    "microsoft/Phi-3.5-mini-instruct",
    encoder_type=EncoderType.TRANSFORMER,
    encoder=TransformerEncoderConfig(
        hidden_size=3072,
        tie_encoders=True,
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

model = Zip2ZipModel(
    config,
    peft_config=LoraConfig(
        r=32,
        lora_alpha=32,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    ),
    device_map={"": current_device},
    torch_dtype=torch.bfloat16,
)

tokenizer = Zip2ZipTokenizer(config)

trainer_args = SFTConfig(
    max_length=2048,
    output_dir="zip2zip-train-debug/",
    packing=False,
    torch_compile=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    eval_strategy="steps",
    eval_steps=100,
    data_seed=42,
    max_steps=18_000,
    learning_rate=1e-5,
    warmup_steps=1_000,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-6},
)

trainer = SFTTrainer(
    model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("zip2zip-train-debug-final/")

```
