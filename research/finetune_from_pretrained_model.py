import os
import shutil
from transformers import AutoTokenizer
from zip2zip import (
    Zip2ZipModel,
    Zip2ZipTokenizer,
    Zip2ZipConfig,
    EncoderType,
    TransformerEncoderConfig,
    CompressionConfig,
)
from zip2zip.utils import (
    setup_seed,
    print_trainable_parameters,
    print_trainable_modules,
)

import torch
from datasets import load_dataset
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType
import transformers

OUTPUT_DIR = "out"

trainer_name = "zip2zip-train-FT-M3-L2-S1024"

trainer_output_dir = os.path.join(OUTPUT_DIR, trainer_name)

if os.path.exists(trainer_output_dir):
    resume_from_checkpoint = True
else:
    resume_from_checkpoint = False

transformers.logging.set_verbosity_info()

# Set random seed for reproducible weight initialization
setup_seed(42, strict_deterministic=False)

current_device = Accelerator().process_index

train_dataset = load_dataset(
    "epfl-dlab/zip2zip-1B", name="default", split="train"
).shuffle(seed=42)
eval_dataset = (
    load_dataset("epfl-dlab/zip2zip-1B", name="default", split="validation")
    .shuffle(seed=42)
    .take(250)
)

config = Zip2ZipConfig(
    "microsoft/Phi-3.5-mini-instruct",
    encoder_type=EncoderType.TRANSFORMER,
    encoder=TransformerEncoderConfig(
        hidden_size=3072,
        tie_encoders=False,
        num_hidden_layers=2,
        position_encoding="learnable",
        intermediate_size=12288,
        num_heads=32,
    ),
    compression=CompressionConfig.from_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct"),
        max_codebook_size=1024,
        max_subtokens=3,
    ),
)

MODE = "Full"  # "Full" or "Lora"


if MODE == "Lora":
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            "up_proj",
        ],
    )
else:
    peft_config = None

model = Zip2ZipModel(
    config,
    peft_config=peft_config,
    device_map={"": current_device},
    torch_dtype=torch.bfloat16,
)

tokenizer = Zip2ZipTokenizer(config)

# Auto-save the current script to output directory
script_path = __file__
output_script_path = os.path.join(trainer_output_dir, "run_trainer.py")
os.makedirs(os.path.dirname(output_script_path), exist_ok=True)
shutil.copy2(script_path, output_script_path)

trainer_args = SFTConfig(
    max_length=1024,
    output_dir=trainer_output_dir,
    packing=False,
    # torch_compile=True,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    save_steps=1000,
    save_total_limit=10,
    eval_strategy="steps",
    eval_steps=100,
    data_seed=42,
    max_steps=36_000,
    learning_rate=3e-5,
    warmup_steps=1000,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-5},
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print_trainable_parameters(model)
print_trainable_modules(model)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)

trainer.save_model(os.path.join(trainer_output_dir, "final"))
