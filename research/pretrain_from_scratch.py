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


transformers.logging.set_verbosity_info()

# Set random seed for reproducible weight initialization
setup_seed(42, strict_deterministic=False)


train_dataset = (
    load_dataset("epfl-dlab/zip2zip-1B", name="default", split="train")
    .shuffle(seed=42)
    .filter(lambda example, indice: indice % 50 == 0, with_indices=True)
)
eval_dataset = (
    load_dataset("epfl-dlab/zip2zip-1B", name="default", split="validation")
    .shuffle(seed=42)
    .take(100)
)

base_model_arch = "llama"
pretrained_model_name = "JackFram/llama-160m"  # to load pretrained config and tokenizer

max_context_length = 512

degenerate = False

# Dynamic import based on architecture
if base_model_arch == "llama":
    from transformers import LlamaConfig, LlamaForCausalLM

    base_model_class = LlamaForCausalLM
    base_model_config_class = LlamaConfig
else:
    raise ValueError(f"Unsupported model architecture: {base_model_arch}")

base_model_config = base_model_config_class.from_pretrained(pretrained_model_name)

base_model = base_model_class(base_model_config)

config = Zip2ZipConfig(
    base_model_name_or_path=pretrained_model_name,
    encoder_type=EncoderType.TRANSFORMER,
    encoder=TransformerEncoderConfig(
        hidden_size=base_model_config.hidden_size,
        tie_encoders=False,
        num_hidden_layers=2,
        position_encoding="learnable",
        intermediate_size=base_model_config.intermediate_size,
        num_heads=base_model_config.num_attention_heads,
    ),
    compression=CompressionConfig.from_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name),
        max_codebook_size=max_context_length,
        max_subtokens=4 if not degenerate else 1,
    ),
)


model = Zip2ZipModel(
    config,
    base_model=base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = Zip2ZipTokenizer(config)

# Auto-save the current script to output directory
script_path = __file__
output_script_path = os.path.join(trainer_output_dir, "run_trainer.py")
os.makedirs(os.path.dirname(output_script_path), exist_ok=True)
shutil.copy2(script_path, output_script_path)

trainer_args = SFTConfig(
    max_length=max_context_length,
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

trainer.train(resume_from_checkpoint=False)

trainer.save_model(os.path.join(trainer_output_dir, "final"))
