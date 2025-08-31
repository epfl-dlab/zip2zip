import os
from zip2zip import (
    Zip2ZipModel,
    Zip2ZipTokenizer,
    Zip2ZipConfig,
)

import torch
from peft import LoraConfig, TaskType

OUTPUT_DIR = "out"

checkpoint_path = os.path.join(
    OUTPUT_DIR, "zip2zip-train-FT-M3-L2-S1024", "checkpoint-36000"
)


MODE = "from_state_dict"  # "from_pretrained" or "from_state_dict"


if MODE == "from_state_dict":
    # ✅ This is the same as the state dict in the checkpoint

    config = Zip2ZipConfig.from_pretrained(checkpoint_path)

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

    model = Zip2ZipModel.from_checkpoint(
        checkpoint_path,
        device_map="auto",
        dtype=torch.float16,
    )

    tokenizer = Zip2ZipTokenizer.from_pretrained(checkpoint_path)


prompt1 = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a MultiHeadAttention layer in TensorFlow"}],
    tokenize=False,
    add_generation_prompt=True,
)


inputs = tokenizer([prompt1], return_tensors="pt", padding="longest").to("cuda")


outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=256,
    use_cache=True,
)
for text in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(text)

for text in tokenizer.color_decode(outputs, color_scheme="finegrained"):
    print(text)
    print("=" * 10)
