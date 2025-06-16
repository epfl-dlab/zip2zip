from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM

from zip2zip.codebook import CodebookManager
from zip2zip.utils import setup_seed, PLATFORM_BEST_DTYPE
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.logging_utils import configure_logging
from zip2zip.config import Zip2ZipConfig

configure_logging()

setup_seed()

model_name = "epfl-dlab/zip2zip-Llama-3.1-8B-Instruct-v0.1"

pretrained_zip2zip_config = Zip2ZipConfig.from_pretrained(model_name)


tokenizer = Zip2ZipTokenizer(pretrained_zip2zip_config)

base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_zip2zip_config.base_model_name_or_path,
    device_map="cuda",
    torch_dtype=PLATFORM_BEST_DTYPE,
)

# load the peft model
peft_model = PeftModel.from_pretrained(base_model, model_name)

zip2zip_model = Zip2ZipModel(
    pretrained_zip2zip_config,
    base_model=peft_model,
    device_map="cuda",
    torch_dtype=PLATFORM_BEST_DTYPE,
)


zip2zip_model.load_pretrained_hyper_encoders(model_name)

# print(zip2zip_model)

prompt1 = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a MultiHeadAttention layer in PyTorch"}],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([prompt1], return_tensors="pt", padding="longest").to("cuda")


outputs = zip2zip_model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=128,
    use_cache=True,
)


output_codebooks = (
    zip2zip_model.codebook_manager.internal_codebook_manager.get_codebooks()
)

for text in tokenizer.color_decode(
    outputs, output_codebooks, color_scheme="finegrained"
):
    print(text)
    print("=" * 10)
