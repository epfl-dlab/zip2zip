import torch

from utils import setup_seed
from zip2zip_batch.model import Zip2ZipModel
from zip2zip_batch.tokenizer import Zip2ZipTokenizer


setup_seed()
torch.set_float32_matmul_precision("high")

model_name = "nathanrchn/zip2zip-test"

model = Zip2ZipModel.from_pretrained(
    model_name,
    with_peft=True,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
).eval()

tokenizer = Zip2ZipTokenizer.from_pretrained(model_name)

prompt1 = tokenizer.apply_chat_template([{"role": "user", "content": "Write a MultiHeadAttention layer in PyTorch"}], tokenize=False, add_generation_prompt=True)
prompt2 = tokenizer.apply_chat_template([{"role": "user", "content": "What is the capital of France and what is the capital of Germany?"}], tokenize=False, add_generation_prompt=True)
prompt3 = tokenizer.apply_chat_template([{"role": "user", "content": "What is your name?"}], tokenize=False, add_generation_prompt=True)

inputs = tokenizer([prompt1, prompt2, prompt3], return_tensors="pt", padding="longest").to("cuda")

outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=128,
    use_cache=True,
)

for text in tokenizer.batch_decode(outputs, codebooks=inputs["codebooks"]):
    print(text)
    print("=" * 10)

print(f"num hyper tokens: {(outputs > model.config.compression.initial_vocab_size).sum().item()}")
