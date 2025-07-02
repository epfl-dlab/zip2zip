import torch

from zip2zip.utils import setup_seed, PLATFORM_BEST_DTYPE
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.logging_utils import configure_logging

configure_logging()

setup_seed()
torch.set_float32_matmul_precision("high")

# model_name = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
model_name = "epfl-dlab/zip2zip-Llama-3.2-3B-Instruct-v0.1"

model = Zip2ZipModel.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=PLATFORM_BEST_DTYPE,
).eval()

tokenizer = Zip2ZipTokenizer.from_pretrained(model_name)

disabled_ids = list(tokenizer.get_added_vocab().values())


prompt1 = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a MultiHeadAttention layer in PyTorch"}],
    tokenize=False,
    add_generation_prompt=True,
)
prompt2 = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Please explain what is messenger ribonucleic acid.",
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)
prompt3 = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Explique-moi l’histoire de la Révolution française.",
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    [prompt1, prompt2, prompt3], return_tensors="pt", padding="longest"
).to("cuda")


outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=128,
    use_cache=True,
)
for text in tokenizer.batch_decode(outputs, skip_special_tokens=True):
    print(text)

for text in tokenizer.color_decode(outputs, color_scheme="finegrained"):
    print(text)
    print("=" * 10)
