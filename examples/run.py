import torch

from zip2zip.utils import setup_seed
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.logging_utils import configure_logging

configure_logging()

setup_seed()
torch.set_float32_matmul_precision("high")

model_name = "Saibo-creator/zip2zip-Phi-3.5-mini-instruct-v0.1"

model = Zip2ZipModel.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
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


input_codebooks = inputs.pop("codebooks")

input_codebooks = [codebook.to_decoding_dict() for codebook in input_codebooks]

outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=512,
    use_cache=True,
)

print(f"outputs: {outputs}")


print(
    f"num hyper tokens: {(outputs > model.zip2zip_config.compression.initial_vocab_size).sum().item()}"
)

# codebooks = [state.codebook for state in model.codebook_manager.internal_codebook_manager.states]
output_codebooks = model.codebook_manager.internal_codebook_manager.get_codebooks()
readable_codebooks = [codebook.to_decoding_dict() for codebook in output_codebooks]

print(f"--decode with codebooks--")

for text in tokenizer.batch_decode(outputs, codebooks=output_codebooks):
    print(text)
    print("=" * 10)


print("--fuzzy decode--")

for text in tokenizer.batch_decode(outputs):
    print(text)
    print("=" * 10)


print(f"output_codebooks: {sorted(readable_codebooks[0].items())}")


for text in tokenizer.color_decode(
    outputs, output_codebooks, color_scheme="finegrained"
):
    print(text)
    print("=" * 10)

exit()


model.codebook_manager.internal_codebook_manager.reset()

outputs_ids = outputs.tolist()

(
    updates,
    updates_indices,
) = model.codebook_manager.internal_codebook_manager.update_codebooks(outputs_ids)

# offline_codebooks = [state.codebook for state in model.codebook_manager.internal_codebook_manager.states]
offline_codebooks = model.codebook_manager.internal_codebook_manager.get_codebooks()
offline_codebooks = [codebook.to_decoding_dict() for codebook in offline_codebooks]


_base_ids = tokenizer._lzw_decode(outputs_ids[0], output_codebooks[0])

canonical_ids, canonical_codebook = tokenizer._lzw_encode(_base_ids)

print("--- canonical codebook ---")
print(sorted(canonical_codebook.to_decoding_dict().items()))

print("--- offline codebook ---")
print(sorted(offline_codebooks[0].items()))

print("--- online codebook ---")
print(sorted(readable_codebooks[0].items()))
