import torch
from peft import LoraConfig
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, upload_file

config = LoraConfig(
    peft_type="LORA",
    base_model_name_or_path="microsoft/Phi-3.5-mini-instruct",
    r=32,
    target_modules=["qkv_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_alpha=32,
)

config.push_to_hub("nathanrchn/zip2zip-test")

p = hf_hub_download(
    "epfl-dlab/online-zip2zip-2",
    "model_7000.safetensors",
    subfolder="evqn",
)

lora_state_dict = {}
with safe_open(p, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith("model"):
            lora_state_dict[
                k.replace("model._orig_mod", "base_model.model")
                .replace("lora_a", "lora_A.weight")
                .replace("lora_b", "lora_B.weight")
            ] = f.get_tensor(k).to(torch.float32)

save_file(lora_state_dict, "adapter_model.safetensors")

upload_file(
    path_or_fileobj="adapter_model.safetensors",
    path_in_repo="adapter_model.safetensors",
    repo_id="nathanrchn/zip2zip-test",
    repo_type="model",
)
