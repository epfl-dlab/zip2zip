from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, upload_file


from zip2zip_batch.config import Zip2ZipConfig, CompressionConfig
from zip2zip_batch.nn.encoders.config import TransformerEncoderConfig

config = Zip2ZipConfig(
    base_model_name_or_path="microsoft/Phi-3.5-mini-instruct",
    encoder_type="transformer",
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

config.push_to_hub("nathanrchn/zip2zip-test")

p = hf_hub_download(
    "epfl-dlab/online-zip2zip-2",
    "model_7000.safetensors",
    subfolder="evqn",
)

metadata = {}
state_dict = {}
with safe_open(p, framework="pt", device="cpu") as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

    for k, v in f.metadata().items():
        metadata[k] = v

input_encoder_state_dict = {}
output_encoder_state_dict = {}

for k, v in state_dict.items():
    if k.startswith("hyper_embedding"):
        nk = k.removeprefix("hyper_embedding._orig_mod.embedding_encoder.")

        if nk == "pos_embed.weight":
            nk = "position_embeddings"

        input_encoder_state_dict[f"input_encoder.{nk}"] = v
    elif k.startswith("hyper_lm_head"):
        nk = k.removeprefix("hyper_lm_head._orig_mod.embedding_encoder.")

        if nk == "pos_embed.weight":
            nk = "position_embeddings"

        output_encoder_state_dict[f"output_encoder.{nk}"] = v

del input_encoder_state_dict["input_encoder.cls_token_vector"]
del output_encoder_state_dict["output_encoder.cls_token_vector"]

state_dict = {**input_encoder_state_dict, **output_encoder_state_dict}

save_file(state_dict, "zip2zip_encoders.safetensors")

upload_file(
    path_or_fileobj="zip2zip_encoders.safetensors",
    path_in_repo="zip2zip_encoders.safetensors",
    repo_id="nathanrchn/zip2zip-test",
    repo_type="model",
)
