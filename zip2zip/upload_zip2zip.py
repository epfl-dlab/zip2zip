import os
import sys
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, upload_file
import tempfile

import torch


from utils import dataclass_from_dict, unflatten
from configs import Config

from zip2zip.config import Zip2ZipConfig, CompressionConfig, CheckpointDetails
from zip2zip.nn.encoders.config import TransformerEncoderConfig


class Zip2ZipUploader:
    def __init__(self, adapter_path: str, repo_id: str, target_dtype=None):
        self.adapter_path = adapter_path
        self.repo_id = repo_id
        self.target_dtype = target_dtype
        self._load()

    def _load(self):
        self.metadata = {}
        self.state_dict = {}
        with safe_open(self.adapter_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                self.state_dict[k] = (
                    f.get_tensor(k).to(self.target_dtype)
                    if self.target_dtype
                    else f.get_tensor(k)
                )
            self.metadata = dict(f.metadata())

        self.dict_config = unflatten(self.metadata)["config"]
        self.old_config = dataclass_from_dict(Config, self.dict_config)

    def upload_config(self):
        zip2zip_config = self.old_config.to_zip2zip_config()
        zip2zip_config.push_to_hub(self.repo_id)

    def upload_encoders(self):
        input_sd = {}
        output_sd = {}

        for k, v in self.state_dict.items():
            if k.startswith("hyper_embedding"):
                nk = k.removeprefix("hyper_embedding._orig_mod.embedding_encoder.")
                input_sd[f"input_encoder.{nk}"] = v
            elif k.startswith("hyper_lm_head"):
                nk = k.removeprefix("hyper_lm_head._orig_mod.embedding_encoder.")
                output_sd[f"output_encoder.{nk}"] = v

        input_sd.pop("input_encoder.cls_token_vector", None)
        output_sd.pop("output_encoder.cls_token_vector", None)

        combined_sd = {**input_sd, **output_sd}

        # save the combined sd to a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "zip2zip_encoders.safetensors")
            save_file(combined_sd, tmp_path)

            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="zip2zip_encoders.safetensors",
                repo_id=self.repo_id,
                repo_type="model",
            )

    def upload_peft(self):
        lora_sd = {}
        for k, v in self.state_dict.items():
            if k.startswith("model"):
                lora_sd[
                    k.replace("model._orig_mod", "base_model.model")
                    .replace("lora_a", "lora_A.weight")
                    .replace("lora_b", "lora_B.weight")
                ] = v

        peft_config = self.old_config._to_PEFT_config()
        peft_config.push_to_hub(self.repo_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "adapter_model.safetensors")
            save_file(lora_sd, tmp_path)
            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="adapter_model.safetensors",
                repo_id=self.repo_id,
                repo_type="model",
            )

    def upload_checkpoint_details(self):
        training_id = self.adapter_path.split("/")[-2]
        checkpoint_id = self.adapter_path.split("/")[-1].split(".")[0]
        checkpoint_details = CheckpointDetails(
            training_id=training_id, checkpoint_id=checkpoint_id
        )
        checkpoint_details.push_to_hub(
            self.repo_id, path_in_repo="zip2zip_checkpoint_details.json"
        )
        # upload_file(
        #     path_or_fileobj=checkpoint_details,
        #     path_in_repo="zip2zip_checkpoint_details.json",
        #     repo_id=self.repo_id,
        #     repo_type="model",
        # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--old-adapter", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--target-dtype", type=str, default=None)
    args = parser.parse_args()

    uploader = Zip2ZipUploader(
        adapter_path=args.old_adapter,
        repo_id=args.repo_id,
        target_dtype=args.target_dtype,
    )
    uploader.upload_config()
    uploader.upload_encoders()
    uploader.upload_peft()  # if needed
    uploader.upload_checkpoint_details()
