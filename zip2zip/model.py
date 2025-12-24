import inspect
import os
from huggingface_hub import hf_hub_download
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from peft import PeftModel, PeftMixedModel
from safetensors.torch import save_file, load_file
from transformers.generation.utils import GenerateOutput

from configs import Config
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.codebook import CodebookManager
from zip2zip.nn.linear import HyperUnembedding
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.config import Zip2ZipConfig, EncoderConfigType
from utils import get_base_vocab_size
from zip2zip.utils.constants import SAFETENSORS_ENCODERS_NAME


class Zip2ZipModel(nn.Module):
    def __init__(
        self,
        config: Zip2ZipConfig,
        model: PreTrainedModel,
    ) -> None:
        super().__init__()
        self.config = config  # TODO, is this ok to override the config? or maybe we should name it zip2zip_config?
        self.base_hf_transformer = model
        self.device = model.device
        self.dtype = model.dtype
        self.input_encoder, self.output_encoder = self.create_encoders(
            config, dtype=self.dtype, device=self.device
        )
        self.codebook_manager = CodebookManager.from_config(
            config, dtype=self.dtype, device=self.device
        )
        self.set_hyper_embeddings(config)

    def __getattr__(self, attr):
        # Here we use super().__getattr__ instead of getattr(self, attr)
        # to avoid infinite recursion
        if attr in [
            "base_hf_transformer",
            "input_encoder",
            "output_encoder",
            "codebook_manager",
        ]:
            return super().__getattr__(attr)
        model = super().__getattr__("base_hf_transformer")
        return getattr(model, attr)

    def set_hyper_embeddings(self, config: Zip2ZipConfig) -> None:

        self.base_hf_transformer.set_input_embeddings(
            HyperEmbedding.from_config(
                self.base_hf_transformer.get_input_embeddings(),
                config,
                self.input_encoder.get_codebook_embedding_fn(),
                self.codebook_manager,
            )
        )

        self.base_hf_transformer.set_output_embeddings(
            HyperUnembedding.from_config(
                self.base_hf_transformer.get_output_embeddings(),
                config,
                self.output_encoder.get_codebook_embedding_fn(),
                self.codebook_manager,
            )
        )

    @staticmethod
    def create_encoders(
        config: Zip2ZipConfig, dtype: torch.dtype, device: torch.device
    ) -> Tuple[BaseEncoder, BaseEncoder]:
        input_encoder = BaseEncoder.from_config(config.encoder, config.compression).to(
            device, dtype
        )
        if config.encoder.tie_encoders:
            output_encoder = input_encoder
        else:
            output_encoder = BaseEncoder.from_config(
                config.encoder, config.compression
            ).to(device, dtype)
        return input_encoder, output_encoder

    def _load_zip2zip_adaptor_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> None:

        input_encoder_state_dict = {}
        output_encoder_state_dict = {}
        model_state_dict = {}

        for key in state_dict.keys():

            if "hyper_embedding.embedding_encoder" in key:
                input_encoder_state_dict[
                    key.replace("hyper_embedding.embedding_encoder.", "")
                ] = state_dict[key]
            elif "hyper_lm_head.embedding_encoder" in key:
                output_encoder_state_dict[
                    key.replace("hyper_lm_head.embedding_encoder.", "")
                ] = state_dict[key]
            else:
                model_state_dict[key.replace("model.", "", 1)] = state_dict[key]

        # del outdated cls_token_vector if it exists
        if "cls_token_vector" in input_encoder_state_dict:
            del input_encoder_state_dict["cls_token_vector"]

        self.input_encoder.load_state_dict(input_encoder_state_dict, strict=True)
        if output_encoder_state_dict:
            # non-empty means the output encoder is not tied to the input encoder
            # so we load it separately
            if "cls_token_vector" in output_encoder_state_dict:
                del output_encoder_state_dict["cls_token_vector"]
            self.output_encoder.load_state_dict(output_encoder_state_dict, strict=True)

        self.base_hf_transformer.load_state_dict(model_state_dict, strict=False)

    def to(self, *args, **kwargs) -> None:
        super().to(*args, **kwargs)
        self.codebook_manager.to(*args, **kwargs)

    def _load_encoders(
        self,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        torch_device: Optional[str] = None,
        **kwargs,
    ) -> None:
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs = {}
        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value

        if os.path.isfile(os.path.join(path, SAFETENSORS_ENCODERS_NAME)):
            encoder_file = os.path.join(path, SAFETENSORS_ENCODERS_NAME)
        else:
            try:
                encoder_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    SAFETENSORS_ENCODERS_NAME,
                    subfolder=subfolder,
                    **hf_hub_download_kwargs,
                )
            except Exception as exc:
                raise ValueError(
                    f"Can't find '{SAFETENSORS_ENCODERS_NAME}' at '{pretrained_model_name_or_path}'"
                ) from exc

        encoders_state_dict = load_file(encoder_file, device=torch_device)

        input_encoder_state_dict = {}
        output_encoder_state_dict = {}

        for k, v in encoders_state_dict.items():
            if k.startswith("input_encoder."):
                input_encoder_state_dict[k.removeprefix("input_encoder.")] = v
            elif k.startswith("output_encoder."):
                output_encoder_state_dict[k.removeprefix("output_encoder.")] = v
        self.input_encoder.load_state_dict(input_encoder_state_dict)
        if not self.config.encoder.tie_encoders:
            self.output_encoder.load_state_dict(output_encoder_state_dict)

    def save_pretrained(
        self, save_directory: str, is_main_process: bool = True, **kwargs
    ) -> None:
        if is_main_process:
            self.config.save_pretrained(save_directory, **kwargs)

            os.makedirs(save_directory, exist_ok=True)
            output_state_dict = {}
            for k, v in self.input_encoder.state_dict().items():
                output_state_dict[f"input_encoder.{k}"] = v

            if not self.config.encoder.tie_encoders:
                for k, v in self.output_encoder.state_dict().items():
                    output_state_dict[f"output_encoder.{k}"] = v

            save_file(
                output_state_dict,
                os.path.join(save_directory, SAFETENSORS_ENCODERS_NAME),
            )

            if isinstance(self.base_model, PeftModel) or isinstance(
                self.base_model, PeftMixedModel
            ):
                self.base_model.save_pretrained(
                    save_directory, is_main_process=is_main_process, **kwargs
                )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        base_model: Optional[PreTrainedModel] = None,
        with_peft: bool = False,
        subfolder: Optional[str] = None,
        peft_subfolder: Optional[str] = None,
        **kwargs,
    ) -> "Zip2ZipModel":
        config = Zip2ZipConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder, **kwargs
        )

        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, **kwargs
            )

        if with_peft:
            base_model = PeftModel.from_pretrained(
                base_model,
                pretrained_model_name_or_path,
                subfolder=peft_subfolder,
                **kwargs,
            )
        model = cls(config, base_model)
        model._load_encoders(pretrained_model_name_or_path, subfolder, **kwargs)
        return model

    def generate(self, *args, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        reset_codebook = kwargs.pop("reset_codebook", True)
        output = self.base_model.generate(*args, **kwargs)

        if reset_codebook:
            self.codebook_manager.reset()
        return output
