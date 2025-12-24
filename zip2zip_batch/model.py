from __future__ import annotations

import os
import torch
import inspect
from torch import nn
from typing import Tuple, Optional, Union
from peft import PeftModel, PeftMixedModel
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from safetensors.torch import save_file, load_file
from transformers.generation.utils import GenerateOutput
from transformers import PreTrainedModel, AutoModelForCausalLM

from zip2zip_batch.config import Zip2ZipConfig
from zip2zip_batch.nn.linear import HyperLinear
from zip2zip_batch.codebook import CodebookManager
from zip2zip_batch.nn.embedding import HyperEmbedding
from zip2zip_batch.nn.encoders.base import BaseEncoder
from zip2zip_batch.nn.encoders.config import EncoderConfigType
from zip2zip_batch.utils.constants import SAFETENSORS_ENCODERS_NAME


class Zip2ZipModel(PushToHubMixin, nn.Module):
    def __init__(
        self, base_model: PreTrainedModel, config: Zip2ZipConfig[EncoderConfigType]
    ) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model

        self.dtype = base_model.dtype
        self.device = base_model.device

        self.codebook_manager = CodebookManager.from_config(
            config, self.dtype, self.device
        )

        self.input_encoder, self.output_encoder = self.get_encoders()
        self.set_hyper_modules()

    def set_hyper_modules(self) -> None:
        model_input_embeddings = self.base_model.get_input_embeddings()
        self.base_model.set_input_embeddings(
            HyperEmbedding.from_embedding(
                model_input_embeddings,
                self.config,
                self.input_encoder,
                self.codebook_manager,
            )
        )

        model_output_embeddings = self.base_model.get_output_embeddings()
        if model_output_embeddings is not None:
            self.base_model.set_output_embeddings(
                HyperLinear.from_linear(
                    model_output_embeddings,
                    self.config,
                    self.output_encoder,
                    self.codebook_manager,
                )
            )

    def get_encoders(self) -> Tuple[BaseEncoder, BaseEncoder]:
        input_encoder = BaseEncoder.from_config(
            self.config.encoder, self.config.compression
        ).to(self.device, self.dtype)

        if self.config.encoder.tie_encoders:
            return input_encoder, input_encoder
        else:
            output_encoder = BaseEncoder.from_config(
                self.config.encoder, self.config.compression
            ).to(self.device, self.dtype)
            return input_encoder, output_encoder

    def load_encoders(
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

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_model":
                raise
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # here we need to handle the train case where we pass the codebooks as tensors
        # we should do that the same as the labels (to be compatible with the Trainer class)
        return self.base_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        codebooks = kwargs.pop("codebooks", None)

        if codebooks is None:
            raise ValueError("`codebooks` must be provided")
        self.codebook_manager.set_codebooks(codebooks)

        output = self.base_model.generate(*args, **kwargs)

        self.codebook_manager.reset()
        return output

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
    ) -> Zip2ZipModel:
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

        model = cls(base_model, config)
        model.load_encoders(pretrained_model_name_or_path, subfolder, **kwargs)
        return model
