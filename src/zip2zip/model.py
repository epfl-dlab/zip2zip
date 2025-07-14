from __future__ import annotations

import os
import torch
import inspect
import logging
from torch import nn
from typing import Tuple, Optional, Union
from peft import PeftModel, PeftMixedModel
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from safetensors.torch import save_file, load_file
from transformers.generation.utils import GenerateOutput
from transformers import PreTrainedModel, AutoModelForCausalLM

from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.linear import HyperLinear
from zip2zip.codebook import CodebookManager
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.nn.encoders.config import EncoderConfigType
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME


logger = logging.getLogger(__name__)


class Zip2ZipModel(PushToHubMixin, nn.Module):
    def __init__(
        self,
        config: Zip2ZipConfig[EncoderConfigType],
        base_model: Optional[PreTrainedModel] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zip2zip_config = config

        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, **kwargs
            )

        # TODO, needs to create peft model here, this would probably require to expand the Zip2ZipConfig class

        self.base_model = base_model

        self.dtype = base_model.dtype
        self.device = base_model.device
        # in case of embedding model(as opposed to generation model), we need to clear the cache after forward, otherwise the cache will cumulate
        self.clear_zip2zip_cache_after_forward = False

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
                self.zip2zip_config,
                self.input_encoder,
                self.codebook_manager,
            )
        )

        model_output_embeddings = self.base_model.get_output_embeddings()
        if model_output_embeddings is not None:
            self.base_model.set_output_embeddings(
                HyperLinear.from_linear(
                    model_output_embeddings,
                    self.zip2zip_config,
                    self.output_encoder,
                    self.codebook_manager,
                )
            )

    def get_encoders(self) -> Tuple[BaseEncoder, BaseEncoder]:
        input_encoder = BaseEncoder.from_config(
            self.zip2zip_config.encoder, self.zip2zip_config.compression
        ).to(self.device, self.dtype)

        if self.zip2zip_config.encoder.tie_encoders:
            return input_encoder, input_encoder
        else:
            output_encoder = BaseEncoder.from_config(
                self.zip2zip_config.encoder, self.zip2zip_config.compression
            ).to(self.device, self.dtype)
            return input_encoder, output_encoder

    def load_pretrained_hyper_encoders(
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
        if not self.zip2zip_config.encoder.tie_encoders:
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

        if self.clear_zip2zip_cache_after_forward:
            self.codebook_manager.reset()

        return self.base_model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        # TODO, we don't need to reset this incase of multi-turn generation
        self.codebook_manager.init_codebooks_and_hyper_weight_cache(batch_size)

        output = self.base_model.generate(*args, **kwargs)

        self.codebook_manager.reset()
        return output

    def save_pretrained(
        self, save_directory: str, is_main_process: bool = True, **kwargs
    ) -> None:
        if is_main_process:
            self.zip2zip_config.save_pretrained(save_directory, **kwargs)

            os.makedirs(save_directory, exist_ok=True)
            output_state_dict = {}
            for k, v in self.input_encoder.state_dict().items():
                output_state_dict[f"input_encoder.{k}"] = v

            if not self.zip2zip_config.encoder.tie_encoders:
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
        **kwargs,
    ) -> Zip2ZipModel:
        config = Zip2ZipConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, **kwargs
            )

        # try to load the peft model
        try:
            base_model = PeftModel.from_pretrained(
                base_model,
                pretrained_model_name_or_path,
                **kwargs,
            )
        except (OSError, FileNotFoundError, ValueError) as e:
            logger.info("[Zip2Zip] No PEFT adapter found — proceeding with base model.")

        model = cls(config, base_model, **kwargs)
        model.load_pretrained_hyper_encoders(pretrained_model_name_or_path, **kwargs)
        return model
