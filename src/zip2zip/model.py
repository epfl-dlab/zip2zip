from __future__ import annotations

import os
import torch
import inspect
import logging
from torch import nn
from typing import Tuple, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from transformers.utils import SAFE_WEIGHTS_NAME
from safetensors.torch import save_file, load_file
from transformers.generation.utils import GenerateOutput
from transformers import PreTrainedModel, AutoModelForCausalLM
from peft import PeftModel, PeftMixedModel, PeftConfig, get_peft_model
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from zip2zip.config import Zip2ZipConfig
from zip2zip.nn.linear import HyperLinear
from zip2zip.codebook import CodebookManager
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.encoders.base import BaseEncoder
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME
from zip2zip.nn.encoders.config import EncoderConfigType


logger = logging.getLogger(__name__)


class Zip2ZipModel(PushToHubMixin, nn.Module):
    def __init__(
        self,
        config: Zip2ZipConfig[EncoderConfigType],
        base_model: Optional[PreTrainedModel] = None,
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.zip2zip_config = config

        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path, **kwargs
            )
        # TODO, should we keep the peft config as part of zip2zip config? how do we handle data, metadata separation?
        if peft_config is not None:
            base_model = get_peft_model(base_model, peft_config)

        self.base_model = base_model

        # in case of embedding model(as opposed to generation model), we need to clear the cache after forward, otherwise the cache will cumulate
        self.clear_zip2zip_cache_after_forward = False

        self.codebook_manager = CodebookManager.from_config(config)
        self.input_encoder, self.output_encoder = self.build_encoders()
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
                    self.output_encoder or self.input_encoder,
                    self.codebook_manager,
                )
            )

    def build_encoders(self) -> Tuple[BaseEncoder, BaseEncoder]:
        # Create encoders with proper device and dtype handling
        input_encoder = BaseEncoder.from_config(
            self.zip2zip_config.encoder, self.zip2zip_config.compression
        )

        if self.zip2zip_config.encoder.tie_encoders:
            output_encoder = None
        else:
            output_encoder = BaseEncoder.from_config(
                self.zip2zip_config.encoder, self.zip2zip_config.compression
            )

        embedding_layer_device = self.base_model.get_input_embeddings().weight.device
        embedding_layer_dtype = self.base_model.get_input_embeddings().weight.dtype

        # Move encoders to the correct device and dtype
        input_encoder.to(device=embedding_layer_device).to(dtype=embedding_layer_dtype)
        if output_encoder is not None:
            output_embedding_layer_device = (
                self.base_model.get_output_embeddings().weight.device
            )
            output_embedding_layer_dtype = (
                self.base_model.get_output_embeddings().weight.dtype
            )
            output_encoder.to(device=output_embedding_layer_device).to(
                dtype=output_embedding_layer_dtype
            )

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

        encoder_filenames = [SAFETENSORS_ENCODERS_NAME, "encoders.safetensors"]
        encoder_file = None

        for filename in encoder_filenames:
            local_candidate = os.path.join(path, filename)
            if os.path.isfile(local_candidate):
                encoder_file = local_candidate
                break

        if encoder_file is None:
            for filename in encoder_filenames:
                try:
                    encoder_file = hf_hub_download(
                        pretrained_model_name_or_path,
                        filename,
                        subfolder=subfolder,
                        **hf_hub_download_kwargs,
                    )
                    break
                except Exception:
                    continue

        if encoder_file is None:
            raise ValueError(
                f"Can't find any encoder file in {encoder_filenames} at '{pretrained_model_name_or_path}'"
            )

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

    @staticmethod
    def _load_pretrained_decoder_weights(
        base_model: PreTrainedModel,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        torch_device: Optional[str] = None,
        **kwargs,
    ) -> bool:
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs = {}
        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value

        if os.path.isfile(os.path.join(path, SAFE_WEIGHTS_NAME)):
            decoder_file = os.path.join(path, SAFE_WEIGHTS_NAME)
        else:
            try:
                decoder_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    SAFE_WEIGHTS_NAME,
                    subfolder=subfolder,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                return False

        decoder_state_dict = load_file(decoder_file, device=torch_device)
        incompatible = base_model.load_state_dict(decoder_state_dict, strict=False)
        if incompatible.missing_keys:
            logger.warning(
                "[Zip2Zip] Missing decoder keys while loading model.safetensors: %s",
                incompatible.missing_keys,
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "[Zip2Zip] Unexpected decoder keys while loading model.safetensors: %s",
                incompatible.unexpected_keys,
            )
        return True

    @staticmethod
    def _load_training_args_metadata(
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> Optional[dict]:
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs = {}
        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value

        meta_file = None
        local_candidate = os.path.join(path, "meta.pt")
        if os.path.isfile(local_candidate):
            meta_file = local_candidate
        else:
            try:
                meta_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    "meta.pt",
                    subfolder=subfolder,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                meta_file = None

        if meta_file is None:
            return None

        try:
            meta = torch.load(meta_file, map_location="cpu", weights_only=False)
            return meta.get("args", {})
        except Exception:
            return None

    @staticmethod
    def _align_encoder_flags_with_training_args(
        config: Zip2ZipConfig,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> None:
        train_args = Zip2ZipModel._load_training_args_metadata(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **kwargs,
        )
        if not train_args:
            return

        expected_causal = bool(train_args.get("encoder_causal", False))
        expected_residual = not bool(train_args.get("no_encoder_residual", False))

        if getattr(config.encoder, "causal", None) != expected_causal:
            logger.warning(
                "[Zip2Zip] Overriding encoder.causal from %s to %s based on meta.pt train args.",
                getattr(config.encoder, "causal", None),
                expected_causal,
            )
            config.encoder.causal = expected_causal

        if getattr(config.encoder, "residual", None) != expected_residual:
            logger.warning(
                "[Zip2Zip] Overriding encoder.residual from %s to %s based on meta.pt train args.",
                getattr(config.encoder, "residual", None),
                expected_residual,
            )
            config.encoder.residual = expected_residual

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base_model":
                raise
            return getattr(self.base_model, name)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        is_training = kwargs.get("labels", None) is not None

        if self.clear_zip2zip_cache_after_forward:
            self.codebook_manager.reset()

        if is_training:
            self.base_model.config.vocab_size += (
                self.zip2zip_config.compression.max_codebook_size
            )

        output = self.base_model.forward(*args, **kwargs)

        if is_training:
            self.base_model.config.vocab_size -= (
                self.zip2zip_config.compression.max_codebook_size
            )
            self.codebook_manager.reset()

        return output

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
        max_codebook_size: Optional[int] = None,
        max_subtokens: Optional[int] = None,
        **kwargs,
    ) -> Zip2ZipModel:
        config = Zip2ZipConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        cls._align_encoder_flags_with_training_args(
            config,
            pretrained_model_name_or_path,
            **kwargs,
        )
        config.compression.max_codebook_size = (
            max_codebook_size
            if max_codebook_size is not None
            else config.compression.max_codebook_size
        )
        config.compression.max_subtokens = (
            max_subtokens
            if max_subtokens is not None
            else config.compression.max_subtokens
        )

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
        except (OSError, FileNotFoundError, ValueError):
            logger.info("[Zip2Zip] No PEFT adapter found — proceeding with base model.")
            decoder_loaded = cls._load_pretrained_decoder_weights(
                base_model,
                pretrained_model_name_or_path,
                **kwargs,
            )
            if decoder_loaded:
                logger.info("[Zip2Zip] Loaded decoder weights from model.safetensors.")
            else:
                logger.info(
                    "[Zip2Zip] No decoder weights found — proceeding with base model."
                )

        model = cls(config, base_model, **kwargs)

        try:
            model.load_pretrained_hyper_encoders(
                pretrained_model_name_or_path, **kwargs
            )
        except Exception as e:
            logger.info(
                "[Zip2Zip] No hyper encoders found — proceeding with base model."
            )

        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device_map: Optional[Union[str, dict]] = "auto",
        dtype: Optional[torch.dtype] = torch.bfloat16,
        **kwargs,
    ) -> Zip2ZipModel:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        config = Zip2ZipConfig.from_pretrained(checkpoint_path, **kwargs)

        # create the model with empty weights
        with init_empty_weights():
            model = cls(config, **kwargs)

        # load the checkpoint and dispatch the model to the correct device
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint_path,
            device_map=device_map,
            dtype=dtype,
            **kwargs,
        )

        return model
