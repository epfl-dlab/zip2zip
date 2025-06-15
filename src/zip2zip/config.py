from __future__ import annotations

import os
import json
import inspect
from transformers.utils import PushToHubMixin
from dataclasses import dataclass, asdict, field
from typing import Generic, Optional, List, Dict
from huggingface_hub import hf_hub_download, ModelCard, ModelCardData

from zip2zip.constants import CONFIG_NAME, MODEL_CARD_ZIP2ZIP_TEXT
from zip2zip.nn.encoders.config import (
    EncoderType,
    EncoderConfigType,
    ENCODER_CONFIG_MAPPING,
)


@dataclass
class CompressionConfig:
    initial_vocab_size: int = field(
        default=None, metadata={"help": "The initial vocabulary size"}
    )
    max_codebook_size: int = field(
        default=None, metadata={"help": "The maximum codebook size"}
    )
    max_subtokens: int = field(
        default=None, metadata={"help": "The maximum number of subtokens"}
    )
    disabled_ids: Optional[List[int]] = field(
        default=None, metadata={"help": "The disabled ids"}
    )


@dataclass
class Zip2ZipConfig(PushToHubMixin, Generic[EncoderConfigType]):
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    encoder_type: EncoderType = field(
        default=None, metadata={"help": "The type of encoder to use."}
    )

    encoder: EncoderConfigType = field(
        default=None, metadata={"help": "The encoder configuration."}
    )
    compression: CompressionConfig = field(
        default=None, metadata={"help": "The compression configuration."}
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        os.makedirs(save_directory, exist_ok=True)
        output_dict = self.to_dict()

        output_path = os.path.join(save_directory, CONFIG_NAME)

        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: bool | None = None,
        commit_message: str | None = None,
        private: bool | None = None,
        token: bool | str | None = None,
        max_shard_size: int | str | None = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: str = None,
        commit_description: str = None,
        tags: List[str] | None = None,
        **deprecated_kwargs,
    ) -> str:
        output = super().push_to_hub(
            repo_id,
            use_temp_dir,
            commit_message,
            private,
            token,
            max_shard_size,
            create_pr,
            safe_serialization,
            revision,
            commit_description,
            tags,
            **deprecated_kwargs,
        )

        model_card = ModelCard.load(repo_id, token=token, ignore_metadata_errors=True)
        model_card_data: ModelCardData = model_card.data

        model_card_data.base_model = self.base_model_name_or_path
        if model_card_data.tags is None:
            model_card_data.tags = []

        if "zip2zip" not in model_card_data.tags:
            model_card_data.tags.append("zip2zip")
            model_card.text += MODEL_CARD_ZIP2ZIP_TEXT

        model_card.data = model_card_data
        model_card.push_to_hub(repo_id=repo_id, token=token)

        return output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> Zip2ZipConfig:
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    subfolder=subfolder,
                    **hf_hub_download_kwargs,
                )
            except Exception as exc:
                raise ValueError(
                    f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'"
                ) from exc

        with open(config_file) as file:
            loaded_attributes = json.load(file)

        kwargs = {**class_kwargs, **loaded_attributes}

        encoder_type = kwargs.pop("encoder_type")
        encoder = ENCODER_CONFIG_MAPPING[encoder_type](**kwargs.pop("encoder"))
        compression = CompressionConfig(**kwargs.pop("compression"))

        return cls(
            **kwargs,
            encoder_type=encoder_type,
            encoder=encoder,
            compression=compression,
        )

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs
