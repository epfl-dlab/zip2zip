import torch
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any

from zip2zip.config import (
    Zip2ZipConfig,
    EncoderConfigType,
    EncoderType,
    ENCODER_CONFIG_MAPPING,
)
import zip2zip.config as zip2zip_config_module
from zip2zip.nn.encoders.config import TransformerEncoderConfig, AttentionEncoderConfig

# Automatically selects bfloat16 if GPU has Compute Capability 8.0+; otherwise, uses float16.
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    BEST_DTYPE = torch.bfloat16 if major >= 8 else torch.float16
else:
    BEST_DTYPE = torch.float32


@dataclass
class WandbConfig:
    # the name of the project in wandb
    project: Optional[str] = None
    # the name of the group in wandb
    group: Optional[str] = None
    # the name of the experiment in wandb
    exp_name: Optional[str] = None
    # the name of the entity in wandb
    entity: Optional[str] = None


@dataclass
class DataConfig:
    # the path to the directory containing the dataset (the dataset will be created in this directory)
    path: str
    # the path to the dataset in the huggingface hub
    dataset_path: str
    # the name of the text column in the dataset
    text_column: str
    # the name of the dataset in the huggingface hub
    dataset_name: Optional[str] = None
    # shuffle the dataset
    shuffle: bool = True


@dataclass
class LoRAConfig:
    # the rank of the LoRA matrices
    rank: int
    # the scaling factor for the LoRA matrices
    alpha: float
    # the modules to apply LoRA to
    target_modules: List[str]
    # the initialization method for the LoRA matrices
    init_lora_weight: Literal["default", "pissa"] = "pissa"
    # whether to use RSLoRA
    use_rslora: bool = False


@dataclass
class CompressionConfig:
    # the compression function to use
    compressor_function_name: str
    # the maximum number of subtokens per hypertoken
    max_subtokens: int


@dataclass
class EmbeddingEncoderConfig:
    # the size of the embedding
    embedding_size: int
    # the embedding encoder class to use (must implement the EmbeddingEncoder interface)
    embedding_encoder_name: str
    # whether to use the same embedding encoder for the input and output embeddings
    tie_embedding_encoder: bool = False
    # config specific to this embedding type
    unsafe_config: Dict[str, Any] = field(default_factory=dict)
    # the position encoding to use
    position_encoding: Optional[str] = None
    # the alpha for the autoencoder loss
    auto_encoder_loss_alpha: float = 0.0

    def __post_init__(self):
        # type cast the unsafe_config values to the correct type: int or float
        for key, value in self.unsafe_config.items():
            if isinstance(value, str):
                if value.isdigit():
                    self.unsafe_config[key] = int(value)
                elif value.lower() in ["true", "false"]:
                    self.unsafe_config[key] = value.lower() == "true"
                else:
                    try:
                        self.unsafe_config[key] = float(value)
                    except ValueError:
                        pass


@dataclass
class Config:
    # the configuration for the dataset
    data: DataConfig
    # the configuration for the compression
    compression: CompressionConfig
    # the configuration for the embedding encoder
    embedding_encoder: EmbeddingEncoderConfig
    # the name of the pretrained model in the huggingface hub
    pretrained_model_name_or_path: str
    # the name of the pretrained tokenizer in the huggingface hub
    pretrained_tokenizer_name_or_path: str
    # the batch size in tokens for training (per device)
    per_device_batch_size: int
    # the sequence length in tokens for training
    seq_length: int
    # the total batch size in tokens for training (the optimizer is updated once per total_batch_size tokens)
    total_batch_size: int
    # the size of the extra vocabulary (maximum number of hypertokens)
    extra_vocab_size: int
    # the size of the initial vocabulary
    initial_vocab_size: int
    # whether to compile the model
    compile_model: bool
    # the maximum learning rate
    max_lr: float
    # the minimum learning rate
    min_lr: float
    # the number of steps between checkpoints
    checkpoint_interval: int
    # the directory to save checkpoints in
    checkpoint_dir: str
    # the schedule to use for the learning rate
    schedule: str
    # the number of epochs to train
    epochs: Optional[int] = None
    # the number of steps to train (calculated from epochs if not provided)
    max_steps: Optional[int] = None
    # warmup steps
    warmup_steps: Optional[int] = None
    # the number of steps between validation
    val_steps: int = 10
    # the number of steps between validation checkpoints
    val_interval: int = 50
    # the path to the checkpoint adapter
    pretrained_adapter_path: Optional[str] = None
    # the path to the hf hub containing the checkpoint adapter
    pretrained_hub_adapter_path: Optional[str] = None
    # the dtype to use for the model
    dtype: torch.dtype = BEST_DTYPE
    # the prompt to use for generation
    generation_prompt: Optional[str] = None
    # the configuration for the LoRA matrices
    lora: Optional[LoRAConfig] = None
    # the configuration for wandb
    wandb_config: Optional[WandbConfig] = None
    # the repository id to upload checkpoints to the huggingface hub at the end of training
    upload_checkpoints_repo_id: Optional[str] = None
    # mask the first occurrence of each hypertoken in the loss
    mask_first_occurrence: bool = False
    # use float8 for the base model
    float8_base_model: bool = False
    # Turn off CLM, only for ablation study
    disable_CLM: bool = False
    # early stopping patience
    early_stopping_patience: int = 5

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        from utils import dataclass_from_file

        return dataclass_from_file(cls, file_path)

    def __post_init__(self):
        if self.epochs is None and self.max_steps is None:
            raise ValueError("Either epochs or max_steps must be set")
        if self.epochs is not None and self.max_steps is not None:
            # take max_steps as precedence
            self.epochs = None

    def _to_zip2zip_compression_config(self) -> zip2zip_config_module.CompressionConfig:
        return zip2zip_config_module.CompressionConfig(
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.extra_vocab_size,
            max_subtokens=self.compression.max_subtokens,
        )

    def _to_zip2zip_encoder_config(self) -> zip2zip_config_module.EncoderConfigType:
        encoder_type = None
        if self.embedding_encoder.embedding_encoder_name == "attention":
            # encoder_type = EncoderType.ATTENTION
            encoder_type = "attention"
            encoder_config = AttentionEncoderConfig(
                hidden_size=self.embedding_encoder.embedding_size,
                tie_encoders=self.embedding_encoder.tie_embedding_encoder,
                num_heads=self.embedding_encoder.unsafe_config["num_heads"],
                position_encoding=self.embedding_encoder.position_encoding,
            )
        elif self.embedding_encoder.embedding_encoder_name == "transformer":
            # encoder_type = EncoderType.TRANSFORMER
            encoder_type = "transformer"
            encoder_config = TransformerEncoderConfig(
                hidden_size=self.embedding_encoder.embedding_size,
                tie_encoders=self.embedding_encoder.tie_embedding_encoder,
                num_heads=self.embedding_encoder.unsafe_config["num_heads"],
                intermediate_size=self.embedding_encoder.unsafe_config.get(
                    "intermediate_size", None
                ),
                num_hidden_layers=self.embedding_encoder.unsafe_config[
                    "num_hidden_layers"
                ],
                position_encoding=self.embedding_encoder.position_encoding,
            )
        else:
            raise ValueError(
                f"Invalid encoder type: {self.embedding_encoder.embedding_encoder_name}"
            )
        return encoder_type, encoder_config

    def _to_PEFT_config(self) -> Dict[str, Any]:
        from peft import LoraConfig

        return LoraConfig(
            peft_type="LORA",
            base_model_name_or_path=self.pretrained_model_name_or_path,
            r=self.lora.rank,
            target_modules=self.lora.target_modules,
            lora_alpha=self.lora.alpha,
        )

    def to_zip2zip_config(self) -> Zip2ZipConfig:
        encoder_type, encoder_config = self._to_zip2zip_encoder_config()
        compression_config = self._to_zip2zip_compression_config()

        return Zip2ZipConfig(
            base_model_name_or_path=self.pretrained_model_name_or_path,
            encoder_type=encoder_type,
            encoder=encoder_config,
            compression=compression_config,
        )


if __name__ == "__main__":
    from utils import dataclass_from_file

    example_config_path = "cfgs/many/c_stable_llama_3.2_1b_instruct_debug.yaml"
    config = dataclass_from_file(Config, example_config_path)
    print(config)

    zip2zip_config = config.to_zip2zip_config()
    print(zip2zip_config)
