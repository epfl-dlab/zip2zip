import os
import omegaconf
import torch
import random
import string
import importlib
from logging import getLogger

logger = getLogger(__name__)
import numpy as np
from torch import nn
from dataclasses import fields
from dataclasses import asdict, is_dataclass
from omegaconf import OmegaConf, DictConfig
from transformers import PreTrainedModel
from collections.abc import MutableMapping
from torchao.float8.float8_linear import Float8Linear
from torch.distributed import init_process_group
from typing import (
    Callable,
    Optional,
    Type,
    Dict,
    Any,
    TypeVar,
    List,
    Tuple,
    get_origin,
    get_args,
    Union,
)
import yaml
from configs import Config
from nn.lora import LoRALinear


T = TypeVar("T")


def nanoid(length: int = 4) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))


def str_of_list_to_list(s: str) -> list[str]:
    s = s.strip("[]")
    result = []
    current = ""
    in_quotes = False

    for char in s:
        if char == "'" or char == '"':
            in_quotes = not in_quotes
        elif char == "," and not in_quotes:
            if current.strip():
                item = current.strip().strip("'\"")
                result.append(item)
            current = ""
        else:
            current += char

    if current.strip():
        item = current.strip().strip("'\"")
        result.append(item)

    return result


def get_class_from_string(class_path: str) -> Type:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Could not import class {class_path}: {str(e)}")


def is_optional_dataclass(field_type) -> bool:
    """Check if a field type is a dataclass or an optional dataclass."""
    if get_origin(field_type) is Union:
        # Check if the field type is Optional (Union with NoneType)
        args = get_args(field_type)
        # Check if one of the arguments is NoneType and the other is a dataclass
        return any(is_dataclass(arg) for arg in args if arg is not type(None))
    return is_dataclass(field_type)


def get_underlying_type(field_type: Type) -> Type:
    """Return the underlying type of an Optional type, ensuring it is a valid Optional."""
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        # Check if the Union has exactly two arguments and one is NoneType
        if len(args) == 2 and type(None) in args:
            # Return the first non-NoneType argument
            for arg in args:
                if arg is not type(None):
                    return arg
    return field_type


def dataclass_from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    cls = get_underlying_type(cls)
    if not is_dataclass(cls):
        return d
    # rename batch_size from old config to per_device_batch_size
    if "batch_size" in d:
        d["per_device_batch_size"] = d.pop("batch_size")

    fieldtypes = {f.name: f.type for f in fields(cls)}
    parsed_dict = {}

    for f, value in d.items():
        field_type = fieldtypes.get(f)
        field_type = get_underlying_type(field_type)

        # Handle None values for Optional fields
        if value is None or value == "None":
            parsed_dict[f] = None
            continue

        if field_type == bool:
            parsed_value = (
                value.lower() == "true" if isinstance(value, str) else bool(value)
            )

        elif isinstance(value, str) and field_type in (int, float):
            logger.info(f"Parsing {f} as {field_type} from string: {value}")
            parsed_value = (
                field_type(value) if value != "None" else None
            )  # Convert string to int/float

        elif isinstance(value, str) and str(field_type).startswith("typing.List[str]"):
            parsed_value = str_of_list_to_list(value)  # Convert string to list

        elif f in ("compressor_type", "embedding_encoder_type", "dtype") and isinstance(
            value, str
        ):
            parsed_value = get_class_from_string(value)  # Import class dynamically

        # Check if it's a dataclass or Optional[dataclass]
        elif field_type and (
            is_dataclass(field_type)
            or (
                str(field_type).startswith("typing.Optional")
                and is_dataclass(get_optional_inner_type(field_type))
            )
        ):
            inner_type = (
                get_optional_inner_type(field_type)
                if str(field_type).startswith("typing.Optional")
                else field_type
            )
            # Ensure value is a dictionary before recursing
            if isinstance(value, omegaconf.DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
            if not isinstance(value, dict):
                parsed_value = None if value == "None" else value
            else:
                parsed_value = dataclass_from_dict(inner_type, value)

        else:
            parsed_value = (
                None if value == "None" else value
            )  # Default: Use value as-is

        parsed_dict[f] = parsed_value

    return cls(**parsed_dict)


def get_optional_inner_type(field_type) -> Type:
    """Extract the inner type from Optional[Type]."""
    # If it's already a type (not a string representation), return it
    if isinstance(field_type, type):
        return field_type

    # Handle typing.Optional[Type] format
    type_str = str(field_type)
    if type_str.startswith("typing.Optional["):
        inner_type = type_str[len("typing.Optional[") : -1]
        # Don't try to import complex typing types
        if inner_type.startswith("typing."):
            return field_type
        # Only try to import actual class paths
        if "." in inner_type and not inner_type.startswith("typing."):
            return get_class_from_string(inner_type)
    return field_type


def dataclass_from_file(cls: Type[T], path: str) -> T:
    return dataclass_from_dict(cls, OmegaConf.load(path))


def setup_seed(value: int = 42, strict_deterministic: bool = False) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(value)

    # Make cuDNN deterministic, this does help and it doesn't slow down the run
    # but it's not sufficient to make the run deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_deterministic:
        # Force deterministic algorithms, this is NECESSARY to make the run deterministic but it can slow down the run by 40% on H100
        torch.use_deterministic_algorithms(True)
        # Set up env variables is necessary to enable deterministic algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_seed() -> int:
    if torch.cuda.is_available():
        return torch.cuda.initial_seed()
    return torch.initial_seed()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_distributed() -> tuple[bool, int, int, int, bool, str, str]:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = get_device()

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    return (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        master_process,
        device,
        device_type,
    )


def setup_wandb(config: Config, run_id: str):
    if config.wandb_config is not None:
        import wandb

        wandb.init(
            project=config.wandb_config.project,
            group=config.wandb_config.group,
            name=f"{config.wandb_config.exp_name}-{run_id}",
            id=run_id,
            entity=config.wandb_config.entity,
            config=asdict(config),
            resume="allow",
        )


def wandb_log(config: Config, items: Dict[str, Any]):
    if config.wandb_config is not None:
        import wandb

        wandb.log(items)


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def print_trainable_modules(model: nn.Module) -> None:
    """
    Prints all modules and parameters in a PyTorch model that require gradients.

    Args:
        model (nn.Module): The model to inspect.
    """
    print("\n Trainable Modules & Parameters:\n" + "=" * 40)

    for name, module in model.named_modules():
        # Check if the module has trainable parameters
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            print(f"🟢 Module: {name} ({module.__class__.__name__})")


def print_grad_min_max(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(
                f"{name}: min={param.grad.min().item()}, max={param.grad.max().item()}, mean={param.grad.mean().item()}"
            )


def flatten(
    dictionary: Dict[str, Any], parent_key: str = "", separator: str = "/"
) -> Dict[str, Any]:
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten(dictionary: Dict[str, Any], separator: str = "/") -> Dict[str, Any]:
    result = {}
    for key, value in dictionary.items():
        keys = key.split(separator)
        d = result
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return result


def to_str_dict(dictionary: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    return {f"{prefix}/{k}": str(v) for k, v in dictionary.items()}


def adapt_model(model: PreTrainedModel, config: Config, merge: bool = False) -> None:
    if config.lora is None:
        return

    for name, param in model.named_parameters():
        param.requires_grad = False

    lora_config = config.lora
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            target_module in name for target_module in lora_config.target_modules
        ):
            parent_name, child_name = name.rsplit(".", 1)
            parent = model
            for part in parent_name.split("."):
                parent = getattr(parent, part)

            lora_layer = LoRALinear.from_pretrained(
                module,
                lora_config.rank,
                lora_config.alpha,
                lora_config.init_lora_weight,
            )

            if merge:
                lora_layer.merge()

            setattr(parent, child_name, lora_layer)


def pad_codebook(
    codebook_dict: Dict[str, int],
    initial_vocab_size: int,
    extra_vocab_size: int,
    max_subtokens: int,
    pad_token_id: int,
) -> Tuple[List[List[int]], int, float]:
    padded_codebook_list = [
        [pad_token_id] * max_subtokens for _ in range(extra_vocab_size)
    ]
    real_codebook_size = len(codebook_dict)

    sum_hypertoken_size = sum(
        len(subtoken_str.split(",")) for subtoken_str, _ in codebook_dict.items()
    )

    for subtoken_str, hypertoken_id in codebook_dict.items():
        subtokens_list = [int(x) for x in subtoken_str.split(",")]
        padded_codebook_list[hypertoken_id - initial_vocab_size][
            : len(subtokens_list)
        ] = subtokens_list

    mean_hyper_token_size = (
        sum_hypertoken_size / real_codebook_size if real_codebook_size > 0 else 0
    )

    return (padded_codebook_list, real_codebook_size, mean_hyper_token_size)


def find_latest_checkpoint(checkpoint_dir: str, run_id: str) -> Optional[int]:
    """
    Finds the latest checkpoint step for a specific run_id.
    Returns the step number or None if no checkpoint found.
    """
    run_dir = os.path.join(checkpoint_dir, run_id)
    if not os.path.exists(run_dir) or not os.path.isdir(run_dir):
        return None

    latest_step = -1

    # Find model checkpoints in this run
    for fname in os.listdir(run_dir):
        if not fname.startswith("model_") or not fname.endswith(".safetensors"):
            continue

        try:
            step = int(
                fname[6:-12]
            )  # Extract step number from 'model_<step>.safetensors'
            latest_step = max(latest_step, step)
        except ValueError:
            continue

    return latest_step if latest_step >= 0 else None


def upload_checkpoints(
    config: Config, run_id: str, only_last_checkpoint: bool = False
) -> None:
    if config.upload_checkpoints_repo_id is not None:
        from huggingface_hub import upload_folder, repo_exists, create_repo

        if not repo_exists(config.upload_checkpoints_repo_id):
            create_repo(config.upload_checkpoints_repo_id)
        if only_last_checkpoint:
            last_checkpoint = find_latest_checkpoint(config.checkpoint_dir, run_id)
            if last_checkpoint is not None:
                upload_folder(
                    repo_id=config.upload_checkpoints_repo_id,
                    folder_path=os.path.join(config.checkpoint_dir, run_id),
                    path_in_repo=f"/{run_id}",
                    allow_patterns=[f"model_{last_checkpoint}.safetensors"],
                )
            else:
                print(f"No valid checkpoint found for run {run_id}")
        else:
            upload_folder(
                repo_id=config.upload_checkpoints_repo_id,
                folder_path=os.path.join(config.checkpoint_dir, run_id),
                path_in_repo=f"/{run_id}",
                allow_patterns=["model_*.safetensors"],
            )


def adjust_kv_cache(
    past_key_values: Tuple, input_ids: torch.Tensor, past_input_ids: torch.Tensor
) -> Tuple:
    """
    Adjusts KV cache based on changes in input_ids.

    - If input_ids is a continuation of past_input_ids → Reuse the cache.
    - If input_ids has the same length but a different last token → Update the cache.

    Args:
        kv_cache (Tuple): The stored key-value cache.
        input_ids (torch.Tensor): Current input tokens.
        past_input_ids (torch.Tensor): Previously cached input tokens.

    Returns:
        tuple: (new_input_ids, adjusted_kv_cache)
    """
    if past_input_ids is None:
        # in case kv_cache disabled, always return the full input_ids and None for the past_key_values
        # this falls back to compute the full sequence
        return input_ids, past_key_values
    # we want to know which tokens are new, N.B. the input_ids may have different shape than last_input_ids
    # Two cases (1) the input_ids is a continuation of the last_input_ids by one token -> No force-merge
    # (2) the input_ids is the same length as last_input_ids but the last token is different. -> a force-merge happened

    # for case (1) we can reuse all the past_key_values
    # for case (2) we need to eject the last past_key_values

    input_ids_len = input_ids.size(1)
    past_input_ids_len = past_input_ids.size(1)

    min_len = min(input_ids_len, past_input_ids_len)

    trimmed_input_ids = input_ids[:, :min_len]
    past_input_ids = past_input_ids[:, :min_len]
    # TODO, check do two generation with the same model ,see the behavior

    matching = torch.all(trimmed_input_ids == past_input_ids, dim=0)
    if not torch.all(matching):
        # find the first index where the input_ids and past_input_ids differ
        first_diff_idx = int(
            torch.where(~matching)[0][0].item()
        )  # always an int, the type cast is to make pyright happy
    else:
        first_diff_idx = min_len
    num_to_trim = past_input_ids_len - first_diff_idx
    new_input_ids = input_ids[:, first_diff_idx:]

    active_kv_cache = (
        trim_kv_cache(past_key_values, num_to_trim)
        if past_key_values is not None
        else None
    )

    return new_input_ids, active_kv_cache


def trim_kv_cache(kv_cache: Tuple, n: int) -> Tuple:
    """
    Trims the KV cache by removing the last n elements.

    Args:
        kv_cache (Tuple): The stored key-value cache.
        n (int): Number of elements to remove.

    Returns:
        Tuple: The trimmed key-value cache.
    """
    if n == 0 or kv_cache is None:
        return kv_cache

    trimmed_cache = []
    for layer_kv_cache in kv_cache:
        if isinstance(layer_kv_cache, tuple) and len(layer_kv_cache) == 2:
            # Case 1: Standard (k, v) format
            k, v = layer_kv_cache
            trimmed_cache.append((k[:, :, :-n, :], v[:, :, :-n, :]))
        else:
            raise ValueError("The KV cache has an invalid format")

    return tuple(trimmed_cache)


#########################
# Compression routines
#########################


def decompress(token_ids: List[int], codebook_dict: Dict[int, List[int]]) -> List[int]:
    output_ids = []
    for token_id in token_ids:
        if token_id in codebook_dict:
            output_ids.extend(codebook_dict[token_id])
        else:
            output_ids.append(token_id)
    return output_ids


def describe_lzw(
    lzw_token_ids: list[int],
    base_vocab_size: int,
    standard_lzw_token_ids=None,
    hyper_vocab_size=None,
):

    decomp_token_ids, codebook_dict = lzw_decomp(
        lzw_token_ids, base_vocab_size, hyper_vocab_size
    )
    codebook: Dict[int, List[int]] = {v: k.split(",") for k, v in codebook_dict.items()}
    num_lzw_tokens = len(lzw_token_ids)
    num_decomp_tokens = len(decomp_token_ids)
    num_hyper_tokens = sum([1 if x >= base_vocab_size else 0 for x in lzw_token_ids])

    compression_ratio = num_lzw_tokens / num_decomp_tokens
    compression_factor = 1 / compression_ratio
    compression_saving = 1 - compression_ratio

    num_hyper_token_built = sum([1 if len(v) > 1 else 0 for v in codebook.values()])

    num_active_hyper_tokens = len(
        set([x for x in lzw_token_ids if x >= base_vocab_size])
    )

    metadata = {
        "num_tokens": num_lzw_tokens,
        "num_original_tokens": num_decomp_tokens,
        "num_hyper_tokens": num_hyper_tokens,
        "hyper_token_ratio": num_hyper_tokens / num_lzw_tokens,
        "compression_ratio": compression_ratio,
        "compression_factor": compression_factor,
        "compression_saving": compression_saving,
        "num_hyper_token_built": num_hyper_token_built,
        "num_active_hyper_tokens": num_active_hyper_tokens,
        "optimal_num_lzw_tokens": len(standard_lzw_token_ids)
        if standard_lzw_token_ids
        else None,
        "optimal_compression_ratio": len(standard_lzw_token_ids) / num_decomp_tokens
        if standard_lzw_token_ids
        else None,
    }

    return metadata


def build_new_token_mask(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Builds a mask indicating where a token appears for the first time in the sequence.
    The first token in each sequence is always marked as new.

    Args:
        input_tensor (torch.Tensor): Shape (b, s), tokenized sequences.

    Returns:
        torch.Tensor: Binary mask of shape (b, s), with 1 indicating first occurrences.
    """
    batch_size, seq_length = input_tensor.shape
    new_token_mask = torch.zeros_like(input_tensor, dtype=torch.bool)

    # Iterate over batch
    for b in range(batch_size):
        seen_tokens = set()  # Track seen tokens
        for i in range(seq_length):
            token = input_tensor[b, i].item()
            if token not in seen_tokens:
                new_token_mask[b, i] = 1  # Mark first occurrence
                seen_tokens.add(token)  # Add token to seen set

    return new_token_mask


def real_size_of_codebook(
    codebook_tensor: torch.Tensor, padding_token_id: int = None
) -> torch.tensor:
    assert (
        codebook_tensor.dim() == 3
    ), "Codebook tensor must have 3 dimensions, (B, V_E, M)"
    batch_size, extra_vocab_size, max_subtokens = codebook_tensor.shape
    if padding_token_id is None and max_subtokens > 2:
        # the first lzw merge is always 2 subtokens, so the 3rd subtoken is surely the padding token
        padding_token_id = codebook_tensor[0, 0, -1]
    elif padding_token_id is None:
        raise ValueError(
            "The padding token id must be provided if the codebook tensor has 2 subtokens"
        )

    is_padding_token = (codebook_tensor == padding_token_id).all(
        dim=-1
    )  # shape (B, V_E)
    num_empty_slots = is_padding_token.sum(dim=-1)  # shape (B,)
    size = extra_vocab_size - num_empty_slots  # shape (B,)
    return size


def lzw_decomp(
    compressed_ids: List[int], initial_vocab_size: int, extra_vocab_size=None
) -> Tuple[List[int], Dict[str, int]]:
    """
    Decompresses a sequence of token IDs using LZW algorithm without needing a predefined vocab.

    Args:
        compressed_ids: List of compressed token IDs.

    Returns:
        A list of decompressed token IDs.
    """
    extra_vocab_size = (
        extra_vocab_size if extra_vocab_size is not None else float("inf")
    )
    if not compressed_ids:
        return [], {}

    # Initialize dictionary with single-character entries
    dict_size = initial_vocab_size
    codebook_list = {i: (i,) for i in range(dict_size)}

    # Decompression starts
    prev_seq = codebook_list[compressed_ids[0]]  # First sequence
    decompressed_ids = list(prev_seq)

    for i in range(1, len(compressed_ids)):
        curr_id = compressed_ids[i]

        if curr_id in codebook_list:
            curr_seq = codebook_list[curr_id]
        else:
            # Handle special case where the sequence isn't in the dictionary
            # (happens when a sequence is encountered before being added)
            curr_seq = prev_seq + (prev_seq[0],)

        decompressed_ids.extend(curr_seq)

        # Add new sequence to the dictionary
        if dict_size < extra_vocab_size:
            codebook_list[dict_size] = prev_seq + (curr_seq[0],)
            dict_size += 1

        # Update previous sequence
        prev_seq = curr_seq

    extra_codebook = {
        ",".join(map(str, v)): k
        for k, v in codebook_list.items()
        if k >= initial_vocab_size
    }

    return decompressed_ids, extra_codebook


def compute_compression_rates(
    compressed_ids: List[int],
    initial_vocab_size: int,
    extra_vocab_size: int = None,
    max_seq_length: int = 1024,
) -> np.ndarray:
    """Compute compression rates for a sequence of compressed IDs by incrementally decompressing the sequence.

    Args:
        compressed_ids (List[int]): The list of compressed token IDs.
        initial_vocab_size (int): The initial size of the vocabulary.
        extra_vocab_size (int, optional): The maximum size of the vocabulary. Defaults to None.
        max_seq_length (int, optional): The maximum sequence length to consider. Defaults to 1024.

    Returns:
        np.ndarray: An array representing the compression rates with respect to base tokens.
    """
    # Set extra_vocab_size to infinity if not provided
    extra_vocab_size = (
        extra_vocab_size if extra_vocab_size is not None else float("inf")
    )

    if not compressed_ids:
        return np.array([])

    # Initialize the codebook with single-character entries
    dict_size = initial_vocab_size
    codebook = {i: (i,) for i in range(initial_vocab_size)}

    # Begin decompression with the first token
    base_tokens = codebook[compressed_ids[0]]
    decompressed_ids = list(base_tokens)

    compression_rates = np.zeros(max_seq_length)
    compression_rates[: len(base_tokens)] = 1 / len(decompressed_ids)

    for i in range(1, min(len(compressed_ids), max_seq_length)):
        current_id = compressed_ids[i]

        # Retrieve the base tokens from the codebook
        if current_id in codebook:
            base_tokens = codebook[current_id]
        else:
            # Handle special case where the sequence isn't in the dictionary
            # (happens when a hypertoken is encountered before being added)
            base_tokens = base_tokens + (base_tokens[0],)

        previous_length = len(decompressed_ids)
        decompressed_ids.extend(base_tokens)

        # Add new hypertoken to the dictionary
        if dict_size < extra_vocab_size:
            codebook[dict_size] = base_tokens + (base_tokens[0],)
            dict_size += 1

        # Update compression rates
        compression_rates[previous_length : previous_length + len(base_tokens)] = (
            i + 1
        ) / len(decompressed_ids)
    return compression_rates


def get_platform_best_dtype():
    """Automatically selects bfloat16 if GPU has Compute Capability 8.0+; otherwise, uses float16."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32  # Use float32 if no CUDA device is available


PLATFORM_BEST_DTYPE = get_platform_best_dtype()


def support_float8():
    """Check if the current device supports float8."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    return False


# https://github.com/pytorch/ao/issues/1132
def swap_linear_layers(
    module: nn.Module,
    target_module: nn.Module,
    swap_func: Callable[[nn.Linear], nn.Linear],
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """
    Generic function to swap linear layers in a module with a new type of linear layer.

    Note:
        If applied to a root-level nn.Linear, the module will not be modified in place
        and returned instead

    Args:
        module: Module to modify.
        target_module: Replace these modules
        from_float_func: Function that accepts a linear layer and returns a new type of linear layer.
        module_filter_fn: If specified, only the `torch.nn.Linear` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    if isinstance(module, target_module) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root {target_module} with children: {module}"
            )
        return swap_func(module)

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)

        if isinstance(module, target_module) and (
            module_filter_fn is None or module_filter_fn(module, cur_fqn)
        ):
            assert (
                parent_module is not None
            ), f"{target_module} root module should return early: {module}"
            new_module = swap_func(module)
            cur_module_name = cur_fqn.split(".")[-1]
            setattr(parent_module, cur_module_name, new_module)

    post_order_traversal(root_module)
    return root_module


def dequantize_float8_training(
    model: nn.Module, dtype: torch.dtype, device: Union[str, torch.device]
) -> nn.Module:
    """
    Converts `Float8Linear` modules in `model` to `torch.nn.Linear`.
    """

    def dequant_func(mod: Float8Linear) -> nn.Linear:
        new_module = nn.Linear(
            mod.in_features, mod.out_features, dtype=dtype, device=device
        )
        new_module.weight = mod.weight
        new_module.bias = mod.bias
        return new_module

    return swap_linear_layers(
        model,
        Float8Linear,
        dequant_func,
    )


def get_base_vocab_size(tokenizer) -> int:
    return len(tokenizer.vocab)


def save_lm_eval_results_to_yaml(results: Dict[str, Any], filepath: str) -> None:
    """
    Safely save a nested results dictionary into a YAML file.

    Args:
        results (dict): The results dictionary you want to save.
        filepath (str): Path to the output YAML file.
    """
    # Optional: configure nice YAML formatting
    yaml_dump_settings = {
        "allow_unicode": True,  # allow non-ASCII (French accents etc.)
        "default_flow_style": False,  # use block style (prettier)
        "sort_keys": False,  # keep dictionary key order
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(results, f, **yaml_dump_settings)
        print(f"Successfully saved results to {filepath}")
    except Exception as e:
        print(f"Error saving YAML: {e}")
