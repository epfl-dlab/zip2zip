import os
import omegaconf
import torch
import random
import string
import importlib
from logging import getLogger
import hashlib

logger = getLogger(__name__)
import numpy as np
from torch import nn
from dataclasses import fields
from dataclasses import asdict, is_dataclass
from omegaconf import OmegaConf, DictConfig
from transformers import PreTrainedModel
from collections.abc import MutableMapping
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


# TODO: needs to polish
def hash_module_bottom_up(model: torch.nn.Module) -> dict[str, str]:
    """Recursively hash submodules bottom-up and return name → hash map."""
    hash_map = {}

    for name, module in reversed(list(model.named_modules())):
        h = hashlib.sha256()
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name in hash_map:
                h.update(hash_map[full_name].encode())

        for p in module.parameters(recurse=False):
            h.update(p.detach().cpu().numpy().tobytes())

        hash_map[name] = h.hexdigest()[:8]

    return hash_map


def print_model_hashes(model: torch.nn.Module, depth: int = 2):
    """Print module structure and hashes up to the given depth, efficiently."""
    hash_map = hash_module_bottom_up(model)

    print("Module Hash Summary:\n")
    for name, module in model.named_modules():
        if name == "":
            current_depth = 0
        else:
            current_depth = name.count(".") + 1
        if current_depth > depth:
            continue
        print(
            f"{name or '[root]':<40} | {module.__class__.__name__:<30} | hash={hash_map[name]}"
        )
