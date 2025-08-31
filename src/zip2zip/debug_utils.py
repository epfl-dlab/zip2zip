import hashlib
from safetensors.torch import load_file
import torch
import torch.nn as nn

from pathlib import Path
from typing import Dict, List


def compare_state_dicts(
    state_dict_1: Dict[str, torch.Tensor] | str | Path,
    state_dict_2: Dict[str, torch.Tensor] | str | Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Compare two state dictionaries and return a detailed comparison report.

    Args:
        state_dict_1: First state dict or path to .safetensors file
        state_dict_2: Second state dict or path to .safetensors file
        rtol: Relative tolerance for torch.allclose
        atol: Absolute tolerance for torch.allclose
        verbose: Whether to print results to console

    Returns:
        Dictionary with keys 'same', 'different', 'missing_1', 'missing_2' containing tensor names
    """
    # Load state dicts if paths are provided
    if isinstance(state_dict_1, (str, Path)):
        state_dict_1 = load_file(str(state_dict_1))
    if isinstance(state_dict_2, (str, Path)):
        state_dict_2 = load_file(str(state_dict_2))

    results = {
        "same": [],
        "different": [],
        "missing_1": [],  # keys in state_dict_2 but not in state_dict_1
        "missing_2": [],  # keys in state_dict_1 but not in state_dict_2
    }

    # Check all keys in state_dict_1
    for name, tensor in state_dict_1.items():
        if name in state_dict_2:
            if torch.allclose(tensor, state_dict_2[name], rtol=rtol, atol=atol):
                results["same"].append(name)
                if verbose:
                    print(f"{name} is the same ✅")
            else:
                results["different"].append(name)
                if verbose:
                    print(f"{name} is different ❌")
        else:
            results["missing_2"].append(name)
            if verbose:
                print(f"{name} is not in state_dict_2 ⚠️")

    # Check for keys in state_dict_2 that are not in state_dict_1
    for name in state_dict_2.keys():
        if name not in state_dict_1:
            results["missing_1"].append(name)
            if verbose:
                print(f"{name} is not in state_dict_1 ⚠️")

    if verbose:
        print(f"\nSummary:")
        print(f"Same tensors: {len(results['same'])}")
        print(f"Different tensors: {len(results['different'])}")
        print(f"Missing in state_dict_1: {len(results['missing_1'])}")
        print(f"Missing in state_dict_2: {len(results['missing_2'])}")

    return results


def compare_checkpoint_files(
    checkpoint_1: str | Path,
    checkpoint_2: str | Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """
    Compare two checkpoint files by loading their model.safetensors.

    Args:
        checkpoint_1: Path to first checkpoint directory
        checkpoint_2: Path to second checkpoint directory
        rtol: Relative tolerance for torch.allclose
        atol: Absolute tolerance for torch.allclose
        verbose: Whether to print results to console

    Returns:
        Dictionary with comparison results
    """
    checkpoint_1 = Path(checkpoint_1)
    checkpoint_2 = Path(checkpoint_2)

    model_path_1 = checkpoint_1 / "model.safetensors"
    model_path_2 = checkpoint_2 / "model.safetensors"

    if not model_path_1.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_1}")
    if not model_path_2.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_2}")

    return compare_state_dicts(model_path_1, model_path_2, rtol, atol, verbose)


def print_model_devices(
    model: nn.Module,
    verbose: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Print the device of all parameters and buffers in a model with emoji indicators.
    Returns separate summaries for parameters and buffers.

    Returns:
        {
          "parameters": { "<device>": [tensor_names...] },
          "buffers":    { "<device>": [tensor_names...] }
        }
    """

    results = {"parameters": {}, "buffers": {}}  # device -> [names]

    def record(name: str, tensor: torch.Tensor, kind: str):
        device_key = str(tensor.device)
        results[kind].setdefault(device_key, []).append(name)

        if verbose:
            emoji = (
                "🟢"
                if tensor.device.type == "cuda"
                else ("🔴" if tensor.device.type == "cpu" else "⚪")
            )
            tag = "P" if kind == "parameters" else "B"
            print(f"[{tag}] {name} is on {tensor.device} {emoji}")

    # Parameters
    for name, param in model.named_parameters(recurse=True):
        record(name, param, "parameters")

    # Buffers
    for name, buf in model.named_buffers(recurse=True):
        record(name, buf, "buffers")

    if verbose:
        # Parameters summary
        print("\nSummary (parameters):")
        if results["parameters"]:
            for device_key, names in results["parameters"].items():
                print(f"{device_key}: {len(names)} tensors")
        else:
            print("No parameters found.")

        # Buffers summary
        print("\nSummary (buffers):")
        if results["buffers"]:
            for device_key, names in results["buffers"].items():
                print(f"{device_key}: {len(names)} buffers")
        else:
            print("No buffers found.")

    return results


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
