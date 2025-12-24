import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union
from dataclasses import dataclass


def high_compression_rho(S):
    return 1.2 - 0.2 * np.log10(S + 10)


def low_compression_rho(S):
    return 1.1 - 0.1 * np.log10(S + 10)


@dataclass
class ModelConfig:
    M: int
    l: int
    L: int
    d: int
    C: int  # number of codebook entries
    const_beta: float  # C/S

    compression_mode: str = "high"

    # add an attribute method to compute \alpha
    def alpha(self):
        return (self.M * self.l) / self.L

    def beta(self, S: int):
        if self.const_beta is not None:
            return self.const_beta
        else:
            return self.C / S

    def rho(self, S: int):
        if self.compression_mode == "high":
            return high_compression_rho(S)
        elif self.compression_mode == "low":
            return low_compression_rho(S)
        else:
            raise ValueError(f"Invalid compression mode: {self.compression_mode}")

    def compute_flops_ratio(self, S: Union[int, np.ndarray]):
        alpha = self.alpha()
        beta = self.beta(S)
        rho = self.rho(S)
        return (alpha * beta * (6 * self.d) + rho * (6 * self.d + S * rho)) / (
            6 * self.d + S
        )


def plot_flops_ratio(
    models: dict[str, ModelConfig],
    seq_range: tuple[int, int] = (10, 16_000),
    output_dir: str = "plots/flop_plots",
):
    """
    Plots the FLOPs ratio between the hyper and base modules for multiple models as a function of sequence length S.

    Parameters:
        models (dict): Dictionary where keys are model names and values are parameter dictionaries.
    """
    plt.figure(figsize=(8, 6))

    linestyle = {
        "high": "-",
        "low": "--",
    }

    for i, (model_name, model_config) in enumerate(models.items()):
        flops_ratio = model_config.compute_flops_ratio(
            np.linspace(seq_range[0], seq_range[1], 1000)
        )
        plt.plot(
            np.linspace(seq_range[0], seq_range[1], 1000),
            flops_ratio,
            label=model_name,
            linestyle=linestyle[model_config.compression_mode],
            color=f"C{i}",
        )

        # set model mode to "low"
        model_config.compression_mode = "low"
        flops_ratio = model_config.compute_flops_ratio(
            np.linspace(seq_range[0], seq_range[1], 1000)
        )
        plt.plot(
            np.linspace(seq_range[0], seq_range[1], 1000),
            flops_ratio,
            label="low compression",
            linestyle=linestyle[model_config.compression_mode],
            color=f"C{i}",
        )

    plt.axhline(y=0, color="black", linestyle=":")
    plt.xlabel("Sequence Length (S)")
    plt.ylabel("FLOPS zip2zip / FLOPS base")
    plt.ylim(0.0, 1.0)
    plt.xlim(seq_range)
    plt.title("FLOPS improvement from zip2zip")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "flops_improvement_from_zip2zip.png"))
    plt.show()


if __name__ == "__main__":

    Llama_3_1_3B = ModelConfig(
        M=4, l=1, L=28, d=3072, const_beta=0.5, C=None, compression_mode="high"
    )
    Llama_3_1_8B = ModelConfig(
        M=4, l=1, L=32, d=4096, const_beta=0.5, C=None, compression_mode="high"
    )
    Llama_3_1_70B = ModelConfig(
        M=4, l=2, L=80, d=8192, const_beta=0.5, C=None, compression_mode="high"
    )
    Llama_3_1_405B = ModelConfig(
        M=6, l=2, L=126, d=16384, const_beta=0.5, C=None, compression_mode="high"
    )

    plot_flops_ratio(
        {
            "Llama 3.1 3B": Llama_3_1_3B,
            "Llama 3.1 8B": Llama_3_1_8B,
            "Llama 3.1 70B": Llama_3_1_70B,
            "Llama 3.1 405B": Llama_3_1_405B,
        },
        seq_range=(10, 16_000),
    )

    print(Llama_3_1_3B.compute_flops_ratio(np.linspace(10, 16_000, 1000)))
