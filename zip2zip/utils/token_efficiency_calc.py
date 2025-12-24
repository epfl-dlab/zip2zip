import pandas as pd
from typing import List
from datasets import IterableDataset
from transformers import PreTrainedTokenizer
import numpy as np
from tqdm import tqdm


class TokenEfficiencyCalculator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def compute_for_text(self, text: str) -> float:
        """Compute UTF-8 bytes per token for a single text string."""
        num_tokens = len(self.tokenizer.encode(text))
        num_bytes = len(text.encode("utf-8"))
        return num_bytes / num_tokens if num_tokens > 0 else float("inf")

    def compute_for_dataset(
        self,
        dataset: IterableDataset,
        num_samples: int = 1000,
        column_name: str = "text",
    ) -> float:
        """Compute average bytes per token for a dataset."""
        total_efficiency = 0.0
        count = 0

        for sample in tqdm(
            dataset, desc="Computing token efficiency", total=num_samples
        ):
            text = sample[column_name]
            efficiency = self.compute_for_text(text)
            total_efficiency += efficiency
            count += 1
            if count >= num_samples:
                break

        return total_efficiency / count if count > 0 else 0.0

    @staticmethod
    def compute_matrix(
        tokenizers: List[PreTrainedTokenizer],
        datasets: List[IterableDataset],
        column_name: str = "text",
        num_samples: int = 100,
        relative: bool = False,
        base_index: int = 0,
    ) -> np.ndarray:
        """Compute a matrix of token efficiency for multiple tokenizers and datasets."""
        token_efficiency_matrix = np.zeros((len(tokenizers), len(datasets)))
        for i, tokenizer in enumerate(tokenizers):
            calc = TokenEfficiencyCalculator(tokenizer)
            for j, dataset in enumerate(datasets):
                token_efficiency_matrix[i, j] = calc.compute_for_dataset(
                    dataset, num_samples, column_name
                )

        if relative:
            token_efficiency_matrix = (
                token_efficiency_matrix / token_efficiency_matrix[base_index]
            )

        return token_efficiency_matrix

    @staticmethod
    def prettyprint_matrix(
        matrix: np.ndarray, column_names: List[str], row_names: List[str]
    ) -> None:
        df = pd.DataFrame(matrix, index=row_names, columns=column_names).round(2)
        print(df)

    @staticmethod
    def bar_chart(
        matrix: np.ndarray,
        column_names: list,
        row_names: list,
        figsize: tuple = (15, 6),
        save_path: str = "token_efficiency_bar_chart.png",
        title: str = "Token Efficiency",
        xlabel: str = "Dataset",
        ylabel: str = "Average Token Length (Bytes)",
        show: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        num_tokenizers, num_datasets = matrix.shape
        bar_width = 0.2
        index = np.arange(num_datasets)

        plt.figure(figsize=figsize)

        for i in range(num_tokenizers):
            offset = i * bar_width
            bars = plt.bar(
                index + offset, matrix[i], width=bar_width, label=row_names[i]
            )
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.05,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(index + bar_width * (num_tokenizers - 1) / 2, column_names)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
