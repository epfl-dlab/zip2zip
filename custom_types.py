from dataclasses import dataclass
import torch
from typing import Dict, List, Tuple, Union, overload, Protocol, Iterator, Optional
from hashlib import sha256


@dataclass
class Codebook:
    codes: List[List[int]]
    initial_vocab_size: int
    extra_vocab_size: int
    max_subtokens: int
    pad_token_id: int

    @classmethod
    def from_token_map(
        cls,
        token_map: Dict[str, int],
        initial_vocab_size: int,
        extra_vocab_size: int,
        max_subtokens: int,
        pad_token_id: int,
    ) -> "Codebook":
        codes = [[] for _ in range(len(token_map))]
        for subtoken_str, hypertoken_id in token_map.items():
            subtokens_list = [int(x) for x in subtoken_str.split(",")]
            assert len(subtokens_list) <= max_subtokens
            codes[hypertoken_id - initial_vocab_size] = subtokens_list
        return cls(
            codes, initial_vocab_size, extra_vocab_size, max_subtokens, pad_token_id
        )

    def pad(self) -> torch.Tensor:
        padded_codebook_list = [
            [self.pad_token_id] * self.max_subtokens
            for _ in range(self.extra_vocab_size)
        ]

        for hypertoken_id, subtokens_list in enumerate(self.codes):
            padded_codebook_list[hypertoken_id][: len(subtokens_list)] = subtokens_list

        return torch.tensor(padded_codebook_list, dtype=torch.long)

    @property
    def size(self) -> int:
        return len(self.codes)

    @property
    def mean_code_size(self) -> float:
        if self.size == 0:
            return 0
        return sum(len(c) for c in self.codes) / self.size

    @property
    def codebook_stats(self) -> Dict[str, float]:
        return {
            "size": self.size,
            "utilization": self.size / self.extra_vocab_size
            if self.extra_vocab_size
            else 0,
            "mean_code_size": self.mean_code_size,
            "max_code_size": max(len(c) for c in self.codes) if self.size else 0,
            "min_code_size": min(len(c) for c in self.codes) if self.size else 0,
        }

    def get_hash(self) -> str:
        """Generate a unique hash for this codebook."""
        # Convert codes to strings for consistent hashing
        codes_str = ";".join(",".join(str(x) for x in code) for code in self.codes)
        params = f"{codes_str}_{self.initial_vocab_size}_{self.extra_vocab_size}_{self.max_subtokens}_{self.pad_token_id}"
        return sha256(params.encode()).hexdigest()


class TokenizationContainer(Protocol):
    """Protocol for tokenization statistics."""

    @property
    def stats(self) -> Dict[str, float]:
        return {
            "num_tokens": round(self.num_tokens, 2),
            "num_hypertokens": round(self.num_hypertokens, 2),
            "num_active_hypertokens": round(self.num_active_hypertokens, 2),
            "num_unique_hypertokens": round(self.num_unique_hypertokens, 2),
            "hypertoken_activeness": round(self.hypertoken_activeness, 2),
            "hypertoken_density": round(self.hypertoken_density, 2),
        }

    @property
    def num_tokens(self) -> int:
        ...

    @property
    def num_hypertokens(self) -> int:
        ...

    @property
    def num_active_hypertokens(self) -> int:
        ...

    @property
    def num_unique_hypertokens(self) -> int:
        ...

    @property
    def hypertoken_activeness(self) -> float:
        ...

    @property
    def hypertoken_density(self) -> float:
        ...


@dataclass
class LZWTokenization(TokenizationContainer):
    token_ids: List[int]
    codebook: Codebook

    def get_compression_rate(self) -> float:
        return len(self.token_ids) / self.get_num_tokens_decompressed()

    def get_token_ids_decompressed(self) -> List[int]:
        raw_token_ids = []
        for token_id in self.token_ids:
            if token_id >= self.codebook.initial_vocab_size:
                raw_token_ids.extend(
                    self.codebook.codes[token_id - self.codebook.initial_vocab_size]
                )
            else:
                raw_token_ids.append(token_id)
        return raw_token_ids

    def get_num_tokens_decompressed(self) -> int:
        code_size: Dict[int, int] = {
            token_id: len(code) for token_id, code in enumerate(self.codebook.codes)
        }
        return sum(
            code_size[token_id - self.codebook.initial_vocab_size]
            if token_id >= self.codebook.initial_vocab_size
            else 1
            for token_id in self.token_ids
        )

    @property
    def token_frequencies(self) -> torch.Tensor:
        return torch.bincount(
            torch.tensor(self.token_ids),
            minlength=self.codebook.initial_vocab_size + self.codebook.extra_vocab_size,
        )

    @property
    def num_tokens(self) -> int:
        "Total number of tokens in the tokenization"
        return len(self.token_ids)

    @property
    def hypertoken_frequencies(self) -> torch.Tensor:
        "Distribution of hypertoken occurrences in the tokenization"
        return self.token_frequencies[self.codebook.initial_vocab_size :]

    @property
    def num_hypertokens(self) -> int:
        "Total number of hypertoken occurrences in the tokenization"
        return sum([1 for _ in self.token_ids if _ > self.codebook.initial_vocab_size])

    @property
    def num_unique_hypertokens(self) -> int:
        "Number of unique hypertokens used in the tokenization"
        return int(
            (self.hypertoken_frequencies > 0).sum().item()
        )  # cast to int to make pylance happy

    @property
    def num_active_hypertokens(self) -> int:
        "Number of active Unique hypertokens in the tokenization"
        return int((self.hypertoken_frequencies > 0).sum().item())

    @property
    def hypertoken_activeness(self) -> float:
        "Metric of how many hypertokens are used in the tokenization"
        return (
            self.num_unique_hypertokens / self.codebook.size
            if self.codebook.size > 0
            else 0
        )

    @property
    def hypertoken_density(self) -> float:
        "Metric of how many tokens are hypertokens in the tokenization"
        return self.num_hypertokens / self.num_tokens if self.num_tokens > 0 else 0

    def get_padded_token_ids(self, target_length: int) -> List[int]:
        return self.token_ids + [self.codebook.pad_token_id] * (
            target_length - len(self.token_ids)
        )

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class BatchedLZWTokenization(TokenizationContainer):
    token_ids: List[List[int]]
    codebooks: List[Codebook]

    def to_tensor(
        self, target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert batched token_ids and codebooks to tensor format.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - input_ids tensor of shape (B, S) where B is batch size and S is max sequence length
                - codebook tensor of shape (B, V_E, M) where V_E is extra vocab size and M is max subtokens
        """
        # Convert token_ids to tensor
        batched_compressed_ids = self.get_padded_token_ids(target_length)
        input_ids = torch.tensor(batched_compressed_ids, dtype=torch.long)  # (B, S)

        # Convert codebooks to tensor
        batched_codebooks = [codebook.pad() for codebook in self.codebooks]
        codebook_tensor = torch.stack(batched_codebooks, dim=0)  # (B, V_E, M)

        return input_ids, codebook_tensor

    def get_padded_token_ids(
        self, target_length: Optional[int] = None
    ) -> List[List[int]]:
        if target_length is None:
            target_length = max(len(ids) for ids in self.token_ids)
        return [
            ids + [self.codebooks[i].pad_token_id] * (target_length - len(ids))
            for i, ids in enumerate(self.token_ids)
        ]

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> LZWTokenization:
        return LZWTokenization(self.token_ids[idx], self.codebooks[idx])

    def __iter__(self) -> Iterator[LZWTokenization]:
        """Iterate over the batch, returning LZWTokenization objects."""
        return (
            LZWTokenization(ids, codebook)
            for ids, codebook in zip(self.token_ids, self.codebooks)
        )

    @property
    def num_tokens(self) -> int:
        "Total number of tokens in the tokenization"
        return sum([len(ids) for ids in self.token_ids])

    @property
    def num_hypertokens(self) -> int:
        "Total number of hypertokens in the tokenization"
        return sum([tokenization.num_hypertokens for tokenization in self])

    @property
    def num_active_hypertokens(self) -> int:
        "Total number of active hypertokens in the tokenization"
        return sum([tokenization.num_active_hypertokens for tokenization in self])

    @property
    def num_unique_hypertokens(self) -> int:
        "Total number of unique hypertokens in the tokenization"
        return sum([tokenization.num_unique_hypertokens for tokenization in self])

    @property
    def hypertoken_activeness(self) -> float:
        "Metric of how many hypertokens are used in the tokenization"
        denominator = sum([tokenization.codebook.size for tokenization in self])
        return self.num_unique_hypertokens / denominator if denominator > 0 else 0

    @property
    def hypertoken_density(self) -> float:
        "Metric of how many tokens are hypertokens in the tokenization"
        return self.num_hypertokens / self.num_tokens if self.num_tokens > 0 else 0
