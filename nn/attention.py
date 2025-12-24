import math
import torch
import torch.nn.functional as F
from torch import nn


def py_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    is_causal=False,
    scale=None,
) -> torch.Tensor:
    """
    Simplified implementation of scaled dot-product attention.

    Args:
        query, key, value: shape (batch_size, num_heads, seq_len, head_dim)
        attn_mask: shape (batch_size, 1, seq_len, seq_len), boolean mask where False means masked
        is_causal: whether to apply a causal mask

    Returns:
        Tuple (attention output, attention weights)
    """
    scale_factor = 1 / (query.size(-1) ** 0.5) if scale is None else scale

    # Compute attention scores
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # Apply causal mask if needed
    if is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
        attn_mask = (
            causal_mask if attn_mask is None else attn_mask & causal_mask
        )  # Combine with user mask

    # Apply attention mask if provided
    if attn_mask is not None:
        attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
    # Apply softmax
    attn_weight = F.softmax(attn_weight, dim=-1)
    # replace NaN with 0. The NaN happes when all the scores in a row are -infs
    attn_weight = torch.nan_to_num(attn_weight, nan=0.0)

    # Compute attention output
    output = torch.matmul(attn_weight, value)

    return output, attn_weight


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        hidden_size: int,
        pe_dim: int = 64,
        scale_factor: float = 10.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Sinusoidal positional encoding module.

        Args:
            max_seq_len (int): Maximum sequence length.
            pe_dim (int): Lower-dimensional positional encoding (e.g., 64).
            hidden_size (int): Target hidden size (e.g., 1024).
            scale_factor (float): Scaling factor for sinusoidal frequencies (default: 1000.0).
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pe_dim = pe_dim
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        self.dtype = dtype
        self.register_buffer(
            "pe", self.compute_positional_encoding(max_seq_len, dtype, device)
        )

    def compute_positional_encoding(self, seq_len: int, dtype, device) -> torch.Tensor:
        """
        Computes sinusoidal positional encoding dynamically.

        Args:
            seq_len (int): Actual sequence length (≤ max_seq_len).
            device (torch.device): Device to place tensor.

        Returns:
            torch.Tensor: Positional encoding tensor of shape (1, seq_len, hidden_size).
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.pe_dim, 2, dtype=torch.float, device=device)
            * (-math.log(self.scale_factor) / self.pe_dim)
        )

        pe = torch.zeros(seq_len, self.pe_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Expand `pe_dim` to `hidden_size` using tiling
        repeat_factor = (
            self.hidden_size // self.pe_dim
        )  # How many times we need to repeat
        pe = pe.repeat(1, repeat_factor)  # Repeat along hidden dim
        pe = pe[:, : self.hidden_size]  # Trim if needed

        return pe.unsqueeze(0).to(
            dtype
        )  # Shape: (1, seq_len, hidden_size) for broadcasting

    def forward(self, seq_len: int):
        """
        Retrieves precomputed positional encoding up to the required sequence length.

        Args:
            seq_len (int): The actual sequence length to retrieve.

        Returns:
            torch.Tensor: Positional encoding of shape (1, seq_len, hidden_size).
        """
        return self.pe[:, :seq_len, :]  # Extract only the required positions
