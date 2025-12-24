import torch
import torch.nn.functional as F
import pytest
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nn.attention import py_scaled_dot_product_attention


def test_basic_attention():
    """Basic test for correct shape and similarity to PyTorch version"""
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)

    custom_out, attn_weight = py_scaled_dot_product_attention(query, key, value)
    torch_out = F.scaled_dot_product_attention(query, key, value)

    assert custom_out.shape == torch_out.shape, "Output shapes do not match"
    assert torch.allclose(
        custom_out, torch_out, atol=1e-5
    ), "Outputs are not numerically close"


def test_attention_with_mask():
    """Test applying an attention mask"""
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)
    mask = torch.tensor(
        [[[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]]], dtype=torch.bool
    )

    custom_out, attn_weight = py_scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )
    torch_out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

    assert custom_out.shape == torch_out.shape, "Output shapes do not match with mask"
    assert torch.allclose(
        custom_out, torch_out, atol=1e-5
    ), "Outputs differ when using mask"


def test_attention_causal():
    """Test causal attention prevents future information leakage"""
    query = torch.randn(1, 1, 4, 8)
    key = torch.randn(1, 1, 4, 8)
    value = torch.randn(1, 1, 4, 8)

    custom_out, attn_weight = py_scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    torch_out = F.scaled_dot_product_attention(query, key, value, is_causal=True)

    assert (
        custom_out.shape == torch_out.shape
    ), "Output shapes do not match for causal attention"
    assert torch.allclose(
        custom_out, torch_out, atol=1e-5
    ), "Causal attention does not match PyTorch version"


if __name__ == "__main__":
    pytest.main()
