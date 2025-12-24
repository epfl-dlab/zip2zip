import os
import sys
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model import OnlineZZModel
from utils import setup_seed, real_size_of_codebook


def visualize_attention_grid(
    attn_tensor,
    attn_mask=None,
    save_path="attention_plots/attention_grid.png",
    tokens=None,
    cmap="Reds",
):
    """
    Visualizes a (L, H, S, S) attention tensor as a single figure with L rows and H columns of heatmaps.

    Args:
        attn_tensor (torch.Tensor or np.ndarray): Attention tensor of shape (L, H, S, S)
        save_path (str, optional): Path to save the figure (default: "attention_plots/attention_grid.png")
        tokens (list, optional): List of tokens to label axes (default: None)
        cmap (str, optional): Color map for heatmaps (default: "viridis")
    """
    if attn_mask is not None:
        attn_tensor = attn_tensor.masked_fill(attn_mask.logical_not(), float("nan"))
    if isinstance(attn_tensor, torch.Tensor):
        attn_tensor = attn_tensor.detach().cpu().to(torch.float32).numpy()

    L, H, seq_len, _ = attn_tensor.shape  # Layers, Heads, Seq_len, Seq_len

    os.makedirs(
        os.path.dirname(save_path), exist_ok=True
    )  # Ensure save directory exists

    fig, axes = plt.subplots(
        L, H, figsize=(H * 3, L * 3)
    )  # Create L x H grid of subplots

    # Progress bar tracking
    total_plots = L * H
    with tqdm(total=total_plots, desc="Plotting Attention Heads") as pbar:
        for layer in range(L):
            for head in range(H):
                ax = (
                    axes[layer, head] if L > 1 and H > 1 else axes[max(layer, head)]
                )  # Handle edge cases
                attn_map = attn_tensor[layer, head]
                im = ax.imshow(attn_map, cmap=cmap, aspect="auto")  # Plot the heatmap

                # Add numerical values to heatmap
                for i in range(seq_len):
                    for j in range(seq_len):
                        text_color = "white" if attn_map[i, j] > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{attn_map[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=6,
                        )

                ax.set_title(f"L{layer+1} H{head+1}", fontsize=8)

                if tokens and layer == L - 1:  # Only label x-axis on last row
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                else:
                    ax.set_xticks([])

                if tokens and head == 0:  # Only label y-axis on first column
                    ax.set_yticks(range(len(tokens)))
                    ax.set_yticklabels(tokens, fontsize=6)
                else:
                    ax.set_yticks([])
                pbar.update(1)  # Update progress bar

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Attention grid saved at: {save_path}")


def main():
    """Main function to inspect hypertoken embeddings and visualize attention weights."""
    setup_seed()

    # Argument parsing
    parser = ArgumentParser(
        description="Inspect hypertoken embeddings and visualize attention maps."
    )
    parser.add_argument(
        "--adapter", type=str, required=True, help="Path to model adapter."
    )
    parser.add_argument(
        "--hub-adapter", type=str, required=False, help="Optional hub adapter path."
    )
    parser.add_argument("--prompt", type=str, required=False, help="Input prompt text.")
    parser.add_argument(
        "--extra-vocab-size", type=int, default=None, help="Size of extra vocabulary."
    )
    args = parser.parse_args()

    # Load model
    model = OnlineZZModel.load_pretrained(
        args.adapter, args.hub_adapter, args.extra_vocab_size
    )

    # Enable debug mode for attention weights
    model.config.embedding_encoder.unsafe_config["attn_implementation"] = "debug"

    # Print vocabulary size
    print(f"Base Vocabulary Size: {model.config.initial_vocab_size}")
    print(f"Extra Vocabulary Size: {model.config.extra_vocab_size}")

    # Define input text
    input_text = "squeeze the juice squeeze the juice"
    input_ids = model.tokenizer([input_text])["input_ids"]
    print("Input IDs:", input_ids)

    # Compress input using LZW
    lzw_input_ids, codebook_tensor = model.lzw_compress(input_ids)

    # Print LZW token statistics
    real_size = real_size_of_codebook(codebook_tensor).item()
    print(f"LZW Input Shape: {lzw_input_ids.shape}")  # (B, L)
    print(f"Codebook Tensor Shape: {codebook_tensor.shape}")  # (B, V_E, M)
    print(f"Number of Hyper Tokens: {real_size}")

    # Compute embeddings
    base_token_embeddings = model.compute_codebook_embeddings(
        codebook_tensor
    )  # (B, V_E, M, D)
    all_hypertoken_embeddings, metadata = model.compute_all_hypertoken_embeddings(
        codebook_tensor
    )  # (B, V_E, D)

    print(
        "First Hyper Token Embedding:", all_hypertoken_embeddings[0, 0, :]
    )  # shape (D,)

    # Extract non-empty attention weights and mask
    non_empty_attn_tensor = metadata["attn_weight"][:real_size, ...]  # (B, S, S)
    non_empty_attn_mask = metadata["attn_mask"][:real_size, ...]  # (B, S, S)

    # Visualize attention
    visualize_attention_grid(
        attn_tensor=non_empty_attn_tensor, attn_mask=non_empty_attn_mask
    )


if __name__ == "__main__":
    main()
