"""
Visualizations for MiniLLM concepts.

Each function produces a matplotlib figure that illustrates
a specific LLM concept.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


def plot_attention(model, tokenizer, text, save_path=None):
    """
    Visualize attention patterns for a given input.

    Shows which tokens 'look at' which other tokens,
    making the self-attention mechanism tangible.
    """
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    ids = ids[:model.max_seq_len]
    tokens = [tokenizer.idx2word.get(i, "?") for i in ids]
    x = torch.tensor([ids], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        model(x)

    attn_maps = model.get_attention_maps()
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[1]
    T = len(ids)

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(5 * n_heads, 4 * n_layers))
    if n_layers == 1:
        axes = [axes]
    if n_heads == 1:
        axes = [[ax] for ax in axes]

    cmap = LinearSegmentedColormap.from_list("attn", ["white", "#2196F3"])

    for layer_i, layer_attn in enumerate(attn_maps):
        for head_i in range(n_heads):
            ax = axes[layer_i][head_i]
            weights = layer_attn[0, head_i, :T, :T].numpy()

            ax.imshow(weights, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_title(f"Layer {layer_i+1}, Head {head_i+1}", fontsize=10)
            ax.set_xlabel("Attends to →")
            ax.set_ylabel("Token ↓")

    fig.suptitle(f'Attention Pattern: "{text}"', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def plot_training_loss(pretrain_losses, finetune_losses=None, save_path=None):
    """
    Plot training loss curves.

    When fine-tuning losses are included, shows how the model
    adapts from general knowledge to a specific task.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(pretrain_losses, color="#1976D2", linewidth=2,
            label="Pretraining (general NBA facts)")

    if finetune_losses is not None:
        offset = len(pretrain_losses)
        x_ft = range(offset, offset + len(finetune_losses))
        ax.plot(x_ft, finetune_losses, color="#E53935", linewidth=2,
                label="Fine-tuning (team profiles)")
        ax.axvline(x=offset, color="gray", linestyle="--", alpha=0.5,
                   label="Fine-tuning starts")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: Pretraining → Fine-tuning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def plot_embeddings(model, tokenizer, words=None, save_path=None):
    """
    Visualize token embeddings in 2D using PCA.

    Shows how the model organizes words in its internal space:
    similar words should cluster together.
    """
    if words is None:
        words = [
            "lakers", "celtics", "bulls", "warriors",
            "la", "boston", "chicago", "sf",
            "california", "massachusetts", "illinois",
            "east", "west",
            "yellow", "green", "red", "white",
            "from", "wears",
        ]

    # Get embeddings
    ids = [tokenizer.word2idx.get(w, tokenizer.unk_id) for w in words]
    with torch.no_grad():
        emb = model.token_emb.weight[ids].numpy()

    # Simple PCA to 2D
    emb_centered = emb - emb.mean(axis=0)
    U, S, Vt = np.linalg.svd(emb_centered, full_matrices=False)
    coords = emb_centered @ Vt[:2].T

    # Color by category
    categories = {
        "team": ["lakers", "celtics", "bulls", "warriors"],
        "city": ["la", "boston", "chicago", "sf",
                 "california", "massachusetts", "illinois"],
        "conference": ["east", "west"],
        "color": ["yellow", "green", "red", "white"],
        "verb": ["from", "wears"],
    }
    colors = {"team": "#1976D2", "city": "#43A047", "conference": "#9C27B0",
              "color": "#E53935", "verb": "#FB8C00"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat, cat_words in categories.items():
        mask = [w in cat_words for w in words]
        idxs = [i for i, m in enumerate(mask) if m]
        if idxs:
            ax.scatter(coords[idxs, 0], coords[idxs, 1],
                       c=colors[cat], label=cat, s=80, zorder=3)

    for i, w in enumerate(words):
        ax.annotate(w, (coords[i, 0], coords[i, 1]),
                    fontsize=9, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 6))

    ax.set_title("Token Embeddings (PCA → 2D)")
    ax.legend(title="Category")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def plot_temperature_comparison(model, tokenizer, prompt, save_path=None):
    """
    Show how temperature affects generation.

    Generates text at different temperatures and displays
    the probability distributions side by side.
    """
    temps = [0.01, 0.5, 1.0, 1.5]
    labels = ["T=0 (greedy)", "T=0.5 (focused)", "T=1.0 (balanced)",
              "T=1.5 (creative)"]

    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = ids[:model.max_seq_len]
    x = torch.tensor([ids], dtype=torch.long)

    fig, axes = plt.subplots(1, len(temps), figsize=(4 * len(temps), 4))

    model.eval()
    with torch.no_grad():
        logits = model(x)[:, -1, :]  # logits for next token

    for i, (temp, label) in enumerate(zip(temps, labels)):
        scaled = logits / max(temp, 1e-8)
        probs = torch.softmax(scaled, dim=-1).squeeze().numpy()

        # Show top 10 tokens
        top_idx = np.argsort(probs)[-10:][::-1]
        top_words = [tokenizer.idx2word.get(j, "?") for j in top_idx]
        top_probs = probs[top_idx]

        ax = axes[i]
        bars = ax.barh(range(len(top_words)), top_probs,
                       color="#1976D2", alpha=0.8)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()

    fig.suptitle(f'Next-Token Probabilities: "{prompt} ..."',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig


def plot_context_window(model, tokenizer, system_prompt, user_prompt,
                        save_path=None):
    """
    Visualize the context window layout.

    Shows how system prompt, separator, and user prompt
    fill the fixed-size context window — illustrating why
    context engineering matters.
    """
    sys_ids = tokenizer.encode(system_prompt, add_bos=True, add_eos=False)
    sep_ids = [tokenizer.sep_id]
    usr_ids = tokenizer.encode(user_prompt, add_bos=False, add_eos=False)
    all_ids = sys_ids + sep_ids + usr_ids

    # Pad to max length
    padded = tokenizer.pad_sequence(all_ids, model.max_seq_len)
    tokens = [tokenizer.idx2word.get(i, "?") for i in padded]

    fig, ax = plt.subplots(figsize=(14, 2.5))

    colors = []
    for i, tid in enumerate(padded):
        if i < len(sys_ids):
            colors.append("#1976D2")   # system prompt = blue
        elif i < len(sys_ids) + len(sep_ids):
            colors.append("#E53935")   # separator = red
        elif i < len(all_ids):
            colors.append("#43A047")   # user prompt = green
        else:
            colors.append("#E0E0E0")   # padding = gray

    for i, (tok, col) in enumerate(zip(tokens, colors)):
        rect = plt.Rectangle((i, 0), 0.9, 1, facecolor=col, alpha=0.7,
                              edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.5, tok, ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if col != "#E0E0E0" else "gray")

    ax.set_xlim(-0.1, model.max_seq_len + 0.1)
    ax.set_ylim(-0.3, 1.5)
    ax.set_xticks(range(model.max_seq_len))
    ax.set_xticklabels(range(model.max_seq_len), fontsize=7)
    ax.set_xlabel("Position in Context Window")
    ax.set_yticks([])
    ax.set_title(f"Context Window Layout ({model.max_seq_len} tokens)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1976D2", alpha=0.7, label="System Prompt"),
        Patch(facecolor="#E53935", alpha=0.7, label="Separator"),
        Patch(facecolor="#43A047", alpha=0.7, label="User Prompt"),
        Patch(facecolor="#E0E0E0", alpha=0.7, label="Padding (unused)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    return fig
