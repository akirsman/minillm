"""
Training loop for MiniLLM.

Supports both pretraining and fine-tuning with loss tracking
for visualization.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_llm.data import TextDataset


def train(model, corpus, tokenizer, epochs=100, lr=3e-3,
          batch_size=8, max_len=16, verbose=True, label="Training"):
    """
    Train (or fine-tune) the model on a corpus.

    Returns:
        losses: list of per-epoch average losses (for plotting)
    """
    dataset = TextDataset(corpus, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    losses = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for x, y in loader:
            logits = model(x)
            # Reshape for cross-entropy: (B*T, vocab) vs (B*T,)
            loss = criterion(
                logits.view(-1, model.vocab_size),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs}  "
                  f"Loss: {avg_loss:.4f}")

    if verbose:
        print(f"  [{label}] Final loss: {losses[-1]:.4f}")

    return losses
