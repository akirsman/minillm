"""
Training data for MiniLLM.

Loads corpora and RAG documents from text files in the data/ directory:
  data/pretrain.txt  - one sentence per line (base knowledge)
  data/finetune.txt  - one sentence per line (chained facts)
  data/rag.txt       - key | document per line (knowledge base)

Lines starting with # and blank lines are ignored.
"""

import os
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_corpus(filename: str) -> list[str]:
    """Load a text corpus: one sentence per line, skip comments/blanks."""
    path = os.path.join(DATA_DIR, filename)
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
    return sentences


def _load_rag(filename: str) -> dict[str, str]:
    """Load RAG documents: 'key | document' per line."""
    path = os.path.join(DATA_DIR, filename)
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "|" in line:
                key, doc = line.split("|", 1)
                docs[key.strip()] = doc.strip()
    return docs


PRETRAIN_CORPUS = _load_corpus("pretrain.txt")
FINETUNE_CORPUS = _load_corpus("finetune.txt")
RAG_DOCUMENTS = _load_rag("rag.txt")


class TextDataset(Dataset):
    """Converts text corpus into training examples for next-token prediction."""

    def __init__(self, corpus, tokenizer, max_len=16):
        self.examples = []
        for sentence in corpus:
            ids = tokenizer.encode(sentence, add_bos=True, add_eos=True)
            ids = tokenizer.pad_sequence(ids, max_len)
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)  # input
        y = torch.tensor(ids[1:], dtype=torch.long)    # target (shifted by 1)
        return x, y
