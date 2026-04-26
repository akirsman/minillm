"""
Simple word-level tokenizer for MiniLLM.

Minimal vocabulary with a strict grammar:
  <team> from <city>
  <team> wears <color>

No English glue words — every token carries meaning.
"""


# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"  # Used for context engineering (separates system/user)

VOCABULARY = [
    # Special tokens (indices 0-4)
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    # Teams (4)
    "lakers", "celtics", "bulls", "warriors",
    # Cities (4)
    "la", "boston", "chicago", "sf",
    # States (3 — california is shared by lakers & warriors)
    "california", "massachusetts", "illinois",
    # Conferences (2 — shared attributes)
    "east", "west",
    # Colors (4 — one per team)
    "yellow", "green", "red", "white",
    # Grammar (2 verbs only)
    "from", "wears",
]

# Remove duplicates while preserving order
_seen = set()
VOCAB_LIST = []
for w in VOCABULARY:
    if w not in _seen:
        _seen.add(w)
        VOCAB_LIST.append(w)


class Tokenizer:
    """Word-level tokenizer with a fixed vocabulary."""

    def __init__(self):
        self.word2idx = {w: i for i, w in enumerate(VOCAB_LIST)}
        self.idx2word = {i: w for i, w in enumerate(VOCAB_LIST)}
        self.vocab_size = len(VOCAB_LIST)

        # Store special token indices
        self.pad_id = self.word2idx[PAD_TOKEN]
        self.unk_id = self.word2idx[UNK_TOKEN]
        self.bos_id = self.word2idx[BOS_TOKEN]
        self.eos_id = self.word2idx[EOS_TOKEN]
        self.sep_id = self.word2idx[SEP_TOKEN]

    def encode(self, text: str, add_bos=True, add_eos=True) -> list[int]:
        """Convert text to token IDs."""
        words = text.lower().strip().split()
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for w in words:
            ids.append(self.word2idx.get(w, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special=True) -> str:
        """Convert token IDs back to text."""
        special = {self.pad_id, self.bos_id, self.eos_id}
        words = []
        for idx in ids:
            if skip_special and idx in special:
                continue
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        """Pad or truncate a sequence to max_len."""
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))

    def __repr__(self):
        return f"Tokenizer(vocab_size={self.vocab_size})"
