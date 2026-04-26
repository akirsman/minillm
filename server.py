#!/usr/bin/env python3
"""
MiniLLM Web Server

Serves the interactive web UI and provides API endpoints
for inference with full visibility into the model's internals.

Usage:
    python server.py
    # Then open http://localhost:8000
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mini_llm.model import MiniLLM
from mini_llm.tokenizer import Tokenizer
from mini_llm.data import PRETRAIN_CORPUS, RAG_DOCUMENTS
from mini_llm.trainer import train

# ── Boot: create and pretrain the model ────────────────────────────
print("Initializing MiniLLM...")
tok = Tokenizer()
model = MiniLLM(vocab_size=tok.vocab_size)
print(f"  Vocab: {tok.vocab_size} tokens, Params: {model.count_parameters():,}")
print("  Pretraining on NBA corpus...")
train(model, PRETRAIN_CORPUS, tok, epochs=200, lr=3e-3, verbose=False)
print("  Ready!\n")

app = FastAPI(title="MiniLLM")

# ── Serve static files ─────────────────────────────────────────────
static_dir = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


# ── API Models ─────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.8
    max_tokens: int = 10
    use_rag: bool = False
    system_prompt: str = ""


class TokenizeRequest(BaseModel):
    text: str


# ── Helpers ────────────────────────────────────────────────────────
def get_top_k_probs(logits, k=10):
    """Return top-k token probabilities from logits."""
    probs = F.softmax(logits, dim=-1).squeeze().detach().numpy()
    top_idx = np.argsort(probs)[-k:][::-1]
    return [
        {"token": tok.idx2word.get(int(i), "?"),
         "token_id": int(i),
         "probability": round(float(probs[i]), 4)}
        for i in top_idx
    ]


def get_attention_data(token_labels):
    """Extract attention weights from all layers/heads."""
    maps = model.get_attention_maps()
    layers = []
    for li, layer_attn in enumerate(maps):
        heads = []
        for hi in range(layer_attn.shape[1]):
            T = len(token_labels)
            w = layer_attn[0, hi, :T, :T].numpy()
            heads.append({
                "head": hi,
                "weights": [[round(float(v), 4) for v in row]
                            for row in w],
            })
        layers.append({"layer": li, "heads": heads})
    return layers


def get_embedding_coords(token_ids):
    """Get 2D PCA projection of token embeddings."""
    with torch.no_grad():
        emb = model.token_emb.weight.numpy()

    # PCA on full vocab
    centered = emb - emb.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    coords_all = centered @ Vt[:2].T

    # Return coords for input tokens + some reference words
    ref_words = [
        "lakers", "celtics", "bulls", "warriors",
        "la", "boston", "chicago", "sf",
        "california", "massachusetts", "illinois",
        "east", "west",
        "yellow", "green", "red", "white",
        "from", "wears",
    ]
    ref_ids = [tok.word2idx.get(w, tok.unk_id) for w in ref_words]

    points = []
    # Reference words (dimmed)
    for w, rid in zip(ref_words, ref_ids):
        points.append({
            "word": w, "x": round(float(coords_all[rid, 0]), 4),
            "y": round(float(coords_all[rid, 1]), 4),
            "type": "reference",
        })
    # Input tokens (highlighted)
    for tid in token_ids:
        w = tok.idx2word.get(tid, "?")
        if tid < len(coords_all):
            points.append({
                "word": w, "x": round(float(coords_all[tid, 0]), 4),
                "y": round(float(coords_all[tid, 1]), 4),
                "type": "input",
            })
    return points


# ── API Endpoints ──────────────────────────────────────────────────
@app.get("/api/info")
async def model_info():
    """Return model architecture info."""
    return {
        "vocab_size": tok.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.layers[0].attn.n_heads,
        "n_layers": len(model.layers),
        "max_seq_len": model.max_seq_len,
        "parameters": model.count_parameters(),
        "vocabulary": [tok.idx2word[i] for i in range(tok.vocab_size)],
    }


@app.post("/api/tokenize")
async def tokenize(req: TokenizeRequest):
    """Show how text is tokenized into IDs."""
    ids = tok.encode(req.text, add_bos=True, add_eos=False)
    tokens = [{"token": tok.idx2word.get(i, "?"), "id": i} for i in ids]
    return {"tokens": tokens, "ids": ids}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """
    Generate text with full internals visibility.

    Returns each generation step with:
    - The token chosen
    - Top-k probability distribution
    - Attention maps for the final state
    - Embedding positions
    """
    # Build input IDs
    input_ids = []

    # System prompt
    if req.system_prompt:
        sys_ids = tok.encode(req.system_prompt, add_bos=True, add_eos=False)
        input_ids.extend(sys_ids)
        input_ids.append(tok.sep_id)

    # RAG: retrieve and prepend
    rag_doc = None
    if req.use_rag:
        # Simple keyword retrieval
        words = req.prompt.lower().split()
        for w in words:
            if w in RAG_DOCUMENTS:
                rag_doc = RAG_DOCUMENTS[w]
                break
        if rag_doc:
            doc_ids = tok.encode(rag_doc, add_bos=not bool(req.system_prompt),
                                 add_eos=False)
            if not req.system_prompt:
                input_ids.extend(doc_ids)
            else:
                input_ids.extend(doc_ids[1:])  # skip extra <bos>
            input_ids.append(tok.sep_id)

    # User prompt
    prompt_ids = tok.encode(req.prompt,
                            add_bos=not (req.system_prompt or req.use_rag),
                            add_eos=False)
    if req.system_prompt or req.use_rag:
        prompt_ids = prompt_ids[1:] if prompt_ids[0] == tok.bos_id else prompt_ids
    input_ids.extend(prompt_ids)

    # Truncate to context window
    input_ids = input_ids[:model.max_seq_len]

    # Token labels for the input
    input_tokens = [{"token": tok.idx2word.get(i, "?"), "id": i}
                    for i in input_ids]

    # ── Step-by-step generation ──
    steps = []
    current_ids = list(input_ids)
    model.eval()

    for step_i in range(min(req.max_tokens, model.max_seq_len - len(current_ids))):
        x = torch.tensor([current_ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        next_logits = logits[:, -1, :] / max(req.temperature, 1e-8)
        probs = F.softmax(next_logits, dim=-1)

        # Sample
        if req.temperature < 0.01:
            next_id = torch.argmax(probs, dim=-1).item()
        else:
            next_id = torch.multinomial(probs, num_samples=1).item()

        top_k = get_top_k_probs(next_logits)

        steps.append({
            "step": step_i,
            "chosen_token": tok.idx2word.get(next_id, "?"),
            "chosen_id": next_id,
            "chosen_prob": round(float(probs[0, next_id].item()), 4),
            "top_k": top_k,
        })

        current_ids.append(next_id)

        if next_id == tok.eos_id:
            break

    # ── Final attention maps ──
    x = torch.tensor([current_ids[:model.max_seq_len]], dtype=torch.long)
    with torch.no_grad():
        model(x)

    all_labels = [tok.idx2word.get(i, "?") for i in current_ids[:model.max_seq_len]]
    attention = get_attention_data(all_labels)
    embeddings = get_embedding_coords(current_ids[:model.max_seq_len])

    # Generated text
    gen_ids = current_ids[len(input_ids):]
    generated_text = tok.decode(gen_ids, skip_special=True)

    return {
        "input_tokens": input_tokens,
        "steps": steps,
        "generated_text": generated_text,
        "full_text": tok.decode(current_ids, skip_special=True),
        "attention": attention,
        "token_labels": all_labels,
        "embeddings": embeddings,
        "rag_document": rag_doc,
        "context_map": {
            "total_slots": model.max_seq_len,
            "used": len(current_ids),
            "input_len": len(input_ids),
            "generated_len": len(gen_ids),
        },
    }


@app.get("/api/rag_documents")
async def rag_documents():
    """List available RAG documents."""
    return {"documents": RAG_DOCUMENTS}


# ── Run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting MiniLLM server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
