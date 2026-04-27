"""
Interactive demonstrations of LLM concepts using MiniLLM.

Each demo function illustrates a specific concept with
text output and visualizations.
"""

import copy
import torch

from mini_llm.model import MiniLLM
from mini_llm.tokenizer import Tokenizer
from mini_llm.data import PRETRAIN_CORPUS, FINETUNE_CORPUS, RAG_DOCUMENTS
from mini_llm.trainer import train
from mini_llm.visualizer import (
    plot_attention,
    plot_training_loss,
    plot_embeddings,
    plot_temperature_comparison,
    plot_context_window,
)


def create_model():
    """Create a fresh MiniLLM instance."""
    tok = Tokenizer()
    model = MiniLLM(vocab_size=tok.vocab_size)
    return model, tok


def generate_text(model, tok, prompt, temperature=0.8, max_tokens=8):
    """Helper to generate and display text."""
    ids = tok.encode(prompt, add_bos=True, add_eos=False)
    ids = ids[:model.max_seq_len]
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_tokens,
                         temperature=temperature, eos_id=tok.eos_id)
    return tok.decode(out[0].tolist())


# ═══════════════════════════════════════════════════════════════════
# DEMO 1: Pretraining
# ═══════════════════════════════════════════════════════════════════
def demo_pretraining():
    print("\n" + "=" * 60)
    print("  DEMO 1: PRETRAINING")
    print("  Learning: <team> from <city>, <team> wears <color>")
    print("=" * 60)

    model, tok = create_model()
    print(f"\n  Model parameters: {model.count_parameters():,}")
    print(f"  Vocabulary size:  {tok.vocab_size}")
    print(f"  Context window:   {model.max_seq_len} tokens")

    print("\n  ── Before Training (random weights) ──")
    for prompt in ["lakers", "bulls from", "warriors wears"]:
        text = generate_text(model, tok, prompt, temperature=0.8)
        print(f'    "{prompt}" → {text}')

    print("\n  ── Training on NBA facts (32 sentences) ──")
    losses = train(model, PRETRAIN_CORPUS, tok, epochs=200, lr=3e-3)

    print("\n  ── After Training ──")
    for prompt in ["lakers", "bulls from", "warriors wears"]:
        text = generate_text(model, tok, prompt, temperature=0.01)
        print(f'    "{prompt}" → {text}')

    plot_training_loss(losses, save_path="output/01_pretrain_loss.png")
    plot_embeddings(model, tok, save_path="output/01_embeddings.png")

    return model, tok, losses


# ═══════════════════════════════════════════════════════════════════
# DEMO 2: Fine-tuning
# ═══════════════════════════════════════════════════════════════════
def demo_finetuning(base_model, tok, pretrain_losses):
    print("\n" + "=" * 60)
    print("  DEMO 2: FINE-TUNING")
    print("  Teaching chained facts: <team> from <city> wears <color>")
    print("=" * 60)

    ft_model = copy.deepcopy(base_model)

    print("\n  ── Before Fine-tuning ──")
    for prompt in ["lakers from", "warriors from", "celtics from"]:
        text = generate_text(ft_model, tok, prompt, temperature=0.01)
        print(f'    "{prompt}" → {text}')

    print("\n  ── Fine-tuning on chained facts ──")
    ft_losses = train(ft_model, FINETUNE_CORPUS, tok,
                      epochs=100, lr=1e-3, label="Fine-tune")

    print("\n  ── After Fine-tuning ──")
    for prompt in ["lakers from", "warriors from", "celtics from"]:
        text = generate_text(ft_model, tok, prompt, temperature=0.01)
        print(f'    "{prompt}" → {text}')

    plot_training_loss(pretrain_losses, ft_losses,
                       save_path="output/02_finetune_loss.png")
    return ft_model


# ═══════════════════════════════════════════════════════════════════
# DEMO 3: Prompt Engineering
# ═══════════════════════════════════════════════════════════════════
def demo_prompt_engineering(model, tok):
    print("\n" + "=" * 60)
    print("  DEMO 3: PROMPT ENGINEERING")
    print("  Same model, different prompts → different outputs")
    print("=" * 60)

    prompts = [
        ("lakers",        "Just team → model picks from or wears"),
        ("lakers from",   "Team + from → city? state? conference?"),
        ("lakers wears",  "Team + wears → color"),
        ("bulls from",    "Different team → different location options"),
        ("california",    "Ambiguous! lakers or warriors?"),
        ("east",          "Shared attribute → celtics or bulls?"),
    ]

    print()
    for prompt, description in prompts:
        text = generate_text(model, tok, prompt, temperature=0.7)
        print(f'  [{description}]')
        print(f'    "{prompt}" → {text}\n')


# ═══════════════════════════════════════════════════════════════════
# DEMO 4: Context Engineering
# ═══════════════════════════════════════════════════════════════════
def demo_context_engineering(model, tok):
    print("\n" + "=" * 60)
    print("  DEMO 4: CONTEXT ENGINEERING")
    print("  Managing the context window: system + user prompts")
    print("=" * 60)

    scenarios = [
        ("lakers from la", "lakers wears",
         "System gives city → user asks color"),
        ("lakers from california warriors from california", "bulls",
         "Long system context → less room"),
        ("lakers", "from",
         "Minimal system + user"),
    ]

    for sys_prompt, usr_prompt, description in scenarios:
        print(f"\n  ── {description} ──")
        print(f"    System: \"{sys_prompt}\"")
        print(f"    User:   \"{usr_prompt}\"")

        sys_ids = tok.encode(sys_prompt, add_bos=True, add_eos=False)
        sep_ids = [tok.sep_id]
        usr_ids = tok.encode(usr_prompt, add_bos=False, add_eos=False)
        all_ids = sys_ids + sep_ids + usr_ids
        used = len(all_ids)
        remaining = model.max_seq_len - used
        print(f"    Tokens used: {used}/{model.max_seq_len} "
              f"({remaining} remaining)")

        padded = all_ids[:model.max_seq_len]
        x = torch.tensor([padded], dtype=torch.long)
        out = model.generate(x, max_new_tokens=min(remaining, 4),
                             temperature=0.01, eos_id=tok.eos_id)
        generated = tok.decode(out[0].tolist()[len(padded):])
        print(f"    Generated: {generated}")

    plot_context_window(model, tok,
                        scenarios[0][0], scenarios[0][1],
                        save_path="output/04_context_window.png")


# ═══════════════════════════════════════════════════════════════════
# DEMO 5: RAG
# ═══════════════════════════════════════════════════════════════════
def demo_rag(model, tok):
    print("\n" + "=" * 60)
    print("  DEMO 5: RAG (Retrieval-Augmented Generation)")
    print("  Injecting retrieved knowledge into the context")
    print("=" * 60)

    queries = ["lakers", "bulls", "warriors"]

    for query in queries:
        print(f'\n  ── Query: "{query} wears" ──')

        print("  [Without RAG]")
        text_no_rag = generate_text(model, tok, f"{query} wears",
                                    temperature=0.7)
        print(f"    → {text_no_rag}")

        doc = RAG_DOCUMENTS.get(query, "")
        print(f"  [Retrieved]: \"{doc}\"")

        doc_ids = tok.encode(doc, add_bos=True, add_eos=False)
        sep_ids = [tok.sep_id]
        query_ids = tok.encode(f"{query} wears", add_bos=False, add_eos=False)
        rag_ids = doc_ids + sep_ids + query_ids
        rag_ids = rag_ids[:model.max_seq_len]

        x = torch.tensor([rag_ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=4, temperature=0.01,
                             eos_id=tok.eos_id)
        text_rag = tok.decode(out[0].tolist()[len(rag_ids):])
        print(f"  [With RAG]")
        print(f"    → {text_rag}")

    doc = RAG_DOCUMENTS["lakers"]
    rag_text = f"{doc} <sep> lakers wears"
    plot_attention(model, tok, rag_text,
                   save_path="output/05_rag_attention.png")


# ═══════════════════════════════════════════════════════════════════
# DEMO 6: Temperature
# ═══════════════════════════════════════════════════════════════════
def demo_temperature(model, tok):
    print("\n" + "=" * 60)
    print("  DEMO 6: TEMPERATURE & SAMPLING")
    print("  How randomness affects text generation")
    print("=" * 60)

    prompt = "lakers"
    temps = [0.01, 0.5, 1.0, 1.5]

    for temp in temps:
        print(f"\n  Temperature = {temp}")
        for i in range(3):
            text = generate_text(model, tok, prompt, temperature=temp)
            print(f"    Run {i+1}: {text}")

    plot_temperature_comparison(model, tok, "lakers",
                                save_path="output/06_temperature.png")


# ═══════════════════════════════════════════════════════════════════
# DEMO 7: Attention
# ═══════════════════════════════════════════════════════════════════
def demo_attention(model, tok):
    print("\n" + "=" * 60)
    print("  DEMO 7: ATTENTION VISUALIZATION")
    print("  See which tokens attend to which")
    print("=" * 60)

    sentences = [
        "lakers from california wears yellow",
        "celtics from east wears green",
        "warriors from sf",
    ]

    for i, sent in enumerate(sentences):
        print(f'\n  Visualizing: "{sent}"')
        plot_attention(model, tok, sent,
                       save_path=f"output/07_attention_{i+1}.png")
