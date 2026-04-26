# MiniLLM - A Minimal Transformer for Teaching LLM Concepts

A tiny but complete transformer-based language model designed to illustrate core LLM concepts to technical audiences. Trainable on a CPU in seconds.

## Architecture

- **Vocabulary**: 24 words (4 teams, 4 cities, 3 states, 2 conferences, 4 colors, 2 verbs, 5 special)
- **Grammar**: `<team> from <city|state|conference>` and `<team> wears <color>`
- **Embedding dimension**: 32
- **Attention heads**: 2
- **Transformer layers**: 2
- **Context window**: 16 tokens
- **Total parameters**: ~18,100

## Concepts Illustrated

| Concept | Demo |
|---------|------|
| **Pretraining** | Train the base model on a small corpus |
| **Fine-tuning** | Adapt the model to a specific style/task |
| **Prompt Engineering** | Show how different prompts change outputs |
| **Context Engineering** | Demonstrate system prompts and context windows |
| **RAG** | Retrieve relevant "documents" and inject into context |
| **Attention Visualization** | See which tokens attend to which |
| **Temperature & Sampling** | Compare greedy vs. creative generation |

## Quick Start

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Run specific demos

```bash
python main.py all      # run all 7 demos
python main.py 1        # pretraining only
python main.py 1 3 5    # pick specific demos
```

Visualizations are saved as PNGs in the `output/` folder.

## Interactive Web UI

A browser-based interface that lets you chat with the model while watching its internals in real time.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
# Open http://localhost:8000
```

### macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
# Open http://localhost:8000
```

The web UI shows three visualization tabs:

- **Pipeline** — Step-by-step view: tokenization → context window → next-token probabilities → decoded output
- **Attention** — Heatmaps for every layer and head, showing which tokens attend to which
- **Embeddings** — 2D PCA scatter plot of token embeddings, with your input tokens highlighted

Controls: temperature slider, max tokens, RAG toggle, and an optional system prompt field.

## Project Structure

```
data/
├── pretrain.txt      # Pretraining corpus (one sentence per line)
├── finetune.txt      # Fine-tuning corpus (chained facts)
├── rag.txt           # RAG knowledge base (key | document per line)
mini_llm/
├── model.py          # Transformer architecture
├── tokenizer.py      # Simple word-level tokenizer
├── data.py           # Data loading from text files
├── trainer.py        # Training loop
├── visualizer.py     # Attention and weight visualizations
├── demos.py          # Interactive concept demonstrations
web/
├── index.html        # Single-page app
├── style.css         # Dark-theme styles
├── app.js            # Visualization logic (Canvas-based)
main.py               # Entry point with menu-driven demos
server.py             # FastAPI web server for interactive UI
requirements.txt
```
