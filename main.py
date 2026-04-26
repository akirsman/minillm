#!/usr/bin/env python3
"""
MiniLLM - A Minimal Transformer for Teaching LLM Concepts

Run all demos sequentially, or pick individual ones.

Usage:
    python main.py          # Run all demos
    python main.py 1        # Run only demo 1 (pretraining)
    python main.py 3 5      # Run demos 3 and 5
"""

import os
import sys

# Create output directory for saved figures
os.makedirs("output", exist_ok=True)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ███╗   ███╗██╗███╗   ██╗██╗██╗     ██╗     ███╗   ███╗║
║   ████╗ ████║██║████╗  ██║██║██║     ██║     ████╗ ████║║
║   ██╔████╔██║██║██╔██╗ ██║██║██║     ██║     ██╔████╔██║║
║   ██║╚██╔╝██║██║██║╚██╗██║██║██║     ██║     ██║╚██╔╝██║║
║   ██║ ╚═╝ ██║██║██║ ╚████║██║███████╗███████╗██║ ╚═╝ ██║║
║   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚══════╝╚══════╝╚═╝     ╚═╝║
║                                                          ║
║   A Minimal Transformer for Teaching LLM Concepts        ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    print("""
  Available Demos:
  ─────────────────────────────────────────────────
  1. Pretraining      - Train the base model
  2. Fine-tuning      - Adapt to a specific style
  3. Prompt Eng.      - Different prompts → outputs
  4. Context Eng.     - System prompts & context window
  5. RAG              - Retrieval-augmented generation
  6. Temperature      - Sampling & randomness
  7. Attention        - Visualize attention patterns
  ─────────────────────────────────────────────────
  all  Run all demos sequentially
  q    Quit
    """)


def run_demos(selections):
    """Run selected demos."""
    from mini_llm.demos import (
        demo_pretraining,
        demo_finetuning,
        demo_prompt_engineering,
        demo_context_engineering,
        demo_rag,
        demo_temperature,
        demo_attention,
    )

    model, tok, pretrain_losses = None, None, None

    # Demos 2-7 need a pretrained model
    needs_pretrain = any(d in selections for d in [1, 2, 3, 4, 5, 6, 7])

    if needs_pretrain:
        if 1 in selections:
            model, tok, pretrain_losses = demo_pretraining()
        else:
            # Quick silent pretrain for other demos
            print("\n  Pretraining base model (quick)...")
            from mini_llm.demos import create_model
            from mini_llm.trainer import train as do_train
            from mini_llm.data import PRETRAIN_CORPUS
            model, tok = create_model()
            pretrain_losses = do_train(
                model, PRETRAIN_CORPUS, tok,
                epochs=150, lr=3e-3, verbose=False
            )
            print(f"  Done. Final loss: {pretrain_losses[-1]:.4f}")

    if 2 in selections:
        demo_finetuning(model, tok, pretrain_losses)

    if 3 in selections:
        demo_prompt_engineering(model, tok)

    if 4 in selections:
        demo_context_engineering(model, tok)

    if 5 in selections:
        demo_rag(model, tok)

    if 6 in selections:
        demo_temperature(model, tok)

    if 7 in selections:
        demo_attention(model, tok)

    print("\n" + "=" * 60)
    print("  All selected demos complete!")
    print("  Check the output/ folder for saved visualizations.")
    print("=" * 60)


def main():
    print_banner()

    # Command-line arguments
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if "all" in args:
            run_demos([1, 2, 3, 4, 5, 6, 7])
        else:
            try:
                selections = sorted(set(int(a) for a in args))
                run_demos(selections)
            except ValueError:
                print("  Usage: python main.py [1-7 | all]")
        return

    # Interactive mode
    while True:
        print_menu()
        choice = input("  Select demo(s) [1-7, all, q]: ").strip().lower()

        if choice == "q":
            print("  Goodbye!")
            break
        elif choice == "all":
            run_demos([1, 2, 3, 4, 5, 6, 7])
        else:
            try:
                nums = [int(c) for c in choice.replace(",", " ").split()]
                valid = [n for n in nums if 1 <= n <= 7]
                if valid:
                    run_demos(sorted(set(valid)))
                else:
                    print("  Please enter numbers 1-7.")
            except ValueError:
                print("  Please enter numbers 1-7, 'all', or 'q'.")


if __name__ == "__main__":
    main()
