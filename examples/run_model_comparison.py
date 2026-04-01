#!/usr/bin/env python3
"""
Example: Run a match between two ChessBot models.

Models can be loaded from a local weights file (.bin or .safetensors)
or directly from a HuggingFace repo ID.
"""

from chessbot.models import MODEL_REGISTRY
from chessbot.inference import run_match


def main():
    # Load models — swap in any model name, local path, or HF repo ID
    model_a = MODEL_REGISTRY.load_with_weights(
        "swin_chessbot",
        "/path/to/model_a.safetensors",  # or a HF repo: "KeithG33/swin_chessbot"
    )

    model_b = MODEL_REGISTRY.load_with_weights(
        "swin_chessbot",
        "KeithG33/swin_chessbot",
    )

    print("Starting match: Model A vs Model B")
    print("=" * 50)

    score_a, score_b = run_match(
        model_a,
        model_b,
        best_of=11,
        search=True,      # use MCTS for stronger play
        num_sims=500,
        visualize=False,
        sample=False,    # when MCTS is off, sample moves from model output distribution
    )

    total = score_a + score_b
    print("=" * 50)
    print(f"Model A: {score_a}  |  Model B: {score_b}")
    if total > 0:
        print(f"Model A win rate: {score_a / total * 100:.1f}%")


if __name__ == "__main__":
    main()
