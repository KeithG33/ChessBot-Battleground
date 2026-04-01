#!/usr/bin/env python3
"""
Example: Evaluate a ChessBot model on the test dataset.

Reports policy and value metrics: CE loss, MSE, MAE, accuracy, top-5 accuracy.
The model can be loaded from a local weights file (.bin or .safetensors)
or directly from a HuggingFace repo ID.
"""

from chessbot.models import MODEL_REGISTRY
from chessbot.inference.evaluate import evaluate_model


def main():
    model = MODEL_REGISTRY.load_with_weights(
        "swin_chessbot",
        "KeithG33/swin_chessbot",  # or a local path: "/path/to/pytorch_model.bin"
    )

    evaluate_model(
        model,
        batch_size=1024,
        num_workers=4,
        device="cuda",
        save_path=None,  # set to a .json path to save results
    )


if __name__ == "__main__":
    main()
