from chessbot.inference import evaluate_model
from chessbot.common import DEFAULT_MODEL_DIR

from simple_chessbot import SimpleChessBot

dataset_dir = DEFAULT_MODEL_DIR
model = SimpleChessBot(hidden_dim=512)
evaluate_model(
    model,
    dataset_dir=DEFAULT_MODEL_DIR,
    batch_size=64,
    num_processes=4,
    device="cuda",
    num_chunks=None,
)