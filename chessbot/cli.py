import json
import os
from typing import Dict, List
import torch
import typer

from chessbot.data.download import download as download_fn
from chessbot.inference.evaluate import evaluate_model
from chessbot.inference import selfplay as selfplay_fn, run_match as run_match_fn
from chessbot.train.trainer import train_fn_hf, train_fn_local
from chessbot.common import DEFAULT_DATASET_DIR, DEFAULT_MODEL_DIR
from chessbot.models import align_state_dict
from chessbot.models.base import BaseChessBot
from huggingface_hub import hf_hub_download

from chessbot.app import play as play_fn

app = typer.Typer(help="ChessBot CLI Tool")


def parse_kwargs(kwargs_str: str) -> Dict:
    try:
        return json.loads(kwargs_str)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON for kwargs: {e}")


def find_and_load_from_register(
    model_name, model_dir, model_args=None, model_kwargs="{}"
):
    extra_kwargs = parse_kwargs(model_kwargs)

    if model_args is None:
        model_args = []

    try:
        from chessbot.models.registry import ModelRegistry

        model = ModelRegistry.load_model(
            model_name, model_dir, *model_args, **extra_kwargs
        )
        typer.echo("Model loaded successfully.")
    except Exception as e:
        typer.echo(f"Error loading model: {e}")
        raise typer.Exit(code=1)
    return model


def load_weights(model: BaseChessBot, weights_id: str, hf_filename: str = "pytorch_model.bin") -> None:
    """Load weights from a local path or HuggingFace repo."""
    if os.path.exists(weights_id):
        weights = align_state_dict(torch.load(weights_id))
        model.load_state_dict(weights)
        return

    try:
        path = hf_hub_download(repo_id=weights_id, filename=hf_filename)
    except Exception as e:
        raise typer.BadParameter(f"Could not download weights from {weights_id}: {e}")

    weights = align_state_dict(torch.load(path, weights_only=True))
    model.load_state_dict(weights)


@app.command()
def evaluate(
    model_name: str = typer.Argument(..., help="Name of the model to load"),
    model_dir: str = typer.Option(
        None, "--model-dir", help="Directory with model definitions"
    ),
    model_weights: str = typer.Option(
        None, "--model-weights", "-w", help="Path to model weights"
    ),
    hf_filename: str = typer.Option(
        "pytorch_model.bin",
        "--model-filename",
        "-f",
        help="Filename of the model weights to load (default: pytorch_model.bin)",
    ),
    model_args: List[str] = typer.Option(
        None,
        "--model-arg",
        "-a",
        help="Additional positional arguments for the model's constructor. This option can be used multiple times.",
    ),
    model_kwargs: str = typer.Option(
        "{}",
        "--model-kwargs",
        "-k",
        help="JSON string of extra keyword arguments for the model's constructor",
    ),
    batch_size: int = typer.Option(
        1024, "--batch-sz", help="Batch size for evaluation"
    ),
    num_workers: int = typer.Option(
        1, "--num-workers", help="Number of workers to use when dataloading"
    ),
    save_json: str = typer.Option(
        None, "--save-json", "-j", help="Path to save evaluation results as JSON"
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to use for evaluation"
    )
):
    """
    Evaluate a model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.
    """
    model = find_and_load_from_register(model_name, model_dir, model_args, model_kwargs)
    if model_weights:
        load_weights(model, model_weights, hf_filename)

    results = evaluate_model(
        model,
        batch_size=batch_size,
        num_workers=num_workers,
        save_path=save_json,
        device=device
    )
    typer.echo(f"Evaluation results: {results}")


@app.command()
def download(
    output_dir: str = typer.Option(
        None, "--output-dir", "-o", help="Path where the dataset should be saved"
    ),
    keep_raw_data: bool = typer.Option(
        False, "--keep-raw-data", "-k", help="Download the original raw pgn data"
    ),
):
    """
    Download the dataset.
    """
    download_fn(output_dir, keep_raw_data)


@app.command()
def play(
    model_name: str = typer.Argument(..., help="Name of the model to load"),
    model_dir: str = typer.Option(
        None, "--model-dir", help="Directory with model definitions"
    ),
    model_weights: str = typer.Option(
        None, "--model-weights", "-w", help="Path to model weights file"
    ),
    hf_filename: str = typer.Option(
        "pytorch_model.bin",
        "--model-filename",
        "-f",
        help="Filename of the model weights to load (default: pytorch_model.bin)",
    ),
    model_args: List[str] = typer.Option(
        None,
        "--model-arg",
        "-a",
        help="Additional positional arguments for the model's constructor. This option can be used multiple times.",
    ),
    model_kwargs: str = typer.Option(
        "{}",
        "--model-kwargs",
        "-k",
        help="JSON string of extra keyword arguments for the model's constructor",
    ),
    port: int = typer.Option(
        5000, "--port", "-p", help="Port to run the game server on"
    ),
):
    """
    Play a game against the bot using a loaded model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.
    """
    model = find_and_load_from_register(model_name, model_dir, model_args, model_kwargs)

    if model_weights:
        load_weights(model, model_weights, hf_filename)

    play_fn(model, port)


@app.command()
def selfplay(
    model_name: str = typer.Argument(..., help="Name of the model to load"),
    model_dir: str = typer.Option(None, "--model-dir", help="Directory with model definitions"),
    model_weights: str = typer.Option(None, "--model-weights", "-w", help="Path to model weights"),
    hf_filename: str = typer.Option(
        "pytorch_model.bin",
        "--model-filename",
        "-f",
        help="Filename of the model weights to load (default: pytorch_model.bin)",
    ),
    model_args: List[str] = typer.Option(None, "--model-arg", "-a", help="Additional positional arguments for the model's constructor. This option can be used multiple times."),
    model_kwargs: str = typer.Option("{}", "--model-kwargs", "-k", help="JSON string of extra keyword arguments for the model's constructor"),
    search: bool = typer.Option(False, "--search", "-s", help="Use MCTS search"),
    num_sims: int = typer.Option(250, "--num-sims", help="Number of MCTS simulations"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Visualize the game"),
    sample: bool = typer.Option(False, "--sample", help="Sample from the policy distribution"),
):
    """Run a selfplay game with a model."""
    model = find_and_load_from_register(model_name, model_dir, model_args, model_kwargs)
    if model_weights:
        load_weights(model, model_weights, hf_filename)
    outcome = selfplay_fn(model, search=search, num_sims=num_sims, visualize=visualize, sample=sample)
    typer.echo(f"Selfplay outcome (white perspective): {outcome}")


@app.command(name="run-match")
def run_match(
    player1_name: str = typer.Argument(..., help="Model name for player 1"),
    player2_name: str = typer.Argument(..., help="Model name for player 2"),
    player1_dir: str = typer.Option(None, "--player1-dir", help="Directory with player1 model"),
    player2_dir: str = typer.Option(None, "--player2-dir", help="Directory with player2 model"),
    player1_weights: str = typer.Option(None, "--player1-weights", help="Weights path or HF repo for player1"),
    player2_weights: str = typer.Option(None, "--player2-weights", help="Weights path or HF repo for player2"),
    hf_filename: str = typer.Option(
        "pytorch_model.bin",
        "--model-filename",
        "-f",
        help="Filename of the model weights to load (default: pytorch_model.bin)",
    ),
    player1_args: List[str] = typer.Option(None, "--player1-arg", help="Positional args for player1 model", show_default=False),
    player1_kwargs: str = typer.Option("{}", "--player1-kwargs", help="JSON kwargs for player1 model"),
    player2_args: List[str] = typer.Option(None, "--player2-arg", help="Positional args for player2 model", show_default=False),
    player2_kwargs: str = typer.Option("{}", "--player2-kwargs", help="JSON kwargs for player2 model"),
    best_of: int = typer.Option(1, "--best-of", "-b", help="Number of games"),
    search: bool = typer.Option(False, "--search", "-s", help="Use MCTS search"),
    num_sims: int = typer.Option(250, "--num-sims", help="Number of MCTS simulations"),
    visualize: bool = typer.Option(True, "--visualize", "-v", help="Visualize games"),
    sample: bool = typer.Option(False, "--sample", help="Sample from the policy distribution"),
):
    """Run a match between two models."""
    player1 = find_and_load_from_register(player1_name, player1_dir, player1_args, player1_kwargs)
    player2 = find_and_load_from_register(player2_name, player2_dir, player2_args, player2_kwargs)
    if player1_weights:
        load_weights(player1, player1_weights, hf_filename)
    if player2_weights:
        load_weights(player2, player2_weights, hf_filename)
    score1, score2 = run_match_fn(
        player1,
        player2,
        best_of=best_of,
        search=search,
        num_sims=num_sims,
        visualize=visualize,
        sample=sample,
    )
    typer.echo(f"Final score: {score1} - {score2}")



@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to the configuration YAML file"),
    override: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Override any config variable using dot notation, e.g., training.lr=0.001. This option can be used multiple times.",
    ),
    local_training: bool = typer.Option(
        False,
        "--local-training",
        "-l",
        help="Use local training from pgn files instead of huggingface streaming the dataset",
    ),
):
    """
    Train a model using the provided configuration file and optional overrides.
    """
    if local_training:
        train_fn_local(config_path, override)
        return

    train_fn_hf(config_path, override)


if __name__ == "__main__":
    app()
