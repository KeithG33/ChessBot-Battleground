import json
import os
from typing import Dict, List
from omegaconf import OmegaConf
import torch
import typer

from chessbot.data.download import DEFAULT_DATASET_DIR, download as download_fn
from chessbot.inference.evaluate import evaluate_model as evaluate_fn
from chessbot.train.trainer import train_fn

from webapp.app import play as play_fn


app = typer.Typer(help="ChessBot CLI Tool")


def parse_kwargs(kwargs_str: str) -> Dict:
    try:
        return json.loads(kwargs_str)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON for kwargs: {e}")
    

def find_and_load_from_register(model_name, model_dir, model_args=None, model_kwargs="{}"):
    extra_kwargs = parse_kwargs(model_kwargs)

    if model_args is None:
        model_args = []

    try:
        from chessbot.models.registry import ModelRegistry
        # Pass both positional and keyword arguments to load_model
        model = ModelRegistry.load_model_from_directory(model_name, model_dir, *model_args, **extra_kwargs)
        typer.echo("Model loaded successfully.")
    except Exception as e:
        typer.echo(f"Error loading model: {e}")
        raise typer.Exit(code=1)
    return model


@app.command()
def evaluate(
    model_name: str = typer.Argument(..., help="Name of the model to load"),
    model_dir: str = typer.Option(..., "--model-dir", help="Directory with model definitions"),
    model_weights: str = typer.Option(None, "--model-weights", "-w", help="Path to model weights"),
    data_dir: str = typer.Option(DEFAULT_DATASET_DIR, "--data-dir", help="Directory containing dataset"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size for evaluation"),
    num_threads: int = typer.Option(4, "--num-threads", help="Number of threads to use"),
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
):
    """
    Evaluate a model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.
    """
    model = find_and_load_from_register(model_name, model_dir, model_args, model_kwargs)
    if model_weights:
        model.load_state_dict(torch.load(model_weights))

    # Evaluate the model (replace with your actual evaluation logic)
    results = evaluate_fn(model, data_dir, batch_size=batch_size, num_threads=num_threads)
    typer.echo(f"Evaluation results: {results}")


@app.command()
def download(
    tag: str = typer.Argument(None, help="Tag of the GitHub release (default: latest)"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Path where the dataset should be saved"),
    dataset_name: str = typer.Option(None, "--dataset-name", "-d", help="Custom dataset filename (default: ChessBot-Dataset-{tag}.zip)")
):
    """
    Download a dataset from a GitHub release.
    """
    download_fn(tag, output_dir, dataset_name)


@app.command()
def play(
    model_name: str = typer.Argument(..., help="Name of the model to load"),
    model_dir: str = typer.Option(None, "--model-dir", help="Directory with model definitions"),
    model_weights: str = typer.Option(None, "--model-weights", "-w", help="Path to model weights file"),
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
):
    """
    Play a game against the bot using a loaded model. Pass additional positional arguments with --model-arg and keyword arguments as a JSON string via --model-kwargs.
    """
    model = find_and_load_from_register(model_name, model_dir, model_args, model_kwargs)
    
    if model_weights:
        model.load_state_dict(torch.load(model_weights))

    play_fn(model)


@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to the configuration YAML file"),
    override: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Override any config variable using dot notation, e.g., training.lr=0.001. This option can be used multiple times."
    )
):
    """
    Train a model using the provided configuration file and optional overrides.
    """
    train_fn(config_path, override)



if __name__ == "__main__":
    app()
