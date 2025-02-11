import os
import sys
import argparse
import tempfile
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import subprocess

from chessbot.data import ChessDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def download_dataset(temp_dir):
    """Download the dataset into a temporary directory."""
    logging.info(f"Downloading dataset to temporary directory: {temp_dir}")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "chessbot.data.download",
                "download",
                "--output-dir",
                temp_dir,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download dataset: {e}")
        sys.exit(1)


def ensure_dataset_exists(test_dir):
    """Ensure the dataset is available. If `test_dir` is None, download it to a temp directory."""
    temp_download = False
    
    # Download if needed
    if test_dir is None:
        test_dir = tempfile.mkdtemp
        download_dataset(test_dir)
        temp_download = True

    # Check if the directory contains PGN files
    elif not os.path.exists(test_dir) or not any(
        f.endswith(".pgn") for f in os.listdir(test_dir)
    ):
        logging.error(f"Test dataset not found at {test_dir}. Exiting")
        sys.exit(1)

    return test_dir, temp_download  # Return both the path and flag


def evaluate_model(
    model, batch_size: int, num_threads: int, test_dir=None, device="cuda"
):
    """Evaluate the model on the test dataset."""
    test_dir, temp_download = ensure_dataset_exists(test_dir)

    test_data = [pgn.path for pgn in os.scandir(test_dir) if pgn.name.endswith(".pgn")]

    # Create dataset and dataloader
    dataset = ChessDataset(test_data, num_threads=num_threads)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Loaded validation dataset with {len(dataset)} positions.")

    model.eval()

    total_policy_loss = 0
    total_mse_loss = 0
    total_mae_loss = 0
    total_acc = 0
    total_top5_acc = 0
    num_samples = 0
    total_r2 = 0

    with torch.inference_mode():
        for i, (state, action, result) in enumerate(dataloader):
            state = state.float().to(device)
            action = action.to(device)
            result = result.float().to(device)

            # Forward pass
            policy_output, value_output = model(state.unsqueeze(1))

            # Compute losses
            policy_loss = F.cross_entropy(policy_output, action, reduction="mean")
            value_loss_l2 = F.mse_loss(value_output.squeeze(), result, reduction="mean")
            value_loss_l1 = F.l1_loss(value_output.squeeze(), result, reduction="mean")
            total_policy_loss += policy_loss.item()
            total_mse_loss += value_loss_l2.item()
            total_mae_loss += value_loss_l1.item()

            total_acc += (policy_output.argmax(dim=1) == action.argmax(dim=1)).float().mean().item()

            total_top5_acc += (
                (policy_output.topk(5, dim=1)[1] == action.argmax(dim=1, keepdim=True))
                .any(dim=1)
                .float()
                .mean()
                .item()
            )
            total_r2 += r2_score(result.cpu().numpy(), value_output.squeeze().cpu().numpy())

            print(f"Batch {i+1}/{len(dataloader)}: Policy Loss: {policy_loss.item()}, Value Loss: {value_loss_l2.item()}") 

    policy_loss = total_policy_loss / len(dataloader)
    mse = total_mse_loss / len(dataloader)
    mae = total_mae_loss / len(dataloader)
    r2 = total_r2 / len(dataloader)
    accuracy = total_acc / len(dataloader)
    top5_accuracy = total_top5_acc / len(dataloader)

    # Print results
    logging.info("\nClassification (Policy) Metrics:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
    logging.info(f"  Cross-Entropy Loss: {policy_loss:.4f}")

    logging.info("\nRegression (Value) Metrics:")
    logging.info(f"  MSE: {mse:.4f}")
    logging.info(f"  MAE: {mae:.4f}")
    logging.info(f"  R² Score: {r2:.4f}")

    # Cleanup temp directory if it was used
    if temp_download:
        logging.info(f"Cleaning up temporary dataset directory: {test_dir}")
        shutil.rmtree(test_dir)

    return {
        "policy_loss": policy_loss,
        "value_loss": mse,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "mae": mae,
        "r2": r2,
    }


# def main():
#     """Command-line interface for evaluating a model."""
#     parser = argparse.ArgumentParser(description="Evaluate a trained model on the ChessBot test dataset.")
#     parser.add_argument("--test-dir", type=str, default=None, help="Directory where test PGN files are stored. If not provided, dataset will be downloaded.")
#     parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
#     parser.add_argument("--num-threads", type=int, default=4, help="Number of threads for dataset loading.")
#     parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (default: cuda).")

#     args = parser.parse_args()

#     # Load model (Assumes a function `load_model` exists)
#     model = load_model()  # You need to define this function elsewhere

#     # Evaluate model
#     eva
