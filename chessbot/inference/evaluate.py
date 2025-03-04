import os
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chessbot.data import ChessDataset
from chessbot.common import setup_logger
from chessbot.train.utils import MetricsTracker


LOGGER = setup_logger('chessbot.evaluate')


def calculate_data_chunks(num_chunks, all_files):
    """ Handles chunking the data into smaller parts for evaluation.

    Returns the chunk size and the starting index of each chunk.
    """
    if num_chunks is None or num_chunks == 0:
        num_chunks = 1  # Ensure there's at least one chunk
    elif num_chunks > len(all_files):
        num_chunks = len(all_files)

    chunk_size = len(all_files) // num_chunks
    remainder = len(all_files) % num_chunks
    chunk_starts = [i * chunk_size + min(i, remainder) for i in range(num_chunks)]
    return chunk_size, chunk_starts


def evaluate_model(
    model,
    dataset_dir,
    batch_size: int,
    num_threads: int,
    device: str = "cuda",
    num_chunks: int = None,
):
    """Evaluate the model
     
    For flexibility, the evaluation can be done in chunks (of files). There are 100 equally sized
    files in the test set, loading all is slightly over 32Gb of memory. Chunk if necessary.
    
    The main performance metrics are:
      - CE loss,
      - mse loss,
      - mae loss,
      - accuracy,
      - top5 accuracy,
      - top10 accuracy,

    Also logs additional model statistics:
      - batch size
      - inference time per batch (average)
      - total model parameters
    """
    test_data = os.path.join(dataset_dir, 'test')
    all_files = [f.path for f in os.scandir(test_data) if f.name.endswith(".pgn")]

    LOGGER.info(f"Found {len(all_files)} PGN files in the test directory.")

    chunk_size, chunk_starts = calculate_data_chunks(num_chunks, all_files)

    tracker = MetricsTracker()
    tracker.add(
        "policy_loss",
        "mse_loss",
        "mae_loss",
        "accuracy",
        "top5_accuracy",
        "top10_accuracy",
        "inference_time",
    )

    num_model_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model Parameters: {num_model_params}")
    LOGGER.info(f"Batch Size: {batch_size}")

    model.eval()
    model = model.to(device)

    t_eval = time.perf_counter()
    pbar = tqdm(total=0, desc="Evaluating")
    for i, start in enumerate(chunk_starts):
        chunk_files = all_files[start : start + chunk_size]
        dataset = ChessDataset(chunk_files, num_threads=num_threads)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        pbar.set_description(f"Chunk {i+1}/{len(chunk_starts)}")
        pbar.total = len(dataloader)
        pbar.refresh()

        with torch.inference_mode():
            for state, action, result in dataloader:
                state = state.float().to(device)
                action = action.to(device)
                action_inds = action.argmax(dim=1)
                result = result.float().to(device)

                start_time = time.perf_counter()
                policy_out, value_out = model(state.unsqueeze(1))
                end_time = time.perf_counter()
                inference_time = end_time - start_time

                # Losses
                policy_loss = F.cross_entropy(policy_out, action)
                value_loss_l2 = F.mse_loss(value_out.squeeze(), result)
                value_loss_l1 = F.l1_loss(value_out.squeeze(), result)

                # Accuracy, Top-5, Top-10
                pred_top5 = policy_out.topk(5, dim=1)[1]
                pred_top10 = policy_out.topk(10, dim=1)[1]
                accuracy = (policy_out.argmax(dim=1) == action_inds).float().mean().item()
                top5_accuracy = (pred_top5 == action_inds.unsqueeze(1)).any(dim=1).float().mean().item()
                top10_accuracy = (pred_top10 == action_inds.unsqueeze(1)).any(dim=1).float().mean().item()

                tracker.update("policy_loss", policy_loss.item())
                tracker.update("mse_loss", value_loss_l2.item())
                tracker.update("mae_loss", value_loss_l1.item())
                tracker.update("accuracy", accuracy)
                tracker.update("top5_accuracy", top5_accuracy)
                tracker.update("top10_accuracy", top10_accuracy)
                tracker.update("inference_time", inference_time)

                pbar.update(1)

    LOGGER.info(f"Finished evaluation in {time.perf_counter() - t_eval:.2f} seconds.")

    pbar.close()
    averages = tracker.get_all_averages()

    LOGGER.info("\nClassification (Policy) Metrics:")
    LOGGER.info(f"  Top-1 Accuracy: {averages['accuracy']:.4f}")
    LOGGER.info(f"  Top-5 Accuracy: {averages['top5_accuracy']:.4f}")
    LOGGER.info(f"  Top-10 Accuracy: {averages['top10_accuracy']:.4f}")
    LOGGER.info(f"  Cross-Entropy Loss: {averages['policy_loss']:.4f}")

    LOGGER.info("\nRegression (Value) Metrics:")
    LOGGER.info(f"  MSE: {averages['mse_loss']:.4f}")
    LOGGER.info(f"  MAE: {averages['mae_loss']:.4f}")

    LOGGER.info("\nAdditional Model Statistics:")
    LOGGER.info(f"  Batch Size: {batch_size}")
    LOGGER.info(f"  Average Inference Time (per batch): {averages['inference_time']:.6f} seconds")
    LOGGER.info(f"  Total Model Parameters: {num_model_params}")

    return {
        "policy_loss": averages["policy_loss"],
        "value_loss": averages["mse_loss"],
        "accuracy": averages["accuracy"],
        "top5_accuracy": averages["top5_accuracy"],
        "top10_accuracy": averages["top10_accuracy"],
        "mae": averages["mae_loss"],
        "batch_size": batch_size,
        "avg_inference_time": averages["inference_time"],
        "num_model_params": num_model_params,
    }

