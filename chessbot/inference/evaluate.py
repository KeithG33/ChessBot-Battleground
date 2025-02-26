
import sys
import os
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chessbot.data import ChessDataset
from chessbot.common import setup_logger


_logger = setup_logger('chessbot.evaluate')


# TODO: double check this
def mean_reciprocal_rank(logits, targets):
    """Compute the Mean Reciprocal Rank (MRR) for ranked predictions."""
    sorted_preds = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_preds == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # 1-based index
    return (1.0 / ranks.float()).mean().item()


import sys
import os
import logging
import time  # Added for timing
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chessbot.data import ChessDataset
from chessbot.common import setup_logger


_logger = setup_logger('chessbot.evaluate')


# TODO: double check this
def mean_reciprocal_rank(logits, targets):
    """Compute the Mean Reciprocal Rank (MRR) for ranked predictions."""
    sorted_preds = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_preds == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # 1-based index
    return (1.0 / ranks.float()).mean().item()


def evaluate_model(
    model, dataset_dir, batch_size: int, num_threads: int, device="cuda", num_chunks=None
):
    """Evaluate the model with optional chunking for low-memory environments.
    
    Also logs additional model statistics:
      - Batch size
      - Average inference time per batch (forward pass)
      - Total number of model parameters
    """
    
    test_data = os.path.join(dataset_dir, 'test')
    all_files = [f.path for f in os.scandir(test_data) if f.name.endswith(".pgn")]
    
    _logger.info(f"Found {len(all_files)} PGN files in the test directory.")

    if num_chunks is None or num_chunks == 0:
        num_chunks = 1  # Ensure there's at least one chunk
    elif num_chunks > len(all_files):
        num_chunks = len(all_files)

    chunk_size = len(all_files) // num_chunks
    remainder = len(all_files) % num_chunks
    chunk_starts = [i * chunk_size + min(i, remainder) for i in range(num_chunks)]

    # Track total metrics
    total_policy_loss = 0
    total_mse_loss = 0
    total_mae_loss = 0
    total_acc = 0
    total_top5_acc = 0
    total_top10_acc = 0
    total_mrr = 0
    total_inference_time = 0.0  # For timing forward passes
    num_batches = 0

    # Compute total number of model parameters
    num_model_params = sum(p.numel() for p in model.parameters())
    _logger.info(f"Model Parameters: {num_model_params}")
    _logger.info(f"Batch Size: {batch_size}")

    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        for i, start in enumerate(chunk_starts):
            chunk_files = all_files[start:start + chunk_size]
            
            dataset = ChessDataset(chunk_files, num_threads=num_threads)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            _logger.info(f"Processing chunk {i+1} / {len(chunk_starts)}")

            for state, action, result in tqdm(dataloader, desc="Evaluating", leave=False):
                state = state.float().to(device)
                action = action.to(device)
                result = result.float().to(device)

                # Time the forward pass
                start_time = time.perf_counter()
                policy_output, value_output = model(state.unsqueeze(1))
                end_time = time.perf_counter()
                total_inference_time += (end_time - start_time)

                policy_loss = F.cross_entropy(policy_output, action)
                value_loss_l2 = F.mse_loss(value_output.squeeze(), result)
                value_loss_l1 = F.l1_loss(value_output.squeeze(), result)

                total_policy_loss += policy_loss.item()
                total_mse_loss += value_loss_l2.item()
                total_mae_loss += value_loss_l1.item()

                # Accuracy, Top-5, and Top-10
                action_indices = action.argmax(dim=1) 
                pred_top5 = policy_output.topk(5, dim=1)[1]
                pred_top10 = policy_output.topk(10, dim=1)[1]

                total_acc += (policy_output.argmax(dim=1) == action_indices).float().mean().item()
                total_top5_acc += (pred_top5 == action_indices.unsqueeze(1)).any(dim=1).float().mean().item()
                total_top10_acc += (pred_top10 == action_indices.unsqueeze(1)).any(dim=1).float().mean().item()

                # Mean Reciprocal Rank (MRR)
                total_mrr += mean_reciprocal_rank(policy_output, action_indices)

                num_batches += 1

    # Compute averages
    policy_loss = total_policy_loss / num_batches
    mse = total_mse_loss / num_batches
    mae = total_mae_loss / num_batches
    accuracy = total_acc / num_batches
    top5_accuracy = total_top5_acc / num_batches
    top10_accuracy = total_top10_acc / num_batches
    mrr = total_mrr / num_batches
    avg_inference_time = total_inference_time / num_batches

    # Print results
    _logger.info("\nClassification (Policy) Metrics:")
    _logger.info(f"  Top-1 Accuracy: {accuracy:.4f}")
    _logger.info(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
    _logger.info(f"  Top-10 Accuracy: {top10_accuracy:.4f}")
    _logger.info(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    _logger.info(f"  Cross-Entropy Loss: {policy_loss:.4f}")

    _logger.info("\nRegression (Value) Metrics:")
    _logger.info(f"  MSE: {mse:.4f}")
    _logger.info(f"  MAE: {mae:.4f}")

    _logger.info("\nAdditional Model Statistics:")
    _logger.info(f"  Batch Size: {batch_size}")
    _logger.info(f"  Average Inference Time (per batch): {avg_inference_time:.6f} seconds")
    _logger.info(f"  Total Model Parameters: {num_model_params}")

    return {
        "policy_loss": policy_loss,
        "value_loss": mse,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "mrr": mrr,
        "mae": mae,
        "batch_size": batch_size,
        "avg_inference_time": avg_inference_time,
        "num_model_params": num_model_params,
    }
