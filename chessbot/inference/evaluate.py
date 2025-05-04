import os
import sys
import time
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chessbot.data import HFChessDataset
from chessbot.common import setup_logger
from chessbot.train.utils import MetricsTracker


def evaluate_model(
    model,
    batch_size: int,
    num_workers: int,
    device: str = "cuda",
    save_path: str = None,
):
    """Evaluate the model
    
    The main performance metrics are:
      - CE loss,
      - mse loss,
      - mae loss,
      - accuracy,
      - top5 accuracy,
      - top10 accuracy,

    Also logs additional model statistics:
      - batch size
      - inference time per batch and per sample (average)
      - total model parameters
    """
    
    LOGGER = setup_logger('chessbot.evaluate')
    LOGGER.info(f"Starting Evaluation - workers {num_workers}, batch size {batch_size}")
    
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

    model.eval()
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model Parameters: {num_params}")

    dataset = HFChessDataset('test')
    num_examples = dataset.ds.info.splits['test'].num_examples
    total_batches = (num_examples + batch_size - 1) // batch_size  # Calculate total batches
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pbar = tqdm(total=total_batches, desc="Evaluating", leave=False)

    t_eval = time.perf_counter()
    with torch.inference_mode():
        for state, action, result in test_loader:
            state, action, result = state.to(device), action.to(device), result.to(device)
            action_inds = action.argmax(dim=1)

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
            pbar.set_postfix(
                policy_loss=policy_loss.item(),
                mse_loss=value_loss_l2.item(),
                mae_loss=value_loss_l1.item(),
                accuracy=accuracy,
                top5_accuracy=top5_accuracy,
                top10_accuracy=top10_accuracy,
                inference_time=inference_time,
            )

    LOGGER.info(f"Finished evaluation in {time.perf_counter() - t_eval:.2f} seconds.")

    averages = tracker.get_all_averages()
    avg_inference_time_per_sample = averages["inference_time"] / batch_size

    results = {
        "policy_loss": averages["policy_loss"],
        "value_loss": averages["mse_loss"],
        "mae": averages["mae_loss"],
        "accuracy": averages["accuracy"],
        "top5_accuracy": averages["top5_accuracy"],
        "top10_accuracy": averages["top10_accuracy"],
        "batch_size": batch_size,
        "avg_inference_time_per_batch": averages["inference_time"],
        "avg_inference_time_per_sample": avg_inference_time_per_sample,
        "num_model_params": num_params,
    }

    # Pretty console print
    print("\n=== Evaluation Summary ===")
    print("Classification (Policy):")
    print(f"  Top-1 Accuracy     : {results['accuracy']:.4f}")
    print(f"  Top-5 Accuracy     : {results['top5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy    : {results['top10_accuracy']:.4f}")
    print(f"  Cross-Entropy Loss : {results['policy_loss']:.4f}")
    print("\nRegression (Value):")
    print(f"  MSE Loss           : {results['value_loss']:.4f}")
    print(f"  MAE Loss           : {results['mae']:.4f}")
    print("\nPerformance:")
    print(f"  Batch Size         : {batch_size}")
    print(f"  Inference Time/Batch : {results['avg_inference_time_per_batch']:.4f} sec")
    print(f"  Inference Time/Sample: {results['avg_inference_time_per_sample']:.6f} sec")
    print(f"  Model Parameters   : {num_params}")
    print("==========================\n")

    # Optional save to JSON
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "evaluation_results.json")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Saved evaluation results to {save_path}")

    return results
