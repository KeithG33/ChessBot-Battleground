
from chessbot.data import ChessDataset
import os
import torch
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def mean_reciprocal_rank(logits, targets):
    """Compute the Mean Reciprocal Rank (MRR) for ranked predictions."""
    sorted_preds = torch.argsort(logits, dim=1, descending=True)
    ranks = (sorted_preds == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # 1-based index
    return (1.0 / ranks.float()).mean().item()


def evaluate_model(
    model, dataset_dir, batch_size: int, num_threads: int, device="cuda"
):
    """Evaluate the model on the test dataset."""

    # Create dataset and dataloader
    test_data = os.path.join(dataset_dir, 'test')
    dataset = ChessDataset(test_data, num_threads=num_threads)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded validation dataset with {len(dataset)} positions.")

    model.eval()

    total_policy_loss = 0
    total_mse_loss = 0
    total_mae_loss = 0
    total_acc = 0
    total_top5_acc = 0
    total_top10_acc = 0
    total_mrr = 0
    num_batches = len(dataloader)

    with torch.inference_mode():
        for state, action, result in tqdm(dataloader, desc="Evaluating", leave=False):
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

            # Compute Accuracy, Top-5, and Top-10
            action_indices = action.argmax(dim=1)  # Get true class indices
            pred_top5 = policy_output.topk(5, dim=1)[1]
            pred_top10 = policy_output.topk(10, dim=1)[1]

            total_acc += (policy_output.argmax(dim=1) == action_indices).float().mean().item()
            total_top5_acc += (pred_top5 == action_indices.unsqueeze(1)).any(dim=1).float().mean().item()
            total_top10_acc += (pred_top10 == action_indices.unsqueeze(1)).any(dim=1).float().mean().item()

            # Compute Mean Reciprocal Rank (MRR)
            total_mrr += mean_reciprocal_rank(policy_output, action_indices)

    # Compute averages
    policy_loss = total_policy_loss / num_batches
    mse = total_mse_loss / num_batches
    mae = total_mae_loss / num_batches
    accuracy = total_acc / num_batches
    top5_accuracy = total_top5_acc / num_batches
    top10_accuracy = total_top10_acc / num_batches
    mrr = total_mrr / num_batches

    # Print results
    print("\nClassification (Policy) Metrics:")
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"  Top-10 Accuracy: {top10_accuracy:.4f}")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  Cross-Entropy Loss: {policy_loss:.4f}")

    print("\nRegression (Value) Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")

    return {
        "policy_loss": policy_loss,
        "value_loss": mse,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "mrr": mrr,
        "mae": mae,
    }
