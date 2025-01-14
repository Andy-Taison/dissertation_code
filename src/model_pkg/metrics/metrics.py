"""
Metric calculation functions
"""

import torch
from ..config import NUM_CLASSES, DEVICE

def compute_prediction_table(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction table where each row corresponds to a descriptor value (class) and each column represents the predicted (reconstructed) value.
    Values represent counts.
    Along the diagonal is correct predictions.

    :param x: Input tensor
    :param x_reconstructed: Reconstructed tensor with same shape as input tensor
    :return: Prediction table
    """
    # Flatten tensors
    x_flat = x.view(-1).to(torch.long)  # Long for indexing
    x_reconstructed_flat = x_reconstructed.view(-1).to(torch.long)  # Long for indexing
    
    prediction_table = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64).to(DEVICE)
    
    for idx, cls in enumerate(x_flat):
        prediction_table[cls, x_reconstructed_flat[idx]] += 1

    return prediction_table


def compute_class_weights(batch_support: torch.Tensor) -> torch.Tensor:
    """
    Calculates class weights using batch support, used to account for class imbalance (descriptor values are sparse).

    :param batch_support: Batch support 1D tensor
    :return: Class weights tensor
    """
    # Inverses batch support to obtain weights
    weights = 1.0 / batch_support
    weights[torch.isinf(weights)] = 1  # Replaces inf with 1

    # Normalises
    weights /= weights.sum()

    return weights


def compute_accuracy(prediction_table: torch.Tensor) -> float:
    """
    Compute accuracy from prediction table.
    Accuracy = correct predictions / total samples.
    Returned value will be between 0 (no correct predictions) and 1 (all predictions correct).

    :param prediction_table: Prediction table tensor where rows represent classes (descriptor values), and columns represent predictions (reconstructed)
    :return: Accuracy as float
    """
    correct = torch.sum(torch.diag(prediction_table))
    total = torch.sum(prediction_table)

    return (correct / total).item()


def compute_recall(prediction_table: torch.Tensor) -> torch.Tensor:
    """
    Compute recall for each class from prediction table.
    Recall = true positives / (true positives + false negatives).
    Returned value will be between 0 (no actual positives were correctly predicted) and 1 (all actual positives were correctly predicted).

    :param prediction_table: Prediction table tensor where rows represent classes (descriptor values), and columns represent predictions (reconstructed)
    :return: Recall as float tensor
    """
    tp = torch.diag(prediction_table)
    tp_fn = torch.sum(prediction_table, dim=1)

    recall = tp / tp_fn
    recall[torch.isnan(recall)] = 0  # Replace nan values from division by zero

    return recall


def compute_precision(prediction_table: torch.Tensor) -> torch.Tensor:
    """
    Compute precision for each class from prediction table.
    Precision = true positives / (true positives + false positives).
    Returned value will be between 0 (no true positive predictions) and 1 (all positive predictions were correct).

    :param prediction_table: Prediction table tensor where rows represent classes (descriptor values), and columns represent predictions (reconstructed)
    :return: Precision as float tensor
    """
    tp = torch.diag(prediction_table)
    tp_fp = torch.sum(prediction_table, dim=0)

    precision = tp / tp_fp
    precision[torch.isnan(precision)] = 0  # Replace nan values from division by zero

    return precision


def compute_f1_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    """
    Compute F1 score from precision and recall tensors.
    Returned value will be between 0 (precision and/or recall are zero (poor performance)) and 1 (perfect precision and recall).

    :param precision: Precision tensor
    :param recall: Recall tensor
    :return: F1 per class (descriptor value) float tensor
    """
    f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score[torch.isnan(f1_score)] = 0  # Replace nan values from division by zero

    return f1_score


def calculate_metrics(x: torch.Tensor, x_reconstructed: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main function to calculate accuracy, recall, precision, F1 score, and prediction table metrics.

    :param x: Input tensor
    :param x_reconstructed: Reconstructed tensor, scaled to have original descriptor values
    :return: accuracy, recall, precision, f1_score, prediction_table
    """
    prediction_table = compute_prediction_table(x, x_reconstructed)
    accuracy = compute_accuracy(prediction_table)
    recall = compute_recall(prediction_table)
    precision = compute_precision(prediction_table)
    f1_score = compute_f1_score(precision, recall)

    return accuracy, recall, precision, f1_score, prediction_table


def get_best_tradeoff_score(recon: list, kl: list, beta: list, f1_weighted_avg: list, loss_f1_tradeoff: float = 0.7) -> tuple[int, float]:
    """
    Finds best tradeoff score (loss_f1_tradeoff x loss + (1 - loss_f1_tradeoff) x (1 - best_f1_avg)) and the epoch it occurred in.

    :param recon: List of epoch reconstruction losses
    :param kl: List of epoch KL divergence values
    :param beta: List of epoch betas used
    :param f1_weighted_avg: List of epoch F1 weighted average scores
    :param loss_f1_tradeoff: Tradeoff value to use, higher puts emphasis on loss, lower emphasises F1
    :return: Best epoch, Best tradeoff score
    """
    # Convert lists to tensors
    recon_tensor = torch.tensor(recon)
    kl_tensor = torch.tensor(kl)
    beta_tensor = torch.tensor(beta)
    f1_tensor = torch.tensor(f1_weighted_avg)

    # Calculates loss for each epoch
    loss = recon_tensor + beta_tensor * kl_tensor

    # Inverts F1 score for combination with loss
    score = loss_f1_tradeoff * loss + (1 - loss_f1_tradeoff) * (1 - f1_tensor)
    best_epoch = torch.argmin(score).item() + 1
    best_score = torch.min(score).item()

    return best_epoch, best_score
