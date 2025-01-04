"""
Metric calculation functions
"""

import torch
from ..config import NUM_CLASSES

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
    x_flat = x.view(-1).to(torch.long)
    x_reconstructed_flat = x_reconstructed.view(-1).to(torch.long)

    prediction_table = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    for idx, cls in enumerate(x_flat):
        prediction_table[cls, x_reconstructed_flat[idx]] += 1

    return prediction_table

def compute_accuracy(prediction_table: torch.Tensor) -> float:
    """
    Compute accuracy from prediction table.
    Accuracy = correct predictions / total samples.

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
