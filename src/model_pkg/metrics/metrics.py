"""
Metric calculation functions
"""

import torch
from ..config import NUM_CLASSES, DEVICE

"""
Coordinate metrics
"""
def euclidean_distance(x: torch.Tensor, x_reconstructed: torch.Tensor) -> float:
    """
    Calculates the average Euclidean distance between original input coordinates and reconstructed coordinates.

    :param x: Input tensor with shape (batch_size, *input_dim)
    :param x_reconstructed: Reconstructed input with shape (batch_size, *input_dim)
    :return: Mean Euclidean distance of reconstructed coordinates to original input coordinates
    """
    distance = torch.norm(x[:, :, :3] - x_reconstructed[:, :, :3], dim=-1).mean().item()

    return distance


"""
Descriptor value metrics
"""
def descriptor_metrics(x: torch.Tensor, x_reconstructed: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates metrics for descriptor values.

    :param x: Input tensor
    :param x_reconstructed: Reconstructed tensor, scaled to have original descriptor values
    :return: accuracy, recall, precision, f1
    """
    preds = torch.argmax(x_reconstructed[:, :, 3:], dim=-1)
    targets = torch.argmax(x[:, :, 3:], dim=-1)

    tp = (preds == targets) & (targets != 0)
    fp = (preds != targets) & (preds != 0)
    fn = (preds != targets) & (targets != 0)

    correct = (preds == targets).sum().item()
    total = preds.numel()
    accuracy = correct / total

    recall = tp.sum().item() / (tp.sum().item() + fn.sum())
    recall[torch.isnan(recall)] = 0  # Replace nan values from division by zero

    precision = tp.sum().item() / (tp.sum().item() + fp.sum())
    precision[torch.isnan(precision)] = 0  # Replace nan values from division by zero

    f1 = 2 * (precision * recall) / (precision + recall)
    f1[torch.isnan(f1)] = 0  # Replace nan values from division by zero

    return accuracy, recall, precision, f1


def get_batch_support(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Obtains batch support based on descriptor values.

    :param x: Input tensor
    :param x_reconstructed: Reconstructed tensor, scaled to have original descriptor values
    :return:
    """
    preds = torch.argmax(x_reconstructed[:, :, 3:], dim=-1)
    targets = torch.argmax(x[:, :, 3:], dim=-1)

    support = torch.zeros(NUM_CLASSES).to(DEVICE)

    for cls in range(NUM_CLASSES):
        support[cls] = ((targets == cls).sum() + (preds == cls).sum()).item()

    return support


def calculate_metrics(x: torch.Tensor, x_reconstructed: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Director function to calculate accuracy, recall, precision, and F1 score metrics.

    :param x: Input tensor
    :param x_reconstructed: Reconstructed tensor, scaled to have original descriptor values
    :return: accuracy, recall, precision, f1_score, euclidean_distance
    """
    euclid = euclidean_distance(x, x_reconstructed)
    accuracy, recall, precision, f1_score = descriptor_metrics(x, x_reconstructed)

    return accuracy, recall, precision, f1_score, euclid


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
