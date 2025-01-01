"""
Training functions
"""

import torch
from ..config import DEVICE
from ..metrics.metrics import calculate_metrics

def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, optimizer: torch.optim.Optimizer, num_classes: int) -> dict:
    """
    Single epoch training loop.
    Uses decoder output to calculate loss (differentiable), and reconstructed output (decoder output scaled, rounded and clamped) for metrics.

    :param model: VAE model
    :param dataloader: DataLoader object
    :param loss_fn: Loss function, should return reconstruction loss and KL div individually
    :param optimizer: Optimizer object
    :param num_classes: Number of classes/descriptor values
    :return: Dictionary of epoch metrics
    """
    # Training mode
    model.train()

    size = len(dataloader.dataset)
    processed = 0

    # Cumulative totals
    total_recon_loss = 0
    total_kl_div = 0
    total_accuracy = 0
    total_recall = torch.zeros(num_classes).to(DEVICE)  # Total weighted recall for each class/descriptor value
    total_precision = torch.zeros(num_classes).to(DEVICE)  # Total weighted precision for each class/descriptor value
    total_f1 = torch.zeros(num_classes).to(DEVICE)  # Total weighted F1 score for each class/descriptor value
    total_support = torch.zeros(num_classes).to(DEVICE)  # Support for each class/descriptor value

    for batch_idx, (ids, grid_data) in enumerate(dataloader):
        # Move to GPU if available
        x = grid_data.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        x_reconstructed, x_decoder, z, z_mean, z_log_var = model(x)

        # Compute loss
        recon_loss, kl_div = loss_fn(x, x_decoder, z_mean, z_log_var)
        loss = recon_loss + kl_div

        # Backpropagation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Metrics
        total_recon_loss += recon_loss.item()
        total_kl_div += kl_div.item()
        accuracy, recall, precision, f1_score, prediction_table = calculate_metrics(x, x_reconstructed, num_classes)
        total_accuracy += accuracy
        batch_support = torch.sum(prediction_table, dim=1)  # True positives + False negatives
        total_recall += recall * batch_support  # Recall weighted by batch support
        total_precision += precision * batch_support  # Precision weighted by batch support
        total_f1 += f1_score * batch_support  # F1 scores weighted by batch support
        total_support += batch_support

        # Update count
        processed += len(ids)

        # print progress every 5 batches
        if batch_idx % 5 == 0:
            print(f"[{processed:>5d}/{size:>5d}]")
            print(f"Batch metrics (train):")
            print(f"\tLoss = {loss.item():>8.4f}")
            print(f"\tAccuracy = {accuracy * 100:>6.2f}%")
            print(f"\tF1 score (unweighted average across all classes for batch) = {f1_score.mean().item():>6.4f}\n")

    # Averages
    epoch_recon_loss = total_recon_loss / len(dataloader)
    epoch_kl_div = total_kl_div / len(dataloader)
    epoch_accuracy = total_accuracy / len(dataloader)
    epoch_recall_per_class = total_recall / total_support  # average recall score for each class
    epoch_recall_per_class[torch.isnan(epoch_recall_per_class)] = 0  # Replace nan values from division by zero
    epoch_recall_weighted_avg = (total_recall / total_support.sum()).sum().item()  # normalises weighted recall and sums to get weighted average
    epoch_precision_per_class = total_precision / total_support  # average precision score for each class
    epoch_precision_per_class[torch.isnan(epoch_precision_per_class)] = 0  # Replace nan values from division by zero
    epoch_precision_weighted_avg = (total_precision / total_support.sum()).sum().item()  # normalises weighted precision and sums to get weighted average
    epoch_f1_per_class = total_f1 / total_support  # average f1 score for each class
    epoch_f1_per_class[torch.isnan(epoch_f1_per_class)] = 0  # Replace nan values from division by zero
    epoch_f1_weighted_avg = (total_f1 / total_support.sum()).sum().item()  # normalises weighted f1 and sums to get weighted average

    print(f"Train metrics (averages):")
    print(f"\tRecon loss = {epoch_recon_loss:>8.4f}")
    print(f"\tKL div = {epoch_kl_div:>8.4f}")
    print(f"\tAccuracy = {epoch_accuracy * 100:>6.2f}%")
    print(f"\tF1 score (weighted average) = {epoch_f1_weighted_avg:>6.4f}\n")

    return {
        "recon": epoch_recon_loss,
        "kl": epoch_kl_div,
        "accuracy": epoch_accuracy,
        "class_recall": epoch_recall_per_class.tolist(),
        "weighted_recall": epoch_recall_weighted_avg,
        "class_precision": epoch_precision_per_class.tolist(),
        "weighted_precision": epoch_precision_weighted_avg,
        "class_f1": epoch_f1_per_class.tolist(),
        "weighted_f1": epoch_f1_weighted_avg
    }


def validate():
    pass


def train_val(model, dataloader, loss_fn, optimizer, epochs):
    pass
