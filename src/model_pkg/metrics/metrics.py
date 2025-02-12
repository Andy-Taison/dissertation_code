"""
Metric calculation functions
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from datetime import datetime
from ..model.history_checkpoint import TrainingHistory, extract_epoch_number, load_model_checkpoint
from ..visualisation.latent import analyse_latent_space
from ..config import NUM_CLASSES, DEVICE, OUTPUTS_DIR

"""
Coordinate metrics
"""
def euclidean_distance(x: torch.Tensor, x_reconstructed: torch.Tensor) -> float:
    """
    Calculates the average Euclidean distance between original input coordinates and reconstructed coordinates.
    Note this does NOT apply the transformation matrix.

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


def log_metrics(history: TrainingHistory, train_dataloader: DataLoader, val_dataloader: DataLoader, k: int, log: str = 'loss', filename: str = "metrics_table"):
    """
    Logs validation metrics to csv file sorted by total loss.
    Epoch number must be extractable from best model stored.
    If model checkpoint can be loaded, it is used to obtain latent space metrics which are included in the csv.

    :param history: History object to add best metrics to file
    :param train_dataloader: Used to normalise the sampled latent vector, and train PCA and UMAP
    :param val_dataloader: Used to sample latent vector, perform PCA and UMAP reduction and obtain latent space metrics via kmeans
    :param k: K used for kmeans clustering for latent space metrics
    :param log: Best performing 'loss' or 'f1' epoch to obtain and log
    :param filename: Filename, if exists, metrics will be appended before sorting
    """
    match log.lower():
        case 'loss':
            epoch = extract_epoch_number(history.bests['best_loss_model'])
        case 'f1':
            epoch = extract_epoch_number(history.bests['best_f1_avg_model'])
        case _:
            raise ValueError("'log' must be either 'loss' or 'f1'.")

    if epoch is None:
        raise ValueError(f"Best '{log.lower()}' model stored is None.")

    idx = epoch - 1

    new_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'name': history.alt_history_filename if history.alt_history_filename is not None else history.model_name,
        'batch_size': history.batch_size,
        'latent_dim': history.latent_dim,
        'lambda_coord': history.criterion.lambda_coord,
        'lambda_desc': history.criterion.lambda_desc,
        'lambda_pad': history.criterion.lambda_pad,
        'lambda_collapse': history.criterion.lambda_collapse,
        'beta': history.val['beta'][idx],
        'recon_loss': history.val['recon'][idx],
        'kl': history.val['kl'][idx],
        'scaled_kl': history.val['kl'][idx] * history.val['beta'][idx],
        'total_loss': history.val['recon'][idx] + (history.val['kl'][idx] * history.val['beta'][idx]),
        'scaled_coord': history.val['scaled_coor_loss'][idx],
        'scaled_desc': history.val['scaled_desc_loss'][idx],
        'scaled_pad': history.val['scaled_pad_penalty'][idx],
        'scaled_col': history.val['scaled_collapse_penalty'][idx]
    }

    try:
        # Load corresponding model
        loaded_model, _, _, _ = load_model_checkpoint(history, log.lower())

        latent_metrics = analyse_latent_space(loaded_model, train_dataloader, val_dataloader, k)
        new_data.update(latent_metrics)
    except FileNotFoundError as e:
        print(e)

    # Convert to DataFrame
    new_row_df = pd.DataFrame([new_data])

    filepath = Path(OUTPUTS_DIR) / f"{filename}.csv"
    if not filepath.parent.exists():
        print(f"\nCreating directory '{filepath.parent}'...")
        filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check if CSV exists
    if filepath.is_file():
        # Read CSV
        df = pd.read_csv(filepath)
        # Append new row
        df = pd.concat([df, new_row_df], ignore_index=True)
        print("\nData appended to existing file. Sorting and saving...")
    else:
        df = new_row_df
        print("\nFile not found. Saving to new file...")

    # Sort the DataFrame by 'total_loss' in ascending order
    df = df.sort_values(by='total_loss', ascending=True).reset_index(drop=True)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved to '{filepath.name}'")
