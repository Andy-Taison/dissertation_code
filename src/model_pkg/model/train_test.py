"""
Training functions
"""

import torch
from pathlib import Path
import time
import math
from ..config import DEVICE, NUM_CLASSES, MODEL_CHECKPOINT_DIR
from ..metrics.metrics import calculate_metrics, get_batch_support
from .history_checkpoint import TrainingHistory, remove_old_improvement_models, extract_epoch_number

def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, optimizer: torch.optim.Optimizer, beta: int = 1) -> dict:
    """
    Single epoch training loop.
    Reconstruction loss is sum of coordinate loss and descriptor loss, averaged across batches.
    Displayed loss, reconstruction loss and KL divergence are scaled for easier interpretability due to weighted recon and averaged KL.

    :param model: VAE model
    :param dataloader: DataLoader object
    :param loss_fn: Loss function, should return reconstruction loss and KL div individually as tensors
    :param optimizer: Optimizer object
    :param beta: KL divergence scaling factor, higher values lead to a more constrained latent space, lower values lead to a more flexible latent space representation, default 1 (standard VAE)
    :return: Dictionary of epoch metrics
    """
    # Training mode
    model.train()

    num_samples = len(dataloader.dataset)  # type: ignore
    processed = 0

    # Cumulative totals
    total_recon_loss = 0
    total_kl_div = 0
    total_accuracy = 0
    total_recall = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted recall for each class/descriptor value
    total_precision = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted precision for each class/descriptor value
    total_f1 = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted F1 score for each class/descriptor value
    total_support = torch.zeros(NUM_CLASSES).to(DEVICE)  # Support for each class/descriptor value
    total_distance = 0  # Total Euclidean distance for coordinate values

    time_to_train = []  # Maintains average time to train from each batch loop for progress statements

    for batch_idx, (ids, grid_data) in enumerate(dataloader):
        start_batch_timer = time.perf_counter()

        # Move to GPU if available
        x = grid_data.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        x_reconstructed, z, z_mean, z_log_var = model(x)

        # Get metrics
        accuracy, recall, precision, f1_score, euclid_dist = calculate_metrics(x, x_reconstructed)
        batch_support = get_batch_support(x, x_reconstructed)  # Based on descriptor values

        # Compute loss
        recon_loss, kl_div = loss_fn(x, x_reconstructed, z_mean, z_log_var)
        loss = recon_loss + beta * kl_div

        # Backpropagation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Update metrics
        total_recon_loss += recon_loss.item()
        total_kl_div += kl_div.item()
        total_accuracy += accuracy
        total_recall += recall * batch_support  # Recall weighted by batch support
        total_precision += precision * batch_support  # Precision weighted by batch support
        total_f1 += f1_score * batch_support  # F1 scores weighted by batch support
        total_support += batch_support
        total_distance += euclid_dist

        # Update count
        processed += len(ids)

        stop_batch_timer = time.perf_counter()
        time_to_train.append(stop_batch_timer - start_batch_timer)

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"Processed [{processed:>5d}/{num_samples:>5d}]")
            if time_to_train:
                average_train_time = sum(time_to_train) / len(time_to_train)
                estimated_completion = time.time() + average_train_time * math.ceil((num_samples - processed)/dataloader.batch_size)
                formatted_estimate = time.strftime('%d-%b-%Y %H:%M:%S', time.localtime(estimated_completion))
                print(f"Estimated train loop completion: {formatted_estimate}")
            print(f"Batch metrics (train):")
            print(f"\tLoss = {loss.item():>8.4f}")
            print(f"\tAccuracy = {accuracy:>6.2f}%")
            print(f"\tF1 score (unweighted average across all classes for batch) = {f1_score.mean().item():>6.4f}")
            print(f"\tAverage coordinate Euclidean distance: {euclid_dist:>6.4f}\n")

    # Averages
    epoch_recon_loss = total_recon_loss / len(dataloader)
    epoch_kl_div = total_kl_div / len(dataloader)
    epoch_accuracy = total_accuracy / len(dataloader)
    epoch_recall_per_class = total_recall / total_support  # average recall score for each class
    epoch_recall_per_class[torch.isnan(epoch_recall_per_class)] = 0  # Replace nan values from division by zero
    epoch_recall_weighted_avg = (total_recall / total_support.sum()).sum().item()  # normalises weighted recall and sums to get weighted average
    epoch_precision_per_class = total_precision / total_support  # Average precision score for each class
    epoch_precision_per_class[torch.isnan(epoch_precision_per_class)] = 0  # Replace nan values from division by zero
    epoch_precision_weighted_avg = (total_precision / total_support.sum()).sum().item()  # Normalises weighted precision and sums to get weighted average
    epoch_f1_per_class = total_f1 / total_support  # Average f1 score for each class
    epoch_f1_per_class[torch.isnan(epoch_f1_per_class)] = 0  # Replace nan values from division by zero
    epoch_f1_weighted_avg = (total_f1 / total_support.sum()).sum().item()  # Normalises weighted f1 and sums to get weighted average
    epoch_euclid_dist = total_distance / len(dataloader)

    print(f"Train metrics (averages):")
    print(f"\tRecon loss = {epoch_recon_loss:>8.4f}")
    print(f"\tKL div = {epoch_kl_div:>8.4f}")
    print(f"\tAccuracy = {epoch_accuracy * 100:>6.2f}%")
    print(f"\tF1 score (weighted average) = {epoch_f1_weighted_avg:>6.4f}")
    print(f"\tCoordinate Euclidean distance: {epoch_euclid_dist:>6.4f}\n")

    return {
        "coor_euclid": epoch_euclid_dist,
        "recon": epoch_recon_loss,
        "kl": epoch_kl_div,
        "beta": beta,
        "accuracy": epoch_accuracy,
        "class_recall": epoch_recall_per_class,
        "weighted_recall": epoch_recall_weighted_avg,
        "class_precision": epoch_precision_per_class,
        "weighted_precision": epoch_precision_weighted_avg,
        "class_f1": epoch_f1_per_class,
        "weighted_f1": epoch_f1_weighted_avg,
        "lr": optimizer.param_groups[0]['lr']
    }


def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, beta: int = 1) -> dict:
    """
    Single epoch test/validation loop.
    Reconstruction loss is sum of coordinate loss and descriptor loss, averaged across batches.
    Displayed loss, reconstruction loss and KL divergence are scaled for easier interpretability due to weighted recon and averaged KL.

    :param model: VAE model
    :param dataloader: DataLoader object
    :param loss_fn: Loss function, should return reconstruction loss and KL div individually as tensors
    :param beta: KL divergence scaling factor, higher values lead to a more constrained latent space, lower values lead to a more flexible latent space representation, default 1 (standard VAE)
    :return: Dictionary of epoch metrics
    """
    # Evaluation mode
    model.eval()

    num_samples = len(dataloader.dataset)  # type: ignore
    processed = 0

    # Cumulative totals
    total_recon_loss = 0
    total_kl_div = 0
    total_accuracy = 0
    total_recall = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted recall for each class/descriptor value
    total_precision = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted precision for each class/descriptor value
    total_f1 = torch.zeros(NUM_CLASSES).to(DEVICE)  # Total weighted F1 score for each class/descriptor value
    total_support = torch.zeros(NUM_CLASSES).to(DEVICE)  # Support for each class/descriptor value
    total_distance = 0  # Total Euclidean distance for coordinate values

    time_to_train = []  # Maintains average time to train from each batch loop for progress statements

    with torch.no_grad():
        for batch_idx, (ids, grid_data) in enumerate(dataloader):
            start_batch_timer = time.perf_counter()

            # Move to GPU if available
            x = grid_data.to(DEVICE)

            # Forward pass
            x_reconstructed, z, z_mean, z_log_var = model(x)

            # Get metrics
            accuracy, recall, precision, f1_score, euclid_dist = calculate_metrics(x, x_reconstructed)
            batch_support = get_batch_support(x, x_reconstructed)  # Based on descriptor values

            # Compute loss
            recon_loss, kl_div = loss_fn(x, x_reconstructed, z_mean, z_log_var)
            loss = recon_loss + beta * kl_div

            # Update metrics
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()
            total_accuracy += accuracy
            total_recall += recall * batch_support  # Recall weighted by batch support
            total_precision += precision * batch_support  # Precision weighted by batch support
            total_f1 += f1_score * batch_support  # F1 scores weighted by batch support
            total_support += batch_support
            total_distance += euclid_dist

            # Update count
            processed += len(ids)

            stop_batch_timer = time.perf_counter()
            time_to_train.append(stop_batch_timer - start_batch_timer)

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Processed [{processed:>5d}/{num_samples:>5d}]")
                if time_to_train:
                    average_train_time = sum(time_to_train) / len(time_to_train)
                    estimated_completion = time.time() + average_train_time * math.ceil((num_samples - processed)/dataloader.batch_size)
                    formatted_estimate = time.strftime('%d-%b-%Y %H:%M:%S', time.localtime(estimated_completion))
                    print(f"Estimated test loop completion: {formatted_estimate}")
                print(f"Batch metrics (test):")
                print(f"\tLoss = {loss.item():>8.4f}")
                print(f"\tAccuracy = {accuracy * 100:>6.2f}%")
                print(f"\tF1 score (unweighted average across all classes for batch) = {f1_score.mean().item():>6.4f}")
                print(f"\tAverage coordinate Euclidean distance: {euclid_dist:>6.4f}\n")

    # Averages
    epoch_recon_loss = total_recon_loss / len(dataloader)
    epoch_kl_div = total_kl_div / len(dataloader)
    epoch_accuracy = total_accuracy / len(dataloader)
    epoch_recall_per_class = total_recall / total_support  # average recall score for each class
    epoch_recall_per_class[torch.isnan(epoch_recall_per_class)] = 0  # Replace nan values from division by zero
    epoch_recall_weighted_avg = (total_recall / total_support.sum()).sum().item()  # normalises weighted recall and sums to get weighted average
    epoch_precision_per_class = total_precision / total_support  # average precision score for each class
    epoch_precision_per_class[torch.isnan(epoch_precision_per_class)] = 0  # Replace nan values from division by zero
    epoch_precision_weighted_avg = (total_precision / total_support.sum()).sum().item()  # Normalises weighted precision and sums to get weighted average
    epoch_f1_per_class = total_f1 / total_support  # Average f1 score for each class
    epoch_f1_per_class[torch.isnan(epoch_f1_per_class)] = 0  # Replace nan values from division by zero
    epoch_f1_weighted_avg = (total_f1 / total_support.sum()).sum().item()  # Normalises weighted f1 and sums to get weighted average
    epoch_euclid_dist = total_distance / len(dataloader)

    print(f"Test metrics (averages):")
    print(f"\tRecon loss = {epoch_recon_loss:>8.4f}")
    print(f"\tKL div = {epoch_kl_div:>8.4f}")
    print(f"\tAccuracy = {epoch_accuracy * 100:>6.2f}%")
    print(f"\tF1 score (weighted average) = {epoch_f1_weighted_avg:>6.4f}")
    print(f"\tCoordinate Euclidean distance: {epoch_euclid_dist:>6.4f}\n")

    return {
        "coor_euclid": epoch_euclid_dist,
        "recon": epoch_recon_loss,
        "kl": epoch_kl_div,
        "beta": beta,
        "accuracy": epoch_accuracy,
        "class_recall": epoch_recall_per_class,
        "weighted_recall": epoch_recall_weighted_avg,
        "class_precision": epoch_precision_per_class,
        "weighted_precision": epoch_precision_weighted_avg,
        "class_f1": epoch_f1_per_class,
        "weighted_f1": epoch_f1_weighted_avg
    }


def train_val(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, loss_fn, optimizer: torch.optim.Optimizer, epochs: int, beta: int = 1, training_history: TrainingHistory = None, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None, prune_old_checkpoints: bool = True) -> TrainingHistory:
    """
    Train and validate dataloader objects should use the same batch size (train dataloader batch size used for history tracking).
    Training continues from last_updated_model if training_history provided.

    Utilises early stopping, see check_and_save_model_improvement in TrainingHistory

    :param model: Model to train
    :param train_dataloader: Training dataloader
    :param val_dataloader: Validation dataloader
    :param loss_fn: Criterion loss function
    :param optimizer: Optimizer
    :param epochs: Maximum number of epochs to run (if early stopping not triggered)
    :param beta: Optional beta to scale KL divergence, defaults 1 (standard VAE)
    :param training_history: Optional, pass to continue training from saved history
    :param scheduler: Scheduler if using (optional), should be a ReduceLROnPlateau scheduler where the learning rate is adjusted based on (reconstruction loss + beta * KL divergence)
    :param prune_old_checkpoints: Removes old checkpoint files (saves memory)
    :return: TrainingHistory tracking object
    """
    print("=" * 50)
    print("Train / Validation loop started...\n")

    if train_dataloader.__getattribute__("batch_size") != val_dataloader.__getattribute__("batch_size"):
        raise ValueError("Train and validate dataloaders do not have the same batch size.")

    # Initialise training history when training from scratch
    if training_history is None:
        training_history = TrainingHistory(model, train_dataloader, optimizer, loss_fn, scheduler)
        print(f"New training history object created '{training_history.model_name}'\n")
    else:
        # Check passed training history matches other objects
        if training_history.model_name != model.name:
            raise ValueError(f"Training history model name: {training_history.model_name} does not match passed model name: {model.name}.")
        if training_history.batch_size != train_dataloader.__getattribute__("batch_size"):
            raise ValueError(f"Training history batch size: {training_history.batch_size} does not match passed dataloader batch size: {train_dataloader.__getattribute__('batch_size')}.")
        if training_history.optim != optimizer.__class__.__name__:
            raise ValueError(f"Training history optimizer: {training_history.optim} does not match passed optimizer: {optimizer.__class__.__name__}.")
        if training_history.loss_fn != loss_fn.loss_name:
            raise ValueError(f"Training history loss function: {training_history.loss_fn} does not match passed loss function: {loss_fn.loss_name}.")
        if training_history.model_architecture != [(name, module) for name, module in model.named_modules()]:
            raise ValueError(f"Training history model architecture does not match passed model architecture.\nTraining history architecture:\n{training_history.model_architecture}")
        if training_history.scheduler is not None:
            if scheduler is None:
                raise ValueError("A scheduler is in training history, but no scheduler was passed.")
            if training_history.scheduler["patience"] != scheduler.patience:  # type: ignore
                raise ValueError(
                    f"Training history scheduler patience: {training_history.scheduler['patience']} does not match passed scheduler patience: {scheduler.patience}."  # type: ignore
                )
            if training_history.scheduler["factor"] != scheduler.factor:  # type: ignore
                raise ValueError(
                    f"Training history scheduler factor: {training_history.scheduler['factor']} does not match passed scheduler factor: {scheduler.factor}."  # type: ignore
                )
        elif scheduler is not None:
            raise ValueError("A scheduler was passed, but none is recorded in training history.")

        print(f"Continuing training, epochs run so far: {training_history.epochs_run}\n")

    time_to_train = []  # Maintains average time to train from each epoch loop for progress statements

    for epoch_idx in range(training_history.epochs_run, epochs):
        epoch = epoch_idx + 1
        print(f"Epoch {epoch:>3d}/{epochs:>3d}: '{model.name}'")
        if time_to_train:
            average_train_time = sum(time_to_train) / len(time_to_train)
            estimated_completion = time.time() + average_train_time * (epochs - epoch_idx)
            formatted_estimate = time.strftime('%d-%b-%Y %H:%M:%S', time.localtime(estimated_completion))
            print(f"Estimated training/validation cycle completion: {formatted_estimate}")
        start_epoch_timer = time.perf_counter()
        print("-" * 50)

        # Train with timer
        start_timer = time.perf_counter()
        train_metrics = train(model, train_dataloader, loss_fn, optimizer, beta)
        stop_timer = time.perf_counter()
        train_metrics['training_time'] = stop_timer - start_timer

        # Validate with timer
        start_timer = time.perf_counter()
        val_metrics = test(model, val_dataloader, loss_fn, beta)
        stop_timer = time.perf_counter()
        val_metrics['training_time'] = stop_timer - start_timer

        terminate = training_history.check_and_save_model_improvement(val_metrics, epoch, model, optimizer, scheduler)
        training_history.update_epoch(train_metrics, "train", False)
        training_history.update_epoch(val_metrics, "val")
        training_history.save_history()

        # Remove old checkpoint files if new files were added
        if prune_old_checkpoints and training_history.epochs_without_improvement == 0:
            print("Pruning old checkpoints...")
            remove_old_improvement_models(Path(MODEL_CHECKPOINT_DIR) / training_history.model_name, extract_epoch_number(training_history.bests['best_f1_avg_model']), extract_epoch_number(training_history.bests['best_loss_model']))

        if terminate:
            print("Early stop terminating...")
            break

        if scheduler:
            scheduler.step(val_metrics['recon'] + val_metrics['beta'] * val_metrics['kl'])

        stop_epoch_timer = time.perf_counter()
        time_to_train.append(stop_epoch_timer - start_epoch_timer)

    print("Train / Validation loop complete!")
    print("=" * 50 + "\n")

    return training_history
