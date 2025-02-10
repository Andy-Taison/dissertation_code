"""
Includes checkpointing functions to save models, optimizers and schedulers.
TrainingHistory class used to track and save training history. It also calls checkpoint_model function.
"""
import pathlib

import torch
from pathlib import Path, PosixPath, WindowsPath  # For patching torch.load when saved on one machine and loaded on a different
import platform  # For patching torch.load when saved on one machine and loaded on a different
import re
from ..config import DEVICE, NUM_CLASSES, MODEL_CHECKPOINT_DIR, HISTORY_DIR, PATIENCE, EPOCHS, BASE_DIR
from ..metrics.losses import VaeLoss
from . import model as model_module  # Used for reconstructing model in load_model_checkpoint

def avg_and_format_time(times: list[float]) -> str:
    """
    Calculates average time from a list of times in seconds.
    Formats average time into a string in the format 'hours:minutes:seconds'.

    :param times: List of times in seconds
    :return: Formatted string total time
    """
    avg_time = sum(times) / len(times)
    hours = round(avg_time / 3600)
    minutes = round((avg_time % 3600) / 60)
    seconds = round(avg_time % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def remove_old_improvement_models(model_dir: Path | str, f1_epoch: int = -1, loss_epoch: int = -1):
    """
    Deletes old model files from 'model_dir' that start with 'best_f1_avg' or 'best_loss'.
    Keeps the latest valid epoch file along with 'final_model' and 'terminated_model' files.
    When specifying an epoch, will look for valid files to keep up to and including the epoch specified.
    If no valid files when specifying epoch, all will be removed.
    The first epoch starts at 1.

    :param model_dir: Model directory to delete files from
    :param f1_epoch: best_f1_avg checkpoint epoch number to keep (keeps only the latest that exists up to that epoch), use -1 (default) to maintain latest
    :param loss_epoch: best_loss checkpoint epoch number to keep (keeps only the latest that exists up to that epoch), use -1 (default) to maintain latest
    """
    model_dir = Path(model_dir)
    available_files = list(model_dir.glob("*.pth"))

    # Filter keeping files starting with 'best'
    pattern = re.compile(r"^best")
    epoch_files = [(extract_epoch_number(file.name), file) for file in available_files if pattern.search(str(file.name))]

    # Separate into groups, files where epoch number cannot be extracted are not deleted
    f1_files = [epoch_file for epoch_file in epoch_files if "best_f1_avg" in epoch_file[1].name and epoch_file[0] is not None]
    loss_files = [epoch_file for epoch_file in epoch_files if "best_loss" in epoch_file[1].name and epoch_file[0] is not None]

    # Get files suitable to keep
    if f1_epoch > 0:
        valid_f1 = [epoch_file for epoch_file in f1_files if epoch_file[0] <= f1_epoch]
    else:
        valid_f1 = f1_files
    # Removes latest epoch file to keep from deletion list
    if valid_f1:
        keep_f1 = max(valid_f1, key=lambda x: x[0])
        f1_files.remove(keep_f1)

    if loss_epoch > 0:
        valid_loss = [epoch_file for epoch_file in loss_files if epoch_file[0] <= loss_epoch]
    else:
        valid_loss = loss_files
    # Remove latest epoch file to keep from deletion list
    if valid_loss:
        keep_loss = max(valid_loss, key=lambda x: x[0])
        loss_files.remove(keep_loss)

    remove = f1_files + loss_files

    # Delete files
    for file in remove:
        file[1].unlink()

    print("Old improvement files pruned.\n")

def checkpoint_model(filepath: Path | str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None):
    """
    Function to save/checkpoint a PyTorch model, optimizer, and optionally scheduler states along with number of epochs run.
    Creates directory if it does not exist.

    Checkpoint is created such that when loading, initialised objects need not be passed.
    Assumes scheduler (if used) is ReduceLROnPlateau.

    :param filepath: Filepath as either Path object or string
    :param model: Model to save state
    :param optimizer: Optimizer to save state
    :param epoch: Number of epochs run
    :param scheduler: Optional scheduler to save state, should be ReduceLROnPlateau
    """
    print("Checkpointing model and optimizer...")
    checkpoint = {
        'model_state_dict': model.state_dict(),  # Parameters
        'model_config': model.__dict__,  # Attributes
        'model_class': model.__class__.__name__,  # Class name

        'optimizer_state_dict': optimizer.state_dict(),  # Parameters
        'optimizer_class': optimizer.__class__.__name__,  # Class name
        'optimizer_args': optimizer.defaults,  # Initialisation arguments

        'epoch': epoch,
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()  # Parameters
        checkpoint['scheduler_config'] = scheduler.__dict__  # Attributes
        print("Saving scheduler.")

    checkpoint_path = Path(filepath)
    if not checkpoint_path.parent.exists():
        print(f"Creating directory '{checkpoint_path.parent}'...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to '{Path(*checkpoint_path.parts[-3:])}'\n")


def extract_epoch_number(filename: str | Path) -> int | None:
    """
    Extracts the epoch number from a filename.
    Filename must have the epoch number immediately prior to the extension '.pth' to be extracted, else None will be returned.

    :param filename: Filename to search
    :return: Extracted epoch number
    """
    filename = str(filename)
    pattern = re.compile(r"_(\d+)\.pth$")
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    else:
        return None


class TrainingHistory:
    """
    A class to track training and validation history during training.
    Handles saving and loading history to/from file.
    Method check_and_save_model_improvement checks for improvements, handles early stopping condition and calls checkpointing functions to save models.
    check_and_save_model_improvement should be called prior to updating epoch history.
    update_epoch should be called prior to saving history.
    """
    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: VaeLoss, scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None):
        """
        Initialises TrainingHistory object for tracking metrics.
        Use TrainingHistory.load() to load and initialise saved objects.

        :param model: Model to track history for, must have attribute 'name'
        :param dataloader: Used to obtain batch size
        :param optimizer: Optimizer name tracked
        :param criterion: Loss function, should have attribute 'loss_name'
        :param scheduler: Scheduler if using (optional), should be a ReduceLROnPlateau scheduler where the learning rate is adjusted based on (reconstruction loss + beta * KL divergence)
        """
        self.model_name = model.name
        self.alt_history_filename = None  # Used to save TrainingHistory to file under different name to model_name

        self.epochs_run = 0
        self.epochs_without_improvement = 0
        self.last_updated_model = None
        self.last_improved_model = None

        self.batch_size = dataloader.batch_size
        self.optim = optimizer.__class__.__name__
        self.weight_decay = optimizer.param_groups[0]['weight_decay']
        self.loss_fn = criterion.loss_name
        self.scheduler = {
            "patience": scheduler.patience,
            "factor": scheduler.factor
        } if scheduler is not None else None
        self.latent_dim = model.latent_dim
        self.model_architecture = [(name, module) for name, module in model.named_modules()]

        self.bests = {
            'best_loss_model': None,
            'best_loss': None,
            'best_f1_avg_model': None,
            'best_f1_avg': None
        }

        self.train = {
            'coor_euclid': [],  # Average coordinate Euclidean distance per epoch
            'recon': [],  # Average reconstruction loss per epoch
            'kl': [],  # Average KL divergence per epoch
            'accuracy': [],  # Average accuracy per epoch
            'recall_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'recall_weighted_avg': [],  # Average recall per epoch, weighted by descriptor value support
            'precision_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'precision_weighted_avg': [],  # Average precision per epoch, weighted by descriptor value support
            'f1_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'f1_weighted_avg': [],  # Average F1 score per epoch, weighted by descriptor value support
            'beta': [],  # Beta applied to KL divergence per epoch
            'lr': [],  # Learning rate used by optimizer per epoch
            'training_time': [],  # Training loop time in seconds
            'desc_loss': [],  # Average descriptor loss per epoch
            'coor_loss': [],  # Average coordinate loss per epoch
            'dup_pad_avg': [],  # Average duplicate and padded penalty per epoch
            'transform_reg_avg': []  # Average transformation regularising term per epoch
        }

        # No learning rate required for validation due to not using optimizer
        self.val = {
            'coor_euclid': [],  # Average coordinate Euclidean distance per epoch
            'recon': [],  # Average reconstruction loss per epoch
            'kl': [],  # Average KL divergence per epoch
            'accuracy': [],  # Average accuracy per epoch
            'recall_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'recall_weighted_avg': [],  # Average recall per epoch, weighted by descriptor value support
            'precision_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'precision_weighted_avg': [],  # Average precision per epoch, weighted by descriptor value support
            'f1_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'f1_weighted_avg': [],  # Average F1 score per epoch, weighted by descriptor value support
            'beta': [],  # Beta applied to KL divergence per epoch
            'training_time': [],  # Validation loop time in seconds
            'desc_loss': [],  # Average descriptor loss per epoch
            'coor_loss': [],  # Average coordinate loss per epoch
            'dup_pad_avg': [],  # Average duplicate and padded penalty per epoch
            'transform_reg_avg': []  # Average transformation regularising term per epoch
        }

    def check_and_save_model_improvement(self, val_epoch_metrics: dict, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler = None) -> bool:
        """
        Should be used with validation metrics only, not training metrics.
        Checks for improvement of total loss or weighted F1 average.
        If improvement metrics changed, also adjust rollback method.
        Model is checkpointed in either condition along with first epoch, early stopping and final epoch.
        Note TrainingHistory status is not saved here, neither is history updated, however metric bests are updated.
        This method should be carried out prior to updating epoch history for value comparisons.
        Models are saved using MODEL_CHECKPOINT_DIR from config along with model name and epoch.
        Epoch number should be immediately prior to the extension for rollback and extract_epoch_number methods.

        Early stopping terminate condition True returned when epoch reaches config PATIENCE with no improvement.

        :param val_epoch_metrics: Validation metrics dictionary
        :param epoch: Current epoch
        :param model: Model for checkpointing
        :param optimizer: Optimizer for checkpointing
        :param scheduler: Scheduler if using (optional) for checkpointing
        :return: True when patience epoch reached with no improvement, or final epoch reached
        """
        improved = False
        epoch_loss = val_epoch_metrics['recon'] + val_epoch_metrics['beta'] * val_epoch_metrics['kl']

        # Loss
        if self.last_updated_model is None or epoch_loss < self.bests['best_loss']:
            filepath = Path(MODEL_CHECKPOINT_DIR) / self.model_name / f"best_loss_epoch_{epoch}.pth"
            checkpoint_model(filepath, model, optimizer, epoch, scheduler)
            self.bests['best_loss_model'] = filepath
            self.bests['best_loss'] = epoch_loss
            self.last_improved_model = filepath
            self.last_updated_model = filepath
            self.epochs_without_improvement = 0
            improved = True

        # Weighted F1 average
        if self.bests['best_f1_avg'] is None or val_epoch_metrics['weighted_f1'] > self.bests['best_f1_avg']:
            filepath = Path(MODEL_CHECKPOINT_DIR) / self.model_name / f"best_f1_avg_epoch_{epoch}.pth"
            checkpoint_model(filepath, model, optimizer, epoch, scheduler)
            self.bests['best_f1_avg_model'] = filepath
            self.bests['best_f1_avg'] = val_epoch_metrics['weighted_f1']
            self.last_improved_model = filepath
            self.last_updated_model = filepath
            self.epochs_without_improvement = 0
            improved = True

        # Early stopping counter
        if not improved:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= PATIENCE:
            # Checkpoint early stop model
            filepath = Path(MODEL_CHECKPOINT_DIR) / self.model_name / f"terminated_model_epoch_{epoch}.pth"
            checkpoint_model(filepath, model, optimizer, epoch, scheduler)
            self.last_updated_model = filepath

            return True  # Terminate training condition
        elif epoch >= EPOCHS:
            # Checkpoint final model
            filepath = Path(MODEL_CHECKPOINT_DIR) / self.model_name / f"final_model_epoch_{epoch}.pth"
            checkpoint_model(filepath, model, optimizer, epoch, scheduler)
            self.last_updated_model = filepath

            return False
        else:
            return False

    def update_epoch(self, epoch_metrics: dict, mode: str, increment_epochs_run: bool = True):
        """
        Update history with epoch metrics.
        Method does not update bests.
        check_and_save_model_improvement should be called prior to calling this method to check for improvements and update bests.

        :param epoch_metrics: Dictionary containing metrics
        :param mode: 'train' or 'val' history to update
        :param increment_epochs_run: Boolean to update epochs run (method may be called to update train and validate metrics)
        """
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        history = self.train if mode == 'train' else self.val

        if increment_epochs_run:
            self.epochs_run += 1
        history['coor_euclid'].append(epoch_metrics['coor_euclid'])
        history['recon'].append(epoch_metrics['recon'])
        history['kl'].append(epoch_metrics['kl'])
        history['accuracy'].append(epoch_metrics['accuracy'])
        history['recall_classes'] = torch.cat([history['recall_classes'], epoch_metrics['class_recall'].unsqueeze(0)])
        history['recall_weighted_avg'].append(epoch_metrics['weighted_recall'])
        history['precision_classes'] = torch.cat([history['precision_classes'], epoch_metrics['class_precision'].unsqueeze(0)])
        history['precision_weighted_avg'].append(epoch_metrics['weighted_precision'])
        history['f1_classes'] = torch.cat([history['f1_classes'], epoch_metrics['class_f1'].unsqueeze(0)])
        history['f1_weighted_avg'].append(epoch_metrics['weighted_f1'])
        history['beta'].append(epoch_metrics['beta'])
        history['training_time'].append(epoch_metrics['training_time'])
        history['desc_loss'].append(epoch_metrics['desc_loss'])
        history['coor_loss'].append(epoch_metrics['coor_loss'])
        history['dup_pad_avg'].append(epoch_metrics['dup_pad_avg'])
        history['transform_reg_avg'].append(epoch_metrics['transform_reg_avg'])
        if 'lr' in history:
            history['lr'].append(epoch_metrics['lr'])
        if 'patience' in history:
            history['patience'].append(PATIENCE)

    def rollback(self, reset_to: int | str | Path):
        """
        Inplace function.
        Rolls back TrainingHistory to a specified state.
        Cannot forward history.
        Last improved model is updated based on validation loss and validation weighted F1 average.

        Does not save history. Call save_history method to save rolled back TrainingHistory.
        When saving, this may overwrite previous TrainingHistory files if not changing filename when saving.

        :param reset_to: An epoch number to reset to the end of that epoch. Alternatively a string/Path referring to a filename or a stored attribute ['last_improved_model', 'best_loss_model', 'best_f1_avg_model']
        """
        model_dir = Path(MODEL_CHECKPOINT_DIR) / self.model_name
        epoch = None

        if isinstance(reset_to, int):
            # Reset to epoch number
            if reset_to >= self.epochs_run:
                raise ValueError(f"Not enough epochs run ({self.epochs_run}) to reset to epoch {reset_to}.")
            elif reset_to <= 0:
                raise ValueError(f"When resetting to an epoch number, 'reset_to' must be greater than 0 and less than epochs run: {self.epochs_run}.")
            epoch = reset_to
            print(f"Resetting history to epoch {epoch}...")
        elif isinstance(reset_to, str):
            # Reset by attribute name
            if reset_to == "last_improved_model":
                if self.last_improved_model is None:
                    raise ValueError(f"'{reset_to}' is None and cannot be used to rollback.")
                epoch = extract_epoch_number(self.last_improved_model.name)
                print(f"Resetting history to last_improved_model: {self.last_improved_model.name}...")
            elif reset_to in self.bests:
                attr_value = self.bests[reset_to]
                if attr_value is None:
                    raise ValueError(f"'{reset_to}' is None and cannot be used to rollback.")
                epoch = extract_epoch_number(attr_value.name)
                print(f"Resetting history to {reset_to}: {attr_value.name}...")
            else:
                # reset_to assumed to be a Path
                reset_to = Path(reset_to)

        if isinstance(reset_to, Path):
            # Reset by filename or absolute path
            if not reset_to.is_absolute():
                reset_to = model_dir / reset_to
            if not reset_to.exists():
                raise FileNotFoundError(f"Checkpoint file '{reset_to}' not found.\nPlease enter either '<filename>.pth' or an absolute path.")
            epoch = extract_epoch_number(reset_to.name)
            if epoch is None:
                raise ValueError(f"Cannot extract epoch number from '{reset_to.name}', check filename format.")
            if epoch >= self.epochs_run:
                raise ValueError(f"Not enough epochs run ({self.epochs_run}) to reset to epoch {reset_to}.")
            print(f"Resetting history based on provided filename: {reset_to}...")

        # Reset epochs_run
        if epoch:
            self.epochs_run = epoch
        else:
            # Avoids potential damage to history
            raise ValueError("'reset_to' must be an int, str, or Path.")

        # Rollback train/val dictionaries
        for key in self.train.keys():
            self.train[key] = self.train[key][:epoch]
        for key in self.val.keys():
            self.val[key] = self.val[key][:epoch]

        # Find and update bests
        val_recon = torch.tensor(self.val['recon'], device=DEVICE)
        beta = torch.tensor(self.val['beta'], device=DEVICE)
        val_kl = torch.tensor(self.val['kl'], device=DEVICE)

        loss = val_recon + beta * val_kl
        best_loss_epoch = torch.argmin(loss).item() + 1

        weighted_f1 = torch.tensor(self.val['f1_weighted_avg'], device=DEVICE)
        best_weighted_f1_epoch = torch.argmax(weighted_f1).item() + 1

        best_loss_model_path = model_dir / f"best_loss_epoch_{best_loss_epoch}.pth"
        if best_loss_model_path.exists():
            self.bests['best_loss_model'] = best_loss_model_path
        else:
            print(f"\nBest loss model checkpoint for epoch {best_loss_epoch} does not exist.")
            self.bests['best_loss_model'] = None
        self.bests['best_loss'] = loss[best_loss_epoch - 1].item()  # type: ignore

        best_f1_model_path = model_dir / f"best_f1_avg_epoch_{best_weighted_f1_epoch}.pth"
        if best_f1_model_path.exists():
            self.bests['best_f1_avg_model'] = best_f1_model_path
        else:
            print(f"\nBest weighted F1 average model checkpoint for epoch {best_weighted_f1_epoch} does not exist.")
            self.bests['best_f1_avg_model'] = None
        self.bests['best_f1_avg'] = weighted_f1[best_weighted_f1_epoch - 1].item()  # type: ignore

        # Last improved
        if best_weighted_f1_epoch >= best_loss_epoch:
            self.last_improved_model = model_dir / f"best_f1_avg_epoch_{best_weighted_f1_epoch}.pth"
        else:
            self.last_improved_model = model_dir / f"best_loss_epoch_{best_loss_epoch}.pth"

        last_improved_epoch = extract_epoch_number(self.last_improved_model.name)
        self.epochs_without_improvement = epoch - last_improved_epoch
        if not self.last_improved_model.exists():
            print(f"\nLast improved model checkpoint file '{self.last_improved_model}' does not exist.")
            self.last_improved_model = None

        # Last updated and check that checkpoint file exists for epoch
        available_files = list(model_dir.glob("*.pth"))
        epoch_files = [(extract_epoch_number(file.name), file) for file in available_files]
        valid_files = [(file_epoch, file) for file_epoch, file in epoch_files if file_epoch <= epoch]
        if not valid_files:
            print(f"\nNo valid checkpoint file found in '{model_dir}' for epoch <= {epoch}.")
            print("Loading checkpoint will not be possible.")
            self.last_updated_model = None
        else:
            self.last_updated_model = max(valid_files, key=lambda x: x[0])[1]

        print(f"\nTrainingHistory rolled back to epoch {self.epochs_run}. Updated attributes:")
        print(f"\t- Last updated model: '{self.last_updated_model.name if self.last_updated_model else 'None'}'")
        print(f"\t- Last improved model: '{self.last_improved_model.name if self.last_improved_model else 'None'}' Epoch: {last_improved_epoch}")
        print(f"\t- Best loss: {self.bests['best_loss']:.4f}, Model: '{self.bests['best_loss_model'].name if self.bests['best_loss_model'] else 'None'}' Epoch: {best_loss_epoch}")  # type: ignore
        print(f"\t- Best F1 average: {self.bests['best_f1_avg']:.4f}, Model: '{self.bests['best_f1_avg_model'].name if self.bests['best_f1_avg_model'] else 'None'}' Epoch: {best_weighted_f1_epoch}\n")  # type: ignore

    def save_history(self, change_filename_to: str = None):
        """
        Save training history to file using PyTorch's save functionality.
        History is saved under '<alt_history_filename>.pth' if set, or '<model_name>_history.pth'.
        Saved in 'HISTORY_DIR' as set in config.

        :param change_filename_to: Change and save under alternative filename (provide without extension)
        """
        if change_filename_to is not None:
            self.alt_history_filename = change_filename_to
            print(f"Alternative history filename updated: '{change_filename_to}'")

        if self.alt_history_filename is not None:
            filepath = Path(HISTORY_DIR) / f"{self.alt_history_filename}.pth"
            print(f"Saving to alternative history filename: '{filepath.name}'...")
        else:
            filepath = Path(HISTORY_DIR) / f"{self.model_name}_history.pth"
            print(f"Saving history to: '{filepath.name}'...")

        # Check directories exist
        if not filepath.parent.exists():
            print(f"Creating directory '{filepath.parent}'...")
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save the state of the object
        torch.save(self.__dict__, filepath)
        print(f"History saved to '{Path(*filepath.parts[-2:])}'\n")

    @classmethod
    def load_history(cls, file: str):
        """
        Class method to instantiate training history from file.
        Allows history to be loaded and inspected prior to loading checkpoint (model, optimizer, scheduler).

        :param file: Filename to be loaded within HISTORY_DIR as specified in config file
        :return: TrainingHistory populated from loaded file
        """
        filepath = Path(HISTORY_DIR) / file

        # Check directories exist
        if not filepath.exists():
            raise FileNotFoundError(f"No file found at {filepath}")

        # Load history dictionary
        try:
            # Attempt normal load
            history_dict = torch.load(filepath, weights_only=False, map_location=DEVICE)
        except NotImplementedError as e:
            # Get current OS
            current_os = platform.system()

            # Patch path
            if 'PosixPath' in str(e) and current_os == 'Windows':
                # For Windows
                pathlib.PosixPath = WindowsPath
            elif 'WindowsPath' in str(e) and current_os != 'Windows':
                # For Linux/macOS
                pathlib.WindowsPath = PosixPath
            else:
                raise e
            # Retry loading the file after patching
            history_dict = torch.load(filepath, weights_only=False, map_location=DEVICE)

        # Initialise a skeleton object
        history_obj = cls.__new__(cls)  # Bypass __init__

        # Update object's attributes with loaded dictionary
        for key, value in history_dict.items():
            if isinstance(value, dict):
                # Handle bests, train, and val dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        # Move tensors to correct device
                        value[sub_key] = sub_value.to(DEVICE)
                setattr(history_obj, key, value)
            else:
                # Other attributes
                setattr(history_obj, key, value)

        return history_obj

    def __str__(self) -> str:
        """
        String representation of TrainingHistory object.
        Displayed loss, reconstruction loss and KL divergence are scaled for easier interpretability due to weighted recon and averaged KL.
        """
        summary = [
            f"Training History Summary for Model: '{self.model_name}'",
            f"History Saved Under Alternative Filename: '{self.alt_history_filename}.pth'" if self.alt_history_filename else None,
            f"Model Directory: '{self.last_updated_model.parent}'" if self.last_updated_model else f"Model Directory (as per config): '{MODEL_CHECKPOINT_DIR / self.model_name}'",
            f"{'-' * 50}",
            f"Epochs Run: {self.epochs_run}",
            f"Last Updated Model: '{self.last_updated_model.name}'\n" if self.last_updated_model else "Last Updated Model Path: None\n",
            f"Epochs Without Improvement: {self.epochs_without_improvement}",
            f"Last Improved Model: '{self.last_improved_model.name}'\n" if self.last_improved_model else "Last Improved Model Path: None\n",
            f"Batch Size: {self.batch_size}",
            f"Optimizer: {self.optim}",
            f"Weight Decay: {self.weight_decay}",
            f"Loss Function: {self.loss_fn}",
            f"Scheduler:\n\t- Patience: {self.scheduler['patience']}\n\t- Factor: {self.scheduler['factor']}\n" if self.scheduler is not None else "Scheduler: None\n",
            f"Best Validation Loss: {self.bests['best_loss'] * 100:.4f}" if self.bests['best_loss'] is not None else "Best Validation Loss: None",  # type: ignore
            f"Best Validation Loss Model: '{self.bests['best_loss_model'].name if self.bests['best_loss_model'] else 'None'}'",  # type: ignore
            f"Best Validation Weighted F1: {self.bests['best_f1_avg']:.4f}" if self.bests['best_f1_avg'] is not None else "Best Validation Weighted F1: None",
            f"Best Validation Weighted F1 Model: '{self.bests['best_f1_avg_model'].name if self.bests['best_f1_avg_model'] else 'None'}'",  # type: ignore
            f"{'-' * 50}",
            "Training Metrics (Last Epoch):",
            f"\t- Coordinate Euclidean Distance: {self.train['coor_euclid'][-1]:.4f}" if self.train['coor_euclid'] else "\t- Coordinate Euclidean Distance: None",
            f"\t- Reconstruction Loss, Averaged Across Batches: {self.train['recon'][-1] * 100:.4f}" if self.train['recon'] else "\t- Reconstruction Loss: None",
            f"\t- KL Divergence, Averaged Across Batches: {self.train['kl'][-1] * 100:.4f}" if self.train['kl'] else "\t- KL Divergence: None",
            f"\t- Descriptor Loss, Averaged Across Batches: {self.train['desc_loss'][-1]}" if self.train['desc_loss'] else "\t- Descriptor Loss: None",
            f"\t- Coordinate Loss, Averaged Across Batches: {self.train['coor_loss'][-1]}" if self.train['coor_loss'] else "\t- Coordinate Loss: None",
            f"\t- Beta: {self.train['beta'][-1]}",
            f"\t- Accuracy: {self.train['accuracy'][-1]:.4f}" if self.train['accuracy'] else "\t- Accuracy: None",
            f"\t- Weighted F1: {self.train['f1_weighted_avg'][-1]:.4f}" if self.train['f1_weighted_avg'] else "\t- Weighted F1: None",
            f"\t- Learning Rate: {self.train['lr'][-1]}",
            f"\t- Average Training Time: {avg_and_format_time(self.train['training_time'])}" if self.train['training_time'] else "\t- Average Training Time: None",
            f"{'-' * 50}",
            "Validation Metrics (Last Epoch):",
            f"\t- Coordinate Euclidean Distance: {self.val['coor_euclid'][-1]:.4f}" if self.val['coor_euclid'] else "\t- Coordinate Euclidean Distance: None",
            f"\t- Reconstruction Loss, Averaged Across Batches: {self.val['recon'][-1] * 100:.4f}" if self.val['recon'] else "\t- Reconstruction Loss: None",
            f"\t- KL Divergence, Averaged Across Batches: {self.val['kl'][-1] * 100:.4f}" if self.val['kl'] else "\t- KL Divergence: None",
            f"\t- Descriptor Loss, Averaged Across Batches: {self.val['desc_loss'][-1]}" if self.val['desc_loss'] else "\t- Descriptor Loss: None",
            f"\t- Coordinate Loss, Averaged Across Batches: {self.val['coor_loss'][-1]}" if self.val['coor_loss'] else "\t- Coordinate Loss: None",
            f"\t- Beta: {self.val['beta'][-1]}",
            f"\t- Accuracy: {self.val['accuracy'][-1]:.4f}" if self.val['accuracy'] else "\t- Accuracy: None",
            f"\t- Weighted F1: {self.val['f1_weighted_avg'][-1]:.4f}" if self.val['f1_weighted_avg'] else "\t- Weighted F1: None",
            f"\t- Average Validation Time: {avg_and_format_time(self.val['training_time'])}" if self.val['training_time'] else "\t- Average Validation Time: None",
            f"{'-' * 50}",
            "Model Architecture Used:",
            f"Latent Dim: {self.latent_dim}\n",
            "\n".join([f"- '{name}':\n{module}\n" for name, module in self.model_architecture]).rstrip("\n"),  # Removes final newline
            f"{'-' * 50}"
        ]

        # Remove any None entries from list
        summary = [line for line in summary if line is not None]

        return "\n".join(summary)


def load_model_checkpoint(source: Path | str | TrainingHistory, load: str = "updated") -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau | None, int]:
    """
    Function to load a PyTorch model, optimizer, and optionally scheduler along with number of epochs run.
    States as saved will be restored.
    Model class structure should match the structure of the model as it was saved (architecture is stored in TrainingHistory).
    Scheduler is assumed to be ReduceLROnPlateau.

    :param source: Filepath to checkpoint or TrainingHistory to load, when TrainingHistory, will load 'last_updated_model' by default (use rollback first or load parameter to load different checkpoints)
    :param load: Model to load when used with TrainingHistory object [default: 'updated', 'improved', 'f1', 'loss']
    :return: model, optimizer, scheduler, epochs_run
    """
    if isinstance(source, TrainingHistory):
        match load:
            case 'updated':
                print("Attempting to load last updated model...")
                model_path = source.last_updated_model
            case 'improved':
                print("Attempting to load last improved model...")
                model_path = source.last_improved_model
            case 'f1':
                print("Attempting to load best weighted F1 average model...")
                model_path = source.bests['best_f1_avg_model']
            case 'loss':
                print("Attempting to load best total loss model...")
                model_path = source.bests['best_loss_model']
            case _:
                raise ValueError("Invalid model to load, choose from [default: 'updated', 'improved', 'f1', 'loss'].")
        if model_path is None:
            raise ValueError("Model path attribute is None, cannot load checkpoint.")
        checkpoint_path = model_path
    else:
        print("Attempting to load model from provided path...")
        checkpoint_path = Path(source)

    # Resolve paths for comparison to check if set correctly
    checkpoint_path = checkpoint_path.resolve()
    base_dir = BASE_DIR.resolve()

    # Replace with BASE_DIR if it doesn't match
    try:
        # Path correct
        checkpoint_path.relative_to(base_dir)
    except ValueError:
        # If not, find 'outputs' in path
        if "outputs" in checkpoint_path.parts:
            outputs_index = checkpoint_path.parts.index("outputs")
            # Get relative path starting from 'outputs'
            relative_path = Path(*checkpoint_path.parts[outputs_index:])
            # Rebuild path under BASE_DIR
            checkpoint_path = BASE_DIR / relative_path
        else:
            raise ValueError("'outputs' directory not found in the checkpoint path.")

    if not checkpoint_path.parent.exists():
        raise FileNotFoundError(f"Directory {checkpoint_path.parent} not found.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path.name}' does not exist in '{checkpoint_path.parent}'.")

    print(f"Loading model and optimizer checkpoint from '{checkpoint_path}'...")
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)  # map_location prevents errors when checkpoint was saved with GPU, but loaded with CPU
    except NotImplementedError as e:
        # Get current OS
        current_os = platform.system()

        # Patch path
        if 'PosixPath' in str(e) and current_os == 'Windows':
            # For Windows
            pathlib.PosixPath = WindowsPath
        elif 'WindowsPath' in str(e) and current_os != 'Windows':
            # For Linux/macOS
            pathlib.WindowsPath = PosixPath
        else:
            raise e
        # Retry loading the file after patching
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)

    # Restore model
    model_class = getattr(model_module, checkpoint['model_class'])  # Convert stored string class to class object
    model = model_class.__new__(model_class)  # Initialise skeleton object bypassing __init__
    model.__dict__.update(checkpoint['model_config'])  # Restores attributes
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])  # Restores parameters

    # Restore optimizer
    optimizer_class = getattr(torch.optim, checkpoint['optimizer_class'])  # Convert stored string class to class object
    optimizer = optimizer_class(model.parameters(), **checkpoint['optimizer_args'])  # Initialise optimizer using original args
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restores parameters

    # Restore the scheduler (always assumed to be ReduceLROnPlateau)
    scheduler = None
    if 'scheduler_state_dict' in checkpoint:
        print("Loading scheduler.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler.__dict__.update(checkpoint['scheduler_config'])  # Restore attributes
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restores parameters

    print("Checkpoint loaded.\n")

    return model, optimizer, scheduler, checkpoint['epoch']
