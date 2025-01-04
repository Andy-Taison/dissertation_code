"""
Includes checkpointing functions to save models, optimizers and schedulers.
TrainingHistory class used to track and save training history. It also calls checkpoint_model function.
"""

import torch
from pathlib import Path
from ..config import DEVICE, NUM_CLASSES, MODEL_DIR, HISTORY_DIR, PATIENCE, EPOCHS
from ..metrics.losses import VaeLoss

def checkpoint_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: Path | str, scheduler: torch.optim.lr_scheduler.LRScheduler = None):
    """
    Function to save/checkpoint a PyTorch model, optimizer, and optionally scheduler states along with number of epochs run.
    Creates directory if it does not exist.

    :param model: Model to save state
    :param optimizer: Optimizer to save state
    :param epoch: Number of epochs run
    :param filepath: Filepath as either Path object or string
    :param scheduler: Optional scheduler to save state
    """
    print("Checkpointing...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    if scheduler:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint_path = Path(filepath)
    if not checkpoint_path.parent.exists():
        print(f"Creating directory {checkpoint_path.parent}...")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved as '{checkpoint_path.name}' in '{checkpoint_path.parent}'\n")


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: Path | str, scheduler: torch.optim.lr_scheduler.LRScheduler = None) -> int:
    """
    Function to load a PyTorch model, optimizer, and optionally scheduler states along with number of epochs run.
    Model, optimizer and schedulers should be initialised before calling this function.

    :param model: Initialised model to load state into
    :param optimizer: Initialised optimizer to load state into
    :param filepath: Filepath to checkpoint to load
    :param scheduler: Optional, initialised scheduler to load state into if scheduler was checkpointed
    :return: Epochs run
    """
    checkpoint_path = Path(filepath)
    if not checkpoint_path.parent.exists():
        raise FileNotFoundError(f"Directory {checkpoint_path.parent} not found.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path.name}' does not exist in '{checkpoint_path.parent}'")

    print(f"Loading checkpoint from '{checkpoint_path}'...\n")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)  # map_location prevents errors when checkpoint was saved with GPU, but loaded with CPU

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    print(f" {model.name} model loaded has run {epoch} epochs.")
    print(f"{optimizer.__class__.__name__} optimizer loaded using learning rate {optimizer.param_groups[0]['lr']}.")

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"{type(scheduler).__name__} scheduler loaded.\n")
    else:
        print("No scheduler loaded.\n")

    return epoch


class TrainingHistory:
    """
    A class to track training and validation history during training.
    Handles saving and loading history to/from file.
    Method check_and_save_model_improvement checks for improvements, handles early stopping condition and calls checkpointing functions to save models.
    check_and_save_model_improvement should be called prior to updating epoch history.
    update_epoch should be called prior to saving history.
    """
    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: VaeLoss):
        """
        Initialises TrainingHistory object for tracking metrics.
        Use TrainingHistory.load() to load and initialise saved objects.

        :param model: Model to track history for, must have attribute 'name'
        :param dataloader: Used to obtain batch size
        :param optimizer: Optimizer name tracked
        :param criterion: Loss function, should have attribute 'loss_name'
        """
        self.epochs_run = 0
        self.epochs_without_improvement = 0
        self.last_updated_model = None
        self.last_improved_model = None

        self.model_name = model.name
        self.batch_size = dataloader.__getattribute__("batch_size")
        self.optim = optimizer.__class__.__name__
        self.loss_fn = criterion.loss_name
        self.model_architecture = [(name, module) for name, module in model.named_modules()]

        self.bests = {
            'best_loss_model': None,
            'best_loss': None,
            'best_f1_avg_model': None,
            'best_f1_avg': None,
        }

        self.train = {
            'recon': [],
            'kl': [],
            'accuracy': [],
            'recall_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'recall_weighted_avg': [],
            'precision_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'precision_weighted_avg': [],
            'f1_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'f1_weighted_avg': [],
            'beta': [],
            'lr': []
        }

        # No learning rate required for validation due to not using optimizer
        self.val = {
            'recon': [],
            'kl': [],
            'accuracy': [],
            'recall_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'recall_weighted_avg': [],
            'precision_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'precision_weighted_avg': [],
            'f1_classes': torch.empty(0, NUM_CLASSES, device=DEVICE),
            'f1_weighted_avg': [],
            'beta': []
        }

    def check_and_save_model_improvement(self, val_epoch_metrics: dict, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler = None) -> bool:
        """
        Should be used with validation metrics only, not training metrics.
        Checks for improvement of total loss or weighted F1 average.
        Model is checkpointed in either condition along with first epoch, early stopping and final epoch.
        Note TrainingHistory status is not saved here, neither is history updated, however metric bests are updated.
        This method should be carried out prior to updating epoch history for value comparisons.
        Models are saved using MODEL_DIR from config along with model name and epoch.

        Early stopping terminate condition True returned when epoch reaches config PATIENCE with no improvement.

        :param val_epoch_metrics: Validation metrics dictionary
        :param epoch: Current epoch
        :param model: Model
        :param optimizer: Optimizer
        :param scheduler: Scheduler if using (optional)
        :return: True when patience epoch reached with no improvement, or final epoch reached
        """
        improved = False
        epoch_loss = val_epoch_metrics['recon'] + val_epoch_metrics['beta'] * val_epoch_metrics['kl']

        # Loss
        if self.last_updated_model is None or epoch_loss < self.bests['best_loss']:
            filepath = Path(MODEL_DIR) / model.name / f"best_loss_epoch_{epoch}.pth"
            checkpoint_model(model, optimizer, epoch, filepath, scheduler)
            self.bests['best_loss_model'] = filepath
            self.bests['best_loss'] = epoch_loss
            self.last_improved_model = filepath
            self.last_updated_model = filepath
            self.epochs_without_improvement = 0
            improved = True

        # Weighted F1 average
        if self.bests['best_f1_avg'] is None or val_epoch_metrics['weighted_f1'] > self.bests['best_f1_avg']:
            filepath = Path(MODEL_DIR) / model.name / f"best_f1_avg_epoch_{epoch}.pth"
            checkpoint_model(model, optimizer, epoch, filepath, scheduler)
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
            filepath = Path(MODEL_DIR) / model.name / f"terminated_model_epoch_{epoch}.pth"
            checkpoint_model(model, optimizer, epoch, filepath, scheduler)
            self.last_updated_model = filepath
            return True  # Terminate training condition
        elif epoch >= EPOCHS:
            # Checkpoint final model
            filepath = Path(MODEL_DIR) / model.name / f"final_model_epoch_{epoch}.pth"
            checkpoint_model(model, optimizer, epoch, filepath, scheduler)
            self.last_updated_model = filepath
            return False
        else:
            return False

    def update_epoch(self, epoch_metrics: dict, mode: str):
        """
        Update history with epoch metrics.
        Method does not update bests.
        check_and_save_model_improvement should be called prior to calling this method to check for improvements and update bests.

        :param epoch_metrics: Dictionary containing metrics
        :param mode: 'train' or 'val' history to update
        """
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        history = self.train if mode == 'train' else self.val

        self.epochs_run += 1
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
        if 'lr' in history:
            history['lr'].append(epoch_metrics['lr'])

    def save_history(self):
        """
        Save training history to file using PyTorch's save functionality.
        History is saved using model_name attribute and HISTORY_DIR set in config.
        """
        filepath = Path(HISTORY_DIR) / f"{self.model_name}_history_.pth"

        # Check directories exist
        if not filepath.parent.exists():
            print(f"Creating directory {filepath.parent}...")
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save the state of the object
        torch.save(self.__dict__, filepath)

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
        history_dict = torch.load(filepath, map_location=DEVICE)

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

    def roll_back(self):
        pass
