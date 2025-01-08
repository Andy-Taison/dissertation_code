"""
Functions for performing grid search to find optimal hyperparameters which allows easier comparison of models.

Things to trial later are in comments.
Focus initially should be on getting baseline on different architectural changes:
 - hidden units, layers
 - activation functions
 - dropout rates
 - convolution
Also want to trial using schedulers adjusting patience and factor.
Maybe trial different early stopping patience.
"""
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from pathlib import Path
from ..config import DEVICE, INPUT_DIM, EPOCHS, HISTORY_DIR
from .model import VAE
from ..metrics.losses import VaeLoss
from ..metrics.metrics import get_best_tradeoff_score
from .train_test import train_val
from .history_checkpoint import TrainingHistory

def create_grid() -> list[dict]:
    """
    Creates a grid of training configurations as a list of dictionaries containing:
    - 'batch_size'
    - 'latent_dim'
    - 'loss'
    - 'optimizer' (list of dictionaries with keys 'type', 'params' and 'model_name')
    - 'lr'
    - 'decay'
    - 'beta'

    :return: Grid of training configurations
    """
    print("Creating grid...")
    batch_sizes = [32, 64]  # Possibly trial 16, and 128 later
    latent_dims = [2, 4, 8]  # Possibly trial 16 later
    loss_functions = ["mse", "bce"]  # Possibly later trail "smoothL1", bce expects probabilities in range [0,1], otherwise use bcewithlogitsloss as internally applies sigmoid
    optimizer = [
        {"type": optim.Adam, "params": {}, "model_name": "adam"},
        # {"type": optim.SGD, "params": {"nesterov": True}, "model_name": "sgd"},  # Potential for later
        {"type": optim.SGD, "params": {"nesterov": True, "momentum": 0.9}, "model_name": "sgdmom.9"},
        # {"type": optim.RMSprop, "params": {}, "model_name": "rms"},  # Potential for later
        {"type": optim.RMSprop, "params": {"momentum": 0.9}, "model_name": "rmsmom.9"}
    ]
    learning_rates = [1e-4, 1e-3]  # Later potentially trial 5e-4, 5e-3, and 1e-2 for fine-tuning
    weight_decay = [0, 1e-4]  # Possibly later trial 1e-2
    betas = [0.1, 1]  # Possibly trial 0.5, 2, and 4 later

    grid = [
        {"batch_size": batch_size, "latent_dim": latent_dim, "loss": loss, "optimizer": opt, "lr": lr, "decay": decay, "beta": beta}
        for batch_size in batch_sizes
        for latent_dim in latent_dims
        for loss in loss_functions
        for opt in optimizer
        for lr in learning_rates
        for decay in weight_decay
        for beta in betas
    ]
    print(f"Grid created with {len(grid)} configuration(s).")

    return grid


def train_grid_search(train_ds: TensorDataset, val_ds: TensorDataset, model_architecture_name: str, clear_history_list: bool = False, history_list_filename: str = "grid_search_list"):
    """
    Conducts training for grid search.
    Grid is created based on 'create_grid' function.
    Stores list of history filenames to '<history_list_filename>.txt' which is stored in the 'HISTORY_DIR' as specified in config.
    This file is used when conducting the actual grid search.

    Ensure to clear or delete list file when starting a new grid search to not include other training history files in search.

    :param train_ds: Training dataset
    :param val_ds: Validation dataset
    :param model_architecture_name: Unique prefix for history file and checkpointing
    :param clear_history_list: When False, files already listed in '<history_list_filename>.txt' are skipped, when True '<history_list_filename>.txt' is overwritten
    :param history_list_filename: Filename without extension for storing paths to trained history files
    """
    print("*" * 50)
    print("Starting grid search training...")
    grid = create_grid()

    search_list_path = Path(HISTORY_DIR) / f"{history_list_filename}.txt"

    # Check directories exist
    search_list_path.parent.mkdir(parents=True, exist_ok=True)

    # Open file in write or append mode
    mode = "w" if clear_history_list else "a"
    completed_histories = set()

    # Read existing entries if not clearing list
    if not clear_history_list and search_list_path.exists():
        with search_list_path.open("r") as file:
            completed_histories = set(line.strip() for line in file.readlines())

    # Open file for appending new entries
    with search_list_path.open(mode) as file:
        # Train for each configuration
        for i, setup in enumerate(grid):
            # Assign unique name containing test information
            model_name = f"{model_architecture_name}_bs{setup['batch_size']}_ld{setup['latent_dim']}_{setup['loss']}_{setup['optimizer']['model_name']}_lr{setup['lr']}_wd{setup['decay']}_be{setup['beta']}"

            # Check if history path is already in file
            if f"{model_name}_history.pth" in completed_histories:
                print(f"Skipping configuration, already completed: '{model_name}'")
                completed_histories.remove(f"{model_name}_history.pth")  # Completed histories set used to check if files in search_list_path were not found
                continue
            else:
                print(f"\nConfiguration [{i + 1:>4d}/{len(grid):>4d}]:")

            # Create dataloaders updating batch_size
            train_loader = DataLoader(train_ds, batch_size=setup['batch_size'], shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=setup['batch_size'], shuffle=False)

            # Create model
            vae = VAE(INPUT_DIM, setup['latent_dim'], model_name).to(DEVICE)

            # Initialise loss function
            criterion = VaeLoss(setup['loss'])

            # Initialise optimizer
            optim_params = setup['optimizer']['params']
            optim_params["lr"] = setup['lr']
            optim_params["weight_decay"] = setup['decay']
            optimizer = setup['optimizer']['type'](vae.parameters(), **optim_params)

            # Train VAE
            history = train_val(vae, train_loader, val_loader, criterion, optimizer, EPOCHS, setup['beta'])  # History will be to the latest model, which most likely will not be the best model

            # Add path to file
            history_path = f"{history.model_name}_history.pth"
            file.write(history_path + "\n")
            print(f"Added '{history_path}' to {Path(*search_list_path.parts[-2:])}")

    print("\nGrid search training complete!")
    if completed_histories:
        print(f"\nFiles were found in '{search_list_path.name}' that were not part of the grid training configuration:")
        for file in completed_histories:
            print(file)
        print("\nEnsure to remove these before searching grid if not wanting to include them.")
    print("*" * 50 + "\n")


def search_grid_history(history_list_filename: str = "grid_search_list", loss_f1_tradeoff: int = 0.7) -> tuple[TrainingHistory, float, int] | None:
    """
    Finds best tradeoff score (loss_f1_tradeoff x_best loss + (1 - loss_f1_tradeoff) x (1 - best_f1_avg)) of the models listed in 'history_list_filename'.
    'history_list_filename' should exist in 'HISTORY_DIR' as specified in config.

    Note in certain cases where tradeoff score epoch does not match either best weighted F1 average epoch or best loss epoch, loading of the checkpoint will not be possible.
    This is only the case when checkpointing files are pruned (default).
    However, the TrainingHistory object can still be loaded to obtain metrics.

    :param history_list_filename: txt file listing all TrainingHistory filenames to perform grid search on
    :param loss_f1_tradeoff: Balances loss and weighted F1 average to compare histories for the grid search
    :return: TrainingHistory found with best score, best score, best epoch
    """
    print(">" * 50)
    print("Searching grid...\n")
    # Get all history filenames
    history_list_path = Path(HISTORY_DIR) / f"{history_list_filename}.txt"
    with history_list_path.open("r") as file:
        history_filenames = set(filename.strip() for filename in file.readlines())

    best_score = None
    best_history = None
    best_epoch = None

    if not history_filenames:
        print(f"No filenames found in '{history_list_filename}'\n")
        return

    # Grid search
    for filename in history_filenames:
        history = TrainingHistory.load_history(filename)

        epoch, score = get_best_tradeoff_score(history.val['recon'], history.val['kl'], history.val['beta'], history.val['f1_weighted_avg'], loss_f1_tradeoff)

        if best_score is None or score < best_score:
            best_score = score
            best_history = history
            best_epoch = epoch

    print("Search complete!\n")
    print(f"Best Configuration tradeoff score: {best_score:.4f} at epoch: {best_epoch}")
    print(best_history)
    print(">" * 50)

    return best_history, best_score, best_epoch
