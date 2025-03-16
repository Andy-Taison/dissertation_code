"""
Configuration file
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Backend, for use locally, interactive backend
# matplotlib.use('AGG')  # Backend, for use with HPC, headless environment

# Repository/project directory
# When running locally, use this:
BASE_DIR = Path.cwd().parent  # Can also use Path("../") for relative address, may need adjusting depending on project setup - sho$
# When running on the HPC use this:
# BASE_DIR = Path("/users/40538519/sharedscratch")

# Show plots
PLOT = False

# Directories
DATA_DIR = BASE_DIR / "data" / "raw"  # Path to raw data CSV files
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Path to store processed CSV files
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT_DIR = OUTPUTS_DIR / "model_checkpoints"  # Path to store checkpointed trained models (files include optimizer and optional scheduler if used)
HISTORY_DIR = OUTPUTS_DIR / "metric_history"  # Path to store training history metrics
PLOT_DIR = BASE_DIR / "plots"  # Path to store generated plots

# Training configurations
BATCH_SIZE = 64  # Not used in main package, intended to be used when calling the train_val function without using grid search
LEARNING_RATE = 0.001  # Not used in main package, intended to be used when calling the train_val function without using grid search
EPOCHS = 500  # Maximum number of epochs to run (for dataset size, ideal will be between 50-200), used in grid search, history checkpoint
PATIENCE = 31  # 20  # How many epochs to run before stopping with no improvement in F1 score or loss (reconstruction + beta * KL), used in history checkpoint
SCHEDULER_PATIENCE = 10  # How many epochs to run with no improvement to loss (reconstruction + beta * KL) before scheduler (if using) adjusts learning rate. Lower value is more aggressive.
NUM_CLASSES = 5  # Descriptor values including 0, used in dataset, losses, metrics, train test, and history checkpoint

# VAE and dataset configurations
EXPANDED_GRID_SIZE = 11  # Assumed square/cube
INPUT_DIM = (8, 8)  # Used in grid search
COORDINATE_DIMENSIONS = 3
MAX_VOXELS = 8  # Maximum number of voxels for any robot in the full dataset - if this changes, review deconvolutional layers in model decoder
LATENT_DIM = 2  # Not used in main package, intended to be used when constructing a model without using grid search

# Seeds for repeatability
RANDOM_STATE = 42  # Used in multiple files
torch.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_STATE)

# Check if GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Used in multiple files
print(f"Using {DEVICE} device\n")
