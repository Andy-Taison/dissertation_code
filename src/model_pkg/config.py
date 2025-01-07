"""
Configuration file
"""
import torch
import numpy as np
from pathlib import Path

# Repository/project directory
BASE_DIR = Path.cwd().parent  # Can also use Path("../") for relative address, may need adjusting depending on project setup - should point to

# Directories
DATA_DIR = BASE_DIR / "data" / "raw"  # Path to raw data CSV files
PROCESSED_DIR = BASE_DIR / "data" / "processed"  # Path to store processed CSV files
MODEL_CHECKPOINT_DIR = BASE_DIR / "outputs" / "model_checkpoints"  # Path to store checkpointed trained models (files include optimizer and optional scheduler if used)
HISTORY_DIR = BASE_DIR / "outputs" / "metric_history"  # Path to store training history metrics

# Training configurations
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100  # Maximum number of epochs to run (for dataset size, ideal will be between 50-200)
PATIENCE = 16  # How many epochs to run before stopping with no improvement in F1 score or loss (reconstruction + beta * KL)
SCHEDULER_PATIENCE = 5  # How many epochs to run with no improvement to loss (reconstruction + beta * KL) before scheduler (if using) adjusts learning rate. Lower value is more aggressive.
NUM_CLASSES = 5  # Descriptor values including 0

# VAE configurations
INPUT_DIM = (11, 11, 11)
LATENT_DIM = 2

# Seeds for repeatability
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_STATE)

# Check if GPU available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device\n")
