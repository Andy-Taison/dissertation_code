"""
Configuration file
"""

import torch
import numpy as np
from pathlib import Path

# Repository/project directory
BASE_DIR = Path("../")

# Directories
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Training configurations
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
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

