"""
Defines public API for src package
"""

from . import config
from .data.preprocess import combine_csv_files, split_and_save_data
from .data.loaders import create_dataset_and_loader
from .visualisation.robot import load_grid_from_file, visualise_robot
from .model.model import VAE
from .metrics.losses import VaeLoss
from .model.train_test import train_val, test
from .model.history_checkpoint import TrainingHistory, load_model_checkpoint
from .model.grid_search import train_grid_search, perform_grid_search

__version__ = "0.1.0"

# Define what should be available when `from code import *` is used
__all__ = ["config",
           "combine_csv_files",
           "split_and_save_data",
           "create_dataset_and_loader",
           "load_grid_from_file",
           "visualise_robot",
           "VAE",
           "VaeLoss",
           "train_val",
           "test",
           "TrainingHistory",
           "load_model_checkpoint",
           "train_grid_search",
           "perform_grid_search"]
