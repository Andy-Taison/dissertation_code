"""
Defines public API for src package
"""

from . import config
from .data.preprocess import combine_csv_files, split_data, save_split_datasets, summarise_dataset, clean_data
from .data.loaders import create_dataset_and_loader
from .visualisation.robot import load_grid_from_file, visualise_robot
from .visualisation.plots import plot_metrics_vs_epochs, plot_loss_tradeoffs
from .visualisation.latent import analyse_latent_space
from .model.model import VAE
from .metrics.losses import VaeLoss
from .model.train_test import train_val, test
from .model.history_checkpoint import TrainingHistory, load_model_checkpoint
from .model.grid_search import train_grid_search, search_grid_history

__version__ = "0.1.0"

# Define what should be available when `from code import *` is used
__all__ = ["config",
           "combine_csv_files",
           "split_data",
           "save_split_datasets",
           "summarise_dataset",
           "clean_data",
           "create_dataset_and_loader",
           "load_grid_from_file",
           "visualise_robot",
           "plot_metrics_vs_epochs",
           "plot_loss_tradeoffs",
           "analyse_latent_space",
           "VAE",
           "VaeLoss",
           "train_val",
           "test",
           "TrainingHistory",
           "load_model_checkpoint",
           "train_grid_search",
           "search_grid_history"]
