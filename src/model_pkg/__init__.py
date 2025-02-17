"""
Defines public API for src package
"""

from . import config
from .data.preprocess import combine_csv_files, split_data, save_datasets, summarise_dataset, clean_data, load_processed_datasets, split_diverse_sets
from .data.dataset import VoxelDataset, sparse_to_dense
from .visualisation.robot import load_grid_from_file, visualise_robot, compare_reconstructed
from .visualisation.plots import plot_metrics_vs_epochs, plot_loss_tradeoffs, generate_plots
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
           "save_datasets",
           "summarise_dataset",
           "clean_data",
           "load_processed_datasets",
           "split_diverse_sets",
           "VoxelDataset",
           "sparse_to_dense",
           "load_grid_from_file",
           "visualise_robot",
           "compare_reconstructed",
           "plot_metrics_vs_epochs",
           "plot_loss_tradeoffs",
           "generate_plots",
           "analyse_latent_space",
           "VAE",
           "VaeLoss",
           "train_val",
           "test",
           "TrainingHistory",
           "load_model_checkpoint",
           "train_grid_search",
           "search_grid_history"]
