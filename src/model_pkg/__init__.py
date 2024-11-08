"""
Defines public API for src package
"""

# from .model.model import VAE, vae_loss
from .model.train import train
from .data.preprocess import combine_csv_files, split_and_save_data
from .data.loaders import create_dataloader
from . import config

__version__ = "0.1.0"

# define what should be available when `from code import *` is used
__all__ = ["config", "combine_csv_files", "split_and_save_data", "create_dataloader"]  #, "train"]  # "VAE", "vae_loss", "train"]
