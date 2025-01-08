from .model.model import VAE, vae_loss
from .model.train import main

__version__ = "0.1.0"

# define what should be available when `from code import *` is used
__all__ = ["VAE", "vae_loss", "main"]
