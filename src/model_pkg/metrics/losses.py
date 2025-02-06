"""
Defines loss functions
"""

import torch
import torch.nn as nn
from ..config import NUM_CLASSES

class VaeLoss:
    def __init__(self, recon_loss_name: str):
        """
        Initialise VAE Loss with a specific reconstruction loss function.
        BCE expects values to be in range [0,1]

        :param recon_loss_name: Name of reconstruction loss to use ("mse", "bce", "smoothl1")
        """
        self.loss_name = recon_loss_name.lower()

        # Assign reconstruction loss function, used for coordinates
        if self.loss_name == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='mean')
        elif self.loss_name == "bce":
            self.recon_loss_fn = nn.BCELoss(reduction='mean')
        elif self.loss_name == "smoothl1":
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_loss_name}")

        self.desc_loss_fn = nn.CrossEntropyLoss(reduction='mean')  # Used for descriptor values

        self.loss_name = f"VAE Loss: {type(self.recon_loss_fn).__name__} + {type(self.desc_loss_fn).__name__}, KL Divergence"

    def __call__(self, x: torch.Tensor, x_reconstructed: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates VAE loss (reconstruction loss + beta * KL divergence), each is returned individually as tensors.
        Reconstruction loss is the sum of coordinate loss and descriptor loss.
        Coordinate loss uses the defined recon_loss_fn with mean reduction.
        Descriptor loss uses CrossEntropyLoss also with mean reduction.
        KL divergence is averaged across batch size.
        Beta is applied in train/test loops.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_reconstructed: Decoder output with shape (batch_size, *input_dim)
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :return: Reconstruction loss with mean reduction, KL divergence
        """
        # Reconstruction loss for coordinates and descriptors
        coor_loss = self.recon_loss_fn(x_reconstructed[:, :, :3], x[:, :, :3])  # For [x, y, z]
        desc_loss = self.desc_loss_fn(x_reconstructed[:, :, 3:].view(-1, NUM_CLASSES), x[:, :, 3:].argmax(dim=-1).view(-1))  # For one-hot descriptor values, internally applies softmax

        # Combine coordinate loss and descriptor loss
        recon_loss = coor_loss + desc_loss

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.shape[0]  # Averaged across batch size

        return recon_loss, kl_div
