"""
Defines loss functions
"""

import torch
import torch.nn as nn

class VaeLoss:
    def __init__(self, recon_loss_name: str):
        """
        Initialise VAE Loss with a specific reconstruction loss function.

        :param recon_loss_name: Name of reconstruction loss to use ("mse", "smoothl1")
        """
        self.loss_name = recon_loss_name.lower()

        # Assign reconstruction loss function
        if self.loss_name == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='sum')
        elif self.loss_name == "smoothl1":
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='sum')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_loss_name}")

        self.loss_name = f"VAE Loss: {type(self.recon_loss_fn).__name__}, KL Divergence"

    def __call__(self, x: torch.Tensor, x_decoder: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates VAE loss (reconstruction loss + KL divergence), each is returned individually as tensors.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_decoder: Decoder output reshaped to (batch_size, *input_dim), (sigmoid) normalized in range [0, 1]
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :return: Reconstruction loss, KL divergence
        """
        # Normalize input to match decoder output range [0, 1]
        x_norm = x / 4

        recon_loss = self.recon_loss_fn(x_decoder, x_norm)
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return recon_loss, kl_div
