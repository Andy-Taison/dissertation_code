"""
Defines loss functions
"""

import torch
import torch.nn.functional as F

def vae_loss(x: torch.Tensor, x_decoder: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates VAE loss (reconstruction loss + KL divergence), each is returned individually.

    :param x: Input tensor with shape (batch_size, *input_dim)
    :param x_decoder: Decoder output reshaped to (batch_size, *input_dim), (sigmoid) normalized in range [0, 1]
    :param z_mean: Latent space mean with shape (batch_size, latent_dim)
    :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
    :return: Reconstruction loss, KL divergence
    """
    # Normalize input to match decoder output range [0, 1]
    x_norm = x / 4

    recon_loss = F.mse_loss(x_decoder, x_norm, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

    return recon_loss, kl_div
