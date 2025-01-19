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

        # Assign reconstruction loss function
        # Reduction is none for applying class weights element wise in __call__
        if self.loss_name == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='none')
        elif self.loss_name == "bce":
            self.recon_loss_fn = nn.BCELoss(reduction='none')
        elif self.loss_name == "smoothl1":
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_loss_name}")

        self.loss_name = f"VAE Loss: {type(self.recon_loss_fn).__name__}, KL Divergence"

    def __call__(self, x: torch.Tensor, x_reconstructed: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor, class_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates VAE loss (weighted reconstruction loss + beta * KL divergence), each is returned individually as tensors.
        Reconstruction loss is weighted by class imbalance (loss reduction must be 'none' when initialised for element wise application). Mean reduction applied.
        KL divergence is averaged across batch size.
        Beta is applied in train/test loops.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_reconstructed: Decoder output with shape (batch_size, *input_dim)
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :param class_weights: Class weight tensor to weight loss to account for class imbalance (descriptor values are sparse)
        :return: Reconstruction loss weighted by class imbalance with mean reduction, KL divergence
        """
        # Calculate reconstruction loss, element wise due to reduction being 'none'

        # Normalise inputs to range [0, 1] for BCE, and for other losses for grid search comparison between losses
        x_normalised = x / (NUM_CLASSES - 1)
        x_reconstructed_normalised = torch.sigmoid(x_reconstructed)

        raw_loss = self.recon_loss_fn(x_reconstructed_normalised, x_normalised)

        # Apply class weights
        x_flat = x.view(-1).to(torch.long)  # Flatten x for indexing class_weights, long for indexing
        weights_map = class_weights[x_flat]
        weighted_loss = raw_loss.view(-1) * weights_map

        # Mean reduction
        recon_loss = weighted_loss.mean()

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.shape[0]  # Averaged across batch size

        return recon_loss, kl_div
