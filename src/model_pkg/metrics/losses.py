"""
Defines loss functions
"""

import torch
import torch.nn as nn

class VaeLoss:
    def __init__(self, recon_loss_name: str):
        """
        Initialise VAE Loss with a specific reconstruction loss function.

        :param recon_loss_name: Name of reconstruction loss to use ("mse", "smoothl1", "bce")
        """
        self.loss_name = recon_loss_name.lower()

        # Assign reconstruction loss function
        # Reduction is none for applying class weights element wise in __call__
        if self.loss_name == "mse":
            self.recon_loss_fn = nn.MSELoss(reduction='none')
        elif self.loss_name == "smoothl1":
            self.recon_loss_fn = nn.SmoothL1Loss(reduction='none')
        elif self.loss_name == "bce":
            self.recon_loss_fn = nn.BCELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_loss_name}")

        self.loss_name = f"VAE Loss: {type(self.recon_loss_fn).__name__}, KL Divergence"

    def __call__(self, x: torch.Tensor, x_decoder: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor, class_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates VAE loss (weighted reconstruction loss + beta * KL divergence), each is returned individually as tensors.
        Reconstruction loss is weighted (loss reduction must be 'none' for element wise application). Mean reduction applied.
        Beta is applied in train/test loops.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_decoder: Decoder output reshaped to (batch_size, *input_dim), (sigmoid) normalized in range [0, 1]
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :param class_weights: Class weight tensor to weight loss to account for class imbalance (descriptor values are sparse)
        :return: Reconstruction loss weighted with mean reduction, KL divergence
        """
        # Ensure tensors are flat for applying weights
        x_flat = x.view(-1).to(torch.long)  # Long for indexing
        x_decoder_flat = x_decoder.view(-1)

        weights_map = class_weights[x_flat]

        raw_loss = self.recon_loss_fn(x_decoder_flat, x_flat.to(torch.float))  # Element wise due to reduction being 'none'
        weighted_loss = weights_map * raw_loss
        recon_loss = weighted_loss.mean()

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return recon_loss, kl_div
