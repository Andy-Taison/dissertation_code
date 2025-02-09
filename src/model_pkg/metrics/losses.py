"""
Defines loss functions
"""

import torch
import torch.nn as nn
from ..config import NUM_CLASSES, COORDINATE_DIMENSIONS, DEVICE


def duplicate_and_padded_penalty(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a penalty for not having the correct number of padded voxels (identified via original descriptor values).
    A further penalty is calculated for duplicate coordinates in the reconstructed tensor.
    Penalty is averaged across the batch.

    :param x: Original input tensor with shape (batch_size, num_voxels, features)
    :param x_reconstructed: Reconstructed tensor with shape (batch_size, num_voxels, features)
    :return: Average penalty per batch
    """
    coords_recon = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]
    descriptors_orig = x[:, :, COORDINATE_DIMENSIONS:]
    batch_size, num_voxels, _ = coords_recon.shape

    # Descriptors are raw logits, argmax used to get predicted class
    descriptors_recon = x_reconstructed[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1)

    penalty = 0.0

    for orig_desc, recon_desc, recon_coords in zip(descriptors_orig, descriptors_recon, coords_recon):
        # Count padded voxels
        num_padded_orig = (orig_desc.argmax(dim=-1) == 0).sum().item()
        num_padded_recon = (recon_desc == 0).sum().item()

        # Penalty for different number of padded voxels
        padded_penalty = abs(num_padded_recon - num_padded_orig)

        # Filter out padded voxels from reconstructed coordinates
        non_padded_coords = recon_coords[recon_desc != 0]

        # Penalty for duplicate coordinates
        unique_coords = torch.unique(non_padded_coords, dim=0)
        redundant = non_padded_coords.size(0) - unique_coords.size(0)

        # Total penalty for sample
        total_sample_penalty = redundant + padded_penalty
        penalty += total_sample_penalty

    # Average penalty across batch
    return penalty / batch_size


class VaeLoss:
    def __init__(self, recon_loss_name: str, alpha=0.3, dup_pad_penalty_scale=0.1, lambda_reg=0.001):
        """
        Initialise VAE Loss with a specific reconstruction loss function.
        BCE expects values to be in range [0,1]

        When alpha is high, it emphasises descriptor accuracy, lower focuses on coordinate reconstruction.
        dup_pad_penalty_scale scales duplicate pad penalty.
        lambda_reg used to scale transformation regularisation term.

        :param recon_loss_name: Name of reconstruction loss to use ("mse", "bce", "smoothl1")
        :param alpha: Balancing factor for descriptor loss and coordinate loss, high emphasises descriptor accuracy, low focuses on coordinate reconstruction
        :param dup_pad_penalty_scale: Penalty scaling factor for duplicate pad penalty
        :param lambda_reg: Scaling factor of transformation regularising term
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

        self.alpha = alpha
        self.dup_pad_penalty_scale = dup_pad_penalty_scale
        self.lambda_reg = lambda_reg

    def __call__(self, x: torch.Tensor, x_reconstructed: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor, transform_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates VAE loss (reconstruction loss + beta * KL divergence)
        Reconstruction loss and KL divergence are returned individually as tensors.
        Reconstruction loss is the sum of coordinate loss and descriptor loss (scaled by alpha), with duplication penalty and transformation regulariser.

        Reconstruction loss is the sum of coordinate loss and descriptor loss, scaled by alpha with a transformation regularisation term added.
        When alpha is high, it emphasises descriptor accuracy, lower focuses on coordinate reconstruction.
        Coordinate loss uses the defined recon_loss_fn with mean reduction.
        Descriptor loss uses CrossEntropyLoss also with mean reduction.
        KL divergence is averaged across batch size.

        Reconstruction loss includes a penalty for duplicate coordinates and incorrect number of padded voxels scaled by dup_pad_penalty_scale.

        Beta is applied in train/test loops for tracking purposes.

        Lambda reg scales the transformation regularising term which is used to encourage orthogonality.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_reconstructed: Decoder output with shape (batch_size, *input_dim)
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :param transform_matrix: Transformation matrix used to calculate regularisation term to encourage orthogonality
        :return: Reconstruction loss with mean reduction, KL divergence, desc_loss (applied to recon loss), coor_loss (applied to recon loss)
        """
        # Reconstruction loss for coordinates and descriptors
        coor_loss = self.recon_loss_fn(x_reconstructed[:, :, :COORDINATE_DIMENSIONS], x[:, :, :COORDINATE_DIMENSIONS])  # For [x, y, z]
        desc_loss = self.desc_loss_fn(
            x_reconstructed[:, :, COORDINATE_DIMENSIONS:].view(-1, NUM_CLASSES),
            x[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1).view(-1))  # For one-hot descriptor values, internally applies softmax

        # Calculate a penalty for duplicate coordinates and padded voxels
        duplicate_pad_penalty = duplicate_and_padded_penalty(x, x_reconstructed) * self.dup_pad_penalty_scale

        # Regularisation term to encourage orthogonality - based on https://medium.com/@itberrios6/point-net-for-classification-968ca64c57a9
        batch_size = transform_matrix.size(0)
        identity = torch.eye(transform_matrix.size(-1)).to(DEVICE)
        transform_reg = torch.linalg.norm(identity - torch.bmm(transform_matrix, transform_matrix.transpose(2, 1)))
        transform_reg = self.lambda_reg * transform_reg / batch_size

        # Combine coordinate loss, descriptor loss, with duplicate penalty, alpha balancing term and transformation regularising term
        # High alpha emphasises descriptor accuracy, lower focuses on coordinate reconstruction
        recon_loss = self.alpha * desc_loss + (1 - self.alpha) * coor_loss + duplicate_pad_penalty + transform_reg

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.shape[0]  # Averaged across batch size

        return recon_loss, kl_div, desc_loss, coor_loss
