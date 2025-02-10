"""
Defines loss functions
"""

import torch
import torch.nn as nn
from ..config import NUM_CLASSES, COORDINATE_DIMENSIONS, DEVICE
import torch.nn.functional as F


def padded_voxel_penalty(orig_descriptors: torch.Tensor, recon_descriptors: torch.Tensor) -> torch.Tensor:
    """
    Calculates penalty for the number of padded voxels differing from original input.
    Calculated in a differentiable way to help the model learn.

    :param orig_descriptors: Original batched one-hot descriptors
    :param recon_descriptors: Reconstructed batched logits
    :return: Penalty
    """
    # Class probabilities - softmax is differentiable
    orig_probs = F.softmax(orig_descriptors, dim=-1)
    recon_probs = F.softmax(recon_descriptors, dim=-1)

    # Probability of padded voxel
    orig_padded_prob = orig_probs[:, :, 0]
    recon_padded_prob = recon_probs[:, :, 0]

    # Difference in expected number of padded voxels
    penalty = torch.abs(orig_padded_prob.sum(dim=1) - recon_padded_prob.sum(dim=1)).mean()

    return penalty


def coordinate_matching_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a penalty based on pairwise distance to original coordinates.
    Creates competition between points to avoid clustering around a single original point.
    Masks out coordinates marked as padded voxels by the descriptor values to not include in the calculation.

    :param x: Original input
    :param x_reconstructed: Reconstructed
    :return: Penalty
    """
    total_penalty = torch.tensor(0.0, device=DEVICE)

    orig_coords = x[:, :, :COORDINATE_DIMENSIONS]
    recon_coords = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]

    # Masks for non-padded voxels
    orig_mask = (x[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1) != 0)  # Shape: (batch_size, num_voxels)
    recon_mask = (x_reconstructed[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1) != 0)

    for orig, recon, o_mask, r_mask in zip(orig_coords, recon_coords, orig_mask, recon_mask):
        # Skip if no non-padded voxels
        if o_mask.sum() == 0 or r_mask.sum() == 0:
            continue

        # Pairwise distance matrix between all original and reconstructed coordinates
        dist_matrix = torch.cdist(recon, orig, p=2)

        # Apply masks to ignore padded voxels
        dist_matrix[~r_mask, :] = float('inf')  # Mask padded rows (reconstructed)
        dist_matrix[:, ~o_mask] = float('inf')  # Mask padded columns (original)

        # Gets probability of the closest point, softmin is differentiable
        neighbour_weights = F.softmin(dist_matrix * 100, dim=1)  # Scaling distance encourages model to focus on a single point
        neighbour_weights = torch.nan_to_num(neighbour_weights)  # Replace nan with 0

        # Normalise probability weights to create competition to avoid clustering around a single point
        competitive_weights = neighbour_weights / (neighbour_weights.sum(dim=0, keepdim=True) + 1e-8)  # Avoid division by 0
        competitive_weights = torch.nan_to_num(competitive_weights)  # Replace nan with 0

        # Calculate penalty (averaged over sample voxels)
        dist_matrix[torch.isinf(dist_matrix)] = 0.0  # Avoid multiplication with 'inf'
        penalty = (competitive_weights * dist_matrix).sum() / dist_matrix.size(0)

        total_penalty += penalty

    batch_penalty = total_penalty / x.size(0)  # Averaged over batch

    return batch_penalty


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

    def __call__(self, x: torch.Tensor, x_reconstructed: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor, transform_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
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
        :return: Reconstruction loss with mean reduction, KL divergence, desc_loss (applied to recon loss), coor_loss (applied to recon loss), scaled duplicate_pad_penalty, transform_reg
        """
        pad_penalty = padded_voxel_penalty(x[:, :, COORDINATE_DIMENSIONS:], x_reconstructed[:, :, COORDINATE_DIMENSIONS:])
        coor_match_loss = coordinate_matching_loss(x, x_reconstructed)

        # Reconstruction loss for coordinates and descriptors
        coor_loss = self.recon_loss_fn(x_reconstructed[:, :, :COORDINATE_DIMENSIONS], x[:, :, :COORDINATE_DIMENSIONS])  # For [x, y, z]
        desc_loss = self.desc_loss_fn(
            x_reconstructed[:, :, COORDINATE_DIMENSIONS:].view(-1, NUM_CLASSES),
            x[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1).view(-1))  # For one-hot descriptor values, internally applies softmax

        # Calculate a penalty for duplicate coordinates and padded voxels
        duplicate_pad_penalty_adjusted = duplicate_and_padded_penalty(x, x_reconstructed) * self.dup_pad_penalty_scale

        # Regularisation term to encourage orthogonality - based on https://medium.com/@itberrios6/point-net-for-classification-968ca64c57a9
        batch_size = transform_matrix.size(0)
        identity = torch.eye(transform_matrix.size(-1)).to(DEVICE)
        transform_reg = torch.linalg.norm(identity - torch.bmm(transform_matrix, transform_matrix.transpose(2, 1)))
        transform_reg = self.lambda_reg * transform_reg / batch_size

        # Combine coordinate loss, descriptor loss, with duplicate penalty, alpha balancing term and transformation regularising term
        # High alpha emphasises descriptor accuracy, lower focuses on coordinate reconstruction
        coor_loss_adjusted = coor_loss * 10
        recon_loss = self.alpha * desc_loss + (1 - self.alpha) * coor_loss_adjusted + duplicate_pad_penalty_adjusted + transform_reg

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.shape[0]  # Averaged across batch size

        return recon_loss, kl_div, desc_loss, coor_loss_adjusted, duplicate_pad_penalty_adjusted, transform_reg
