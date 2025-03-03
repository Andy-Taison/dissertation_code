"""
Defines loss functions
"""

import torch
import torch.nn.functional as F
from ..config import COORDINATE_DIMENSIONS, DEVICE, EXPANDED_GRID_SIZE


def coordinate_matching_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a penalty based on spatial alignment between original and reconstructed coordinates.
    Penalty is the difference between max entropy (perfect alignment) and the entropy over
    the original point attention distribution. Padded voxels are encoded as a coordinate outside
    the expanded grid (EXPANDED_GRID_SIZE), and contribute to the loss.

    :param x: Original input
    :param x_reconstructed: Reconstructed
    :return: Penalty averaged over batch
    """
    total_penalty = torch.tensor(0.0, device=DEVICE)

    orig_coords = x[:, :, :COORDINATE_DIMENSIONS]
    recon_coords = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]

    for orig, recon in zip(orig_coords, recon_coords):

        # Pairwise distance matrix between all original and reconstructed coordinates
        dist_matrix = torch.cdist(recon, orig, p=2)  # (rows: distances from reconstructed, columns: distances from original)

        # Gets probability of the closest point, softmin is differentiable
        neighbour_weights = F.softmin(dist_matrix * 100, dim=1)  # Scaling distance encourages model to focus on a single point

        # Sum of attention each original point (columns) receives from all reconstructed points
        attention_per_original = neighbour_weights.sum(dim=0)  # (num_original_points)

        # Normalise attention to form a probability distribution over original points
        attention_distribution = attention_per_original / (attention_per_original.sum() + 1e-8)  # Avoid division by 0

        # Shannon entropy for attention distribution (amount of uncertainty) - clustered points will have lower entropy
        entropy = -(attention_distribution * torch.log(attention_distribution + 1e-8)).sum()  # log(0) is undefined

        # Maximum entropy (perfect alignment)
        # Happens when distribution is uniform: p(x) = 1/n
        # H(x)max = -sum_{i..N}((1/n)*log(1/n))
        # log(1/n) = -log(n)
        # H(x)max = log(n)
        max_entropy = torch.log(torch.tensor(attention_distribution.size(0), device=DEVICE))

        # Clustered points will have lower entropy
        entropy_penalty = (max_entropy - entropy) ** 2  # Avoid negatives

        # Count padded voxel coordinates in original
        orig_scaled = orig * EXPANDED_GRID_SIZE  # Scale normalised coordinates back to grid indices, padded values encoded as EXPANDED GRID SIZE (outside of grid)
        orig_rounded = orig_scaled.round().long()
        orig_clamped = torch.clamp(orig_rounded, min=0, max=EXPANDED_GRID_SIZE)
        orig_padded_num = (orig_clamped == 11).any(dim=1).sum()

        # Count padded voxel coordinates in reconstructed
        recon_scaled = recon * EXPANDED_GRID_SIZE  # Scale normalised coordinates back to grid indices, padded values encoded as EXPANDED GRID SIZE (outside of grid)
        recon_rounded = recon_scaled.round().long()
        recon_clamped = torch.clamp(recon_rounded, min=0, max=EXPANDED_GRID_SIZE)
        recon_padded_num = (recon_clamped == 11).any(dim=1).sum()

        padded_diff = ((orig_padded_num - recon_padded_num) ** 2 / (1 + (orig_padded_num - recon_padded_num) ** 2)) * 2

        total_penalty += entropy_penalty + padded_diff

    batch_penalty = total_penalty / x.size(0)  # Averaged over batch

    return batch_penalty


def descriptor_matching_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a loss for descriptor values based on nearest neighbour.
    Calculates the pairwise distance between original and reconstructed points.
    The most likely descriptor value based on nearest original voxel is then obtained.
    Loss is calculated by taking the cross entropy (model must output raw logits).

    :param x: Original input
    :param x_reconstructed: Reconstructed
    :return: Descriptor loss averaged over batch
    """
    total_loss = torch.tensor(0.0, device=DEVICE)

    orig_coords = x[:, :, :COORDINATE_DIMENSIONS]
    orig_descriptors = x[:, :, COORDINATE_DIMENSIONS:]

    recon_coords = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]
    recon_descriptors = x_reconstructed[:, :, COORDINATE_DIMENSIONS:]

    for orig_coor, recon_coor, orig_desc, recon_desc in zip(orig_coords, recon_coords, orig_descriptors, recon_descriptors):

        # Pairwise distance matrix between all original and reconstructed coordinates
        dist_matrix = torch.cdist(recon_coor, orig_coor, p=2)  # (rows: distances from reconstructed, columns: distances from original)

        # Get probability of the closest point, softmin is differentiable
        neighbour_weights = F.softmin(dist_matrix * 100, dim=1)  # Scaling distance helps focus on a single point

        # Matrix multiplication to get predicted descriptors
        matched_target_descriptors = torch.matmul(neighbour_weights, orig_desc)  # (recon_voxels, descriptor_dim)

        # Extract the most likely class index for each matched descriptor
        target_cls = matched_target_descriptors.argmax(dim=-1)  # (recon_voxels)

        # Apply CrossEntropyLoss
        descriptor_loss = F.cross_entropy(recon_desc, target_cls, reduction='mean')

        total_loss += descriptor_loss

    batch_loss = total_loss / x.size(0)  # Average over batch

    return batch_loss


def overlap_penalty(x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates penalty for overlapping non-padded voxels that would collapse into
    one another when scaled to the full descriptor matrix.
    The penalty is calculated from the pairwise distance between matching coordinates.
    The negative sum of the log of the upper triangle of the distance matrix (excluding the diagonal)
    is calculated as the penalty and then averaged over the batch.

    :param x_reconstructed: Reconstructed
    :return: Overlap penalty averaged over batch
    """
    total_penalty = torch.tensor(0.0, device=DEVICE)

    recon_coords = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]

    # Mask for non-padded voxels
    recon_mask = (x_reconstructed[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1) != 0)  # (Batch_size, num_voxels)

    for coords, mask in zip(recon_coords, recon_mask):
        # Scale, round, and clamp to grid space range
        scaled_coords = coords * (EXPANDED_GRID_SIZE - 1)
        rounded_coords = torch.clamp(torch.round(scaled_coords), min=0, max=EXPANDED_GRID_SIZE - 1)

        # Mask padded voxels with 'inf'
        rounded_coords[~mask, :] = float('inf')

        # Tracks what has been found to not double count
        overlapping_idx = set()

        # Iterate over each coordinate set (x,y,z) and check for matching
        for i in range(rounded_coords.size(0)):
            current_coord = rounded_coords[i]

            # Skip if padded (masked with inf) or already found overlapping
            if torch.isinf(current_coord).any() or i in overlapping_idx:
                continue

            # Find matching coordinates (overlaps)
            matches = (rounded_coords == current_coord).all(dim=1)
            match_indices = matches.nonzero(as_tuple=True)[0]  # Returns indices of True values as a 1D tensor

            # Calculate pairwise distance, and take the negative sum of the log of the upper triangle of the distance matrix (excluding the diagonal)
            if match_indices.numel() > 1:
                dist = torch.cdist(coords[match_indices, :], coords[match_indices, :])

                total_penalty += -torch.triu(torch.log(dist + 1e-8), diagonal=1).sum()  # Avoid log(0)

                # Add to set to not check matched coordinates
                overlapping_idx.update(match_indices.tolist())

    batch_penalty = total_penalty / x_reconstructed.size(0)  # Averaged over batch

    return batch_penalty


class VaeLoss:
    def __init__(self, lambda_coord: float, lambda_desc: float, lambda_collapse: float, lambda_reg: float = 0.001):
        """
        Initialise VAE Loss with lambda scaling terms.

        :param lambda_coord: Scaling factor of coordinate matching loss (clustering around original points, and nearest neighbour weighted pairwise distance)
        :param lambda_desc: Scaling factor of descriptor loss (cross entropy loss based on raw logits and nearest original voxel descriptor)
        :param lambda_collapse: Scaling factor of overlap penalty (negative sum of the log of the upper triangle of the distance matrix (excluding diagonal) of reconstructed voxel coordinates that would collapse into a single point when scaled)
        :param lambda_reg: Scaling factor of transformation regularising term
        """
        self.lambda_coord = lambda_coord
        self.lambda_desc = lambda_desc
        self.lambda_collapse = lambda_collapse
        self.lambda_reg = lambda_reg

    def __call__(self, x: torch.Tensor, x_reconstructed: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor, beta: float, transform_matrix: torch.Tensor) -> tuple[torch.Tensor, dict[str:float]]:
        """
        Calculates VAE loss (reconstruction loss + beta * KL divergence)
        Reconstruction loss is the sum of coordinate match loss, descriptor loss, padded penalty,
        collapse penalty, and transform regularising term (all scaled).

        KL divergence is averaged across batch size.

        Returned loss parts dictionary contains the following keys with float values:
        - 'recon_loss'
        - 'coor_match_loss'
        - 'scaled_coor_match_loss'
        - 'desc_loss'
        - 'scaled_desc_loss'
        - 'collapse_penalty'
        - 'scaled_collapse_penalty'
        - 'transform_reg'
        - 'kl'
        - 'beta_kl'

        :param x: Input tensor with shape (batch_size, *input_dim)
        :param x_reconstructed: Decoder output with shape (batch_size, *input_dim)
        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :param beta: Beta to scale KL divergence
        :param transform_matrix: Transformation matrix used to calculate regularisation term to encourage orthogonality
        :return: Total loss, Loss parts dictionary
        """
        # Calculate losses and penalties
        coor_match_loss = coordinate_matching_loss(x, x_reconstructed)
        desc_loss = descriptor_matching_loss(x, x_reconstructed)
        collapse_penalty = overlap_penalty(x_reconstructed)

        # Regularisation term to encourage orthogonality - based on https://medium.com/@itberrios6/point-net-for-classification-968ca64c57a9
        batch_size = transform_matrix.size(0)
        identity = torch.eye(transform_matrix.size(-1)).to(DEVICE)
        transform_reg = torch.linalg.norm(identity - torch.bmm(transform_matrix, transform_matrix.transpose(2, 1)))
        transform_reg = transform_reg / batch_size

        recon_loss = (
                self.lambda_coord * coor_match_loss +
                self.lambda_desc * desc_loss +
                self.lambda_collapse * collapse_penalty +
                self.lambda_reg * transform_reg
        )

        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x.shape[0]  # Averaged across batch size

        total_loss = recon_loss + beta * kl_div

        loss_parts = {
            'recon_loss': recon_loss.item(),
            'coor_match_loss': coor_match_loss.item(),
            'scaled_coor_match_loss': (self.lambda_coord * coor_match_loss).item(),
            'desc_loss': desc_loss.item(),
            'scaled_desc_loss': (self.lambda_desc * desc_loss).item(),
            'collapse_penalty': collapse_penalty.item(),
            'scaled_collapse_penalty': (self.lambda_collapse * collapse_penalty).item(),
            'transform_reg': transform_reg.item(),
            'scaled_transform_reg': (self.lambda_reg * transform_reg).item(),
            'kl': kl_div.item(),
            'beta_kl': (beta * kl_div).item()
        }
        
        return total_loss, loss_parts
