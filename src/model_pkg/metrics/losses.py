"""
Defines loss functions
"""

import torch
import torch.nn.functional as F
from ..config import COORDINATE_DIMENSIONS, DEVICE, EXPANDED_GRID_SIZE


def coordinate_matching_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a penalty based spatial alignment between original and reconstructed coordinates.

    Consists of two parts:
    1. Clustering penalty - Penalises clustering of multiple reconstructed points around the same original point.
        Measured using the entropy of the distribution formed based on each original points attention.
        Lower entropy suggests clustering, so the max entropy is calculated based on a uniform distribution,
        and the entropy is subtracted from this value, scaled to prevent over penalising when the model over
        predicts non-padded voxels.
    2. Distance penalty - The pairwise distance matrix is weighted by the likelihood of nearest neighbours, and scaled
        by the average non-padded voxels between the original and reconstructed samples.

    :param x: Original input
    :param x_reconstructed: Reconstructed
    :return: Penalty averaged over batch
    """
    total_penalty = torch.tensor(0.0, device=DEVICE)

    orig_coords = x[:, :, :COORDINATE_DIMENSIONS]
    recon_coords = x_reconstructed[:, :, :COORDINATE_DIMENSIONS]

    # Masks for non-padded voxels
    orig_mask = (x[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1) != 0)  # (Batch_size, num_voxels)
    recon_mask = (x_reconstructed[:, :, COORDINATE_DIMENSIONS:].argmax(dim=-1) != 0)

    for orig, recon, o_mask, r_mask in zip(orig_coords, recon_coords, orig_mask, recon_mask):
        # Skip if no non-padded voxels
        if o_mask.sum() == 0 or r_mask.sum() == 0:
            continue

        # Used to scale penalties
        avg_non_padded = (r_mask.sum() + o_mask.sum()) / 2.0

        # Pairwise distance matrix between all original and reconstructed coordinates
        dist_matrix = torch.cdist(recon, orig, p=2)  # (rows: distances from reconstructed, columns: distances from original)

        # Apply masks to ignore padded voxels
        masked_dist_matrix = dist_matrix.clone()  # Avoid modifying matrix directly (causes error) with autograd
        masked_dist_matrix[~r_mask, :] = float('inf')  # Mask padded rows (reconstructed)
        masked_dist_matrix[:, ~o_mask] = float('inf')  # Mask padded columns (original)

        # Gets probability of the closest point, softmin is differentiable
        neighbour_weights = F.softmin(masked_dist_matrix * 100, dim=1)  # Scaling distance encourages model to focus on a single point
        neighbour_weights = torch.nan_to_num(neighbour_weights)  # Replace nan with 0

        # Sum of attention each original point receives from all reconstructed points
        attention_per_original = neighbour_weights.sum(dim=0)  # (num_original_points)

        # Normalise attention to form a probability distribution over original points
        attention_distribution = attention_per_original / (attention_per_original.sum() + 1e-8)  # Avoid division by 0

        # Shannon entropy for attention distribution (amount of uncertainty) - clustered points will have lower entropy
        entropy = -(attention_distribution * torch.log(attention_distribution + 1e-8)).sum()  # log(0) is undefined

        # Maximum entropy
        # Happens when distribution is uniform: p(x) = 1/n
        # H(x)max = -sum_{i..N}((1/n)*log(1/n))
        # log(1/n) = -log(n)
        # H(x)max = log(n)
        max_entropy = torch.log(torch.tensor(attention_distribution.size(0), device=DEVICE))

        # Clustered points will have lower entropy
        # Scaled to prevent over penalising when the model over predicts non-padded voxels
        voxel_diff = F.smooth_l1_loss(r_mask.sum().float(), o_mask.sum().float())  # Differentiable absolute function
        cluster_scale = 1 + (voxel_diff / (avg_non_padded + 1e-8))  # Avoid division by 0
        clustering_penalty = (max_entropy - entropy) * cluster_scale

        # Distance penalty, balanced for incorrect number of voxels
        masked_dist_matrix[torch.isinf(masked_dist_matrix)] = 0.0  # Avoid 'inf' in calculations
        distance_penalty = (neighbour_weights * masked_dist_matrix).sum() / (avg_non_padded + 1e-8)  # Avoid division by zero

        total_penalty += clustering_penalty + distance_penalty

    batch_penalty = total_penalty / x.size(0)  # Averaged over batch

    return batch_penalty


def descriptor_matching_loss(x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Calculates a loss for descriptor values based on nearest neighbour.
    Calculates the pairwise distance between original and reconstructed points.
    Padded voxels are then masked, before obtaining the most likely descriptor value based on nearest original voxel.
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

    # Masks for non-padded voxels
    orig_mask = (orig_descriptors.argmax(dim=-1) != 0)  # (Batch_size, num_voxels)
    recon_mask = (recon_descriptors.argmax(dim=-1) != 0)

    for orig_coor, recon_coor, orig_desc, recon_desc, o_mask, r_mask in zip(orig_coords, recon_coords, orig_descriptors, recon_descriptors, orig_mask, recon_mask):
        # Skip if no valid voxels
        if o_mask.sum() == 0 or r_mask.sum() == 0:
            continue

        # Pairwise distance matrix between all original and reconstructed coordinates
        dist_matrix = torch.cdist(recon_coor, orig_coor, p=2)  # (rows: distances from reconstructed, columns: distances from original)

        # Apply masks to ignore padded voxels
        masked_dist_matrix = dist_matrix.clone()  # Avoid modifying matrix directly (causes error) with autograd
        masked_dist_matrix[~r_mask, :] = float('inf')  # Mask padded rows (reconstructed)
        masked_dist_matrix[:, ~o_mask] = float('inf')  # Mask padded columns (original)

        # Get probability of the closest point, softmin is differentiable
        neighbour_weights = F.softmin(masked_dist_matrix * 100, dim=1)  # Scaling distance helps focus on a single point
        neighbour_weights = torch.nan_to_num(neighbour_weights)  # Replace nan with 0

        # Matrix multiplication to get predicted descriptors
        matched_target_descriptors = torch.matmul(neighbour_weights, orig_desc)  # (recon_voxels, descriptor_dim)

        # Extract the most likely class index for each matched descriptor
        target_cls = matched_target_descriptors.argmax(dim=-1)  # (recon_voxels)

        # Apply CrossEntropyLoss
        descriptor_loss = F.cross_entropy(recon_desc, target_cls, reduction='mean')

        total_loss += descriptor_loss

    batch_loss = total_loss / x.size(0)  # Average over batch
    return batch_loss


def padded_voxel_penalty(orig_descriptors: torch.Tensor, recon_descriptors: torch.Tensor) -> torch.Tensor:
    """
    Calculates penalty for the smooth L1 (differentiable absolute) padded voxel difference from original input.
    Calculated in a differentiable way to help the model learn.

    :param orig_descriptors: Original batched one-hot descriptors
    :param recon_descriptors: Reconstructed batched logits
    :return: Penalty (mean reduction smooth L1)
    """
    # Class probabilities - softmax is differentiable
    orig_probs = F.softmax(orig_descriptors, dim=-1)
    recon_probs = F.softmax(recon_descriptors, dim=-1)

    # Probability of padded voxel
    orig_padded_prob = orig_probs[:, :, 0]
    recon_padded_prob = recon_probs[:, :, 0]

    # Difference in expected number of padded voxels
    penalty = F.smooth_l1_loss(orig_padded_prob.sum(dim=1), recon_padded_prob.sum(dim=1), reduction='mean')  # Differentiable absolute function

    return penalty


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
        # Scale, round, and clamp to grid space
        scaled_coords = coords * (EXPANDED_GRID_SIZE - 1)
        rounded_coords = torch.clamp(torch.round(scaled_coords), min=0, max=EXPANDED_GRID_SIZE - 1)

        # Mask padded voxels with 'inf'
        rounded_coords[~mask, :] = float('inf')

        # Tracks what has been found to not double count
        overlapping_idx = set()

        # Iterate over each coordinate and check for matching
        for i in range(rounded_coords.size(0)):
            current_coord = rounded_coords[i]

            # Skip if padded (masked with inf) or already found overlapping
            if torch.isinf(current_coord).any() or i in overlapping_idx:
                continue

            # Find matching coordinates (overlaps)
            matches = (rounded_coords == current_coord).all(dim=1)
            match_indices = matches.nonzero(as_tuple=True)[0]

            # Calculate pairwise distance, and take the negative sum of the log of the upper triangle of the distance matrix (excluding the diagonal)
            if match_indices.numel() > 1:
                dist = torch.cdist(coords[match_indices, :], coords[match_indices, :])
                total_penalty += -torch.triu(torch.log(dist + 1e-8), diagonal=1).sum()  # Avoid log(0)

                # Add to set to not check matched coordinates
                overlapping_idx.update(match_indices.tolist())

    batch_penalty = total_penalty / x_reconstructed.size(0)  # Averaged over batch

    return batch_penalty


class VaeLoss:
    def __init__(self, lambda_coord: float, lambda_desc: float, lambda_pad: float, lambda_collapse: float, lambda_reg: float = 0.001):
        """
        Initialise VAE Loss with lambda scaling terms.

        :param lambda_coord: Scaling factor of coordinate matching loss (clustering around original points, and nearest neighbour weighted pairwise distance)
        :param lambda_desc: Scaling factor of descriptor loss (cross entropy loss based on raw logits and nearest original voxel descriptor)
        :param lambda_pad: Scaling factor of padded voxel penalty (smooth L1 (differentiable absolute) padded voxel difference from original input)
        :param lambda_collapse: Scaling factor of overlap penalty (negative sum of the log of the upper triangle of the distance matrix (excluding diagonal) of reconstructed voxel coordinates that would collapse into a single point when scaled)
        :param lambda_reg: Scaling factor of transformation regularising term
        """
        self.lambda_coord = lambda_coord
        self.lambda_desc = lambda_desc
        self.lambda_pad = lambda_pad
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
        - 'desc_loss': desc_loss.item()
        - 'scaled_desc_loss'
        - 'pad_penalty'
        - 'scaled_pad_penalty'
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
        pad_penalty = padded_voxel_penalty(x[:, :, COORDINATE_DIMENSIONS:], x_reconstructed[:, :, COORDINATE_DIMENSIONS:])
        collapse_penalty = overlap_penalty(x_reconstructed)

        # Regularisation term to encourage orthogonality - based on https://medium.com/@itberrios6/point-net-for-classification-968ca64c57a9
        batch_size = transform_matrix.size(0)
        identity = torch.eye(transform_matrix.size(-1)).to(DEVICE)
        transform_reg = torch.linalg.norm(identity - torch.bmm(transform_matrix, transform_matrix.transpose(2, 1)))
        transform_reg = transform_reg / batch_size

        recon_loss = (
                self.lambda_coord * coor_match_loss +
                self.lambda_desc * desc_loss +
                self.lambda_pad * pad_penalty +
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
            'pad_penalty': pad_penalty.item(),
            'scaled_pad_penalty': (self.lambda_pad * pad_penalty).item(),
            'collapse_penalty': collapse_penalty.item(),
            'scaled_collapse_penalty': (self.lambda_collapse * collapse_penalty).item(),
            'transform_reg': transform_reg.item(),
            'scaled_transform_reg': (self.lambda_reg * transform_reg).item(),
            'kl': kl_div.item(),
            'beta_kl': (beta * kl_div).item()
        }

        return total_loss, loss_parts
