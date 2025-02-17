"""
Defines VAE model modules
"""

import torch
import torch.nn as nn
from ..config import DEVICE, NUM_CLASSES, MAX_VOXELS, COORDINATE_DIMENSIONS

class TNet(nn.Module):
    """
    Transformation Net.
    Should only pass the coordinates to TNet.
    Learns a transformation matrix that is applied to each point, helping to make the network invariant to geometric
    variations e.g. rotation, translation and scaling.
    """
    def __init__(self, coordinate_dimensions: int):
        """
        :param coordinate_dimensions: Number of coordinate_dimensions
        """
        super(TNet, self).__init__()
        self.coord = coordinate_dimensions  # Only applied to coordinates

        self.fc = nn.Sequential(
            nn.Linear(self.coord, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.coord * self.coord)
        )

        # Initialise as identity matrix for stability
        self.eye = torch.eye(self.coord).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, coordinate_dimensions)
        :return: Transformation matrix (batch_size, coordinate_dimensions, coordinate_dimensions)
        """
        batch_size = x.size(0)
        transform = self.fc(x.mean(dim=1))  # Learns transformation matrix from average global features
        transform = transform.view(batch_size, self.coord, self.coord) + self.eye.to(DEVICE).view(1, self.coord, self.coord)

        return transform


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)

        return attn_output


class Encoder(nn.Module):
    """
    Encoder module.
    """
    def __init__(self, latent_dim: int, coordinate_dimensions: int = 3):
        """
        :param latent_dim: Dimensionality of latent space
        :param coordinate_dimensions: Number of coordinate dimensions
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.coord = coordinate_dimensions
        self.tnet = TNet(self.coord)

        # MLP for coordinates
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(self.coord, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )

        # MLP for descriptor values (less complex than spatial, so do not need to go as deep)
        self.descriptor_mlp = nn.Sequential(
            nn.Conv1d(NUM_CLASSES, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.combined_activation = nn.LeakyReLU()

        # Reduction before latent space
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )

        self.z_mean_fc = nn.Linear(128, self.latent_dim)
        self.z_log_var_fc = nn.Linear(128, self.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
            transform_matrix - Transforms coordinates for geometric invariance
        """
        # Split coordinates and descriptors
        coord = x[:, :, :self.coord]  # (B, N, coordinates)
        desc = x[:, :, self.coord:]  # (B, N, descriptors)

        # T-Net used for learning spatial alignment matrix - coordinates only
        transform_matrix = self.tnet(coord)
        coord_transformed = torch.bmm(coord, transform_matrix)

        # Transpose for input to 1D convolutional layers
        coord_transposed = coord_transformed.transpose(1, 2)  # (batch, num voxels, coordinate features) -> (batch, coordinate features, num voxels)
        desc_transposed = desc.transpose(1, 2)  # (batch, num voxels, one-hot descriptors) -> (batch, one-hot descriptors, num voxels)

        # Per point features - spatial and descriptors processed separately
        spatial_features = self.spatial_mlp(coord_transposed)
        desc_features = self.descriptor_mlp(desc_transposed)

        # Global feature extraction
        pooled_spatial = self.global_pool(spatial_features).squeeze(-1)
        max_spatial = torch.max(spatial_features, dim=2)[0]
        pooled_desc = self.global_pool(desc_features).squeeze(-1)

        # Concatenate global, local, and descriptor features
        combined_features = self.combined_activation(pooled_spatial + max_spatial + pooled_desc)

        reduced_features = self.fc(combined_features)

        z_mean = self.z_mean_fc(reduced_features)
        z_log_var = self.z_log_var_fc(reduced_features)

        return z_mean, z_log_var, transform_matrix


class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, latent_dim: int, max_voxels: int, coordinate_dimensions: int = 3):
        """
        :param latent_dim: Dimensionality of latent space
        :param max_voxels: Maximum number of voxels per robot in full dataset
        :param coordinate_dimensions: Number of coordinate dimensions
        """
        super(Decoder, self).__init__()
        self.num_voxels = max_voxels
        self.coord = coordinate_dimensions

        # Mirrors encoders mean/logvar branches
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )

        # Upsample coordinates
        self.coord_deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 512, kernel_size=2),  # 1 -> 2
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 1024, kernel_size=2, stride=2),  # 2 -> 4
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(1024, 256, kernel_size=2, stride=2),  # 4 -> 8
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, self.coord, kernel_size=1)  # (B, coord dim, num_voxels) 8 -> 8
        )

        # Refine coordinates
        self.coord_fc = nn.Sequential(
            nn.Linear(self.coord, self.coord),
            nn.LeakyReLU()
        )

        # Upsample descriptors
        self.desc_deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 512, kernel_size=2),  # 1 -> 2
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 128, kernel_size=2, stride=2),  # 2 -> 4
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),  # 4 -> 8
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, NUM_CLASSES, kernel_size=1)  # (B, num descriptors, num_voxels)  8 -> 8
        )

        # Refine descriptors
        self.desc_fc = nn.Sequential(
            nn.Linear(NUM_CLASSES, NUM_CLASSES),
            nn.LeakyReLU()
        )

        # Combined refining
        self.combined_fc = nn.Sequential(
            nn.Linear(self.coord + NUM_CLASSES, self.coord + NUM_CLASSES),
            nn.LeakyReLU()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Normalise coordinates using sigmoid.
        Descriptor values are raw logits.

        :param z: Sampled latent vector with shape (batch_size, latent_dim)
        :return: Reconstructed input with shape (batch_size, *input_dim)
        """
        upsampled_latent = self.fc(z).unsqueeze(-1)  # (B, 256, 1)

        upsampled_coord = self.coord_deconv(upsampled_latent)  # (B, coord dim, num_voxels)
        upsampled_desc = self.desc_deconv(upsampled_latent)  # (B, num descriptors, num_voxels)

        transposed_coord = upsampled_coord.transpose(1, 2)  # (B, num_voxels, coord dim)
        transposed_desc = upsampled_desc.transpose(1, 2)  # (B, num_voxels, descriptor_dim)

        refined_coord = self.coord_fc(transposed_coord)
        refined_desc = self.desc_fc(transposed_desc)

        combined = torch.cat((refined_coord, refined_desc), dim=2)

        x_reconstructed = self.combined_fc(combined)

        # Normalises coordinates
        x_reconstructed[:, :, :self.coord] = torch.sigmoid(x_reconstructed[:, :, :self.coord])

        # Raw logits for descriptors
        return x_reconstructed


class Sample(nn.Module):
    """
    Reparameterization trick to allow backpropagation and sampling.
    """
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param z_mean: Latent space mean with shape (batch_size, latent_dim)
        :param z_log_var: Log variance of latent space with shape (batch_size, latent_dim)
        :return: Sampled latent vector with shape (batch_size, latent_dim)
        """
        epsilon = torch.randn_like(z_mean)

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VAE(nn.Module):
    """
    Implements encoder, sampling (reparameterization trick) and decoder modules for complete VAE architecture.
    """

    def __init__(self, input_dim: tuple[int, int], latent_dim: int, model_name: str, max_voxels: int = MAX_VOXELS, coordinate_dimensions: int = COORDINATE_DIMENSIONS):
        """
        :param input_dim: Dimensions of input tensor (and reconstructed), (e.g., (8, 8))
        :param latent_dim: Dimensionality of latent space
        :param model_name: Name of model instance, used to identify models and name saved files, ENSURE UNIQUE to avoid overwriting
        :param max_voxels: Maximum number of voxels per robot in full dataset
        :param coordinate_dimensions: Number of coordinate dimensions
        """
        super(VAE, self).__init__()
        self.name = model_name
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim, coordinate_dimensions)
        self.sampling = Sample()
        self.decoder = Decoder(latent_dim, max_voxels, coordinate_dimensions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            x_reconstructed - Reconstructed input with shape (batch_size, *input_dim),
            z - Sampled latent vector with shape (batch_size, latent_dim),
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
            transform_matrix - Transforms coordinates for geometric invariance
        """
        z_mean, z_log_var, transform_matrix = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, z_mean, z_log_var, transform_matrix
