"""
Defines VAE model modules
"""

import torch
import torch.nn as nn
from ..config import DEVICE

class TNet(nn.Module):
    """
    Transformation Net.
    Learns a transformation matrix that is applied to each point helping to make the network invariant to geometric
    variations e.g. rotation, translation and scaling.

    Simplified in comparison to pointnet due to small point cloud.
    """
    def __init__(self, in_features: int):  # (x, y, z, one-hot descriptors)
        """
        :param in_features: Number of input features (x, y, z, one-hot descriptors)
        """
        super(TNet, self).__init__()
        self.features = in_features

        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, in_features * in_features)
        )

        # Initialise as identity matrix for stability
        self.eye = torch.eye(in_features).flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return: Transformed input with shape (batch_size, in_features, in_features)
        """
        batch_size = x.size(0)
        transform = self.fc(x.mean(dim=1))  # Learns transformation matrix from average global features
        transform = transform.view(batch_size, self.features, self.features) + self.eye.to(DEVICE).view(1, self.features, self.features)

        # Apply transformation
        transformed_x = torch.bmm(x, transform)

        return transformed_x


class Encoder(nn.Module):
    """
    Encoder module.
    """
    def __init__(self, latent_dim: int, in_features: int):
        """
        :param latent_dim: Dimensionality of latent space
        :param in_features: Number of input features (x, y, z, one-hot descriptors)
        """
        super(Encoder, self).__init__()
        self.tnet = TNet(in_features)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),  # (x, y, z, one-hot descriptors)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU()
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(1024 + 1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.z_mean_fc = nn.Linear(256, latent_dim)
        self.z_log_var_fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
        """
        # T-Net used for spatial alignment
        x = self.tnet(x)

        # Per point features
        local_features = self.mlp(x)

        # Symmetric max pooling for global features, retains only values and drops indices
        global_features = torch.max(local_features, dim=1, keepdim=True)[0]  # (B, 1, 1024)

        # Concatenate global features with local features
        combined_features = torch.cat([local_features, global_features.expand(-1, local_features.size(1), -1)], dim=2)

        combined_out = self.combined_mlp(combined_features)

        avg_global = torch.mean(combined_out, dim=1)

        z_mean = self.z_mean_fc(avg_global)
        z_log_var = self.z_log_var_fc(avg_global)

        return z_mean, z_log_var


class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, latent_dim: int, out_features: int):
        """
        :param latent_dim: Dimensionality of latent space
        :param out_features: Number of output features - should match in features (x, y, z, raw descriptors)
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024 + latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)  # (x, y, z, raw descriptors)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Normalise coordinates using sigmoid.
        Descriptor values are raw logits.

        :param z: Sampled latent vector with shape (batch_size, latent_dim)
        :return: Reconstructed input with shape (batch_size, *input_dim)
        """
        upsampled = self.fc1(z)

        # Mirrors encoder`s mean operation
        expanded = upsampled.unsqueeze(1).expand(-1, 8, -1)

        global_features = self.fc2(expanded)

        # Concatenate with latent vector to mirror encoders local and global feature separation
        latent_expanded = z.unsqueeze(1).expand(-1, 8, -1)
        combined = torch.cat([global_features, latent_expanded], dim=-1)  # Performed along last dimension

        # Per point reconstruction
        x_recon = self.mlp(combined)

        # Sigmoid applied to coordinates to normalise
        x_recon[:, :, :3] = torch.sigmoid(x_recon[:, :, :3])

        # Raw logits for descriptors
        return x_recon


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

    Attributes:
        encoder: Input to latent space (z_mean, z_log_var)
        sampling: Reparameterization trick to sample z
        decoder: Reconstructs input from sampled z
    """
    def __init__(self, input_dim: tuple[int, int], latent_dim: int, model_name: str):
        """
        :param input_dim: Dimensions of input tensor (and reconstructed), (e.g., (8, 8))
        :param latent_dim: Dimensionality of latent space
        :param model_name: Name of model instance, used to identify models and name saved files, ENSURE UNIQUE to avoid overwriting
        """
        super(VAE, self).__init__()
        self.name = model_name
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim, input_dim[1])
        self.sampling = Sample()
        self.decoder = Decoder(latent_dim, input_dim[1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            x_reconstructed - Reconstructed input with shape (batch_size, *input_dim),
            z - Sampled latent vector with shape (batch_size, latent_dim),
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
        """
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)

        return x_reconstructed, z, z_mean, z_log_var
