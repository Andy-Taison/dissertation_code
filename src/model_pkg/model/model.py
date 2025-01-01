"""
Defines VAE model modules
"""

import torch
import torch.nn as nn

class Encoder(nn.Modele):
    """
    Encoder module.
    """
    def __init__(self, input_dim: tuple[int, int, int], latent_dim: int):
        """
        :param input_dim: Dimensions of input tensor (and reconstructed), (e.g., (11, 11, 11))
        :param latent_dim: Dimensionality of latent space
        """
        super(Encoder, self).__init__()
        self.flatten = nn.flatten()
        self.fc1 = nn.Linear(input_dim[0] * input_dim[1] * input_dim[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.z_mean = nn.Linear(256, latent_dim)
        self.z_log_var = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
        """
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var


class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, latent_dim: int, output_dim: tuple[int, int, int]):
        """
        :param latent_dim: Dimensionality of latent space
        :param output_dim: Dimensions of reconstructed output tensor, (e.g., (11, 11, 11))
        """
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim[0] * output_dim[1] * output_dim[2])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns a normalized output in range [0, 1].

        :param z: Sampled latent vector with shape (batch_size, latent_dim)
        :return: Reconstructed input with shape (batch_size, *input_dim)
        """
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Normalize output to [0, 1]
        return x


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
    def __init__(self, input_dim: tuple[int, int, int], latent_dim: int):
        """
        :param input_dim: Dimensions of input tensor (and reconstructed), (e.g., (11, 11, 11))
        :param latent_dim: Dimensionality of latent space
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.sampling = Sample()
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Scales, rounds, clamps and reshapes decoder output to match original shape and descriptor values.
        Note rounding and clamping (partially) are not differentiable, so use x_decoder for loss/backpropagation.

        :param x: Input tensor with shape (batch_size, *input_dim)
        :return:
            x_reconstructed - Reconstructed input with shape (batch_size, *input_dim),
            x_decoder - Decoder output reshaped to (batch_size, *input_dim), (sigmoid) normalized in range [0, 1],
            z - Sampled latent vector with shape (batch_size, latent_dim),
            z_mean - Latent space mean with shape (batch_size, latent_dim),
            z_log_var - Log variance of latent space with shape (batch_size, latent_dim)
        """
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        x_decoder = self.decoder(z)

        x_decoder = x_decoder.view(-1, *self.output_dim)  # Reshape to match original dimensions

        # Adjust values to obtain original descriptor values
        x_reconstructed = x_decoder * 4  # Scale output to [0, 4]
        x_reconstructed = torch.round(x_reconstructed)  # Round to the nearest integer
        x_reconstructed = torch.clamp(x_reconstructed, min=0, max=4)  # Ensure range is [0, 4]

        return x_reconstructed, x_decoder, z, z_mean, z_log_var

