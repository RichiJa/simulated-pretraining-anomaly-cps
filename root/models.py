# models.py
import torch.nn as nn
import torch

"""
models.py

Defines neural network architectures for anomaly detection using Autoencoders:

Classes:
- LinearAE: A fully connected (MLP) autoencoder.
- LinearVAE: A fully connected Variational Autoencoder with reparameterization.
"""

class LinearAE(nn.Module):
    """
        A fully connected (MLP-based) autoencoder for reconstruction tasks.

        Encoder:
            - Linear -> ReLU -> Dropout -> Linear -> ReLU

        Decoder:
            - Linear -> ReLU -> Linear -> Sigmoid

        Args:
            input_dim (int): Total input dimension (e.g., window_size × num_features).
            latent_dim (int): Size of the compressed latent space.
            dropout_prob (float): Dropout probability applied during encoding.
    """

    def __init__(self, input_dim, latent_dim=128, dropout_prob=0.0):
        super(LinearAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


import torch
import torch.nn as nn




class LinearVAE(nn.Module):
    """
       A fully connected (MLP-based) Variational Autoencoder.

       Encoder:
           - Linear -> ReLU -> Dropout
           - Outputs mean (mu) and log variance (logvar) for reparameterization.

       Decoder:
           - Linear -> ReLU -> Linear -> Sigmoid

       Args:
           input_dim (int): Input dimension of each sample.
           latent_dim (int): Dimension of latent space.
           dropout_prob (float): Dropout applied after initial encoding layer.
    """
    def __init__(self, input_dim, latent_dim=128, dropout_prob=0.0):
        super(LinearVAE, self).__init__()
        # Encoder: two-layer MLP with dropout applied before the bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # Separate linear layers to produce mean and log-variance for the latent distribution
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: mirror architecture of the encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Ensures outputs lie between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        """
            Performs the reparameterization trick to sample from N(mu, sigma^2).

            Args:
                mu (Tensor): Mean of the latent distribution.
                logvar (Tensor): Log variance of the latent distribution.

            Returns:
                Tensor: Sampled latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
            Forward pass through the VAE.

            Args:
                x (Tensor): Input tensor.

            Returns:
                Tuple[Tensor, Tensor, Tensor]:
                    - Reconstructed output
                    - Mean of latent distribution
                    - Log variance of latent distribution
        """
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
