import torch
import torch.nn.functional as F

"""
loss.py

Defines the loss function used for training Variational Autoencoders (VAEs).

Functions:
- vae_loss: Computes the combined reconstruction loss (BCE) and KL divergence.
"""

def vae_loss(recon_x, x, mu, logvar):
    """
    Computes the loss for a Variational Autoencoder (VAE).

    The loss consists of two parts:
    1. Binary Cross-Entropy (BCE) between the reconstructed input and the original input.
    2. Kullback-Leibler Divergence (KLD) between the learned latent distribution and the standard normal distribution.

    Args:
        recon_x (Tensor): Reconstructed input from the decoder, shape (B, D).
        x (Tensor): Original input tensor, shape (B, D).
        mu (Tensor): Mean of the latent Gaussian, shape (B, Z).
        logvar (Tensor): Log variance of the latent Gaussian, shape (B, Z).

    Returns:
        Tensor: Scalar loss value (BCE + KLD).
    """
    # Binary Cross-Entropy (reconstruction loss)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # Kullback-Leibler Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Example usage:
# recon_x, mu, logvar = model(input_data)
# loss = vae_loss(recon_x, input_data, mu, logvar)
