import optuna
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from models import LinearVAE  # VAE model (MLP-based VAE)
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence_VAE, determine_threshold
from GitFile.root.data.data import load_data, load_anomaly_data
from GitFile.root.config import defaultsVAE, data_paths  # Use your VAE-specific defaults
from utils import seed_all
from loss import vae_loss  # Custom loss combining BCE and KL divergence

"""
optimize_pretraining_VAE.py

Hyperparameter optimization using Optuna for a Variational Autoencoder (VAE)
trained on simulated and real-world time-series data.

This script optimizes:
- Number of epochs using simulated data (sim_epochs)

Evaluation is based on F1-score by classifying normal vs. anomalous sequences
based on their reconstruction loss and a learned threshold.

Functions:
- objective: The Optuna objective function for evaluating one hyperparameter trial.
- __main__: Runs the optimization loop and prints the best trial.
"""
def objective(trial):
    """
        Objective function for Optuna to optimize VAE training with simulated data.

        Optimizes the number of pretraining epochs on simulated data while keeping
        other hyperparameters fixed from `defaultsVAE`.

        The model is evaluated using F1-score on classifying test and anomaly data.

        Args:
            trial (optuna.Trial): Optuna trial for hyperparameter suggestion.

        Returns:
            float: F1-score for the trial.
    """

    seed_all(42)
    # --- Optimized hyperparameters ---
    window_size = defaultsVAE["window_size"]
    step_size = defaultsVAE["step_size"]
    latent_dim = defaultsVAE["latent_dim"]
    dropout = defaultsVAE["dropout"]
    lr = defaultsVAE["lr"]
    weight_decay = defaultsVAE["weight_decay"]
    batch_size = defaultsVAE["batch_size"]
    real_epochs = defaultsVAE["real_epochs"]
    std_factor = defaultsVAE["std_factor"]
    sim_epochs = trial.suggest_int("sim_epochs", 0, 30)
    # --- Load data ---
    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])
    input_dim = window_size * train_data[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = LinearVAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = vae_loss  # vae_loss should take arguments: (recon, x, mu, logvar)

    # --- Dataloaders ---
    train_loader = DataLoader(SlidingWindowDataset(train_data, window_size, step_size),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size),
                            batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size),
                             batch_size=batch_size, shuffle=True)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size),
                             batch_size=batch_size, shuffle=True)
    sim_loader = DataLoader(SlidingWindowDataset(sim_data, window_size, step_size),
                            batch_size=batch_size, shuffle=True)

    # --- Training ---
    model.train()
    for _ in range(sim_epochs):
        for batch, _ in sim_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = loss_fn(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
    for _ in range(real_epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = loss_fn(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    normal_losses = compute_loss_per_sequence_VAE(model, val_loader, device)
    anomaly_losses = compute_loss_per_sequence_VAE(model, anom_loader, device)
    threshold = determine_threshold(normal_losses, std_factor=std_factor)
    test_losses = compute_loss_per_sequence_VAE(model, test_loader, device)

    y_true = [0] * len(test_losses) + [1] * len(anomaly_losses)
    y_pred = [1 if l > threshold else 0 for l in test_losses + anomaly_losses]

    return f1_score(y_true, y_pred)


# --- Run the optimization ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1)

    print("Best trial (LinearVAE):")
    print(study.best_trial)
