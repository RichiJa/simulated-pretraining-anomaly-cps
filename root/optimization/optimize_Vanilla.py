import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models import LinearAE
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence, determine_threshold
from GitFile.root.data.data import load_data, load_anomaly_data
from GitFile.root.config import defaults, data_paths
from utils import seed_all

"""
optimize_Vanilla.py

Hyperparameter optimization script for training a Linear Autoencoder
on real-world time-series data for anomaly detection.

This script uses Optuna to optimize architectural and training hyperparameters
to maximize the F1 score for classifying normal vs. anomalous sequences.

Functions:
- objective: Defines the training and evaluation process for a single trial.
- __main__: Executes the optimization and prints the best trial result.
"""

def objective(trial):
    """
        Optuna objective function for tuning a Linear Autoencoder.

        The model is trained on real data, and its performance is evaluated by computing
        sequence-level reconstruction losses and classifying anomalies based on a dynamic threshold.

        Args:
            trial (optuna.Trial): Trial object to suggest hyperparameters.

        Returns:
            float: F1 score of the model's anomaly detection performance.
    """

    seed_all(42)
    # --- Optimized hyperparameters ---
    window_size = defaults["window_size"]
    step_size = defaults["step_size"]
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    real_epochs = trial.suggest_int("real_epochs", 15, 30)
    std_factor = trial.suggest_float("std_factor", 0.5, 3.0)

    # --- Load data ---
    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    input_dim = window_size * train_data[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = LinearAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # --- Dataloaders ---
    train_loader = DataLoader(SlidingWindowDataset(train_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size), batch_size=batch_size, shuffle=True)

    # --- Training ---
    model.train()
    for _ in range(real_epochs):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    normal_losses = compute_loss_per_sequence(model, val_loader, device)
    anomaly_losses = compute_loss_per_sequence(model, anom_loader, device)
    threshold = determine_threshold(normal_losses, std_factor=std_factor)
    test_losses = compute_loss_per_sequence(model, test_loader, device)

    y_true = [0] * len(test_losses) + [1] * len(anomaly_losses)
    y_pred = [1 if l > threshold else 0 for l in test_losses + anomaly_losses]

    return f1_score(y_true, y_pred)


# --- Run the optimization ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1)

    print("Best trial:")
    print(study.best_trial)
