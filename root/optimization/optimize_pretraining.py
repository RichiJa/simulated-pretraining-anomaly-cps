import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import LinearAE
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence, determine_threshold
from GitFile.root.data.data import load_data, load_anomaly_data
from GitFile.root.config import defaults, data_paths
from utils import seed_all
from sklearn.metrics import f1_score
from train import train_autoencoder

"""
optimize_pretraining.py

This script runs hyperparameter optimization using Optuna to evaluate the effectiveness of
pretraining a Linear Autoencoder on simulated data before fine-tuning on real-world data.

It optimizes:
- Number of pretraining epochs
- Dropout rate
- Learning rate
- Weight decay
- Thresholding std factor

Evaluation is based on F1-score over normal vs. anomalous reconstruction losses.

Functions:
- objective: The Optuna objective function for evaluating one trial.
- __main__: Launches the optimization process and saves the study.
"""
def objective(trial):
    """
        Optuna objective function to optimize pretraining and fine-tuning setup
        for a Linear Autoencoder model.

        The model is first pretrained on simulated data, then fine-tuned on real data.
        The validation set is used to determine a reconstruction loss threshold, and
        the final F1-score is computed using test and anomaly data.

        Args:
            trial (optuna.Trial): A trial object for suggesting hyperparameters.

        Returns:
            float: F1 score for the current trial.
    """
    seed_all(42)
    total_epochs = 17

    # --- Optimized hyperparameters ---
    pre_epochs = trial.suggest_int("pre_epochs", 5, 20)
    fine_epochs = 17
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    std_factor = trial.suggest_float("std_factor", 0.5, 3.0)

    latent_dim = defaults["latent_dim"]
    batch_size = defaults["batch_size"]
    window_size = defaults["window_size"]
    step_size = defaults["step_size"]

    # --- Load data ---
    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    input_dim = window_size * train_data[0].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = LinearAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    # --- Dataloaders ---
    sim_loader = DataLoader(SlidingWindowDataset(sim_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    real_loader = DataLoader(SlidingWindowDataset(train_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size), batch_size=batch_size, shuffle=False)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size), batch_size=batch_size, shuffle=False)

    # --- Training ---
    if pre_epochs > 0:
        model = train_autoencoder(model, sim_loader, optimizer, loss_fn, epochs=pre_epochs, device=device)
    if fine_epochs > 0:
        model = train_autoencoder(model, real_loader, optimizer, loss_fn, epochs=fine_epochs, device=device)

    # --- Evaluation ---
    model.eval()
    val_losses = compute_loss_per_sequence(model, val_loader, device)
    threshold = determine_threshold(val_losses, std_factor=std_factor)

    normal_losses = compute_loss_per_sequence(model, test_loader, device)
    anomaly_losses = compute_loss_per_sequence(model, anom_loader, device)

    y_true = [0] * len(normal_losses) + [1] * len(anomaly_losses)
    y_pred = [1 if l > threshold else 0 for l in normal_losses + anomaly_losses]

    return f1_score(y_true, y_pred)


# --- Run the optimization ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, n_jobs=1)

    print("Best trial:")
    print(study.best_trial)

    # Optional: save the study
    import joblib
    joblib.dump(study, "optuna_pretrain_split_study.pkl")
