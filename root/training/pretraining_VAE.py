import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from models import LinearVAE  # Use VAE instead of LinearAE
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence_VAE, determine_threshold, evaluate_anomalies
from GitFile.root.data.data import load_data, load_anomaly_data
from GitFile.root.config import defaultsVAE, data_paths
from utils import seed_all
from loss import vae_loss  # Loss function that combines BCE and KL divergence

"""
pretraining_VAE.py

Performs a grid search using a Linear Variational Autoencoder (LinearVAE) over combinations of real and simulated data.

Each configuration is evaluated for anomaly detection performance using standard metrics
(F1, accuracy, precision, recall). Models are pretrained on simulated data and optionally fine-tuned on real data.

Functions:
- run_pretraining_grid: Executes training/evaluation for one seed across real/sim split combinations.
- __main__: Launches grid search across multiple seeds and saves evaluation plots and CSVs.
"""


def run_pretraining_grid(seed):
    """
        Runs one grid sweep over real/sim combinations for a given seed using a LinearVAE.

        The model is trained using simulated and/or real data and evaluated on test + anomaly sets.
        Results are collected for each configuration (real, sim).

        Args:
            seed (int): Random seed to control reproducibility.

        Returns:
            list[dict]: Metrics per (real, sim) configuration.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])

    window_size = defaultsVAE["window_size"]
    step_size = defaultsVAE["step_size"]
    input_dim = window_size * train_data[0].shape[-1]
    batch_size = defaultsVAE["batch_size"]
    latent_dim = defaultsVAE["latent_dim"]
    loss_fn = None  # not used directly here since we call vae_loss
    dropout = defaultsVAE["dropout"]
    lr = defaultsVAE["lr"]
    weight_decay = defaultsVAE["weight_decay"]
    std_factor = defaultsVAE["std_factor"]

    # Here, use the epoch numbers from your VAE config
    sim_epochs = defaultsVAE["sim_epochs"]  # simulated pretraining epochs
    real_epochs = defaultsVAE["real_epochs"]  # real-data fine-tuning epochs

    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size), batch_size=batch_size,
                             shuffle=True)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size), batch_size=batch_size,
                             shuffle=True)

    results = []
    real_options = [0, 10, 15, 20, 30]
    max_sim = 100

    for real_size in real_options:
        if real_size > 0:
            real_loader = DataLoader(SlidingWindowDataset(train_data[:real_size], window_size, step_size),
                                     batch_size=batch_size, shuffle=True)

        for sim_size in range(0, max_sim + 1, 2):
            if real_size == 0 and sim_size == 0:
                continue
            elif real_size == 0:
                # If only simulated data is used, fine-tuning on real data is skipped.
                current_real_epochs, current_sim_epochs = 0, sim_epochs
                # Optionally, adjust lr or other hyperparameters here if needed.
            elif sim_size == 0:
                current_real_epochs, current_sim_epochs = real_epochs, 0
            else:
                current_real_epochs, current_sim_epochs = real_epochs, sim_epochs

            model = LinearVAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Pretraining on simulated data
            if sim_size > 0:
                if sim_size > len(sim_data):
                    raise ValueError(f"Sim size {sim_size} exceeds available simulated data {len(sim_data)}.")
                sim_loader = DataLoader(SlidingWindowDataset(sim_data[:sim_size], window_size, step_size),
                                        batch_size=batch_size, shuffle=True)
                model.train()
                for _ in range(current_sim_epochs):
                    for batch, _ in sim_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        recon, mu, logvar = model(batch)
                        loss = vae_loss(recon, batch, mu, logvar)
                        loss.backward()
                        optimizer.step()

            # Fine-tuning on real data
            if real_size > 0:
                model.train()
                for _ in range(current_real_epochs):
                    for batch, _ in real_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        recon, mu, logvar = model(batch)
                        loss = vae_loss(recon, batch, mu, logvar)
                        loss.backward()
                        optimizer.step()

            # Evaluation
            model.eval()
            normal_losses = compute_loss_per_sequence_VAE(model, val_loader, device)
            anomaly_losses = compute_loss_per_sequence_VAE(model, anom_loader, device)
            test_losses = compute_loss_per_sequence_VAE(model, test_loader, device)
            std_loss = np.mean(normal_losses)
            threshold = determine_threshold(normal_losses, std_factor=std_factor)
            _, acc, prec, rec, f1 = evaluate_anomalies(test_losses, anomaly_losses, threshold)
            del model
            torch.cuda.empty_cache()
            model = None

            # Only record valid model configurations
            if acc != 0 and prec != 0 and rec != 0 and f1 != 0:
                results.append({
                    "real": real_size,
                    "sim": sim_size,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "seed": seed,
                    "std_loss": std_loss
                })
            else:
                continue

    return results


import numpy as np

if __name__ == "__main__":
    seed_options = [42, 32, 132, 42, 13, 17, 42, 87, 133, 205, 314, 512, 777, 901, 1024, 56, 420, 32789, 7, 986]
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib

    with tqdm_joblib(tqdm(desc="Running Seeds", total=len(seed_options))):
        all_results = Parallel(n_jobs=5)(
            delayed(run_pretraining_grid)(s) for s in seed_options
        )

    # Flatten the results from all seeds
    results = [r for group in all_results for r in group]
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv("results/pretraining_sweep_results_vae.csv", index=False)

    for metric in ["f1", "precision", "recall"]:
        df.loc[df[metric] == 0.0, metric] = np.nan

    metrics = ["f1", "recall", "accuracy", "precision", "std_loss"]
    titles = {
        "f1": "F1 Score",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "std_loss": "Reconstruction Loss"
    }

    from matplotlib.ticker import MultipleLocator
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 14})

    for metric in metrics:
        # Group data, ignoring NaNs
        grouped = df.groupby(["real", "sim"])[metric].agg(["mean", "std", "count"]).reset_index()
        if metric == "std_loss":
            grouped = grouped[grouped["real"] != 0]  # drop real=0 for loss graph as it skews the scale
        plt.figure(figsize=(12, 6))
        for real_val in sorted(grouped["real"].unique()):
            sub = grouped[grouped["real"] == real_val].sort_values("sim")
            plt.plot(sub["sim"], sub["mean"], label=f"Real = {real_val}")
            plt.fill_between(sub["sim"],
                             sub["mean"] - 0.5 * sub["std"],
                             sub["mean"] + 0.5 * sub["std"],
                             alpha=0.2)
        plt.xlabel("Number of Simulated Samples Used", fontsize=16)
        plt.ylabel(titles[metric], fontsize=16)
        plt.legend(title="Real Samples", fontsize=14, title_fontsize=16)
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"results/final_{metric}_VAE.png", dpi=300)