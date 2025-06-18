import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import numpy as np
from matplotlib.ticker import PercentFormatter, MultipleLocator
from GitFile.root.config import defaultsVAE, data_paths  # VAE-specific defaults and data paths
from models import LinearVAE  # VAE model (MLP-based VAE)
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence_VAE, determine_threshold, evaluate_anomalies
from GitFile.root.data.data import load_data, load_anomaly_data
from utils import seed_all
from loss import vae_loss  # Custom VAE loss: expects (recon, x, mu, logvar)

"""
pretraining_fixed_size_VAE.py

Evaluates a Linear Variational Autoencoder (LinearVAE) for anomaly detection using a fixed data budget.

The script investigates the impact of different real-to-simulated data ratios under a total sample constraint.
For each ratio, it trains a LinearVAE and evaluates its ability to detect anomalies.

Functions:
- run_fixed_budget_split: Runs training and evaluation for a single seed across all real/sim split ratios.
- __main__: Runs the experiment for multiple seeds, stores results, and generates plots.
"""

def run_fixed_budget_split(seed):
    """
        Runs a fixed-budget training split experiment using a LinearVAE.

        For a fixed total number of training samples, iterates through different ratios
        of simulated vs. real samples, trains a VAE, and evaluates anomaly detection performance.

        Args:
            seed (int): Seed for reproducibility.

        Returns:
            list[dict]: Each dictionary contains evaluation metrics for a single sim/real split.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets:
    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])

    print("Train:", len(train_data), "Test:", len(test_data), "Val:", len(val_data))
    window_size = defaultsVAE["window_size"]
    step_size = defaultsVAE["step_size"]
    input_dim = window_size * train_data[0].shape[-1]
    batch_size = defaultsVAE["batch_size"]
    latent_dim = defaultsVAE["latent_dim"]
    dropout = defaultsVAE["dropout"]
    lr = defaultsVAE["lr"]
    weight_decay = defaultsVAE["weight_decay"]
    std_factor = defaultsVAE["std_factor"]

    # Create evaluation dataloaders:
    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size),
                            batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size),
                             batch_size=batch_size, shuffle=True)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size),
                             batch_size=batch_size, shuffle=True)

    results = []
    total_budget = 40  # fixed budget of training samples

    # Iterate over different splits of simulated and real training data
    for sim_count in range(0, total_budget + 1, 1):
        real_count = total_budget - sim_count
        if sim_count > len(sim_data) or real_count > len(train_data):
            continue

        # Initialize VAE model
        model = LinearVAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = vae_loss  # VAE loss takes (recon, batch, mu, logvar)

        # Pretraining on simulated data (if any simulated samples are used)
        if sim_count > 0:
            sim_loader = DataLoader(SlidingWindowDataset(sim_data[:sim_count], window_size, step_size),
                                    batch_size=batch_size, shuffle=True)
            # Use a fixed number of pretraining epochs (e.g., 10)
            for epoch in range(defaultsVAE["sim_epochs"]):
                model.train()
                for batch, _ in sim_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(batch)
                    loss = loss_fn(recon, batch, mu, logvar)
                    loss.backward()
                    optimizer.step()

            # (Optional) Reinitialize the optimizer after pretraining
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Fine-tuning on real data
        if real_count > 0:
            real_loader = DataLoader(SlidingWindowDataset(train_data[:real_count], window_size, step_size),
                                     batch_size=batch_size, shuffle=True)
            for epoch in range(defaultsVAE["real_epochs"]):
                model.train()
                for batch, _ in real_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(batch)
                    loss = loss_fn(recon, batch, mu, logvar)
                    loss.backward()
                    optimizer.step()

        # Evaluation:
        model.eval()
        normal_losses = compute_loss_per_sequence_VAE(model, val_loader, device)
        anomaly_losses = compute_loss_per_sequence_VAE(model, anom_loader, device)
        test_losses = compute_loss_per_sequence_VAE(model, test_loader, device)
        std_loss = np.mean(normal_losses)
        threshold = determine_threshold(normal_losses, std_factor=std_factor)
        _, acc, prec, rec, f1 = evaluate_anomalies(test_losses, anomaly_losses, threshold)

        if acc != 0 and prec != 0 and rec != 0 and f1 != 0:
            results.append({
                "sim_ratio": sim_count / total_budget,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "seed": seed,
                "std_loss": std_loss
            })
        else:
            print(f"❌ Skipping seed {seed} with sim={sim_count} due to broken metrics")

        del model
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    seed_options = [42, 32, 13, 87, 133]
    with tqdm_joblib(tqdm(desc="Running Seeds", total=len(seed_options))):
        all_results = Parallel(n_jobs=1)(
            delayed(run_fixed_budget_split)(s) for s in seed_options
        )

    # Flatten results and save to CSV
    results = [r for group in all_results for r in group]
    df = pd.DataFrame(results)
    df.to_csv("results/fixed_budget_sim_VAE.csv", index=False)

    # Plotting the results with consistent, larger fonts
    plt.rcParams.update({'font.size': 9})
    metrics = ["f1", "recall", "accuracy", "precision"]
    titles = {
        "f1": "F1 Score",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "precision": "Precision"
    }
    plt.figure(figsize=(12, 6))
    formatter = PercentFormatter(1.0)
    for metric in metrics:
        grouped = df.groupby("sim_ratio")[metric].agg(["mean", "std"]).reset_index()
        plt.plot(grouped["sim_ratio"], grouped["mean"], label=titles[metric])
        plt.fill_between(grouped["sim_ratio"],
                         grouped["mean"] - 0.5 * grouped["std"],
                         grouped["mean"] + 0.5 * grouped["std"],
                         alpha=0.2)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("Percentage of Simulated Samples Used", fontsize=9)
    plt.ylabel("Score", fontsize=9)
    plt.title("Anomaly Detection Metrics vs. Simulated Data Proportion\n(Fixed Budget: 40 samples)", fontsize=18)
    plt.legend(title="Metric", fontsize=9, title_fontsize=10)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/final_metrics_VAE.png", dpi=300)
    plt.show()
