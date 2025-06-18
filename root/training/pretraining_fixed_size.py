import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from GitFile.root.config import defaults, data_paths
from models import LinearAE
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence, determine_threshold, evaluate_anomalies
from train import train_autoencoder
from GitFile.root.data.data import load_data, load_anomaly_data
from utils import seed_all

"""
pretraining_fixed_size.py

Evaluates anomaly detection performance under a fixed total data budget, split
between real and simulated training samples.

This script runs multiple seeds to analyze how different ratios of simulated vs. real
training data (with a fixed total size) affect model performance.

Functions:
- run_fixed_budget_split: Executes training and evaluation for one seed and all real/sim ratios.
- __main__: Runs the full experiment across multiple seeds and saves/visualizes results.
"""

def run_fixed_budget_split(seed):
    """
        Trains and evaluates a Linear Autoencoder using a fixed total number of training samples.

        Simulates multiple splits between real and simulated data (total = 40),
        trains the model on each split, and evaluates anomaly detection performance.

        Args:
            seed (int): Random seed to ensure reproducibility.

        Returns:
            list of dict: Evaluation metrics (F1, accuracy, precision, recall) per ratio.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])

    print("Train:", len(train_data), "Test:", len(test_data), "Val:", len(val_data))
    window_size = defaults["window_size"]
    step_size = defaults["step_size"]
    input_dim = window_size * train_data[0].shape[-1]
    batch_size = defaults["batch_size"]
    latent_dim = defaults["latent_dim"]
    dropout = defaults["dropout"]
    lr = defaults["lr"]
    weight_decay = defaults["weight_decay"]
    std_factor = defaults["std_factor"]

    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size), batch_size=batch_size)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size), batch_size=batch_size)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size), batch_size=batch_size)

    results = []
    total_budget = 40

    for sim_count in range(0, total_budget + 1, 1):
        real_count = total_budget - sim_count

        if sim_count > len(sim_data) or real_count > len(train_data):
            continue

        model = LinearAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCELoss()

        if sim_count > 0:
            sim_loader = DataLoader(SlidingWindowDataset(sim_data[:sim_count], window_size, step_size), batch_size=batch_size, shuffle=True)
            model = train_autoencoder(model, sim_loader, optimizer, loss_fn, epochs=10, device=device)

        if real_count > 0:
            real_loader = DataLoader(SlidingWindowDataset(train_data[:real_count], window_size, step_size), batch_size=batch_size, shuffle=True)
            model = train_autoencoder(model, real_loader, optimizer, loss_fn, epochs=17, device=device)

        model.eval()
        val_losses = compute_loss_per_sequence(model, val_loader, device)
        test_losses = compute_loss_per_sequence(model, test_loader, device)
        anomaly_losses = compute_loss_per_sequence(model, anom_loader, device)

        threshold = determine_threshold(val_losses, std_factor=std_factor)
        _, acc, prec, rec, f1 = evaluate_anomalies(test_losses, anomaly_losses, threshold)

        if all(metric != 0 for metric in [acc, prec, rec, f1]):
            results.append({
                "sim_ratio": sim_count / total_budget,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "seed": seed
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

    df = pd.DataFrame([r for group in all_results for r in group])
    df.to_csv("results/fixed_budget_sim_vs_real.csv", index=False)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator

    # Set a global font size
    plt.rcParams.update({'font.size': 14})

    metrics = ["f1", "recall", "accuracy", "precision"]
    titles = {
        "f1": "F1 Score",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "precision": "Precision"
    }

    # Create one combined plot
    plt.figure(figsize=(12, 6))

    # Format x-axis to display percentages
    formatter = PercentFormatter(1.0)

    for metric in metrics:
        grouped = df.groupby("sim_ratio")[metric].agg(["mean", "std"]).reset_index()
        plt.plot(grouped["sim_ratio"], grouped["mean"], label=titles[metric])
        plt.fill_between(grouped["sim_ratio"],
                         grouped["mean"] - 0.5 * grouped["std"],
                         grouped["mean"] + 0.5 * grouped["std"],
                         alpha=0.2)

    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("Percentage of Simulated Samples Used", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.title("Anomaly Detection Metrics vs. Simulated Data Proportion\n(Fixed Budget: 40 samples)", fontsize=18)
    plt.legend(title="Metric", fontsize=14, title_fontsize=16)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/fixed_budget_all_metrics.png", dpi=300)
    plt.show()
