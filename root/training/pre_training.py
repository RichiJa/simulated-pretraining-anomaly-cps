# parallel_pretraining.py
import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from GitFile.root.config import defaults, data_paths
from models import LinearAE
from GitFile.root.data.datasets import SlidingWindowDataset
from evaluate import compute_loss_per_sequence, determine_threshold, evaluate_anomalies
from train import train_autoencoder
from GitFile.root.data.data import load_data, load_anomaly_data
from utils import seed_all
from matplotlib.ticker import MultipleLocator

"""
pre_training.py

Performs a large-scale grid search across different combinations of real and simulated training data
for anomaly detection using a Linear Autoencoder.

The script trains models on varying amounts of real and simulated data, evaluates them across multiple seeds,
and generates CSV/plot output to analyze performance metrics such as F1, accuracy, recall, precision, and std_loss.

Functions:
- run_pretraining_grid: Core logic that trains and evaluates a model for all real/sim sample combinations for a given seed.
- __main__: Executes the grid search across multiple seeds using parallel processing and visualizes the results.
"""




def run_pretraining_grid(seed):
    """
        Runs one full grid of pretraining + fine-tuning combinations for a single random seed.

        Varies the number of real samples (0–30) and simulated samples (0–100) and records
        evaluation metrics on validation and test data. Results are collected per configuration.

        Args:
            seed (int): Random seed to use for reproducibility.

        Returns:
            list of dicts: Each dict contains metrics for one (real_size, sim_size) configuration.
    """
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, test_data, val_data = load_data(data_paths["real_data_folder"])
    anom_data = load_anomaly_data(data_paths["anom_folder_A1"]) + load_anomaly_data(data_paths["anom_folder_A2"])
    sim_data = load_anomaly_data(data_paths["sim_data_folder"])
    #print("Train:", len(train_data), "Test:", len(test_data), "Val:", len(val_data))
    window_size = defaults["window_size"]
    step_size = defaults["step_size"]
    input_dim = window_size * train_data[0].shape[-1]
    batch_size = defaults["batch_size"]
    latent_dim = defaults["latent_dim"]
    loss_fn = torch.nn.BCELoss()
    dropout = defaults["dropout"]
    lr = defaults["lr"]
    weight_decay = defaults["weight_decay"]
    std_factor = defaults["std_factor"]

    val_loader = DataLoader(SlidingWindowDataset(val_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SlidingWindowDataset(test_data, window_size, step_size), batch_size=batch_size, shuffle=True)
    anom_loader = DataLoader(SlidingWindowDataset(anom_data, window_size, step_size), batch_size=batch_size, shuffle=True)

    results = []
    real_options = [0, 10, 15, 20, 30]
    max_sim = 100

    for real_size in real_options:
        if real_size > 0:
            real_loader = DataLoader(SlidingWindowDataset(train_data[:real_size], window_size, step_size), batch_size=batch_size, shuffle=True)

        for sim_size in range(0, max_sim + 1, 2):


            if real_size == 0 and sim_size == 0:
                continue
            elif real_size == 0:
                real_epochs, sim_epochs = 0, 17
                lr = defaults["lr_PT"]
                #dropout = defaults["dropout_PT"]
                #weight_decay = defaults["weight_decay_PT"]
                #std_factor = defaults["std_factor_PT"]
            elif sim_size == 0:
                real_epochs, sim_epochs = 17, 0
                #dropout = defaults["dropout"]
                #lr = defaults["lr"]
                #weight_decay = defaults["weight_decay"]
                #std_factor = defaults["std_factor"]
            else:
                real_epochs, sim_epochs = 17, 10
                #lr = defaults["lr_PT"]
                #dropout = defaults["dropout_PT"]
                #weight_decay = defaults["weight_decay_PT"]
                #std_factor = defaults["std_factor_PT"]

            model = LinearAE(input_dim=input_dim, latent_dim=latent_dim, dropout_prob=dropout)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if sim_size > 0:
                if sim_size > len(sim_data):
                    raise ValueError(f"Sim size {sim_size} exceeds available simulated data {len(sim_data)}.")

                sim_loader = DataLoader(SlidingWindowDataset(sim_data[:sim_size], window_size, step_size), batch_size=batch_size, shuffle=True)
                model = train_autoencoder(model, sim_loader, optimizer, loss_fn, epochs=sim_epochs, device=device)

            if real_size > 0:
                model = train_autoencoder(model, real_loader, optimizer, loss_fn, epochs=real_epochs, device=device)
            model.eval()
            normal_losses = compute_loss_per_sequence(model, val_loader, device)
            anomaly_losses = compute_loss_per_sequence(model, anom_loader, device)
            test_losses = compute_loss_per_sequence(model, test_loader, device)
            std_loss = np.mean(normal_losses)

            threshold = determine_threshold(normal_losses, std_factor=std_factor)
            _, acc, prec, rec, f1 = evaluate_anomalies(test_losses, anomaly_losses, threshold)
            del model
            torch.cuda.empty_cache()  # Optional: if using GPU
            model = None
            #avoid completely broken model due to ill rnd state
            if(acc != 0 and prec != 0 and rec != 0 and f1 != 0):
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


import  numpy as np
if __name__ == "__main__":

    seed_options = [42, 32, 132, 42, 13, 17, 42, 87, 133, 205, 314, 512, 777, 901, 1024, 56, 420, 32789, 7, 986]
    with tqdm_joblib(tqdm(desc="Running Seeds", total=len(seed_options))):
        all_results = Parallel(n_jobs=5)(
            delayed(run_pretraining_grid)(s) for s in seed_options
        )

    # Flatten results
    results = [r for group in all_results for r in group]
    df = pd.DataFrame(results)
    df.to_csv("results/pretraining_sweep_results.csv", index=False)

    # Replace unstable 0s (likely due to no TP, FP, etc.) with NaN for fair stats
    for metric in ["f1", "precision", "recall"]:
        df.loc[df[metric] == 0.0, metric] = np.nan

    metrics = ["f1", "recall", "accuracy", "precision", "std_loss"]
    titles = {
        "f1": "F1 Score",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "std_loss": "std_loss"
    }

    # Set base font size across all plots
    plt.rcParams.update({'font.size': 9})

    for metric in metrics:
        # Group data, ignoring NaNs
        grouped = df.groupby(["real", "sim"])[metric].agg(["mean", "std", "count"]).reset_index()
        if metric == "std_loss":
            grouped = grouped[grouped["real"] != 0]  # Drop real=0 for loss graph as it skews the scale
        plt.figure(figsize=(12, 6))
        for real_val in sorted(grouped["real"].unique()):
            sub = grouped[grouped["real"] == real_val].sort_values("sim")
            plt.plot(sub["sim"], sub["mean"], label=f"Real={real_val}")
            plt.fill_between(sub["sim"], sub["mean"] - 0.5 * sub["std"], sub["mean"] + 0.5 * sub["std"], alpha=0.2)
            print(sub["std"])

        plt.xlabel("Number of Simulated Samples Used", fontsize=16)
        plt.ylabel(titles[metric], fontsize=16)
        plt.legend(title="Real Samples", fontsize=14, title_fontsize=16)
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"results/final_{metric}.png", dpi=300)

