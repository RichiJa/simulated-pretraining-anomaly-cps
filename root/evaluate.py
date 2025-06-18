# evaluate.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
from loss import vae_loss
import warnings
from sklearn.exceptions import UndefinedMetricWarning

"""
evaluate.py

Evaluation utilities for computing reconstruction losses and anomaly detection metrics
for Autoencoders and Variational Autoencoders. Includes support for:
- Sequence-level loss calculation
- Thresholding for anomaly detection
- Evaluation metrics: accuracy, precision, recall, F1 score
- Plotting metric curves across experiments

Functions:
- compute_loss_per_sequence
- compute_loss_per_sequence_VAE
- determine_threshold
- evaluate_anomalies
- plot_metrics
"""

def compute_loss_per_sequence(model, loader, device):
    """
        Computes the average BCE reconstruction loss for each original sequence in the dataset.
        Suitable for Linear AE models using flattened inputs.

        Args:
            model: Trained autoencoder model.
            loader (DataLoader): Dataloader returning (batch, sequence_id).
            device: 'cuda' or 'cpu'.

        Returns:
            List[float]: Average loss per sequence.
    """

    model.eval()
    loss_fn = torch.nn.BCELoss(reduction='none')
    seq_losses = defaultdict(list)

    with torch.no_grad():
        for batch, seq_ids in loader:
            batch = batch.to(device)
            recon = model(batch)
            losses = loss_fn(recon, batch).mean(dim=1)
            for loss, seq_id in zip(losses, seq_ids):
                seq_losses[seq_id.item()].append(loss.item())

    return [np.mean(seq_losses[seq_id]) for seq_id in sorted(seq_losses.keys())]



def compute_loss_per_sequence_VAE(model, dataloader, device):
    """
        Computes total VAE loss (BCE + KL divergence) per batch for a VAE model.

        Args:
            model: Trained VAE model.
            dataloader (DataLoader): Input data.
            device: 'cuda' or 'cpu'.

        Returns:
            List[float]: Loss values per batch.
    """
    loss_fn = vae_loss
    losses = []

    model.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            batch = batch.to(device)
            recon_x, mu, logvar = model(batch)
            loss = loss_fn(recon_x, batch, mu, logvar)
            losses.append(loss.item())  # or extend for per-sample if needed

    return losses

def determine_threshold(losses, std_factor=1.2):
    """
        Computes an anomaly threshold using mean + N * std of normal losses.

        Args:
            losses (List[float]): List of reconstruction losses.
            std_factor (float): Multiplier for standard deviation.

        Returns:
            float: Threshold value.
    """
    return np.mean(losses) + std_factor * np.std(losses)


def evaluate_anomalies(normal_losses, anomaly_losses, threshold):
    """
        Computes confusion matrix and classification metrics for anomaly detection.

        Args:
            normal_losses (List[float]): Losses for normal sequences.
            anomaly_losses (List[float]): Losses for anomalous sequences.
            threshold (float): Decision threshold.

        Returns:
            Tuple: (confusion_matrix, accuracy, precision, recall, F1)
    """
    y_true = [0] * len(normal_losses) + [1] * len(anomaly_losses)
    y_scores = normal_losses + anomaly_losses
    y_pred = [1 if l > threshold else 0 for l in y_scores]

    cm = confusion_matrix(y_true, y_pred)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

    #print("Confusion Matrix:\n", cm)
    #print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return cm, acc, prec, rec, f1


def plot_metrics(df, metric, save_path="results/", show=False):
    """
        Plots a performance metric across different simulated data quantities.

        Args:
            df (DataFrame): Dataframe with columns ["real", "sim", metric].
            metric (str): One of ['f1', 'recall', 'accuracy', 'precision'].
            save_path (str): Path to save the figure.
            show (bool): If True, display the plot.
    """
    plt.figure(figsize=(12, 6))
    title_map = {"f1": "F1 Score", "recall": "Recall", "accuracy": "Accuracy", "precision": "Precision"}

    for real_val in sorted(df["real"].unique()):
        sub = df[df["real"] == real_val].sort_values("sim")
        plt.plot(sub["sim"], sub[metric], label=f"Real={real_val}", marker='o')

    plt.xlabel("Number of Simulated Samples Used")
    plt.ylabel(title_map.get(metric, metric))
    plt.title(f"{title_map.get(metric, metric)} vs Simulated Data")
    plt.legend(title="Real Samples")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}{metric}.png")
    if show:
        plt.show()
    plt.close()
