# datasets.py
from torch.utils.data import Dataset
import torch

"""
datasets.py

Custom PyTorch Dataset classes for applying sliding window segmentation
to time-series data.

Classes:
- SlidingWindowDataset: Returns flattened windows for Linear AE/VAE.
"""

class SlidingWindowDataset(Dataset):
    """
    Dataset that extracts flattened sliding windows from multivariate time-series sequences.

    Each window is returned as a 1D tensor (flattened), which is suitable for use with
    feed-forward autoencoders (e.g., LinearAE or LinearVAE).

    Args:
        sequences (List[Tensor]): List of input sequences (T x F) to slice.
        window_size (int): Number of time steps per window.
        step_size (int): Step size for the sliding window (default: 5).
    """
    def __init__(self, sequences, window_size, step_size=5):
        self.windows = []
        self.seq_indices = []
        for seq_id, sequence in enumerate(sequences):
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.tensor(sequence, dtype=torch.float32)
            else:
                sequence = sequence.float()

            for i in range(0, sequence.shape[0] - window_size + 1, step_size):
                window = sequence[i:i+window_size].flatten()
                self.windows.append(window)
                self.seq_indices.append(seq_id)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.seq_indices[idx]

