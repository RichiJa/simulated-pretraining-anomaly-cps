# data.py
import torch
import numpy as np
import glob
import random
from torch.utils.data import Dataset

"""
data.py

Utility functions for loading and preprocessing time-series data stored as CSV files.
This module supports loading real and simulated sensor data, removing timestamps and headers,
and splitting datasets for training, validation, and testing.

Functions:
- load_csv_tensor(file_path): Load a single CSV file into a torch tensor.
- load_data(folder_path): Load and split real/normal data into train, val, and test sets.
- load_anomaly_data(folder_path): Load and return all anomaly data from a folder.
"""

def load_csv_tensor(file_path):
    """
       Load a CSV file into a PyTorch tensor, removing timestamp and header.

       Args:
           file_path (str): Path to the CSV file.

       Returns:
           torch.FloatTensor: Tensor of shape (time_steps, num_features).
    """

    data = np.genfromtxt(file_path, dtype=float, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data[:, 1:]  # remove timestamp
    data = data[1:, :]  # remove header
    return torch.tensor(data, dtype=torch.float32)


def load_data(folder_path):
    """
        Load all CSV files in a folder, shuffle them, and split into
        training (40), validation (10), and test (30) sequences.
        Used primarily for real - normal data (train, val, and test split).

        Args:
            folder_path (str): Path to the folder containing normal .csv files.

        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor]]:
                - TrainList
                - TestList
                - ValList
    """

    file_paths = glob.glob(f"{folder_path}/*")
    random.shuffle(file_paths)
    data_list = [load_csv_tensor(path) for path in file_paths]

    from sklearn.model_selection import train_test_split
    TrainList, temp_data = train_test_split(data_list, train_size=40, random_state=42)
    ValList, TestList = train_test_split(temp_data, train_size=10, test_size=30, random_state=42)
    return TrainList, TestList, ValList

#available: 100 sim; 80 real-normal; 30 real-anomalous
def load_anomaly_data(folder_path):
    """
        Load all CSV files from a folder containing anomaly data and return them as a list of tensors.
        Used primarily for sim- and anomaly data (no split required)
        Args:
            folder_path (str): Path to the folder containing anomaly .csv files.

        Returns:
            List[Tensor]: Each tensor has shape (time_steps, num_features).
    """
    file_paths = glob.glob(f"{folder_path}/*")
    random.shuffle(file_paths)
    data_list = []
    for path in file_paths:
        try:
            data_list.append(load_csv_tensor(path))
        except Exception as e:
            print(f"[!] Failed loading {path}: {e}")
    return data_list

