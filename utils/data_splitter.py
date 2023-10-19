import numpy as np


def train_val_split(data, labels, val_fraction=0.1):
    """
    Split data into training and validation subsets.

    Parameters:
    - data: np.array, original training data.
    - labels: np.array, original training labels.
    - val_fraction: float, fraction of data to use for validation.

    Returns:
    - splits: tuple, (train_data, train_labels, val_data, val_labels).
    """
    num_val = int(len(data) * val_fraction)
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]

    val_data = data[val_indices]
    val_labels = labels[val_indices]

    return train_data, train_labels, val_data, val_labels
