import numpy as np


def accuracy(predictions, labels):
    """
    Calculate accuracy given predictions and ground truth labels.

    Parameters:
    - predictions: np.array, model predictions.
    - labels: np.array, ground truth labels.

    Returns:
    - accuracy: float, accuracy score.
    """
    return np.mean(predictions == labels)
