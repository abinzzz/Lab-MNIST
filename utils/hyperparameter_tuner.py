import numpy as np
from evaluation import accuracy

def tune_hyperparameters(model_class, data, labels, hyperparameters, num_trials=10):
    """
    Tune hyperparameters by searching over a given range.

    Parameters:
    - model_class: class, classifier class (e.g., SoftmaxClassifier, NNClassifier).
    - data: np.array, training data.
    - labels: np.array, training labels.
    - hyperparameters: dict, hyperparameter ranges to search over.
    - num_trials: int, number of hyperparameter settings to test.

    Returns:
    - best_model: trained model with best hyperparameters.
    """
    best_acc = -1
    best_model = None

    for _ in range(num_trials):
        hp = {k: np.random.choice(v) for k, v in hyperparameters.items()}
        model = model_class(**hp)
        model.train(data, labels)
        predictions = model.predict(data)
        acc = accuracy(predictions, labels)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model
