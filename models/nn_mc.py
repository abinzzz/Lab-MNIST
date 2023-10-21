import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


class NN_MCClassifier:

    def __init__(self, input_dim, hidden_dim, num_classes, loss_function='cross_entropy'):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.001
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, num_classes) * 0.001
        self.bias2 = np.zeros((1, num_classes))
        self.loss_function = loss_function

    def train(self, data, labels, learning_rate=0.001, num_iters=1000):
        num_samples = len(data)
        for _ in range(num_iters):
            # Forward pass
            hidden = relu(np.dot(data, self.weights1) + self.bias1)
            scores = np.dot(hidden, self.weights2) + self.bias2

            if self.loss_function == 'cross_entropy':
                exp_scores = np.exp(scores)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                loss = -np.log(probs[np.arange(num_samples), labels]).mean()
                dscores = probs
                dscores[np.arange(num_samples), labels] -= 1
                dscores /= num_samples
            elif self.loss_function == 'mse':
                one_hot_labels = np.zeros((num_samples, self.bias2.shape[1]))
                one_hot_labels[np.arange(num_samples), labels] = 1
                loss = np.sum((scores - one_hot_labels) ** 2) / (2 * num_samples)
                dscores = (scores - one_hot_labels) / num_samples
            else:
                raise ValueError(f"Invalid loss function: {self.loss_function}")

            # Backward pass
            dhidden = np.dot(dscores, self.weights2.T)
            dhidden[hidden <= 0] = 0  # Apply ReLU backward

            dW2 = np.dot(hidden.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            dW1 = np.dot(data.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)

            # Update weights and biases
            self.weights1 -= learning_rate * dW1
            self.bias1 -= learning_rate * db1
            self.weights2 -= learning_rate * dW2
            self.bias2 -= learning_rate * db2

            return loss

    def predict(self, data):
        hidden = relu(np.dot(data, self.weights1) + self.bias1)
        scores = np.dot(hidden, self.weights2) + self.bias2
        return np.argmax(scores, axis=1)

    def get_weights(self):
        return {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2
        }

    def set_weights(self, weights):
        self.weights1 = weights['weights1']
        self.bias1 = weights['bias1']
        self.weights2 = weights['weights2']
        self.bias2 = weights['bias2']
