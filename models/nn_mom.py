import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


class NN_MOMClassifier:

    def __init__(self, input_dim, hidden_dim, num_classes):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.001
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, num_classes) * 0.001
        self.bias2 = np.zeros((1, num_classes))

        # For Momentum
        self.vW1, self.vb1 = np.zeros_like(self.weights1), np.zeros_like(self.bias1)
        self.vW2, self.vb2 = np.zeros_like(self.weights2), np.zeros_like(self.bias2)

    def train(self, data, labels, optimizer='sgd', learning_rate=0.01, momentum=0.9, batch_size=32, num_iters=1000):
        num_samples = len(data)

        for _ in range(num_iters):
            if optimizer == 'sgd':
                indices = np.random.choice(num_samples, 1)  # choose 1 sample for SGD
            elif optimizer == 'mini-batch-gd':
                indices = np.random.choice(num_samples, batch_size)  # choose a batch of samples for mini-batch GD
            else:  # for the full-batch and momentum we use all the data
                indices = np.arange(num_samples)

            X_batch = data[indices]
            y_batch = labels[indices]

            # Forward pass
            hidden = relu(np.dot(X_batch, self.weights1) + self.bias1)
            scores = np.dot(hidden, self.weights2) + self.bias2

            # Convert labels to one-hot encoding
            one_hot_labels = np.zeros((len(indices), self.bias2.shape[1]))
            one_hot_labels[np.arange(len(indices)), y_batch] = 1

            # Compute MSE loss
            mse_loss = np.sum((scores - one_hot_labels) ** 2) / (2 * len(indices))

            # Backward pass
            dscores = (scores - one_hot_labels) / len(indices)
            dhidden = np.dot(dscores, self.weights2.T)
            dhidden[hidden <= 0] = 0

            dW2 = np.dot(hidden.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)
            dW1 = np.dot(X_batch.T, dhidden)
            db1 = np.sum(dhidden, axis=0, keepdims=True)

            if optimizer == 'momentum':
                # Apply momentum update
                self.vW1 = momentum * self.vW1 - learning_rate * dW1
                self.vb1 = momentum * self.vb1 - learning_rate * db1
                self.vW2 = momentum * self.vW2 - learning_rate * dW2
                self.vb2 = momentum * self.vb2 - learning_rate * db2

                self.weights1 += self.vW1
                self.bias1 += self.vb1
                self.weights2 += self.vW2
                self.bias2 += self.vb2
            else:
                # Apply normal update
                self.weights1 -= learning_rate * dW1
                self.bias1 -= learning_rate * db1
                self.weights2 -= learning_rate * dW2
                self.bias2 -= learning_rate * db2

            return mse_loss

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

