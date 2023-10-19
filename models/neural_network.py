import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


class NNClassifier:

    def __init__(self, input_dim, hidden_dim, num_classes):
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.001
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, num_classes) * 0.001
        self.bias2 = np.zeros((1, num_classes))

    def train(self, data, labels, learning_rate=0.001, num_iters=1000):
        for _ in range(num_iters):
            # 前向传播
            hidden = relu(np.dot(data, self.weights1) + self.bias1)
            scores = np.dot(hidden, self.weights2) + self.bias2
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # 计算loss
            correct_log_probs = -np.log(probs[range(len(data)), labels])
            loss = np.sum(correct_log_probs) / len(data)

            # 反向传播
            dscores = probs
            dscores[range(len(data)), labels] -= 1
            dscores /= len(data)
            dhidden = np.dot(dscores, self.weights2.T)
            dhidden = relu_backward(dhidden, hidden)
            dW2 = np.dot(hidden.T, dscores)
            db2 = np.sum(dscores, axis=0)
            dW1 = np.dot(data.T, dhidden)
            db1 = np.sum(dhidden, axis=0)

            # 更新参数
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
