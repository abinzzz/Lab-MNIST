import numpy as np


class SoftmaxClassifier:

    def __init__(self, input_dim, num_classes):
        self.weights = np.random.randn(input_dim, num_classes) * 0.001
        self.bias = np.zeros((1, num_classes))


    def train(self, data, labels, learning_rate=0.001, num_iters=1000):
        for _ in range(num_iters):
            scores = np.dot(data, self.weights) + self.bias
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # 计算loss
            correct_log_probs = -np.log(probs[range(len(data)), labels])
            loss = np.sum(correct_log_probs) / len(data)

            # 计算gradient
            dscores = probs
            dscores[range(len(data)), labels] -= 1
            dscores /= len(data)
            dW = np.dot(data.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            # 更新参数
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * db

            return loss

    def predict(self, data):
        scores = np.dot(data, self.weights) + self.bias
        return np.argmax(scores, axis=1)

    def get_weights(self):
        return {
            'weights': self.weights,
            'bias': self.bias
        }


    def set_weights(self, weights):
        self.weights = weights['weights']
        self.bias = weights['bias']
