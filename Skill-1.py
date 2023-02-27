import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs, weights=None, bias=0):
        self.X = None
        self.y = None
        self.weights = weights
        self.bias = bias
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation_function = self.unit_step

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def unit_step(x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.random.random(n_features)

        target = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.epochs):
            for index, x in enumerate(X):
                y_predicted = self.activation_function(np.dot(x, self.weights) + self.bias)
                error = self.learning_rate * (target[index] - y_predicted)
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

    def predict(self, X):
        return self.activation_function(np.dot(X, self.weights) + self.bias)
