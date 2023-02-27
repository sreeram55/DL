import numpy as np
from planar_utlis import load_planar_dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.where(x > 0, 1, 0)


def cross_entropy_loss(yHat, y):
    return np.where(y == 1, -np.log(yHat), -np.log(1 - yHat))


class MLP:
    def init(self, input_nodes, hidden_nodes, output_nodes, activation_function=sigmoid, loss=mean_squared_error):
        self.X = None
        self.y = None
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.activation_function = activation_function
        self.loss = loss

        self.w1 = np.random.rand(input_nodes, hidden_nodes)
        self.w2 = np.random.rand(hidden_nodes, output_nodes)
        self.b1 = np.zeros(hidden_nodes)
        self.b2 = np.zeros(output_nodes)

        self.total_loss = []

    def getWeights(self):
        return self.w1, self.w2

    def mse(self):
        return np.sum(self.total_loss) / len(self.X)

    def forward_propagation(self, x):
        hidden_out = self.activation_function(np.dot(x, self.w1) + self.b1)
        out = self.activation_function(np.dot(hidden_out, self.w2) + self.b2)
        return hidden_out, out

    def mini_batches(self, X, y, batch_size=64):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_mini_batches = data.shape[0] // batch_size
        i = 0
        for i in range(n_mini_batches + 1):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def fit(self, X_train, y_train, learning_rate=0.01, epochs=1000, batch_size=64):
        for _ in range(epochs):
       mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_mini_batches = data.shape[0] // batch_size
        i = 0
        for i in range(n_mini_batches + 1):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-1]
            Y_mini = mini_batch[:, -1].reshape((-1, 1))
            mini_batches.append((X_mini, Y_mini))      

            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                Z1 = np.dot(X_mini, self.w1)
                A1 = self.activation_function(Z1)
                Z2 = np.dot(A1, self.w2)
                A2 = self.activation_function(Z2)
                N = len(X_mini)
                self.loss(A2, y_mini)
                E1 = A2 - y_mini
                dW1 = E1 * A2 * (1 - A2)
                E2 = np.dot(dW1, self.w2.T)
                dW2 = E2 * A1 * (1 - A1)
                W2_update = np.dot(A1.T, dW1) / N
                W1_update = np.dot(X_mini.T, dW2) / N

                self.w2 = self.w2 - learning_rate * W2_update
                self.w1 = self.w1 - learning_rate * W1_update

            # print(f'> Epoch: {_ + 1}\tLoss: {loss}')

    def predict(self, x, function=None):
        if function is None:
            return self.forward_propagation(x)[1]
        hidden_out = function(np.dot(x, self.w1) + self.b1)
        out = function(np.dot(hidden_out, self.w2) + self.b2)
        return out


if name == 'main':
    X, y = load_planar_dataset()
