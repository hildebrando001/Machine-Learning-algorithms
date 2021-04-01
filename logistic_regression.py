import numpy as np

class LogisticRegression:

    def __init__(self,learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # first dimension of shape gos to n_samples, second to n_features
        # init parameters
        self.weights = np.zeros(n_features) # create a vector only of zeros of size n_samples
        self.bias = 0  # random numbers may be used but zeros is just fine

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model) # Apply sigmund function

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # exp: Calculate the exponential of all elements in the input array.