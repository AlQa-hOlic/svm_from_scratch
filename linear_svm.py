import numpy as np


# Define the Linear SVM class
class LinearSVM:
    # Initialization method with learning rate, lambda parameter for regularization, and number of iterations
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate  # Learning rate for gradient descent
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.w = None  # Weight vector, to be initialized later
        self.b = None  # Bias term, to be initialized later

    # Method to fit the SVM to the training data
    def fit(self, X, y):
        (
            n_samples,
            n_features,
        ) = X.shape  # Get the number of samples and features in the training data
        y_ = np.where(y <= 0, -1, 1)  # Convert binary labels to -1 and 1

        self.w = np.zeros(n_features)  # Initialize weight vector to zeros
        self.b = 0  # Initialize bias to 0

        # Iterative training over the dataset
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition for checking if the data point is correctly classified
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Update weights for correctly classified points
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Update weights and bias for incorrectly classified points
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    # Method to make predictions using the learned model
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b  # Calculate the decision function
        return np.sign(
            approx
        )  # Return the sign of the decision function as class label
