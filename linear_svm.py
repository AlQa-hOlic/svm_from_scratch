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


class MultiClassLinearSVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for class_label in self.classes:
            # Create a binary target variable for each class
            y_binary = np.where(y == class_label, 1, -1)
            svm = LinearSVM(self.learning_rate, self.lambda_param, self.n_iters)
            svm.fit(X, y_binary)
            self.models.append(svm)

    def predict(self, X):
        # Store the results of each binary classifier
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, svm in enumerate(self.models):
            predictions[:, i] = svm.predict(X)

        # Choose the class with the highest decision function value
        return np.argmax(predictions, axis=1)


# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# iris = load_iris()
# X, Y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.25, random_state=10
# )

# multi_svm = MultiClassLinearSVM()
# multi_svm.fit(X_train, y_train)  # Assuming y_train has multiple classes
# multi_class_predictions = multi_svm.predict(X_test)

# print(f"Accuracy: {np.mean(multi_class_predictions == y_test)}")
