"""
Gradient Descent Algorithms for Linear Regression
===================================================

Three variants of gradient descent for minimizing least squares loss.
"""


class GradientDescentOptimizer:
    """Base class for gradient descent optimizers."""

    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = {
            'coefficients': [],
            'loss': [],
            'gradients': []
        }

    def reset_history(self):
        self.history = {
            'coefficients': [],
            'loss': [],
            'gradients': []
        }

    def compute_loss(self, X, y, coeffs):
        """Compute MSE loss: (1/2n) * ||X*coeffs - y||^2"""
        predictions = self._predict(X, coeffs)
        loss = 0.0
        for i in range(len(y)):
            residual = predictions[i] - y[i]
            loss += residual * residual
        return loss / (2.0 * len(y))

    def _predict(self, X, coeffs):
        """Predict values. X is list of rows, each row is a sample."""
        predictions = []
        for sample in X:
            pred = coeffs[0]  # bias term
            for i in range(1, len(coeffs)):
                pred += coeffs[i] * sample[i-1]
            predictions.append(pred)
        return predictions

    def fit(self, X, y, coeffs_init):
        """Fit the model. Override in subclasses."""
        raise NotImplementedError


class VanillaGradientDescent(GradientDescentOptimizer):
    """
    Vanilla (Batch) Gradient Descent.
    Updates parameters using gradient computed from entire dataset.
    """

    def compute_gradient(self, X, y, coeffs):
        """
        Compute gradient of loss w.r.t. coefficients.
        grad_loss = (1/n) * X^T * (X*coeffs - y)
        Returns list of gradients [grad_c0, grad_c1, ...]
        """
        n = len(y)
        num_features = len(coeffs)
        gradients = [0.0] * num_features

        # Compute predictions
        predictions = self._predict(X, coeffs)

        # Gradient for bias term (c0)
        for i in range(n):
            error = predictions[i] - y[i]
            gradients[0] += error
        gradients[0] /= n

        # Gradient for feature weights (c1, c2, ...)
        for j in range(1, num_features):
            for i in range(n):
                error = predictions[i] - y[i]
                gradients[j] += error * X[i][j-1]
            gradients[j] /= n

        return gradients

    def fit(self, X, y, coeffs_init):
        """
        Vanilla batch gradient descent.

        Args:
            X: List of lists, each inner list is features for one sample
            y: List of target values
            coeffs_init: Initial coefficients [c0, c1, ...]

        Returns:
            Trained coefficients and history
        """
        self.reset_history()
        coeffs = list(coeffs_init)

        for iteration in range(self.max_iterations):
            # Compute loss and gradient
            loss = self.compute_loss(X, y, coeffs)
            gradients = self.compute_gradient(X, y, coeffs)

            # Store history
            self.history['loss'].append(loss)
            self.history['coefficients'].append(list(coeffs))
            self.history['gradients'].append(list(gradients))

            # Update coefficients
            grad_norm = 0.0
            for i in range(len(coeffs)):
                coeffs[i] -= self.learning_rate * gradients[i]
                grad_norm += gradients[i] * gradients[i]

            grad_norm = grad_norm ** 0.5

            if grad_norm < self.tolerance:
                break

        return coeffs


class StochasticGradientDescent(GradientDescentOptimizer):
    """
    Stochastic Gradient Descent (SGD).
    Updates parameters using gradient computed from individual samples.
    """

    def fit(self, X, y, coeffs_init):
        """
        Stochastic gradient descent - one sample at a time.

        Args:
            X: List of lists, each inner list is features for one sample
            y: List of target values
            coeffs_init: Initial coefficients [c0, c1, ...]

        Returns:
            Trained coefficients and history
        """
        self.reset_history()
        coeffs = list(coeffs_init)
        n = len(y)

        for iteration in range(self.max_iterations):
            # Shuffle data (simple in-place shuffle)
            indices = list(range(n))
            for i in range(n-1, 0, -1):
                j = (iteration * (i+1)) % (i+1)  # pseudo-random
                indices[i], indices[j] = indices[j], indices[i]

            total_loss = 0.0
            grad_norm = 0.0

            # Process one sample at a time
            for idx in indices:
                sample = [X[idx]]
                target = [y[idx]]

                # Compute gradient for single sample
                prediction = self._predict(sample, coeffs)[0]
                error = prediction - target[0]

                # Update coefficients
                gradients = [0.0] * len(coeffs)
                gradients[0] = error
                for j in range(1, len(coeffs)):
                    gradients[j] = error * sample[0][j-1]

                for i in range(len(coeffs)):
                    coeffs[i] -= self.learning_rate * gradients[i]
                    grad_norm += gradients[i] * gradients[i]

                # Accumulate loss
                total_loss += error * error / 2.0

            total_loss /= n
            grad_norm = grad_norm ** 0.5

            # Store history
            self.history['loss'].append(total_loss)
            self.history['coefficients'].append(list(coeffs))

            if grad_norm < self.tolerance:
                break

        return coeffs


class MiniBatchGradientDescent(GradientDescentOptimizer):
    """
    Mini-batch Gradient Descent.
    Updates parameters using gradient computed from a batch of samples.
    """

    def __init__(self, learning_rate=0.01, max_iterations=1000,
                 tolerance=1e-8, batch_size=32):
        super().__init__(learning_rate, max_iterations, tolerance)
        self.batch_size = batch_size

    def fit(self, X, y, coeffs_init):
        """
        Mini-batch gradient descent.

        Args:
            X: List of lists, each inner list is features for one sample
            y: List of target values
            coeffs_init: Initial coefficients [c0, c1, ...]

        Returns:
            Trained coefficients and history
        """
        self.reset_history()
        coeffs = list(coeffs_init)
        n = len(y)

        for iteration in range(self.max_iterations):
            # Shuffle indices for each epoch
            indices = list(range(n))
            for i in range(n-1, 0, -1):
                j = (iteration * (i+1)) % (i+1)  # pseudo-random
                indices[i], indices[j] = indices[j], indices[i]

            total_loss = 0.0
            epoch_grad_norm = 0.0

            # Process mini-batches
            for batch_start in range(0, n, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n)
                batch_indices = indices[batch_start:batch_end]
                batch_size_actual = batch_end - batch_start

                # Get batch data
                X_batch = [X[i] for i in batch_indices]
                y_batch = [y[i] for i in batch_indices]

                # Compute gradient for batch
                gradients = [0.0] * len(coeffs)
                predictions = self._predict(X_batch, coeffs)

                for i in range(len(y_batch)):
                    error = predictions[i] - y_batch[i]
                    total_loss += error * error / 2.0

                    gradients[0] += error
                    for j in range(1, len(coeffs)):
                        gradients[j] += error * X_batch[i][j-1]

                # Average gradient over batch
                for j in range(len(gradients)):
                    gradients[j] /= batch_size_actual
                    coeffs[j] -= self.learning_rate * gradients[j]
                    epoch_grad_norm += gradients[j] * gradients[j]

            total_loss /= n
            epoch_grad_norm = epoch_grad_norm ** 0.5

            # Store history
            self.history['loss'].append(total_loss)
            self.history['coefficients'].append(list(coeffs))

            if epoch_grad_norm < self.tolerance:
                break

        return coeffs
