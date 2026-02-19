"""
Gradient Descent Algorithms for Linear Regression
===================================================

Three variants of gradient descent for minimizing least squares loss.
"""

import numpy as np


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
        residuals = predictions - y
        loss = np.sum(residuals ** 2) / (2.0 * len(y))
        return float(loss)

    def _predict(self, X, coeffs):
        """
        Predict values.
        X: (n, m) array where n=samples, m=features
        coeffs: (m+1,) array [bias, w1, w2, ...]
        Returns: (n,) array of predictions
        """
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return X_with_bias @ coeffs

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
        grad = (1/n) * X^T * (predictions - y)
        """
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        predictions = X_with_bias @ coeffs
        errors = predictions - y
        gradients = X_with_bias.T @ errors / len(y)
        return gradients

    def fit(self, X, y, coeffs_init):
        """
        Vanilla batch gradient descent.

        Args:
            X: (n, m) array - n samples, m features
            y: (n,) array - target values
            coeffs_init: (m+1,) array - initial coefficients [bias, w1, w2, ...]

        Returns:
            Trained coefficients
        """
        self.reset_history()
        X = np.array(X)
        y = np.array(y)
        coeffs = np.array(coeffs_init, dtype=float)

        for iteration in range(self.max_iterations):
            # Compute loss and gradient
            loss = self.compute_loss(X, y, coeffs)
            gradients = self.compute_gradient(X, y, coeffs)

            # Store history
            self.history['loss'].append(loss)
            self.history['coefficients'].append(coeffs.copy())
            self.history['gradients'].append(gradients.copy())

            # Update coefficients
            coeffs -= self.learning_rate * gradients
            grad_norm = np.linalg.norm(gradients)

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
            X: (n, m) array
            y: (n,) array
            coeffs_init: (m+1,) array

        Returns:
            Trained coefficients
        """
        self.reset_history()
        X = np.array(X)
        y = np.array(y)
        coeffs = np.array(coeffs_init, dtype=float)
        n = len(y)

        for iteration in range(self.max_iterations):
            # Shuffle indices
            indices = np.random.permutation(n)

            total_loss = 0.0
            grad_norm = 0.0

            # Process one sample at a time
            for idx in indices:
                X_i = X[idx:idx+1]  # Keep 2D shape
                y_i = y[idx:idx+1]

                # Compute gradient for single sample
                X_with_bias = np.column_stack([1.0, X_i])
                prediction = X_with_bias @ coeffs
                error = prediction - y_i

                # Gradient for this sample
                gradient = X_with_bias.T @ error

                # Update coefficients
                coeffs -= self.learning_rate * gradient.ravel()
                grad_norm += np.sum(gradient ** 2)

                # Accumulate loss
                total_loss += float(error[0] ** 2) / 2.0

            total_loss /= n
            grad_norm = np.sqrt(grad_norm)

            # Store history
            self.history['loss'].append(total_loss)
            self.history['coefficients'].append(coeffs.copy())

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
            X: (n, m) array
            y: (n,) array
            coeffs_init: (m+1,) array

        Returns:
            Trained coefficients
        """
        self.reset_history()
        X = np.array(X)
        y = np.array(y)
        coeffs = np.array(coeffs_init, dtype=float)
        n = len(y)

        for _iteration in range(self.max_iterations):
            # Shuffle indices for this epoch
            indices = np.random.permutation(n)

            total_loss = 0.0
            epoch_grad_norm = 0.0

            # Process mini-batches
            for batch_start in range(0, n, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n)
                batch_indices = indices[batch_start:batch_end]

                # Get batch data
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                batch_size_actual = len(batch_indices)

                # Compute gradient for batch
                X_with_bias = np.column_stack(
                    [np.ones(batch_size_actual), X_batch])
                predictions = X_with_bias @ coeffs
                errors = predictions - y_batch

                # Average gradient over batch
                gradients = X_with_bias.T @ errors / batch_size_actual

                # Update coefficients
                coeffs -= self.learning_rate * gradients
                epoch_grad_norm += np.sum(gradients ** 2)

                # Accumulate loss
                total_loss += np.sum(errors ** 2) / 2.0

            total_loss /= n
            epoch_grad_norm = np.sqrt(epoch_grad_norm)

            # Store history
            self.history['loss'].append(total_loss)
            self.history['coefficients'].append(coeffs.copy())

            if epoch_grad_norm < self.tolerance:
                break

        return coeffs
