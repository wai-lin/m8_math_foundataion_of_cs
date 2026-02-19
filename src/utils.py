"""
Utility functions for gradient descent visualization.
"""

import numpy as np


def generate_1d_data(num_samples=50, slope=2.0, intercept=3.0, noise_std=0.5):
    """
    Generate 1D linear regression data.

    Args:
        num_samples: Number of samples to generate
        slope: True slope of the line
        intercept: True intercept of the line
        noise_std: Standard deviation of noise

    Returns:
        (X, y) where X is (n, 1) array, y is (n,) array
    """
    np.random.seed(42)
    X = np.linspace(-5, 5, num_samples).reshape(-1, 1)
    noise = np.random.normal(0, noise_std, num_samples)
    y = slope * X.ravel() + intercept + noise

    return X, y


def generate_2d_data(num_samples=100, noise_std=0.5):
    """
    Generate 2D linear regression data (fitting a plane).

    Args:
        num_samples: Number of samples to generate
        noise_std: Standard deviation of noise

    Returns:
        (X, y) where X is (n, 2) array, y is (n,) array
    """
    np.random.seed(42)
    # True plane: y = 2*x1 - x2 + 3
    X1 = np.random.uniform(-2, 2, num_samples)
    X2 = np.random.uniform(-2, 2, num_samples)
    X = np.column_stack([X1, X2])

    noise = np.random.normal(0, noise_std, num_samples)
    y = 2 * X1 - X2 + 3 + noise

    return X, y


def get_algorithm_info():
    """Return information about available algorithms."""
    return {
        'vanilla_gd': {
            'name': 'Vanilla Gradient Descent',
            'description': 'Batch gradient descent - updates using entire dataset',
            'hyperparameters': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
            }
        },
        'sgd': {
            'name': 'Stochastic Gradient Descent',
            'description': 'Updates using individual samples (noisy but fast)',
            'hyperparameters': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
            }
        },
        'mini_batch': {
            'name': 'Mini-batch Gradient Descent',
            'description': 'Balances batch GD and SGD using small batches',
            'hyperparameters': {
                'learning_rate': 0.01,
                'max_iterations': 1000,
                'batch_size': 32,
            }
        }
    }
