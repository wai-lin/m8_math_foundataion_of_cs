"""
Utility functions for gradient descent visualization.
"""


def generate_1d_data(num_samples=50, slope=2.0, intercept=3.0, noise_std=0.5):
    """
    Generate 1D linear regression data.

    Args:
        num_samples: Number of samples to generate
        slope: True slope of the line
        intercept: True intercept of the line
        noise_std: Standard deviation of noise

    Returns:
        (X, y) where X is list of x values, y is list of y values
    """
    import random

    random.seed(42)
    X = []
    y = []

    for i in range(num_samples):
        x = (i / num_samples) * 10 - 5  # Range [-5, 5]
        noise = random.gauss(0, noise_std)
        y_true = slope * x + intercept
        X.append([x])
        y.append(y_true + noise)

    return X, y


def generate_2d_data(num_samples=100, noise_std=0.5):
    """
    Generate 2D linear regression data (fitting a plane).

    Args:
        num_samples: Number of samples to generate
        noise_std: Standard deviation of noise

    Returns:
        (X, y) where X is list of [x1, x2] pairs, y is list of y values
    """
    import random

    random.seed(42)
    X = []
    y = []

    # True plane: y = 2*x1 - x2 + 3
    for i in range(num_samples):
        x1 = ((i % 10) / 10) * 4 - 2
        x2 = ((i // 10) / 10) * 4 - 2
        noise = random.gauss(0, noise_std)
        y_true = 2 * x1 - x2 + 3
        X.append([x1, x2])
        y.append(y_true + noise)

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
