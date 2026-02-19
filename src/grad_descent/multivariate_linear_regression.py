"""
Generalized Multivariate Linear Regression via Gradient Descent
Supports arbitrary dimensionality with configurable noise and learning rates.

Linear Model: y = c[0]*x[0] + c[1]*x[1] + ... + c[d-1]*x[d-1]
Vectorized: y = X @ c
Loss: L = ||X @ c - y||²
Gradient: ∇L = 2 * X^T @ (X @ c - y)
Update: c := c - α * ∇L
"""

import numpy as np


def generate_data(dim, n_samples, true_coeffs, noise_std=0.0, seed=42):
    """
    Generate synthetic linear regression data.

    Args:
        dim: Dimensionality (number of features)
        n_samples: Number of data points
        true_coeffs: True coefficient vector c (shape: (dim,))
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        X: Data matrix (n_samples, dim)
        y: Target vector (n_samples,)
    """
    np.random.seed(seed)

    # Generate random features in [-1, 1]
    X = np.random.uniform(-1, 1, size=(n_samples, dim))

    # Compute exact targets
    y_exact = X @ true_coeffs

    # Add noise if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=y_exact.shape)
        y = y_exact + noise
    else:
        y = y_exact

    return X, y, y_exact


def gradient_descent(X, y, dim, alpha=0.01, max_iters=10000, eps=1e-6, verbose=True):
    """
    Perform gradient descent for multivariate linear regression.

    Args:
        X: Data matrix (n_samples, dim)
        y: Target vector (n_samples,)
        dim: Number of features
        alpha: Learning rate (step size)
        max_iters: Maximum iterations
        eps: Convergence tolerance (for gradient norm)
        verbose: Whether to print iteration details

    Returns:
        c: Learned coefficients (dim,)
        history: List of (iteration, c, loss, grad_norm)
    """
    # Initialize coefficients
    c = np.ones(dim)
    history = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"Gradient Descent: dim={dim}, α={alpha}")
        print(f"{'='*70}")
        print(f"{'Iter':>6} | {'Loss':>12} | {'||∇L||':>12} | {'||Δc||':>12}")
        print(f"{'-'*70}")

    for i in range(max_iters):
        c_prev = np.copy(c)

        # Compute gradient: ∇L = 2 * X^T @ (X @ c - y)
        residual = X @ c - y
        grad = 2 * X.T @ residual

        # Update step
        c -= alpha * grad

        # Compute loss and gradient norm
        loss = np.linalg.norm(residual) ** 2
        grad_norm = np.linalg.norm(grad)
        step_norm = np.linalg.norm(c - c_prev)

        history.append((i, c.copy(), loss, grad_norm))

        # Print progress
        if verbose and (i % max(1, max_iters//20) == 0 or i < 10 or grad_norm < eps):
            print(f"{i:6d} | {loss:12.6f} | {grad_norm:12.6e} | {step_norm:12.6e}")

        # Check convergence
        if grad_norm < eps:
            if verbose:
                print(
                    f"{i+1:6d} | {loss:12.6f} | {grad_norm:12.6e} | [CONVERGED]")
            break

    if verbose:
        print(f"{'='*70}\n")

    return c, history


def experiment(dim, n_samples=1000, noise_std=0.0, alpha=0.001, seed=42):
    """
    Run a complete experiment: generate data, fit model, and report results.

    Args:
        dim: Dimensionality
        n_samples: Number of samples
        noise_std: Noise standard deviation
        alpha: Learning rate
        seed: Random seed
    """
    # Generate true coefficients: alternating pattern for clarity
    true_c = np.array([2.0 if i % 2 == 0 else -1.0 for i in range(dim)])

    print(f"\n{'#'*70}")
    print(
        f"EXPERIMENT: dim={dim}, n_samples={n_samples}, noise_std={noise_std}, α={alpha}")
    print(f"{'#'*70}")
    print(f"True coefficients: {true_c}")

    # Generate data
    X, y, y_exact = generate_data(
        dim, n_samples, true_c, noise_std=noise_std, seed=seed)

    # Fit model
    c_learned, history = gradient_descent(X, y, dim, alpha=alpha, max_iters=10000,
                                          eps=1e-6, verbose=True)

    # Report results
    error = np.linalg.norm(c_learned - true_c)
    print(f"Learned coefficients:  {c_learned}")
    print(f"True coefficients:     {true_c}")
    print(f"Coefficient errors:    {np.abs(c_learned - true_c)}")
    print(f"Total error ||Δc||:    {error:.6e}")

    # Final loss
    final_residual = X @ c_learned - y
    final_loss = np.linalg.norm(final_residual) ** 2
    print(f"Final loss:            {final_loss:.6f}")

    return c_learned, history, true_c


# ============================================================================
# EXPERIMENTS
# ============================================================================

print("="*70)
print("MULTIVARIATE LINEAR REGRESSION - GRADIENT DESCENT EXPERIMENTS")
print("="*70)

# Experiment 1: 3D (reference - no noise, verify it works)
print("\n" + "▶"*35)
print("EXPERIMENT 1: 3D, NO NOISE (verify reference code works)")
print("▶"*35)
c1, hist1, true_c1 = experiment(
    dim=3, n_samples=1000, noise_std=0.0, alpha=0.001, seed=42)

# Experiment 2: 3D with noise
print("\n" + "▶"*35)
print("EXPERIMENT 2: 3D WITH NOISE (std=0.5)")
print("▶"*35)
c2, hist2, true_c2 = experiment(
    dim=3, n_samples=1000, noise_std=0.5, alpha=0.001, seed=42)

# Experiment 3: 5D with noise (increased dimensionality)
print("\n" + "▶"*35)
print("EXPERIMENT 3: 5D WITH NOISE (std=0.5)")
print("▶"*35)
c3, hist3, true_c3 = experiment(
    dim=5, n_samples=1000, noise_std=0.5, alpha=0.001, seed=42)

# Experiment 4: 10D with noise (higher dimensionality)
print("\n" + "▶"*35)
print("EXPERIMENT 4: 10D WITH NOISE (std=0.5) - Adjusted learning rate")
print("▶"*35)
c4, hist4, true_c4 = experiment(
    dim=10, n_samples=1000, noise_std=0.5, alpha=0.0005, seed=42)

# Experiment 5: 20D with noise (very high dimensionality)
print("\n" + "▶"*35)
print("EXPERIMENT 5: 20D WITH NOISE (std=0.5) - Small learning rate")
print("▶"*35)
c5, hist5, true_c5 = experiment(
    dim=20, n_samples=1000, noise_std=0.5, alpha=0.0001, seed=42)

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF ALL EXPERIMENTS")
print("="*70)
print(f"{'Dim':>4} | {'Noise':>6} | {'α':>8} | {'Iters*':>6} | {'Error':>12} | {'Loss':>12}")
print("-"*70)

experiments = [
    (3, 0.0, 0.001, c1, hist1, true_c1),
    (3, 0.5, 0.001, c2, hist2, true_c2),
    (5, 0.5, 0.001, c3, hist3, true_c3),
    (10, 0.5, 0.0005, c4, hist4, true_c4),
    (20, 0.5, 0.0001, c5, hist5, true_c5),
]

for dim, noise, alpha, c, hist, true_c in experiments:
    iters = len(hist) - 1
    error = np.linalg.norm(c - true_c)
    final_loss = hist[-1][2]
    print(f"{dim:4d} | {noise:6.1f} | {alpha:8.4f} | {iters:6d} | {error:12.6e} | {final_loss:12.6f}")

print("="*70)
print("* Iterations until convergence (||∇L|| < 1e-6)")
print("="*70)

# ============================================================================
# DETAILED OUTPUT FOR MAIN EXPERIMENT (Exp 4: 10D)
# ============================================================================
print("\n" + "="*70)
print("DETAILED ITERATION LOG: EXPERIMENT 4 (10D with noise)")
print("="*70)
print(f"{'Iter':>6} | {'Loss':>12} | {'||∇L||':>12} | Learned coefficients")
print("-"*70)

for i, (iter_num, c_vals, loss, grad_norm) in enumerate(hist4):
    coeff_str = ", ".join([f"{x:7.4f}" for x in c_vals])
    if i % max(1, len(hist4)//15) == 0 or i == len(hist4)-1:
        print(
            f"{iter_num:6d} | {loss:12.6f} | {grad_norm:12.6e} | [{coeff_str}]")

print("="*70)
