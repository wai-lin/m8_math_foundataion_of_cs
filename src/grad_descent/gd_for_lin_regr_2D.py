"""
Gradient Descent for 2D Bowl Function
f(x, y) = (x - 2)² + y²
Hypothesis: Φ(x, y, c) = c[0] + c[1]·(x-2)² + c[2]·y²
"""

import numpy as np


def test_bowl(x, y):
    """The true bowl-shaped function"""
    return (x - 2)**2 + y**2


def Phi(x, y, c):
    """Hypothesis function: Φ(x, y, c) = c[0] + c[1]·(x-2)² + c[2]·y²"""
    return c[0] + c[1]*(x-2)**2 + c[2]*y**2


def grad_Phi(x_vals, y_vals, c):
    """
    Compute gradient of loss function with respect to [c0, c1, c2]
    ∇L = 2 * Σ(residual_i) * [1, (xi-2)², yi²]
    """
    grad_c = np.array([0.0, 0.0, 0.0])

    for i in range(len(x_vals)):
        ri = Phi(x_vals[i], y_vals[i], c) - test_bowl(x_vals[i], y_vals[i])
        grad_c[0] += ri
        grad_c[1] += ri * (x_vals[i] - 2)**2
        grad_c[2] += ri * y_vals[i]**2

    grad_c *= 2
    return grad_c


# Generate 2D data: sample from a grid over the bowl
n = 5  # Smaller grid for faster computation
x_vals = np.linspace(0, 4, n)
y_vals = np.linspace(-2, 2, n)
# Flatten the grid for loss computation
x_data = np.repeat(x_vals, n)
y_data = np.tile(y_vals, n)

print("="*60)
print("GRADIENT DESCENT FOR 2D BOWL FUNCTION")
print("="*60)
print(f"True function: f(x,y) = (x-2)² + y²")
print(f"Hypothesis: Φ(x,y,c) = c[0] + c[1]·(x-2)² + c[2]·y²")
print(f"True parameters: c = [0, 1, 1]")
print(f"Data: {len(x_data)} points from grid [0,4] × [-2,2]")
print(f"Initial guess: c = [1.0, 0.5, 0.5]")
print(f"Step size (α) = 0.0001")
print("="*60 + "\n")

# Gradient descent parameters
c = np.array([1.0, 0.5, 0.5])  # initial guess
c_prev = np.copy(c)
alpha = 0.0001  # Much smaller step size for stability
max_iters = 5_000
eps = 1e-9

print(f"{'Iter':>6} | {'c0':>13} | {'c1':>13} | {'c2':>13} | {'||∇L||':>10}")
print("-"*70)

for i in range(max_iters):
    c_prev = np.copy(c)
    grad = grad_Phi(x_data, y_data, c)
    c -= alpha * grad

    if i % max(1, max_iters//25) == 0 or i < 15:
        grad_norm = np.linalg.norm(grad)
        print(
            f"{i:6d} | {c[0]:13.9f} | {c[1]:13.9f} | {c[2]:13.9f} | {grad_norm:.2e}")

    if np.linalg.norm(c - c_prev) < eps:
        grad_norm = np.linalg.norm(grad)
        print(
            f"{i+1:6d} | {c[0]:13.9f} | {c[1]:13.9f} | {c[2]:13.9f} | [CONV]")
        break

print("="*60)
print(f"Final solution:  c = {c}")
print(f"True solution:   c = [0, 1, 1]")
print(
    f"Errors: Δc0={abs(c[0]):.2e}, Δc1={abs(c[1]-1):.2e}, Δc2={abs(c[2]-1):.2e}")
print("="*60)
