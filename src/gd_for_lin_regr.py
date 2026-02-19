import numpy as np


def test_line(x):
    return 2*x + 3


def Phi(x, c):
    return c[0] + c[1]*x


def grad_Phi(x, c):
    grad_c0 = 0
    grad_c1 = 0
    for i in range(len(x)):
        ri = Phi(x[i], c) - test_line(x[i])
        grad_c0 += ri
        grad_c1 += ri * x[i]
    grad_c0 *= 2
    grad_c1 *= 2
    return np.array([grad_c0, grad_c1])


n = 10
x_vals = np.linspace(0, 5, n)
y_vals = test_line(x_vals)

c_prev = np.array([1.0, 1.0])
alpha = 0.01
max_iters = 1000
eps = 0.000_000_000_1

c = np.copy(c_prev)
for i in range(max_iters):
    c_prev = np.copy(c)
    c -= alpha * grad_Phi(x_vals, c)
    print(f"{i}: c = {c}")

    if np.linalg.norm(c - c_prev) < eps:
        break

print(f"Solution: {c}")
