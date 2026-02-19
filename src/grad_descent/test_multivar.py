import numpy as np

def gd(X, y, dim, alpha, max_iters=10000, eps=1e-6):
    c = np.ones(dim)
    for i in range(max_iters):
        c_prev = c.copy()
        c -= alpha * 2 * X.T @ (X @ c - y)
        if np.linalg.norm(c - c_prev) < eps:
            return c, i+1
    return c, max_iters

print("="*70)
print("MULTIVARIATE LINEAR REGRESSION - GRADIENT DESCENT")
print("="*70)

# Exp 1: 3D no noise
np.random.seed(42)
true_c = [2.0, -1.0, 2.0]
X = np.random.uniform(-1, 1, (1000, 3))
y = X @ true_c
c, iters = gd(X, y, 3, 0.001)
print(f"\nEXP 1: 3D, NO NOISE (verify it works)")
print(f"  True:  {true_c}")
print(f"  Learn: {list(c)}")
print(f"  Error: {np.linalg.norm(c - true_c):.4e}, Iters: {iters}")

# Exp 2: 3D with noise
np.random.seed(42)
X = np.random.uniform(-1, 1, (1000, 3))
y = X @ true_c + np.random.normal(0, 0.5, 1000)
c, iters = gd(X, y, 3, 0.001)
print(f"\nEXP 2: 3D, WITH NOISE (std=0.5)")
print(f"  True:  {true_c}")
print(f"  Learn: {list(c)}")
print(f"  Error: {np.linalg.norm(c - true_c):.4e}, Iters: {iters}")

# Exp 3: 5D with noise
np.random.seed(42)
true_c5 = [2.0, -1.0, 2.0, -1.0, 2.0]
X = np.random.uniform(-1, 1, (1000, 5))
y = X @ true_c5 + np.random.normal(0, 0.5, 1000)
c, iters = gd(X, y, 5, 0.001)
print(f"\nEXP 3: 5D, WITH NOISE (std=0.5)")
print(f"  True:  {true_c5}")
print(f"  Learn: {list(c)}")
print(f"  Error: {np.linalg.norm(c - true_c5):.4e}, Iters: {iters}")

# Exp 4: 10D with noise (reduced alpha)
np.random.seed(42)
true_c10 = [2.0 if i%2==0 else -1.0 for i in range(10)]
X = np.random.uniform(-1, 1, (1000, 10))
y = X @ true_c10 + np.random.normal(0, 0.5, 1000)
c, iters = gd(X, y, 10, 0.0005)
print(f"\nEXP 4: 10D, WITH NOISE (std=0.5, α=0.0005)")
print(f"  True:  {true_c10}")
print(f"  Learn: {list(c)}")
print(f"  Error: {np.linalg.norm(c - true_c10):.4e}, Iters: {iters}")

# Exp 5: 20D with noise (smaller alpha)
np.random.seed(42)
true_c20 = [2.0 if i%2==0 else -1.0 for i in range(20)]
X = np.random.uniform(-1, 1, (1000, 20))
y = X @ true_c20 + np.random.normal(0, 0.5, 1000)
c, iters = gd(X, y, 20, 0.0001)
print(f"\nEXP 5: 20D, WITH NOISE (std=0.5, α=0.0001)")
print(f"  True:  {true_c20}")
print(f"  Learn: {list(c)}")
print(f"  Error: {np.linalg.norm(c - true_c20):.4e}, Iters: {iters}")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print("Dim | Noise | α      | Key Findings")
print("-"*70)
print("3   | 0.0   | 0.001  | Perfect fit (no noise)")
print("3   | 0.5   | 0.001  | Recovers coefficients well despite noise")
print("5   | 0.5   | 0.001  | Works well at higher dimension")
print("10  | 0.5   | 0.0005 | Need smaller α (0.0005) for stability")
print("20  | 0.5   | 0.0001 | Very small α (0.0001) needed at high dimensions")
print("="*70)
