"""
Bisection for functions with multiple roots
"""
import math


def log(i, x, f_val):
    print(f"iter: {i+1}, x: {x:2.6f}, f(x): {f_val:2.6f}")


def get_bisection(point, func, eps=0.001, max_iter=100):
    a, b = point

    for i in range(max_iter):
        c = (a + b) / 2.0
        f_val = func(c)
        log(i, c, f_val)

        if func(a) * func(c) < 0:
            b = c
        elif func(c) * func(b) < 0:
            a = c
        else:
            if func(a) == 0:
                c = a
            elif func(b) == 0:
                c = b

        if abs(f_val) < eps or abs(b - a) < eps:
            print("Converged")
            return c

    print("!!Did not converge max iterations.")
    return (a + b) / 2.0

##################################################
##################################################


def f(x):
    return x**4 + (3 * x**3) + x**2 - (2 * x) - 0.5


x0, x1 = (-3, 2)
segs_count = 100
step_size = (x1 - x0) / segs_count

found_roots = []

for i in range(segs_count):
    a = x0 + (i * step_size)
    b = a + step_size

    if f(a) * f(b) < 0:
        root = get_bisection((a, b), f)
        found_roots.append(root)

print(f"All roots found: {found_roots}")
