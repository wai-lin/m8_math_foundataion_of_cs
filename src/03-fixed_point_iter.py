# g(x) = sqrt(x+2)
# [a, b] = [1, 4]
# x_0 = 3

import math

def g(x):
    return math.sqrt(x + 2)

eps = 0.001
x0 = 3

def log(i, x):
    print(f"iter: {i+1}, x: {x:2.6f}")

def fixed_point():
    max_iter = 100
    x = x0
    for i in range(max_iter):
        x_next = g(x)
        log(i, x_next)
        if abs(x_next - x) < eps:
            return x_next
        x = x_next
    return x

fixed_point()
