##################################################
##################################################

def log(i, x, f_val):
    print(f"iter: {i+1}, x: {x:2.6f}, f(x): {f_val:2.6f}")


def get_fixed_point(algo, x0, eps=0.001, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_next = algo(x)
        log(i, x_next, f(x_next))
        if abs(x_next - x) < eps:
            return x_next
        x = x_next
    print("!!Did not converge max iterations.")
    return x


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


a, b = (0, 3)


def f(x):
    return x**2 - 4


def f_prime(x):
    return 2*x

##################################################
##################################################


alp = -1/10
eps = 0.001

print("Relaxation fixed point algorithm")
get_fixed_point(
    algo=lambda x: x + (alp * f(x)),
    x0=3,
    eps=eps,
)

print("")

# newton algorithm
print("Newton algorithm")
get_fixed_point(
    algo=lambda x: x - f(x) / f_prime(x),
    x0=3,
    eps=eps,
)

print("")

# bisection algorithm
print("Bisection algorithm")
get_bisection(
    point=(a, b),
    func=f,
    eps=eps,
)

##################################################
##################################################
