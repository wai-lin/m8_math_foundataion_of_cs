def get_bisection(func, point):
    a, b = point

    max_iter = 100
    for i in range(max_iter):
        c = (a + b) / 2.0

        if func(a)*func(c) < 0:
            b = c
        elif func(c)*func(b) < 0:
            a = c
        else:
            if func(a) == 0:
                c = a
            elif func(b) == 0:
                c = b

        print(i, c, func(c))


def f(x):
    return x**2 - 4


print("==================================================")
print("f(x) = x**2 - 4")
print("==================================================\n")
point = (0, 3)
get_bisection(f, point)


def f(x):
    return x**3 - 2 + x**2


print("\n\n==================================================")
print("f(x) = x**3 - 2 + x**2")
print("==================================================\n")
point = (-1, 5)
get_bisection(f, point)
