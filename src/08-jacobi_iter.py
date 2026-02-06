import math


def euclidean_dist(n, x1, x2):
    diff_val = 0.0
    for i in range(n):
        diff_val += (x1[i] - x2[i]) ** 2
    return math.sqrt(diff_val)


def jacobi_iter(A, b, x0=None, eps=1e-10, max_iter=100, debug=False):
    len_A = len(A)
    if x0 is None:
        x0 = [0.0] * len_A
    x = x0.copy()

    # x1(k+1) = (b1 - a12 * x2(k)) / a11
    # x2(k+1) = (b2 - a21 * x1(k)) / a22
    # ...
    for k in range(max_iter):
        x_new = [0.0] * len_A

        for i in range(len_A):
            sum_ax = 0.0
            for j in range(len_A):
                if j != i:
                    sum_ax += A[i][j] * x[j]
            x_new[i] = (b[i] - sum_ax) / A[i][i]

        euclid_dist = euclidean_dist(len_A, x_new, x)

        if debug:
            print_x = [f'{xi:.6f}' for xi in x]
            print(f"Iteration {k+1}: x = {print_x}")

        if euclid_dist < eps:
            print("Converged!")
            return x_new

        x = x_new.copy()
    return x


def solve_jacobi(A, b, debug=False):
    x_sol = jacobi_iter(A, b, debug=debug)
    print_x = [f"{x:.2f}" for x in x_sol]
    print(f"\nSolution: {print_x}")
    print("==============================\n")


def gen_jacobi_matrix(n):
    J = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = 2
            elif abs(i - j) == 1:
                J[i][j] = -1
            else:
                J[i][j] = 0
    return J


solve_jacobi(
    debug=True,
    A=[
        [2, 1],
        [1, -3]
    ],
    b=[3, -2],
)

solve_jacobi(
    debug=True,
    A=gen_jacobi_matrix(5),
    b=[1, 0, 0, 0, 1],
)
