# A = [[1,1,1], [3, 2, 1], [2, -1, 4]]
# b = [6, 10, 12]

def gauss_elimination(A, b):
    print(f"\nMatrix A: {A}")
    print(f"\nVector b: {b}")
    n = len(A)

    # Forward pass of elimination
    for i in range(n-1):
        for j in range(i+1, n):
            k_temp = - A[j][i] / A[i][i]
            # print(f"Step {i+1} {j+1}: k = {k_temp:.2f}")

            temp_row = [a * k_temp for a in A[i]]
            # print(f"Temp row: {temp_row}")

            for k in range(n):
                A[j][k] += temp_row[k]

            b[j] += k_temp * b[i]

            # print(f"Updated A = {A}")
            # print(f"Updated b = {b}")

    x_sol = [None for _ in range(n)]
    for i in range(n):
        row_idx = n - i - 1
        res = b[row_idx]
        for j in range(n - i, n):
            res -= A[row_idx][j] * x_sol[j]
        x_sol[row_idx] = res / A[row_idx][row_idx]

        # print(f"x_{n-i-1} = {x_sol[n-i-1]:.2f}")

    print(f"\nSolution: {x_sol}")
    return x_sol


def gen_hilbert_matrix(n):
    H = [[1 / (i + j + 1) for j in range(n)] for i in range(n)]
    return H

# gauss_elimination(A, b)


print("\n==============================")
print("\n==============================")
print("\nn = 5")
A = gen_hilbert_matrix(5)
b = [sum(row) for row in A]
gauss_elimination(A, b)

print("\n==============================")
print("\n==============================")
print("\nn = 10")
A = gen_hilbert_matrix(10)
b = [sum(row) for row in A]
gauss_elimination(A, b)

print("\n==============================")
print("\n==============================")
print("\nn = 20")
A = gen_hilbert_matrix(20)
b = [sum(row) for row in A]
gauss_elimination(A, b)

# A = gen_hilbert_matrix(50)
# b = [sum(row) for row in A]
# gauss_elimination(A, b)

# A = gen_hilbert_matrix(100)
# b = [sum(row) for row in A]
# gauss_elimination(A, b)
