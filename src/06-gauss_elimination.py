zero_check = 1e-9


def guss_elimination(A, b):
    n = len(A)

    for i in range(n-1):

        if abs(A[i][i]) < zero_check:
            print(f"Zero pivot. Row {i}.")
            swapped = False

            for k in range(i+1, n):
                if abs(A[k][i]) > zero_check:
                    A[i], A[k] = A[k], A[i]
                    b[i], b[k] = b[k], b[i]
                    swapped = True
                    break

            if not swapped:
                print("Matrix is singular. No unique solution.")
                return None, None, None

        print(f"\nStep {i+1}:")
        for j in range(i+1, n):
            curr_row = i
            next_row = j

            mul = - (A[next_row][i] / A[curr_row][i])  # multiplicator
            print(f"  Row{next_row} x Row{curr_row}; Mul:{mul}")

            for k in range(i+1, n):
                A[next_row][k] += mul * A[curr_row][k]
            b[next_row] += mul * b[curr_row]

            print(f"  Updated row {next_row}: {A[next_row]}: {b[next_row]}")

    if abs(A[n-1][n-1]) < zero_check:
        print("Matrix is singular. No unique solution.")
        return None, None, None

    x_sol = [0] * n
    for i in range(n - 1, -1, -1):
        sum_xi = 0
        for j in range(i + 1, n):
            sum_xi += A[i][j] * x_sol[j]
        x_sol[i] = (b[i] - sum_xi) / A[i][i]

    return A, b, x_sol


def solve_with_print(A, b):
    print(f"\nMatrix A: {A}")
    print(f"Vector b: {b}")
    A_upper, b_upper, x_solution = guss_elimination(A, b)
    print(f"\n\nA: {A_upper}")
    print(f"b: {b_upper}")
    print(f"Solution: {x_solution}")
    print("\n==============================")
    print("\n==============================")


solve_with_print(
    A=[
        [1, 1, 1],
        [3, 2, 1],
        [2, -1, 4],
    ],
    b=[6, 10, 12]
)

# Should trigger row swap and handle zero pivot
solve_with_print(
    A=[
        [1, 2, 3],
        [2, 5, 2],
        [3, 7, 5],
    ],
    b=[14, 18, 20]
)
