
import simplex


c = np.array([5, 6, 1])
A = np.array([
    [2, 1, 1],
    [1, 2, 0],
    [0, 0.5, 1]
])
b = np.array([5, 3, 8])

c_dual = b
A_dual = -A.T
b_dual =-c
simplex_dual = SimplexMethod(c_dual, A_dual, b_dual, "min")
if simplex_dual.solution():
    simplex_dual.print_table()
    simplex_dual.print_solution()
    simplex_dual.check_solution()
