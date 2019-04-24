from factorio.balancer_solver import solve_balancers
import factorio.solver_core as core


def _test_values(M, N, answer):

    solver = core.get_solver(verbose=False)
    ans = solve_balancers(
        M=M, N=N, max_spls=(answer + 2), min_spls=0, solver=solver
    )
    adjmat = ans[2]
    assert len(adjmat) == answer


for M, N, answer in [
    (1, 1, 0),
    (2, 2, 1),
    (3, 3, 4),
    (4, 4, 4),
    (1, 8, 7),
    (3, 7, 8),
    (4, 5, 11),
    (8, 8, 12),
    (1, 16, 15),
]:

    exec(f"def test_{M}_{N}(): _test_values({M}, {N}, {answer})")
