from balancer_solver import solve_balancers


def test_suite():
    for M, N, answer in [
        (1, 1, 0),
        (2, 2, 1),
        (3, 3, 4),
        (4, 4, 4),
        (1, 8, 7),
        (3, 7, 8),
        (5, 5, 11),
        (8, 8, 12),
        (1, 16, 15),
    ]:

        try:
            ans = solve_balancers(M, N, (answer + 3), 0)
            adjmat = ans[2]
            assert len(adjmat) == answer
        except AssertionError:
            print(f'Failed to find {M} -> {N} solution of {answer}')
            print(f'Found solution with {answer} only')
            return
        except Exception:
            print(f'Failed to find {M} -> {N} solution of {answer}')
            return


if __name__ == '__main__':
    test_suite()
