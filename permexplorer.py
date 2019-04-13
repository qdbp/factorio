from itertools import permutations
from math import factorial

import numpy as np
import numpy.linalg as nlg
import numpy.random as npr
from tqdm import tqdm


def genperms(A):

    assert A.shape[1] == A.shape[0]
    assert A.ndim == 2

    I = np.eye(A.shape[1], dtype=np.int32)

    out = np.zeros((factorial(A.shape[1]), ) + (A.shape), dtype=np.int32)

    for px, perm in enumerate(permutations(range(A.shape[1]), A.shape[1])):
        P = I[list(perm)]
        out[px] = P @ A @ P.T

    return out


POWS = np.array([2**(9 - i) for i in range(10)])


def test():
    def predicate(mat: np.ndarray) -> bool:

        if mat.shape[0] == 1:
            return True

        digits = POWS[-mat.shape[0]:] @ mat

        return np.all((digits[1:] - digits[0]) <= 0) and predicate(mat[1:, 1:])

    for i in tqdm(range(3, 6)):
        for _ in tqdm(range((20 * i)**2)):

            A = npr.binomial(1, 0.5, size=(i, i))

            perms = genperms(A)

            n_true = sum(int(predicate(perm)) for perm in perms)

            if n_true == 0:
                print(A)
                for perm in perms:
                    print(perm)
                    print(predicate(perm))
                    print(nlg.eig(perm))
                raise ValueError(f'FAIL')
            else:
                print(n_true)


if __name__ == '__main__':

    test()
    fail

    A = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 0]])
    B = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]])

    for mat in (A, B):
        n = mat.shape[0]
        powrow = np.array([2**i for i in range(n)])
        perms, pmats = genperms(mat)
        for perm, pmat in zip(perms, pmats):
            # print('perm')
            # print(perm)
            # print('pmat')
            # print(pmat)
            # print('colsum')
            csum = powrow.T @ perm
            print(csum)
            print('')
            if all(np.diff(csum) <= 0):
                print('WIN')
