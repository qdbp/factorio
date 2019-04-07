import typing as ty

import numpy as np
import numpy.linalg as nlg
from numpy.linalg import matrix_power as mpow

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)


def investigate(mat, n_inps, n_outs):

    assert mat.shape[0] == mat.shape[1]
    assert n_inps + n_outs <= mat.shape[0]

    pw = mat
    for i in range(10):
        pw = mat @ pw
        print(pw.sum(axis=0))

    a_inp = np.zeros(mat.shape[0])
    a_inp[0:n_inps] = 1


def extend_splvec(
        splvecs: np.ndarray, input_map: ty.List[int], n_outputs: int
) -> np.ndarray:

    n_inps = len(input_map)

    final_mat = np.zeros(
        (n_inps + splvecs.shape[0] + n_outputs, n_inps + splvecs.shape[1])
    )

    final_mat[-n_outputs:, -n_outputs:] = np.eye(n_outputs)
    final_mat[n_inps:-n_outputs, n_inps:] = splvecs
    for inx, inp in enumerate(input_map):
        if inp >= splvecs.shape[0]:
            print(f'Mapping input {inx} straight to an output!')
        inp += n_inps
        final_mat[inx, inp] = 1

    return final_mat


if __name__ == '__main__':

    # simple self-feeder
    # print('simgple self-feeder')
    # mat = np.array([[0, 1, 0], [0, .5, .5], [0, 0, 1]])
    # investigate(mat)

    # output density = 1/2
    mat = np.array([[0, 1, 0, 0, 0], [0, 0, .5, .5, 0], [0, 1, 0, 0, 0],
                    [0, 0, .5, 0, .5], [0, 0, 0, 0, 1]])

    print('first nonworking case')
    investigate(mat, 1, 1)

    # 2-2 lane balancer
    # | |
    # S^P
    # | |

    print('basic 2-2 balancer')
    spl_vec = np.array([[0, .5, .5]])
    mat = extend_splvec(spl_vec, [0, 0], 2)
    investigate(mat, 2, 2)
