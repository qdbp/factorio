import os
import typing as ty

import numpy as np
import pulp as pp


class Infeasible(Exception):
    pass


def get_solver(which='coin'):
    if which == 'coin':
        return pp.solvers.COIN_CMD(
            msg=1,
            threads=(os.cpu_count() or 1),
        )
    elif which == 'glpk':
        return pp.solvers.GLPK()


def dicts_to_ndarray(
        dicts,
        index_sets: ty.Tuple[ty.Sized, ...],
        dtype=np.float32,
) -> np.ndarray:

    shape = tuple(len(ixset) for ixset in index_sets)

    def _rworker(subdim: np.ndarray, d: ty.Dict, index_sets):
        if len(index_sets) > 1:
            for ex, elem in enumerate(index_sets[0]):
                _rworker(subdim[ex, ...], d[elem], index_sets[1:])
        else:
            for ex, elem in enumerate(index_sets[0]):
                subdim[ex] = pp.value(d[elem])

    out = np.zeros(shape, dtype=dtype)
    _rworker(out, dicts, index_sets)

    return out


def ndarray_to_dicts(
        arr,
        index_sets: ty.Tuple[ty.Sized, ...],
) -> ty.Dict:
    '''
    The inverse of dicts_to_ndarray.
    '''

    assert arr.ndim == len(index_sets)

    def _rworker(subdim: np.ndarray, index_sets) -> ty.Dict[ty.Any, ty.Any]:
        head_set = index_sets[0]
        leaf = subdim.ndim == 1
        return {
            ix: (subdim[ex] if leaf else _rworker(subdim[ex], index_sets[1:]))
            for ex, ix in enumerate(head_set)
        }

    return _rworker(arr, index_sets)
