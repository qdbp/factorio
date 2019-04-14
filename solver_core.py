import os
import typing as ty
from pathlib import Path

import numpy as np
import pulp as pp

SOLDIR = Path('./solutions/')


class Infeasible(Exception):
    pass


class IllSpecified(Exception):
    pass


def in_sol_dir(fn: str) -> Path:
    SOLDIR.mkdir(exist_ok=True)
    return SOLDIR.joinpath(fn)


def get_solver(which='coin', threads: int = None):
    if which == 'coin':
        return pp.solvers.COIN_CMD(
            msg=1,
            threads=(threads or os.cpu_count() or 1),
        )
    if which == 'coinmp':
        return pp.solvers.COINMP_DLL()
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
                try:
                    subdim[ex] = pp.value(d[elem])
                except AttributeError:
                    subdim[ex] = d[elem]

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


def get_dict_depth(d: ty.Any):
    is_dict = isinstance(d, dict)
    return int(is_dict) + (
        0 if not is_dict else get_dict_depth(d[list(d.keys())[0]])
    )
