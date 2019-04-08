import typing as ty
import numpy as np
import pulp as pp


class Infeasible(Exception):
    pass


def dicts_to_ndarray(
        dicts,
        index_sets: ty.Tuple[ty.Sized, ...],
        dtype=np.float32,
) -> np.ndarray:

    shape = tuple(len(ixset) for ixset in index_sets)

    def _rworker(plane: np.ndarray, d: ty.Dict, index_sets):
        if len(index_sets) > 1:
            for ex, elem in enumerate(index_sets[0]):
                _rworker(plane[ex, ...], d[elem], index_sets[1:])
        else:
            for ex, elem in enumerate(index_sets[0]):
                plane[ex] = pp.value(d[elem])

    out = np.zeros(shape, dtype=dtype)
    _rworker(out, dicts, index_sets)

    return out
