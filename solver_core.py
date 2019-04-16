from __future__ import annotations

import os
import typing as ty
from itertools import product
from pathlib import Path

import numpy as np
import pulp as pp
from numpy import einsum as ein

SOLDIR = Path('./solutions/')


class Infeasible(Exception):
    pass


class IllSpecified(Exception):
    pass


def number(it: ty.Iterable) -> ty.List[int]:
    return list(ix for ix, _ in enumerate(it))


def numprod(*its: ty.Iterable
            ) -> ty.Generator[ty.Tuple[ty.Any, ...], None, None]:

    yield from product(*map(number, its))


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


class sumdict(dict):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return sum([self[k] for k in key])
        else:
            return super().__getitem__(key)


class lparray(np.ndarray):
    @classmethod
    def create(cls, name: str, index_sets, *args, **kwargs) -> lparray:
        '''
        Numpy array equivalent of pulp.LpVariable.dicts
        '''

        def _rworker(name: str, plane: np.ndarray, index_sets):
            if len(index_sets) == 1:
                close_paren = name and (')' if '(' in name else '')
                for ix in number(index_sets[0]):
                    plane[ix] = pp.LpVariable(
                        f'{name}{ix}{close_paren}', *args, **kwargs
                    )
            else:
                open_paren = name and ('(' if '(' not in name else '')
                for ix in number(index_sets[0]):
                    _rworker(
                        f'{name}{open_paren}{ix},', plane[ix], index_sets[1:]
                    )

        arr = np.zeros(
            tuple(len(ixset) for ixset in index_sets), dtype=np.object
        )
        _rworker(name, arr, index_sets)

        return arr.view(lparray)

    def __ge__(self, other):
        return np.greater_equal(self, other, dtype=object)

    def __le__(self, other):
        return np.less_equal(self, other, dtype=object)

    def __lt__(self, other):
        raise NotImplementedError('lparrays support only <=, >=, and ==')

    def __gt__(self, other):
        raise NotImplementedError('lparrays support only <=, >=, and ==')

    def __eq__(self, other):
        return np.equal(self, other, dtype=object)

    @property
    def values(self) -> np.ndarray:
        return np.vectorize(lambda x: pp.value(x))(self).view(np.ndarray)

    def sumit(self, *args, **kwargs):
        out = self.sum(*args, **kwargs)
        return out.item()

    def constrain(self, prob: pp.LpProblem, name: str = None) -> None:
        if self.ndim == 0:
            cons = self.item()
            cons.name = name
            prob += cons
            return

        def _rworker(prob, plane, name):
            if plane.ndim == 1:
                close_paren = name and (')' if '(' in name else '')
                for cx, const in enumerate(plane):
                    if not isinstance(const, pp.LpConstraint):
                        raise TypeError(
                            'Attempting to constrain problem with '
                            f'non-constraint {const}'
                        )
                    const.name = name and f'{name}{cx}{close_paren}'
                    prob += const
            else:
                open_paren = name and ('(' if '(' not in name else '')
                for px, subplane in enumerate(plane):
                    subname = name and f'{name}{open_paren}{px},'
                    _rworker(prob, subplane, subname)

        _rworker(prob, self, name)


def get_lparr_value(lparray: lparray):
    return np.vectorize(lambda x: pp.value(x))(lparray).view(np.ndarray)


if __name__ == '__main__':

    var = pp.LpVariable('Y', 0, 1, pp.LpBinary)
    c = var >= 0

    X = lparray.create('X', ([1, 2, 3], ['a', 'b', 'c']), 0, 1, pp.LpBinary)
    Y = np.array([2, 3, 4])

    print(X * Y)
    print(X.sum(1))
    print(X.sum(0))

    con = np.diag(X) == 1
    print(con)
    print(vars(con))

    prob = pp.LpProblem('foo')
    con.constrain(prob, name='diag')

    prob += X.sumit()

    print(prob)
