from __future__ import annotations

import os
import typing as ty
from itertools import product
from pathlib import Path

import numpy as np
import pulp as pp

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


SOLVER_FALLBACK = ['gurobi', 'coin', 'glpk']


def get_solver(which='coin', threads: int = None, _fallback=None):
    threads = threads or os.cpu_count() or 1
    if which == 'coin':
        return pp.solvers.COIN_CMD(
            msg=1,
            threads=threads,
        )
    if which == 'coinmp':
        solver = pp.solvers.COINMP_DLL()
    elif which == 'glpk':
        solver = pp.solvers.GLPK()
    elif which == 'gurobi':
        solver = pp.solvers.GUROBI(
            threads=threads,
            display_interval=60,
        )
    else:
        solver = None

    if solver is not None and solver.available():
        return solver

    if _fallback is None:
        _fallback = SOLVER_FALLBACK.copy()
    _fallback.remove(which)
    if not _fallback:
        print('Could not get a solver!')
        return None
    print(
        f'Solver: {which} solver is not available. '
        f'Falling back on {_fallback[0]}'
    )
    return get_solver(which=_fallback[0], threads=threads, _fallback=_fallback)


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
    @staticmethod
    def bin_and(prob: pp.LpProblem, name: str, out: lparray, *ins: lparray):
        for ix, _in in enumerate(ins):
            (out <= _in).constrain(prob, f'{name}_and_ub{ix}')
        (out >= sum(ins, 1 - len(ins))).constrain(prob, f'{name}_and_lb')

    @staticmethod
    def bin_or(prob: pp.LpProblem, name: str, out: lparray, *ins: lparray):
        for ix, _in in enumerate(ins):
            (out >= _in).constrain(prob, f'{name}_or_lb{ix}')
        (out <= sum(ins)).constrain(prob, f'{name}_and_ub')

    @staticmethod
    def abs_ge(prob: pp.LpProblem, name: str, x: lparray, lim, bigM=1000):

        b = pp.LpVariable(f'{name}_b', 0, 1, pp.LpBinary)

        ((x + bigM * b) >= lim).constrain(prob, f'{name}_ub')
        ((-x + bigM * (1 - b)) >= lim).constrain(prob, f'{name}_lb')

    @classmethod
    def create(cls, name: str, index_sets, *args, **kwargs) -> lparray:
        '''
        Numpy array equivalent of pulp.LpVariable.dicts
        '''

        if len(index_sets) == 1:
            name = name + '('

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

        return arr.view(lparray)  # type: ignore

    @classmethod
    def create_like(cls, name: str, like: lparray, *args, **kwargs) -> lparray:
        return cls.create_anon(name, like.shape, *args, **kwargs)

    @classmethod
    def create_anon(
            cls, name: str, shape: ty.Tuple[int, ...], *args, **kwargs
    ) -> lparray:
        ixsets = tuple(list(range(d)) for d in shape)
        return cls.create(name, ixsets, *args, **kwargs)

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

    def constrain(self, prob: pp.LpProblem, name: str) -> None:
        if not isinstance(prob, pp.LpProblem):
            raise TypeError(
                f'Trying to constrain a {type(prob)}. Did you pass prob?'
            )
        if self.ndim == 0:
            cons = self.item()
            cons.name = name
            prob += cons
            return

        if name and self.ndim == 1:
            name = name + '('

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

    def logical_clip(self, prob: pp.LpProblem, bigM=10000) -> lparray:
        '''
        Assumes self is integer >= 0.

        Returns an array of the same shape as self containing
            z_... = max(self_..., 1)
        using the big M method
        '''

        if self.ndim == 0:
            basename = self.item().name
        else:
            basename = self[(0, ) * self.ndim].split('(')[:-1].join('(')

        clipname = basename + '_lclipped'

        z = self.__class__.create(
            clipname, [range(x) for x in self.shape], 0, 1, pp.LpBinary
        )

        (self >= z).constrain(prob, clipname + '_limit')
        (self <= bigM * z).constrain(prob, clipname + '_reach')

        return z

    def _lp_minmax(
            self,
            name: str,
            prob: pp.LpProblem,
            which,
            categ,
            lb=None,
            ub=None,
            bigM=10000,
            axis: ty.Union[None, int, ty.Tuple[int, ...]] = None,
    ):

        if not np.product(self.shape):
            raise ValueError('No variables given!')

        # if any(v.cat != categ for v in self.ravel()):
        #     raise ValueError(f'This function expects {categ} variables')

        if axis is None:
            axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axis = (axis, )
        elif (not isinstance(axis, tuple) or not axis
              or any(not isinstance(ax, int) or ax < 0 for ax in axis)):
            raise TypeError("Axis must be a tuple of positive integers")

        if categ == pp.LpBinary:
            lb = 0
            ub = 1
        elif lb is None or ub is None:
            assert 0, "Need to supply constraints for non-binary variables!"

        assert which in ('min', 'max')

        mmname = f'{name}_{which}'
        aux_name = f'{name}_{which}_aux'

        # axes of self which the max is indexed by
        keep_axis = tuple(sorted(set(range(self.ndim)) - set(axis)))

        # array of maxes
        minmax_shape = sum((self.shape[ax:ax + 1] for ax in keep_axis), ())
        z = lparray.create_anon(mmname, minmax_shape, lb, ub, categ)

        # broadcastable version for comparison with self
        minmax_br_index = tuple(
            (slice(None, None, None) if ax in keep_axis else None)
            for ax in range(self.ndim)
        )
        z_br = z[minmax_br_index]

        w = self.create_like(
            aux_name, self, lowBound=0, upBound=1, cat=pp.LpBinary
        )

        (w.sum(axis=axis) == 1).constrain(prob, f'{mmname}_auxsum')

        if which == 'max':
            (z_br >= self).constrain(prob, f'{mmname}_lb')
            (z_br <= self + bigM * (1 - w)).constrain(prob, f'{mmname}_ub')
        elif which == 'min':
            (z_br <= self).constrain(prob, f'{mmname}_ub')
            (z_br >= self - bigM * (1 - w)).constrain(prob, f'{mmname}_lb')

        return z

    def _lp_int_minmax(
            self, name: str, prob: pp.LpProblem, which: str, lb: int, ub: int,
            **kwargs
    ) -> pp.LpVariable:

        if lb == 0 and ub == 1:
            cat = pp.LpBinary
        else:
            cat = pp.LpInteger

        return self._lp_minmax(
            prob, name, which=which, categ=cat, lb=lb, ub=ub, **kwargs
        )

    def lp_int_max(
            self, prob: pp.LpProblem, name: str, lb: int, ub: int, **kwargs
    ) -> pp.LpVariable:
        return self._lp_int_minmax(
            prob, name, which='max', lb=lb, ub=ub, **kwargs
        )

    def lp_int_min(
            self, prob: pp.LpProblem, name: str, lb: int, ub: int, *args,
            **kwargs
    ) -> pp.LpVariable:
        return self._lp_int_minmax(
            prob, name, which='min', lb=lb, ub=ub, **kwargs
        )

    def lp_bin_max(self, name: str, prob: pp.LpProblem, *args, **kwargs):
        return self._lp_int_minmax(
            prob, name, lb=0, ub=1, which='max', **kwargs
        )

    def lp_bin_min(self, name: str, prob: pp.LpProblem, *args, **kwargs):
        return self._lp_int_minmax(
            prob, name, lb=0, ub=1, which='min', **kwargs
        )


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

    prob = pp.LpProblem('foo')
    z = X.lp_bin_max('rowmax', prob, axis=0)
    print(prob)

    prob = pp.LpProblem('foo')
    z = X.lp_bin_max('colmax', prob, axis=1)
    print(prob)
