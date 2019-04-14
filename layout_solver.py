# Evgeny Naumov, 2019
'''
Factorio Layout solvers.

The goal of this module is to accept abstract solutions (usually in the form
of graphs) from the other solvers and to realize them in concrete factorio
tile layouts.
'''

from __future__ import annotations

import typing as ty
from abc import abstractmethod
from enum import Enum
from itertools import product

import numpy as np
import pulp as pp

from solver_core import (
    IllSpecified, Infeasible, dicts_to_ndarray, get_dict_depth,
    ndarray_to_dicts
)

T = ty.TypeVar('T', bound="TileSet")


class TileSet(Enum):
    '''
    A class defining a tile set relevant to a particular layout problem.

    Allows of switching to the "minimum relevant tileset" to constrain problem
    size.
    '''

    @classmethod
    def dict(cls):
        return {k: v.value for k, v in cls.__members__.items()}

    @classmethod
    @abstractmethod
    def exclusion_groups(cls: ty.Type[T]) -> ty.Iterable[ty.Set[T]]:
        '''
        Generate the exclusion groups for this tileset. These are subsets
        of the enum such that within each subset, the at most one of that
        tile type can be present.
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def valid_interconns(
            cls: ty.Type[T],
            t: T,
    ) -> ty.Iterable[ty.Tuple[int, int, ty.Iterable[T]]]:
        '''
        For each member of the enum, generate a list of (dx, dy, {valid}).

        Assuming this tile's position to be (x, y), this is for each (dx, dy)
        that requires to be restriced, a list of restrictions such that
        the tile (x + dx, y + dy) must be in {valid}.
        '''
        raise NotImplementedError


class BalancerTiles(TileSet):
    '''
    Represents belts, ubelts and splitters of homogeneous type.
    '''
    # NOTE the empty tile is implicity in the problem!
    # Empty = 0
    BeltN = 1
    BeltE = 2
    BeltS = 3
    BeltW = 4
    UBeltN = 5
    UBeltE = 6
    UBeltS = 7
    UBeltW = 8
    UBeltInN = 9
    UBeltInE = 10
    UBeltInS = 11
    UBeltInW = 12
    UBeltOutN = 13
    UBeltOutE = 14
    UBeltOutS = 15
    UBeltOutW = 16
    SplRN = 17
    SplRE = 18
    SplRS = 19
    SplRW = 20
    SplLN = 21
    SplLE = 22
    SplLS = 23
    SplLW = 24

    @property
    def aboveground(self) -> bool:
        return self not in self.exg_0()

    @classmethod
    def exclusion_groups(cls):
        return [cls.exg_0(), cls.exg_1(), cls.exg_2()]

    # exclusion groups
    @classmethod
    def exg_0(cls):
        return cls.dict().keys() - {
            cls.UBeltN, cls.UBeltE, cls.UBeltS, cls.UBeltW
        }

    @classmethod
    def exg_1(cls):
        return {cls.UBeltN, cls.UBeltS, cls.UBeltInN, cls.UBeltInS}

    @classmethod
    def exg_2(cls):
        return {cls.UBeltE, cls.UBeltW, cls.UBeltInE, cls.UBeltInW}

    @classmethod
    def valid_interconns(cls, t: BalancerTiles):
        pass


def solve_balancer_layout(
        max_x: int,
        max_y: int,
        C: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
):
    '''
    Arguments:
        max_x: maximum x width
        max_y: maximum y width
        C: full adjaceny matrix of the flows
        x0: edge condition on the bottom
        y0: edge condition on the top (max_y'th row)
    '''

    if C.ndim != 2 or C.shape[0] != C.shape[1] or np.max(C) != 1 or np.min(
            C) != 0:
        raise IllSpecified("C must be a square (full) adjacency matrix.")

    if max_y < 3:
        raise IllSpecified(
            "There are no tiles not determined by edge conditions!"
        )

    prob = pp.LpProblem('solve_balancer_layout', sense=pp.LpMinimize)

    Xs = [f'x{ix}' for ix in range(max_x)]
    Ys = [f'y{iy}' for iy in range(max_y)]
    Tiles = [None] + list(BalancerTiles.dict().keys())
    Flows = [f's{fi}' for fi in range(C.shape[0])]

    # these are the coordinates not constained by edge conditions
    # XXX abstract away edge conditions
    FreeXs = Xs[:]
    FreeYs = Ys[1:-1]

    empty_flow = Flows[0]

    for x, name in zip([x0, x1], ['x0', 'x1']):
        if x.shape != (max_x, len(Tiles), len(Flows)):
            raise IllSpecified(
                f'The constraint {name} is the wrong shape. '
                'Must be (max_x, #tiles, #nodes)'
            )

    # the meat and potatoes
    # W[x, y, t, i] == 1 <=> tile (x, y) has component t, carrying flow from
    # node j in the connectivity graph
    W = pp.LpVariable.dicts("W", (Xs, Ys, Tiles, Flows), 0, 1, pp.LpBinary)

    # the "validity map"
    # V[x, y] == 1 if tile t is receiving any flow
    # i.e. the tile at (x, y) is wired correctly
    # V = pp.LpVariable.dicts("V", (Xs, Ys), 0, 1, pp.LpBinary)

    # TODO for now we just want any layout that works
    prob += 0

    # ## CONSTRAINTS
    # # CHAPTER 0: BUILDABILITY
    # 0.0: NO OVERLAPPING TILES
    for x, y in product(Xs, Ys):
        for excl_group in BalancerTiles.exclusion_groups():
            prob += pp.lpSum(
                W[x][y][t][f] for t in excl_group for f in Flows
            ) <= 1

    # 0.1: VALID INTERCONNECTS
    # this is a long one...
    for (ix, x), (iy, y) in product(enumerate(Xs), enumerate(FreeYs)):
        for t in Tiles:
            Wxf = pp.lpSum(W[x][y][t][f] for f in Flows)
            # 0.1.0 Belts are connected properly
            V[x][y] = Wxf[x][y]

    # 0.2: EMPTY TILES HAVE FIXED FLOW
    # to make the optimizer's job easier
    for x, y, f in product(Xs, Ys, Flows):
        prob += W[x][y][None][f] == int(f == empty_flow)

    # # CHAPTER 1: CORRECTNESS
    # 1.0 FLOW EXCLUSIVITY
    for x, y, t in product(Xs, Ys, Tiles):
        prob += pp.lpSum(W[x][y][t][f] for f in Flows) <= 1

    # 1.1 NO USELESS TILES
    # a tile is either empty or valid
    for x, y in product(FreeXs, FreeYs):
        prob += V[x][y] + pp.lpSum(W[x][y][None][f] for f in Flows) == 1

    # ## SOLVE
    prob.solve()


if __name__ == '__main__':
    print(BalancerTiles.dict())
