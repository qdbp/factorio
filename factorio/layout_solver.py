# Evgeny Naumov, 2019
"""
Factorio Layout solvers.

The goal of this module is to accept abstract solutions (usually in the form
of graphs) from the other solvers and to realize them in concrete factorio
tile layouts.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from itertools import product

# import typing as ty
from typing import Iterable, List, Tuple

import numpy as np
import pulp as pp

from solver_core import (
    IllSpecified,
    Infeasible,
    dicts_to_ndarray,
    ndarray_to_dicts,
    sumdict,
)

T = ty.TypeVar("T", bound="TileSet")


class Direction(Enum):
    N = "N"
    E = "E"
    S = "S"
    W = "W"


class Tile(Enum):
    Empty = "Empty"
    Belt = "Belt"
    UBelt = "UBelt"
    UBeltIn = "UBeltIn"
    UBeltOut = "UBeltOut"
    SplR = "SplR"
    SplL = "SplL"

    # def required_inputs(self) -> Iterable[Tuple[int, int,


class TileSet(Enum):
    """
    A class defining a tile set relevant to a particular layout problem.

    Allows of switching to the "minimum relevant tileset" to constrain problem
    size.
    """

    @classmethod
    def dict(cls):
        return {k: v.value for k, v in cls.__members__.items()}

    @classmethod
    @abstractmethod
    def exclusion_groups(cls: ty.Type[T]) -> ty.Iterable[ty.Set[T]]:
        """
        Generate the exclusion groups for this tileset. These are subsets
        of the enum such that within each subset, the at most one of that
        tile type can be present.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def valid_interconns(
        cls: ty.Type[T], t: T
    ) -> ty.Iterable[ty.Tuple[int, int, ty.Iterable[T]]]:
        """
        For each member of the enum, generate a list of (dx, dy, {valid}).

        Assuming this tile's position to be (x, y), this is for each (dx, dy)
        that requires to be restriced, a list of restrictions such that
        the tile (x + dx, y + dy) must be in {valid}.
        """
        raise NotImplementedError


def solve_balancer_layout(
    max_x: int, max_y: int, C: np.ndarray, x0: np.ndarray, x1: np.ndarray
):
    """
    Arguments:
        max_x: maximum x width
        max_y: maximum y width
        C: full adjaceny matrix of the flows
        x0: edge condition on the bottom
        y0: edge condition on the top (max_y'th row)
    """

    if (
        C.ndim != 2
        or C.shape[0] != C.shape[1]
        or np.max(C) != 1
        or np.min(C) != 0
    ):
        raise IllSpecified("C must be a square (full) adjacency matrix.")

    if max_y < 3:
        raise IllSpecified(
            "There are no tiles not determined by edge conditions!"
        )

    prob = pp.LpProblem("solve_balancer_layout", sense=pp.LpMinimize)

    Xs = [f"x{ix}" for ix in range(max_x)]
    Ys = [f"y{iy}" for iy in range(max_y)]
    Tiles = [None] + list(BalTil.dict().keys())
    Flows = [f"s{fi}" for fi in range(C.shape[0])]

    # these are the coordinates not constained by edge conditions
    # XXX abstract away edge conditions
    FreeXs = Xs[:]
    FreeYs = Ys[1:-1]

    empty_flow = Flows[0]

    for x, name in zip([x0, x1], ["x0", "x1"]):
        if x.shape != (max_x, len(Tiles), len(Flows)):
            raise IllSpecified(
                f"The constraint {name} is the wrong shape. "
                "Must be (max_x, #tiles, #nodes)"
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
        for excl_group in BalTil.exclusion_groups():
            prob += (
                pp.lpSum(W[x][y][t][f] for t in excl_group for f in Flows) <= 1
            )

    # 0.1: VALID INTERCONNECTS
    # this is a long one...
    BT = BalTil
    for (ix, x), (iy, y) in product(enumerate(Xs), enumerate(FreeYs)):

        def w(ix: int, iy: int):
            return sumdict(
                {t: pp.lpSum(W[Xs[ix]][Ys[iy]][t][f] for f in Flows)}
            )

        ws: ty.Dict[ty.Optional[BalTil], ty.Any] = {
            t: pp.lpSum(W[x][y][t][f] for f in Flows) for t in Tiles
        }

        conss = [
            ws[BT.BeltE]
            <= sum(
                (
                    w(ix - 1, iy)[BT.BeltE, BT.UBeltOutE, BT.SplLE, BT.SplRE],
                    w(ix, iy - 1)[BT.BeltN, BT.UBeltOutN, BT.SplLN, BT.SplRN],
                    w(ix, iy + 1)[BT.BeltS, BT.UBeltOutS, BT.SplLS, BT.SplRS],
                )
            )
        ]

    # 0.2: EMPTY TILES HAVE FIXED FLOW
    # to make the optimizer's job easier
    for x, y, f in product(Xs, Ys, Flows):
        prob += W[x][y][None][f] == int(f == empty_flow)

    # # CHAPTER 1: CORRECTNESS
    # 1.0 FLOW EXCLUSIVITY
    for x, y, t in product(Xs, Ys, Tiles):
        prob += pp.lpSum(W[x][y][t][f] for f in Flows) <= 1

    # 1.1 NO USELESS TILES
    # FIXME
    # a tile is either empty or valid
    # for x, y in product(FreeXs, FreeYs):
    #     prob += V[x][y] + pp.lpSum(W[x][y][None][f] for f in Flows) == 1

    # ## SOLVE
    prob.solve()
