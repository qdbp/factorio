"""
Use PuLP+CBC to solve the belt assignment problem.
"""
from __future__ import annotations

import typing as ty
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from enum import Enum
from fractions import Fraction as Frac
from itertools import product
from tempfile import NamedTemporaryFile

import pulp as pp

from item import Item

# scale half capacities to keep them integral, ensuring exactness
BELT_CAP_SCALE = 10
BELT_CAPACITIES = {"yellow": 75, "red": 150, "blue": 225}

BELT_COSTS = {"yellow": 10, "red": 10, "blue": 10}
BELT_EXISTS_COST = 5


# XXX is it worthwhile to try and read this from game data?
class Belt(Enum):
    Y = Frac("15")
    R = Frac("30")
    B = Frac("45")

    @classmethod
    def by_name(cls, s: str) -> Belt:
        return {"y": cls.Y, "r": cls.R, "b": cls.B}[s[0].lower()]

    @property
    def name(self):
        if self == self.Y:
            return "yellow belt"
        elif self == self.R:
            return "red belt"
        elif self == self.B:
            return "blue belt"
        else:
            raise AssertionError(f"{self} didn't get a name")

    def __str__(self):
        return {self.Y: "yellow belt", self.R: "red belt", self.B: "blue belt"}[
            self
        ]


def strinsert(base: str, val: str, ix: int) -> str:
    return base[:ix] + val + base[ix:]


@dataclass(frozen=True)
class BeltAssignment:
    class Infeasible(Exception):
        pass

    class Pos(Enum):
        U = "U"
        L = "L"

        def __lt__(self, other):
            return self.value < other.value

    line: int
    pos: Pos
    belt: Belt
    item_left: ty.Optional[Item]
    item_right: ty.Optional[Item]

    @staticmethod
    def normalize(
        assignments: ty.List[BeltAssignment]
    ) -> ty.List[BeltAssignment]:
        """
        Normalize a belt assignment solution set.

        Minimizes hetereogeneous belts.
        """

        by_belt_type = defaultdict(list)  # type: ignore
        for asst in assignments:
            by_belt_type[asst.belt].append(asst)

        out = []
        # within each belt type...
        for bt_assts in by_belt_type.values():
            assts_copy = set(bt_assts)
            while True:
                continue_outer = False
                by_item = defaultdict(set)  # type: ignore
                # ... we map each item to the hetero assts that contain it ...
                for asst in assts_copy:
                    if asst.item_right == asst.item_left:
                        continue
                    by_item[asst.item_left].add(asst)
                    by_item[asst.item_right].add(asst)
                    # ... rechecking by_item every time we add a new asst ...
                    for asst_set in by_item.values():
                        # ... if ever an item has two hetero assets
                        # then they can be swapped to homogenize a belt...
                        if len(asst_set) == 2:
                            x = asst_set.pop()
                            y = asst_set.pop()

                            # so we check which assets to swap between the
                            # belts. Immutability is a guiderail here.
                            if (x.item_right == y.item_right) or (
                                x.item_left == y.item_left
                            ):
                                new_x = dc_replace(x, item_right=y.item_left)
                                new_y = dc_replace(y, item_left=x.item_right)
                            else:
                                new_x = dc_replace(x, item_right=y.item_right)
                                new_y = dc_replace(y, item_right=x.item_right)

                            # we invalidate the originals
                            assts_copy.discard(x)
                            assts_copy.discard(y)
                            # and add the reforged assignments
                            assts_copy.add(new_x)
                            assts_copy.add(new_y)
                            # we continue our search if we did anything
                            continue_outer = True
                            break  # for asst_set in by_item.values()
                    # we continue continuing our search if we did anything
                    if continue_outer:
                        break  # for asst in assts_copy
                # otherwise we are done!
                if not continue_outer:
                    break  # while True

            out.extend(list(assts_copy))
        return sorted(out)

    # TODO this supports only the base symmetric two-line layout
    @staticmethod
    def generate_layout_str(
        assignments: ty.List[BeltAssignment]
    ) -> ty.Tuple[str, str]:
        n_lines = len(set(asst.line for asst in assignments))

        top_half = "^   "
        bot_half = "0 X "

        if n_lines > 1:
            top_half = "^" + top_half
            bot_half = strinsert(bot_half, "1", 1)
        if n_lines > 2:
            top_half += "^"
            bot_half += "3"

        return (
            top_half + "v" + "".join(reversed(top_half)),
            bot_half + "P" + "".join(reversed(bot_half)),
        )

    def __lt__(self, other):
        return (self.line, self.pos, self.belt.name) < (
            other.line,
            other.pos,
            other.belt.name,
        )

    def __post_init__(self):
        # if this fails, we have a problem in our solver
        assert (self.item_left is not None) or (self.item_right is not None)

    def __str__(self) -> str:
        return (
            f"{self.line}:{self.pos.value}"
            f"[{self.belt.name[:3].upper()}] = "
            f"{self.item_left} | {self.item_right}"
        )


# XXX support multiple line layouts
def solve_belts(
    flows: ty.List[Item.Flow], max_lines=3, costs=None, capacities=None
) -> ty.List[BeltAssignment]:
    """
    Solves the optimal belt allocation for the item requirements given by
    `flows`, which should be a list of (items_per_sec, item_name)


    Assumes a symmetric layout as follows:

        L1   L2                   L3   O   L3                      L2  L1

        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^                   ^^^ vvv ^^^                     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
        ^^^ ^^^     XXXXXXXXX     ^^^ vvv ^^^       XXXXXXXXX     ^^^ ^^^
                ...                   ...                     ...

    Namely, flows are allocated among L1, L2 and L3 such that a target output
    in line O can be produced. Each of L1, L2 and L3 can be braided; thus up to
    6 input belts are supported.
    """

    costs = costs or BELT_COSTS.copy()
    capacities = capacities or BELT_CAPACITIES.copy()

    required_flow: ty.Dict[str, int] = defaultdict(int)
    for flow in flows:
        # divide by 2 since we are splitting flows among mirrored lines
        required_flow[flow.item.name] += flow.num * BELT_CAP_SCALE
    nothing_key = "xxx"
    while nothing_key in required_flow:
        nothing_key = "_" + nothing_key
    required_flow[nothing_key] = 0
    required_flow = dict(required_flow)

    if costs.keys() != capacities.keys():
        raise ValueError(
            "Inconsistent belt types between costs and capacities."
        )

    prob = pp.LpProblem(name="solve_belts", sense=pp.LpMinimize)

    s_Line = list(range(max_lines))
    s_Position = list(BeltAssignment.Pos)
    s_BeltType = sorted(costs.keys())
    s_side = ["R", "L"]
    s_Item = sorted(required_flow.keys())

    Z = pp.LpVariable.dicts(
        "Z", (s_Line, s_Position, s_BeltType, s_side, s_Item), 0, 1, pp.LpBinary
    )

    norm_pos_cost = {BeltAssignment.Pos.L: 0, BeltAssignment.Pos.U: 1}
    norm_line_cost = {0: 0, 1: 1, 2: 2}

    # OBJECTIVE
    prob += (
        # complexity
        pp.lpSum(
            Z[l][p][t][s][i] * costs[t]
            for l, p, t, s, i in product(
                s_Line, s_Position, s_BeltType, s_side, s_Item
            )
        )
        # excess flow
        + pp.lpSum(
            Z[l][p][t][s][i] * capacities[t]
            for l, p, t, s, i in product(
                s_Line, s_Position, s_BeltType, s_side, s_Item
            )
        )
        # normalization costs
        # line
        + pp.lpSum(
            Z[l][p][t][s][i] * norm_line_cost[l]
            for l, p, t, s, i in product(
                s_Line, s_Position, s_BeltType, s_side, s_Item
            )
        )
        # position
        + pp.lpSum(
            Z[l][p][t][s][i] * norm_pos_cost[p]
            for l, p, t, s, i in product(
                s_Line, s_Position, s_BeltType, s_side, s_Item
            )
        )
    )

    # CONSTRAINTS
    # 1. satisfaction
    for i in s_Item:
        prob += (
            pp.lpSum(
                Z[l][p][t][s][i] * capacities[t]
                for l, p, t, s in product(
                    s_Line, s_Position, s_BeltType, s_side
                )
            )
            >= required_flow[i]
        )

    # 2. exclusive side type and content
    for l, p, s in product(s_Line, s_Position, s_side):
        prob += (
            pp.lpSum(Z[l][p][t][s][i] for t, i in product(s_BeltType, s_Item))
            <= 1
        )

    # 3. placed belt sides are matched
    for l, p, t in product(s_Line, s_Position, s_BeltType):
        prob += (
            pp.lpSum(Z[l][p][t]["R"][i] - Z[l][p][t]["L"][i] for i in s_Item)
            == 0
        )

    # 4. there are at most two belts (right sides) per line
    for l in s_Line:
        prob += (
            pp.lpSum(
                Z[l][p][t]["R"][i]
                for p, t, i in product(s_Position, s_BeltType, s_Item)
            )
            <= 2
        )

    # 5. at most one belt (right side) of a type per line (braiding)
    for l, t in product(s_Line, s_BeltType):
        prob += (
            pp.lpSum(Z[l][p][t]["R"][i] for p, i in product(s_Position, s_Item))
            <= 1
        )

    with NamedTemporaryFile(suffix=".lp") as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve()

    if "Infeasible" in pp.LpStatus[prob.status]:
        raise BeltAssignment.Infeasible("No solution exists for desired flows")

    out = []
    for l, p, t in product(s_Line, s_Position, s_BeltType):
        items = []
        for s in s_side:
            for i in s_Item:
                if pp.value(Z[l][p][t][s][i]) == 1:
                    items.append(
                        Item.by_name(i) if i is not nothing_key else None
                    )
                    break
        assert not items or any(items)
        if items:
            out.append(
                BeltAssignment(
                    line=l,
                    pos=BeltAssignment.Pos(p),
                    belt=Belt.by_name(t),
                    item_right=items[0],
                    item_left=items[1],
                )
            )

    return BeltAssignment.normalize(out)
