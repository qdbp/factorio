'''
Use PuLP+CBC to solve the belt assignment problem.
'''
from __future__ import annotations

import typing as ty
from collections import defaultdict
from itertools import product
from tempfile import NamedTemporaryFile

import pulp as pp

from core import Item

# scale half capacities to keep them integral, ensuring exactness
BELT_CAP_SCALE = 10
BELT_CAPACITIES = {
    'yellow': 75,
    'red': 150,
    'blue': 225,
}

BELT_COSTS = {
    'yellow': 1,
    'red': 1,
    'blue': 1,
}

POSITION_COSTS = {
    0: 1,
    1: 2,
}

BELT_EXISTS_COST = 5
BeltSolution = ty.List[ty.Tuple[(int, str, str, str, Item)]]


def solve_belts(
    flows: ty.List[ty.Tuple[int, Item]],
    max_lines=4,
    costs=None,
    capacities=None,
) -> BeltSolution:
    '''
    Solves the optimal belt allocation for the item requirements given by
    `flows`, which should be a list of (items_per_sec, item_name)
    '''

    costs = costs or BELT_COSTS.copy()
    capacities = capacities or BELT_CAPACITIES.copy()

    required_flow: ty.Dict[ty.Any, int] = defaultdict(int)
    for need, item in flows:
        required_flow[item] += need * BELT_CAP_SCALE
    nothing_key = 'xxx'
    while nothing_key in required_flow:
        nothing_key = '_' + nothing_key
    required_flow[nothing_key] = 0
    required_flow = dict(required_flow)

    if costs.keys() != capacities.keys():
        raise ValueError(
            'Inconsistent belt types between costs and capacities.')

    prob = pp.LpProblem(name="solve_belts", sense=pp.LpMinimize)

    Line = list(range(max_lines))
    Position = ['U', 'L']
    BeltType = sorted(costs.keys())
    Side = ['R', 'L']
    Item = sorted(required_flow.keys())

    Z = pp.LpVariable.dicts(
        "Z", (Line, Position, BeltType, Side, Item), 0, 1, pp.LpBinary,
    )

    # OBJECTIVE
    prob += (
        # complexity
        pp.lpSum(
            Z[l][p][t][s][i] * costs[t]
            for l, p, t, s, i in product(Line, Position, BeltType, Side, Item)
        )
        # excess flow
        + pp.lpSum(
            Z[l][p][t][s][i] * capacities[t]
            for l, p, t, s, i in product(Line, Position, BeltType, Side, Item)
        )
    )

    # CONSTRAINTS
    # 1. satisfaction
    for i in Item:
        prob += pp.lpSum(
            Z[l][p][t][s][i] * capacities[t]
            for l, p, t, s in product(Line, Position, BeltType, Side)
        ) >= required_flow[i]

    # 2. exclusive side type and content
    for l, p, s in product(Line, Position, Side):
        prob += pp.lpSum(
            Z[l][p][t][s][i]
            for t, i in product(BeltType, Item)
        ) <= 1

    # 3. placed belt sides are matched
    for l, p, t in product(Line, Position, BeltType):
        prob += pp.lpSum(
            Z[l][p][t]['R'][i] - Z[l][p][t]['L'][i]
            for i in Item
        ) == 0

    # 4. there are at most two belts (right sides) per line
    for l in Line:
        prob += pp.lpSum(
            Z[l][p][t]['R'][i]
            for p, t, i in product(Position, BeltType, Item)
        ) <= 2

    # 5. at most one belt (right side) of a type per line (braiding)
    for l, t in product(Line, BeltType):
        prob += pp.lpSum(
            Z[l][p][t]['R'][i]
            for p, i in product(Position, Item)
        ) <= 1

    with NamedTemporaryFile(suffix='.lp') as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve()

    print('Belt Solver: status:', pp.LpStatus[prob.status])
    print('Belt Solver: solution:')

    out = []
    for l, p, t, s, i in product(Line, Position, BeltType, Side, Item):
        if pp.value(Z[l][p][t][s][i]) == 1:
            print('\t', l, p, t, s, i)
            out.append((l, p, t, s, i))

    return out


if __name__ == '__main__':
    solve_belts([(10, 'iron'), (20, 'copper'), (10, 'plastic')])
