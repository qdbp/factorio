# Dicovering the optimal M x N balancers, phase 1: max flow
# Evgeny Naumov, 2019
from __future__ import annotations

import typing as ty
from itertools import permutations, product
from math import ceil
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp as pp
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from solver_core import (
    Infeasible, dicts_to_ndarray, get_solver, ndarray_to_dicts
)


def lowerbound_splitters(M, N) -> int:
    '''
    Guesses a lower bound on the number of splitters needed for an M-N balancer
    '''

    return ceil(np.log2(N)) * ceil(M / 2)


def solve_balancers(
        M: int,
        N: int,
        max_spls: int,
        min_spls: int = 0,
        debug=False,
        exact_counts=False,
        solver='coin',
) -> np.ndarray:
    '''
    Attempt to find a splitter-count optimal M -> N balancer for Factorio.

    More efficient than the naive expanded matrix approach, using "implicit"
    input and output flows with inherent restrictions for the splitter problem.

    Further optimized by restricting the search to N >= M and allowing only
    22 and 12 splitters.
    '''

    if M > N:
        raise ValueError(
            'The problem formulation only allows for fanout designs. '
            'Note that an optimal N -> M solution can be reversed to give '
            'what you want.'
        )

    min_spls = max(lowerbound_splitters(M, N), min_spls)
    if max_spls < min_spls:
        raise ValueError(
            f'Balancing {M} -> {N} requires at least {min_spls}, which is '
            'lower than the max given'
        )

    Inps = [f'i{ix}' for ix in range(M)]
    Outs = [f'o{ix}' for ix in range(N)]

    Splitters = [f's{ix}' for ix in range(max_spls)]

    # Spls22 = [f's[22]{ix}' for ix in range(max_22)]
    # Spls12 = [f's[12]{ix}' for ix in range(max_12)]
    # Spls21 = [f's[21]{ix}' for ix in range(max_21)]
    # Splitters = Spls22 + Spls21 + Spls12

    # splitters enabled
    if not exact_counts:
        S = pp.LpVariable.dicts("S", (Splitters, ), 0, 1, pp.LpBinary)
    else:
        S = {s: 1 for s in Splitters}

    # capacity matrix
    # c[i, j] = 1 <=> directed capacity of 1 exists from i to j
    Conn = pp.LpVariable.dicts("C", (Splitters, Splitters), 0, 1, pp.LpBinary)
    # internal flows
    Fs = pp.LpVariable.dicts("Fs", (Splitters, Splitters, Inps))

    # input map: Imap[i, u] == 1 <=> input i flows into splitter u
    # Imap = pp.LpVariable.dicts("Imap", (Inps, Splitters), 0, 1, pp.LpBinary)
    # XXX experimental, fixed map of inputs
    fixed_imap = np.zeros((M, len(Splitters)))
    for ix in range(M):
        fixed_imap[ix][ix // 2] = 1
    Imap = ndarray_to_dicts(fixed_imap, (Inps, Splitters))

    # input flows
    Fi = pp.LpVariable.dicts("Fi", (Inps, Splitters, Inps))
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    Omap = pp.LpVariable.dicts("Omap", (Splitters, Outs), 0, 1, pp.LpBinary)
    # output flows
    Fo = pp.LpVariable.dicts("Fo", (Splitters, Outs, Inps))

    # ## OBJECTIVE
    prob = pp.LpProblem(
        name="solve_balancer",
        sense=pp.LpMinimize,
    )

    # NOTE: optimizer tuning: add discriminating coefficients
    objective = pp.LpAffineExpression()
    objective += pp.lpSum(100 * S[spl] for spl in Splitters)

    # for exact, we want to keep the objective 0 to quit on first feasible
    if not exact_counts:
        for ix, si in enumerate(Splitters):
            # try to wire inputs to "low" splitters
            # objective += ix * pp.lpSum(Imap[t][si] for t in Inps)
            # try to wire outputs from "late" splitters
            objective += (-ix * pp.lpSum(Omap[si][o] for o in Outs))
            # # overall, we aim for an adjecency matrix that's as
            # "diagonal as possible" while still having the objective
            # dominated by splitter count
            # penalize "backflow"
            for jx, sj in zip(range(ix), Splitters):
                objective += (ix - jx) * Conn[si][sj]
            # penalize "jumps"
            for jx in range(ix + 1, len(Splitters)):
                sj = Splitters[jx]
                objective += (jx - ix) * Conn[si][sj]

    prob += objective

    # ## CONSTRAINTS
    # # CHAPTER 0: FORMULATION RESTRICTIONS
    # 0.1: enable splitters in order
    if not exact_counts:
        for u, v in zip(Splitters, Splitters[1:]):
            S[u] >= S[v]
    # 0.2: min splitters
    # each input must have at least ceil(lg(N)) between it and any output
    # this creates a trivial lower bound of (M // 2)ceil(lg(N)) splitters
    prob += pp.lpSum(S[u] for u in Splitters) >= min_spls

    if not exact_counts:
        for u, v in zip(Splitters, Splitters[1:]):
            S[u] >= S[v]

    # # CHAPTER 1: ADJACENCY RESTRICTIONS
    # 1.1 INPUTS WELL CONNECTED
    # 1.1.1 each input goes into exactly one splitter
    for inp in Inps:
        prob += pp.lpSum(Imap[inp][u] for u in Splitters) == 1
    # 1.1.2 each splitter receives from at most two inputs
    for u in Splitters:
        prob += pp.lpSum(Imap[inp][u] for inp in Inps) <= 2

    #  1.2 OUTPUTS UNIQUELY CONNECTED
    #  1.2.1 each output receives from exactly one splitter
    for out in Outs:
        prob += pp.lpSum(Omap[u][out] for u in Splitters) == 1
    #  1.2.2 each splitter goes to at most two outputs
    for u in Splitters:
        prob += pp.lpSum(Omap[u][out] for out in Outs) <= 2

    for spl in Splitters:
        outs_from_spl = pp.lpSum(Omap[spl][out] for out in Outs)
        inps_into_spl = pp.lpSum(Imap[inp][spl] for inp in Inps)

        #  1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
        outlinks = pp.lpSum(Conn[spl][v] for v in Splitters) + outs_from_spl
        prob += outlinks <= 2 * S[spl]
        prob += outlinks >= S[spl]

        #  1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
        inlinks = pp.lpSum(Conn[u][spl] for u in Splitters) + inps_into_spl
        prob += inlinks <= 2 * S[spl]
        prob += inlinks >= S[spl]

        # 1.5 NO SELF-LOOPS
        prob += Conn[spl][spl] == 0

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    for u, v in product(Splitters, Splitters):
        prob += pp.lpSum(Fs[u][v][t] for t in Inps) <= Conn[u][v]

    # 2.2 INFLOW EDGE CONDITIONS
    for i, v, t in product(Inps, Splitters, Inps):
        prob += Fi[i][v][t] == Imap[i][v] * int(i == t)

    # 2.3 OUTFLOW EDGE CONDITIONS
    for o, v, t in product(Outs, Splitters, Inps):
        prob += N * Fo[v][o][t] == Omap[v][o]

    # 2.2 PROPER FLOW
    for u, v, t in product(Splitters, Splitters, Inps):
        prob += Fs[u][v][t] >= 0
    for i, v, t in product(Inps, Splitters, Inps):
        prob += Fi[i][v][t] >= 0
    for o, v, t in product(Outs, Splitters, Inps):
        prob += Fo[v][o][t] >= 0

    for spl, t in product(Splitters, Inps):
        # 2.3 UNIFORM INCOMPRESSIBILITY
        in_s = pp.lpSum(Fs[u][spl][t] for u in Splitters)
        in_i = pp.lpSum(Fi[i][spl][t] for i in Inps)
        out_s = pp.lpSum(Fs[spl][w][t] for w in Splitters)
        out_o = pp.lpSum(Fo[spl][o][t] for o in Outs)

        prob += in_s + in_i == out_s + out_o

        # 3.1 EQUAL SPLITTING
        for w in Splitters:
            prob += 2 * Fs[spl][w][t] <= in_s + in_i
        for o in Outs:
            prob += 2 * Fo[spl][o][t] <= in_s + in_i

    # ## SOLVING
    # #
    #
    with NamedTemporaryFile(suffix='.lp') as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve(solver=get_solver(which=solver))

    if 'Infeasible' in pp.LpStatus[prob.status]:
        raise Infeasible

    adjmat = dicts_to_ndarray(Conn, (Splitters, Splitters))
    keep_rows = np.where(dicts_to_ndarray(S, (Splitters, )) > 0)[0]
    # keep_rows = np.where(adjmat.sum(axis=0) + adjmat.sum(axis=1) != 0)[0]
    adjmat = adjmat[np.ix_(keep_rows, keep_rows)]

    imap = dicts_to_ndarray(Imap, (Inps, Splitters))[:, keep_rows]
    omap = dicts_to_ndarray(Omap, (Splitters, Outs))[keep_rows]
    labels = np.array(Splitters)[keep_rows]

    if debug:
        print('S')
        print(dicts_to_ndarray(S, (Splitters, )))
        print('adjmat')
        print(adjmat)
        print('imap')
        print(imap)
        print('omap')
        print(omap)
        for i in Inps:
            print(f'flow for {i}')
            flows = dicts_to_ndarray(Fs, (Splitters, Splitters, [i])).squeeze()
            # flows = flows[np.ix_(keep_rows, keep_rows)]
            print(flows)

    return imap, omap, adjmat, labels


def draw_solution(
        imap: np.ndarray,
        omap: np.ndarray,
        splmat: np.ndarray,
        labels=None,
        graphname='graph.png',
        debug=False,
) -> None:

    n_inps = imap.shape[0]
    n_outs = omap.shape[1]
    n_spls = splmat.shape[0]
    assert n_spls == splmat.shape[1]

    #  |n_i |   n_spl  |n_o|
    #  ---------------------
    #  |    |   imap   | 0 |
    #  |    |__________|___|
    #  |    |          |   |
    #  | 0  |          |   |
    #  |    |  splmat  |omap
    #  |    |          |   |
    #  |____|__________|___|
    #  | 0  |     0    | 0 |
    #  |____|__________|___|
    # one can see why I switched to implicit flows - too many zeroes!

    full_adjmat = np.zeros(
        (n_inps + n_spls + n_outs, ) * 2,
        dtype=np.uint8,
    )
    full_adjmat[:n_inps, n_inps:-n_outs] = imap
    full_adjmat[n_inps:-n_outs, n_inps:-n_outs] = splmat
    full_adjmat[n_inps:-n_outs, -n_outs:] = omap

    if debug:
        print(full_adjmat)

    g = nx.convert_matrix.from_numpy_array(
        full_adjmat, create_using=nx.DiGraph
    )
    if labels is not None:
        g = nx.relabel_nodes(
            g, {ix + n_inps: label
                for ix, label in enumerate(labels)}
        )

    values = ['red'] * n_inps + ['black'] * n_spls + ['blue'] * n_outs
    # nx.draw_graphviz(g, node_color=values)
    # plt.savefig('graph.png')
    for ndx, (_, attrs) in enumerate(g.nodes(data=True)):
        attrs['fillcolor'] = attrs['color'] = values[ndx]

    A = to_agraph(g)
    A.layout('dot')
    A.draw(graphname + '.png')


def main():
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()

    parser.add_argument('ni', type=int, help='number of inputs')
    parser.add_argument('no', type=int, help='number of outputs')
    parser.add_argument(
        'maxs', type=int, help='max number of splitters to consider'
    )
    parser.add_argument(
        '--mins',
        type=int,
        default=0,
        help='minimum number of splitters to consider'
    )
    # parser.add_argument(
    #     'm21', type=int, help='max number of 2->1 splitters to consider'
    # )
    parser.add_argument(
        '--exact',
        action='store_true',
        help='splitter counts are treated as exact instead of as upper bounds'
    )
    parser.add_argument(
        '--debug', action='store_true', help='enable debug mode'
    )
    parser.add_argument(
        '--solver', type=str, default='coin', help='specify the solver'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='draw a reversed graph of the solution'
    )

    args = parser.parse_args(sys.argv[1:])

    graphname = (
        f'balancer_{args.ni}-{args.no}-{"x" if args.exact else "le"}-'
        f'{args.maxs}'
    )

    print(f'Solving the optimal {args.ni} -> {args.no} balancer...')
    try:
        imap, omap, adjmat, labels = solve_balancers(
            args.ni,
            args.no,
            max_spls=args.maxs,
            min_spls=args.mins,
            debug=args.debug,
            exact_counts=args.exact,
            solver=args.solver,
        )
    except Infeasible:
        print(f'No feasible solution within the given splitter limits.')
        return

    draw_solution(
        imap,
        omap,
        adjmat,
        labels=labels,
        graphname=graphname,
        debug=args.debug,
    )
    print(f'solution saved to {graphname}.png')


if __name__ == '__main__':
    main()
