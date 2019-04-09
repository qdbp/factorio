# Dicovering the optimal M x N balancers, phase 1: max flow
# Evgeny Naumov, 2019
from __future__ import annotations

import typing as ty
from itertools import product
from math import ceil
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp as pp
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from solver_core import Infeasible, dicts_to_ndarray, get_solver


def lowerbound_splitters(M, N) -> ty.Tuple[int, int, int]:
    '''
    Guesses a lower bound on the number of splitters needed for an M-N balancer
    '''

    # output: (22, 12, 21)

    if M == 1:
        return (0, ceil(N / 2), 0)

    return (M // 2, 0, 0)


def solve_balancers(
        M: int,
        N: int,
        max_22: int,
        max_12: int,
        max_21: int,
        debug=False,
        exact_counts=False,
        solver='coin',
) -> np.ndarray:
    '''
    Attempt to find a splitter-count optimal M -> N balancer for Factorio.

    More efficient than the naive expanded matrix approach, using "implicit"
    input and output flows with inherent restrictions for the splitter problem.
    '''

    Inps = [f'i{ix}' for ix in range(M)]
    Outs = [f'o{ix}' for ix in range(N)]

    Spls22 = [f's[22]{ix}' for ix in range(max_22)]
    Spls12 = [f's[12]{ix}' for ix in range(max_12)]
    Spls21 = [f's[21]{ix}' for ix in range(max_21)]

    Splitters = Spls22 + Spls21 + Spls12

    # tagged internal (between splitters) flows flows, real
    # tags are identified with inputs for the splitter problem, and this
    # identification is integral to the simplifications below
    # XXX might be integralizable from 0 to 2^f(ms), might be faster?
    Flows = pp.LpVariable.dicts("F", (Splitters, Splitters, Inps))

    # splitters enabled
    if not exact_counts:
        S = pp.LpVariable.dicts("S", (Splitters, ), 0, 1, pp.LpBinary)
    else:
        S = {s: 1 for s in Splitters}

    # capacity matrix
    # c[i, j] = 1 <=> directed capacity of 1 exists from i to j
    Conn = pp.LpVariable.dicts("C", (Splitters, Splitters), 0, 1, pp.LpBinary)

    # input map: Imap[i, u] == 1 <=> input i flows into splitter u
    Imap = pp.LpVariable.dicts("Imap", (Inps, Splitters), 0, 1, pp.LpBinary)
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    Omap = pp.LpVariable.dicts("Omap", (Splitters, Outs), 0, 1, pp.LpBinary)

    # ## OBJECTIVE
    prob = pp.LpProblem(
        name="solve_balancer",
        sense=pp.LpMinimize,
    )
    # it starts out gently enough...
    # NOTE: optimizer tuning: add discriminating coefficients
    prob += pp.lpSum(S[spl] * sx for sx, spl in enumerate(Splitters))
    # NOTE: optimizer tuning: EXPERIMENTAL splitter ordering regularization
    for ix, si in enumerate(Splitters):
        # try to were inputs to "low" splitters
        prob += ix * pp.lpSum(Imap[t][si] for t in Inps)
        # try to wire outputs from "late" splitters
        prob += (-ix * pp.lpSum(Omap[si][o] for o in Outs))
        # penalize "backflow"
        for jx, sj in zip(range(ix), Splitters):
            prob += (ix - jx) * Conn[si][sj]

    # ## CONSTRAINTS
    # # CHAPTER 1: ADJACENCY RESTRICTIONS
    #  1.1 INPUTS WELL CONNECTED
    #  1.1.1 each input goes into exactly one splitter
    for inp in Inps:
        prob += pp.lpSum(Imap[inp][u] for u in Splitters) == 1

    #  1.2 OUTPUTS UNIQUELY CONNECTED
    #  1.2.1 each output receives from exactly one splitter
    for out in Outs:
        prob += pp.lpSum(Omap[u][out] for u in Splitters) == 1

    for (icount, ocount), spls in zip([(2, 2), (1, 2), (2, 1)],
                                      [Spls22, Spls12, Spls21]):
        for spl in spls:
            outs_from_spl = pp.lpSum(Omap[spl][out] for out in Outs)
            inps_into_spl = pp.lpSum(Imap[inp][spl] for inp in Inps)
            #  1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
            prob += pp.lpSum(Conn[spl][v] for v in Splitters
                             ) + outs_from_spl == ocount * S[spl]
            #  1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
            prob += pp.lpSum(Conn[u][spl] for u in Splitters
                             ) + inps_into_spl == icount * S[spl]
            # 1.5 NO SELF-LOOPS
            prob += Conn[spl][spl] == 0
            # XXX OBSOLESCENT #  1.5 NO GHOST SPLITTERS
            # #  1.5.1 no splitters without non-self outputs
            # prob += pp.lpSum(Conn[spl][v] for v in Splitters if v != spl
            #                  ) + outs_from_spl >= S[spl]
            # #  1.5.2 no splitters without non-self inputs
            # prob += pp.lpSum(Conn[u][spl] for u in Splitters if u != spl
            #                  ) + inps_into_spl >= S[spl]

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    for u, v in product(Splitters, Splitters):
        prob += pp.lpSum(Flows[u][v][t] for t in Inps) <= Conn[u][v]

    # 2.2 GENERALIZED UNIFORM INCOMPRESSIBILITY
    # forall splitter, tag:
    #    (1)                               output = n_out_cxns / N
    #    (2)                               n_tags = n_inps
    #    (3)                            tot_input = tot_output
    # => (4)                       inflow + input = outflow + output
    # => (5)             inflow + input - outflow = n_out_cxns / N
    # => (6)  n_inps * (inflow + input - outflow) = n_out_cxns
    for spl in Splitters:
        # forall t.
        n_out_cxns = pp.lpSum(Omap[spl][o] for o in Outs)
        for t in Inps:
            inflow_t = pp.lpSum(Flows[u][spl][t] for u in Splitters)
            input_t = Imap[t][spl]
            outflow_t = pp.lpSum(Flows[spl][w][t] for w in Splitters)

            prob += N * (inflow_t + input_t - outflow_t) == n_out_cxns
    # NOTE 2.2 implies both matched inputs and balanced outputs

    # 2.3 PROPER FLOW
    for u, v, t in product(Splitters, Splitters, Inps):
        prob += Flows[u][v][t] >= 0

    # # FACTORIO RESTRICTIONS
    # 3.1 EQUAL SPLITTING
    # The logic: we only need to specify that each non-output connection
    # receives at most half of the total input. Because there are at most
    # two non-output connections, this implies they also receive at least
    # half of the total input. Thus, even splitting.
    # Output connections are handled automatically by the more draconian
    # fixed output constraint + incompressiblity.
    for t, spl in product(Inps, Spls22 + Spls12):
        inflow_t = pp.lpSum(Flows[u][spl][t] for u in Splitters) + Imap[t][spl]
        for w in Splitters:
            prob += 2 * Flows[spl][w][t] <= inflow_t
    # NOTE: 2-1 splitters handled by 2.2
    # NOTE: out-connections handled by 2.2

    # # 3.2 SPLITTER LOWER BOUND
    if not exact_counts:
        for lb, spls in zip(lowerbound_splitters(M, N),
                            [Spls22, Spls12, Spls21]):
            prob += pp.lpSum(S[spl] for spl in spls) >= lb

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
        print('imap')
        print(imap)
        print('omap')
        print(omap)
        for i in Inps:
            print(f'flow for {u}')
            flows = dicts_to_ndarray(Flows,
                                     (Splitters, Splitters, [i])).squeeze()
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
        'm22', type=int, help='max number of 2->2 splitters to consider'
    )
    parser.add_argument(
        'm12', type=int, help='max number of 1->2 splitters to consider'
    )
    parser.add_argument(
        'm21', type=int, help='max number of 2->1 splitters to consider'
    )
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

    args = parser.parse_args(sys.argv[1:])

    graphname = (
        f'balancer_{args.ni}-{args.no}-{"x" if args.exact else "le"}-'
        f'{args.m22}-{args.m12}-{args.m21}'
    )

    print(f'Solving the optimal {args.ni} -> {args.no} balancer...')
    try:
        imap, omap, adjmat, labels = solve_balancers(
            args.ni,
            args.no,
            max_22=args.m22,
            max_12=args.m12,
            max_21=args.m21,
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
