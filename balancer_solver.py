# Dicovering the optimal M x N balancers, phase 1: max flow
# Evgeny Naumov, 2019
from __future__ import annotations

import typing as ty
from itertools import product
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pulp as pp
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from solver_core import Infeasible, dicts_to_ndarray, get_solver


def solve_balancers(
        n_inps: int,
        n_outs: int,
        max_22: int,
        max_12: int,
        max_21: int,
        debug=False,
        exact=False,
) -> np.ndarray:
    '''
    Finds a minimum-splitter balancer using the max-flow formalism with
    extra restrictions.

    Makes no guarantee about the implementability of the topology!
    This is only the beginning...
    '''

    prob = pp.LpProblem(name="solve_balancer", sense=pp.LpMinimize)

    # Inps is simultaneously the item type set
    # XXX in principle they can be separated for more fancy balancers,
    # but there are only so many hours in a day
    Inps = [f'i{ix}' for ix in range(n_inps)]
    Outs = [f'o{ix}' for ix in range(n_outs)]

    Spls22 = [f's[22]{ix}' for ix in range(max_22)]
    Spls12 = [f's[12]{ix}' for ix in range(max_12)]
    Spls21 = [f's[21]{ix}' for ix in range(max_21)]

    AllSpls = Spls22 + Spls21 + Spls12
    Nodes = Inps + Outs + AllSpls

    # flows
    F = pp.LpVariable.dicts("F", (Nodes, Nodes, Inps))

    # splitters enabled
    if not exact:
        S = pp.LpVariable.dicts("S", (AllSpls, ), 0, 1, pp.LpBinary)
    else:
        print('Requiring exact splitter counts!')
        S = {spl: 1 for spl in AllSpls}

    # capacity matrix
    # c[i, j] = 1 <=> directed capacity of 1 exists from i to j
    # XXX binary makes all splitters have to have two distinct inputs and
    # outputs. This should be fine, but making this {0, 1, 2} integer is more
    # general -- at the cost of input/output/splitter distinction complications
    # in the restrictions.
    C = pp.LpVariable.dicts("C", (Nodes, Nodes), 0, 1, pp.LpBinary)

    # it starts out gently enough...
    if not exact:
        prob += pp.lpSum(S[spl] for spl in AllSpls)
    else:
        prob += 0

    # ## RESTRICTIONS
    # # CHAPTER 1: ADJACENCY RESTRICTIONS
    #  1.1 INPUTS WELL CONNECTED
    for inp in Inps:
        # each input flows to one node
        prob += pp.lpSum(C[inp][v] for v in Nodes) == 1
        # nothing flows into an input
        prob += pp.lpSum(C[u][inp] for u in Nodes) == 0

    #  1.2 OUTPUTS UNIQUELY CONNECTED
    for out in Outs:
        # exactly one node flows into an output
        prob += pp.lpSum(C[u][out] for u in Nodes) == 1
        # an output flows into nothing
        prob += pp.lpSum(C[out][v] for v in Nodes) == 0

    #  XXX 1.3 and 1.4 disallow dangling splitter inputs and outputs
    for (icount, ocount), spls in zip([(2, 2), (1, 2), (2, 1)],
                                      [Spls22, Spls12, Spls21]):
        for spl in spls:
            #  1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
            prob += pp.lpSum(C[spl][v] for v in Nodes) == ocount * S[spl]
            #  1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
            prob += pp.lpSum(C[u][spl] for u in Nodes) == icount * S[spl]
            #  1.5 NO GHOST SPLITTERS
            prob += pp.lpSum(C[spl][v] for v in Nodes if v != spl
                             ) >= 1 * S[spl]
            prob += pp.lpSum(C[u][spl] for u in Nodes if u != spl
                             ) >= 1 * S[spl]

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    for u, v in product(Nodes, Nodes):
        prob += pp.lpSum(F[u][v][i] for i in Inps) <= C[u][v]

    # 2.2 INCOMPRESSIBLE PARTIAL FLOWS FOR SPLITTER NODES
    for spl, i in product(AllSpls, Inps):
        prob += (
            pp.lpSum(F[u][spl][i]
                     for u in Nodes) == pp.lpSum(F[spl][w][i] for w in Nodes)
        )

    # 2.3 PROPER FLOW
    for u, v, i in product(Nodes, Nodes, Inps):
        prob += F[u][v][i] >= 0

    # # CHAPTER 3: FACTORIO RESTRICTIONS
    # 3.1 TYPED INPUTS
    for in_node, in_type in product(Inps, Inps):
        prob += pp.lpSum(F[in_node][v][in_type]
                         for v in Nodes) == int(in_node == in_type)

    # 3.2 BALANCED OUTPUTS
    # it's why we're here, no?
    # NOTE inp here is being used as a flow type
    o0 = Outs[0]
    i0 = Inps[0]
    for inp, out in product(Inps, Outs):
        prob += pp.lpSum(F[u][out][inp] for u in Nodes
                         ) == pp.lpSum(F[u][o0][i0] for u in Nodes)

    # 3.3 SPLITTERS EQUALIZE ALL TYPES
    # NOTE: 2-1 splitters just need incompressibility
    for i, spl in product(Inps, Spls22 + Spls12):
        inflow_i = pp.lpSum(F[u][spl][i] for u in Nodes)
        # this with incompressiblity implies flow[spl,dst,i] is either 0 or
        # half the inflow, as needed
        for dst in Nodes:
            prob += 2 * F[spl][dst][i] <= inflow_i

    with NamedTemporaryFile(suffix='.lp') as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve(solver=get_solver())

    if 'Infeasible' in pp.LpStatus[prob.status]:
        raise Infeasible

    adjmat = dicts_to_ndarray(C, (Nodes, Nodes))
    keep_rows = np.where(adjmat.sum(axis=0) + adjmat.sum(axis=1) != 0)[0]
    adjmat = adjmat[np.ix_(keep_rows, keep_rows)]

    if debug:
        print('Debug/Flows:')
        for inp in Inps:
            print(f'flow for {inp}')
            flows = dicts_to_ndarray(F, (Nodes, Nodes, [inp])).squeeze()
            flows = flows[np.ix_(keep_rows, keep_rows)]
            print(flows)

    return adjmat, np.array(Nodes)[keep_rows]


def solve_balancers_2(
        n_inps: int,
        n_outs: int,
        max_22: int,
        max_12: int,
        max_21: int,
        debug=False,
) -> np.ndarray:
    '''
    Attempt to minimize the number of variables by restricting the adjacency
    matrix to be n_spls x n_spls, excluding inputs and outputs.

    The first version was TOO SLOW.
    '''

    prob = pp.LpProblem(name="solve_balancer", sense=pp.LpMinimize)

    # Inps is simultaneously the item type set
    # XXX in principle they can be separated for more fancy balancers,
    # but there are only so many hours in a day
    Inps = [f'i{ix}' for ix in range(n_inps)]
    Outs = [f'o{ix}' for ix in range(n_outs)]

    Spls22 = [f's[22]{ix}' for ix in range(max_22)]
    Spls12 = [f's[12]{ix}' for ix in range(max_12)]
    Spls21 = [f's[21]{ix}' for ix in range(max_21)]

    Nodes = Spls22 + Spls21 + Spls12

    # internal flows, real... (XXX could be integralized from 0 to 2^f(ms),
    # might be faster)
    F = pp.LpVariable.dicts("F", (Nodes, Nodes, Inps))

    # splitters enabled
    S = pp.LpVariable.dicts("S", (Nodes, ), 0, 1, pp.LpBinary)

    # capacity matrix
    # c[i, j] = 1 <=> directed capacity of 1 exists from i to j
    # XXX binary makes all splitters have to have two distinct inputs and
    # outputs. This should be fine, but making this {0, 1, 2} integer is more
    # general -- at the cost of input/output/splitter distinction complications
    # in the restrictions.
    C = pp.LpVariable.dicts("C", (Nodes, Nodes), 0, 1, pp.LpBinary)

    # input map: Imap[i, u] == 1 <=> input i flows into splitter u
    Imap = pp.LpVariable.dicts("Imap", (Inps, Nodes), 0, 1, pp.LpBinary)
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    Omap = pp.LpVariable.dicts("Omap", (Nodes, Outs), 0, 1, pp.LpBinary)

    # it starts out gently enough...
    prob += pp.lpSum(S[spl] for spl in Nodes)

    # ## RESTRICTIONS
    # # CHAPTER 1: ADJACENCY RESTRICTIONS
    #  1.1 INPUTS WELL CONNECTED
    #  1.1.1 each input goes into exactly one splitter
    for inp in Inps:
        prob += pp.lpSum(Imap[inp][u] for u in Nodes) == 1
    #  1.1.2 each splitter receives at most two inputs
    for u in Nodes:
        prob += pp.lpSum(Imap[inp][u] for inp in Inps) <= 2

    #  1.2 OUTPUTS UNIQUELY CONNECTED
    #  1.2.1 each output receives from exactly one splitter
    for out in Outs:
        prob += pp.lpSum(Omap[u][out] for u in Nodes) == 1
    #  1.2.2 each splitter serves at most two outputs
    for u in Nodes:
        prob += pp.lpSum(Omap[u][out] for out in Outs) <= 2

    for (icount, ocount), spls in zip([(2, 2), (1, 2), (2, 1)],
                                      [Spls22, Spls12, Spls21]):
        for spl in spls:
            outs_from_spl = pp.lpSum(Omap[spl][out] for out in Outs)
            ints_into_spl = pp.lpSum(Imap[inp][spl] for out in Inps)
            #  1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
            prob += pp.lpSum(C[spl][v]
                             for v in Nodes) + outs_from_spl == ocount * S[spl]
            #  1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
            prob += pp.lpSum(C[u][spl]
                             for u in Nodes) + ints_into_spl == icount * S[spl]
            #  1.5 NO GHOST SPLITTERS
            #  1.5.1 no splitters without non-self outputs
            prob += pp.lpSum(C[spl][v] for v in Nodes if v != spl
                             ) + outs_from_spl >= S[spl]
            #  1.5.2 no splitters without non-self inputs
            prob += pp.lpSum(C[u][spl] for u in Nodes if u != spl
                             ) + ints_into_spl >= S[spl]

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    for u, v in product(Nodes, Nodes):
        prob += pp.lpSum(F[u][v][i] for i in Inps) <= C[u][v]

    # 2.2 INCOMPRESSIBLE PARTIAL FLOWS FOR SPLITTER NODES
    # this one is a little bit complicated, because of the output.
    # The key to understanding is that the output, for each belt and for
    # each type is equal, and has value #(connected outputs) / n_inps
    # (thus summed across all flow types the external output is just
    # #(connected outputs) as expected.)
    # we can't divide by n_types, so we multiply the lhs instead.
    for spl in Nodes:
        # forall t.
        n_times_output_t = pp.lpSum(Omap[spl][o] for o in Outs)
        for flow_type in Inps:

            inflow_t = pp.lpSum(F[u][spl][flow_type] for u in Nodes)
            input_t = Imap[flow_type][spl]

            outflow_t = pp.lpSum(F[spl][w][flow_type] for w in Nodes)

            prob += (
                n_inps * (inflow_t + input_t - outflow_t) == n_times_output_t
            )

    # 2.3 PROPER FLOW
    for u, v, i in product(Nodes, Nodes, Inps):
        prob += F[u][v][i] >= 0

    # CHAPTER 3: FACTORIO RESTRICTIONS
    # 3.1 Match Inputs
    # for in_node, in_type in product(Inps, Inps):
    #     prob += pp.lpSum(F[in_node][v][in_type]
    #                      for v in Nodes) == int(in_node == in_type)

    # # 3.2 BALANCED OUTPUTS
    # # it's why we're here, no?
    # # NOTE inp here is being used as a flow type
    # o0 = Outs[0]
    # i0 = Inps[0]
    # for inp, out in product(Inps, Outs):
    #     prob += pp.lpSum(F[u][out][inp] for u in Nodes
    #                      ) == pp.lpSum(F[u][o0][i0] for u in Nodes)

    # 3.3 SPLITTERS EQUALIZE ALL TYPES
    # NOTE: 2-1 splitters just need incompressibility
    # FIXME for i, spl in product(Inps, Spls22 + Spls12):
    # FIXME     inflow_i = pp.lpSum(F[u][spl][i] for u in Nodes) + Imap[i][spl]
    # FIXME     # this with incompressiblity implies flow[spl,dst,i] is either 0 or
    # FIXME     # half the inflow, as needed
    # FIXME     for dst in Nodes:
    # FIXME         prob += 2 * F[spl][dst][i] <= inflow_i

    with NamedTemporaryFile(suffix='.lp') as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve(solver=get_solver())

    if 'Infeasible' in pp.LpStatus[prob.status]:
        raise Infeasible

    adjmat = dicts_to_ndarray(C, (Nodes, Nodes))
    keep_rows = np.where(adjmat.sum(axis=0) + adjmat.sum(axis=1) != 0)[0]
    adjmat = adjmat[np.ix_(keep_rows, keep_rows)]

    if debug:
        print('inmap')
        print(dicts_to_ndarray(Imap, (Inps, Nodes)))
        print('omap')
        print(dicts_to_ndarray(Omap, (Nodes, Outs)))
        for i in Inps:
            print(f'flow for {u}')
            flows = dicts_to_ndarray(F, (Nodes, Nodes, [i])).squeeze()
            # flows = flows[np.ix_(keep_rows, keep_rows)]
            print(flows)

    return adjmat, np.array(Nodes)[keep_rows]


def draw_adjmat(
        adjmat: np.ndarray,
        n_inps: int,
        n_outs: int,
        labels=None,
        graphname='graph.png'
) -> None:

    n_spls = adjmat.shape[0] - n_inps - n_outs

    g = nx.convert_matrix.from_numpy_array(adjmat, create_using=nx.DiGraph)
    if labels is not None:
        g = nx.relabel_nodes(g, {ix: label for ix, label in enumerate(labels)})

    values = ['red'] * n_inps + ['blue'] * n_outs + ['black'] * n_spls
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

    args = parser.parse_args(sys.argv[1:])

    # graphname = f'{ni}-{no}'
    graphname = 'graph'

    print(f'Solving the optimal {args.ni} -> {args.no} balancer...')
    try:
        adjmat, labels = solve_balancers_2(
            args.ni,
            args.no,
            max_22=args.m22,
            max_12=args.m12,
            max_21=args.m21,
            debug=args.debug,
            # exact=args.exact,
        )
    except Infeasible:
        print(f'No feasible solution within the given splitter limites.')
        return

    draw_adjmat(adjmat, args.ni, args.no, labels=labels, graphname=graphname)
    print(f'solution saved to {graphname}.png')


if __name__ == '__main__':
    main()
