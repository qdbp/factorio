# Dicovering the optimal M x N balancers, phase 1: max flow
# Evgeny Naumov, 2019
from __future__ import annotations

from itertools import product
from math import ceil
from tempfile import NamedTemporaryFile

import networkx as nx
import numpy as np
import pulp as pp
from networkx.drawing.nx_agraph import to_agraph

from solver_core import (
    Infeasible, dicts_to_ndarray, get_solver, in_sol_dir, ndarray_to_dicts
)

SOL_SUBDIR = './balancers/'


def lowerbound_splitters(M, N) -> int:
    '''
    Guesses a lower bound on the number of splitters needed for an M-N balancer
    '''

    # must place at least ceil(lg(N)) splitters between every input and output
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

    # number of pinned input splitters
    n_s_ipinned = ceil(M / 2)
    # number of pinned output splitters
    # if N >= 3, outputs can't be connected straight to an input splitter. ->
    # min_spls >= ceil(M/2) + (ceil(N/2))
    # For N < 3 (just to be correct!), no splitters are opinned
    n_s_opinned = 0 if N < 3 else ceil(N / 2)

    # we must have at least the input and output pinned splitters
    min_spls = max(min_spls, n_s_ipinned + n_s_opinned)

    Inps = [f'i{ix}' for ix in range(M)]
    Outs = [f'o{ix}' for ix in range(N)]

    Splitters = [f's{ix}' for ix in range(max_spls)]

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

    # Theorem 1. We can pack all inputs without loss of generality.
    # i.e. We can have inputs 0, 1 -> splitter 0, 1, 2 -> splitter 2, etc.
    # Proof:
    # suppose we have a working balancer with input splitter, s0, s1, s.t.
    #   a->[s]<-i1, b->[t]<-i2 where i1, i2 are inputs
    # We know that by the balancer property i1, i2, a, and b are split
    # equally among all outputs. Swapping in connections a and i2 preserves
    # this. Therefore the sysem a->[s]<-b, i1->[s]<-i2 is also a balancer with
    # the same internal topology.
    fixed_imap = np.zeros((M, len(Splitters)), dtype=np.uint8)
    for ix in range(M):
        fixed_imap[ix][ix // 2] = 1
    Imap = ndarray_to_dicts(fixed_imap, (Inps, Splitters))

    # Theorem 2. We can pack all outputs without loss of generality.
    # Proof. Suppos we have a balancer with output splitters s0, s1, such that
    #   a<-[s0]->o0, b<-[s1]->o1 where o0, o1 are output lines.
    # Balancing and equal splitting imply ->a == ->o0 == ->01 == ->b
    # Therefore the system given by swapping the outputs on the two splitters
    # o0<-s0->o1, a<-s1->b is an equivalent balancer.
    if n_s_opinned > 0:
        fixed_omap = np.zeros((len(Splitters), N), dtype=np.uint8)
        for ix in range(N):
            fixed_omap[len(Splitters) - 1 - ix // 2][ix] = 1
        Omap = ndarray_to_dicts(fixed_omap, (Splitters, Outs))
    else:
        Omap = pp.LpVariable.dicts(
            "Omap", (Splitters, Outs), 0, 1, pp.LpBinary
        )

    # input flows
    Fi = pp.LpVariable.dicts("Fi", (Inps, Splitters, Inps))
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    # output flows
    Fo = pp.LpVariable.dicts("Fo", (Splitters, Outs, Inps))

    # ## OBJECTIVE
    prob = pp.LpProblem(
        name="solve_balancer",
        sense=pp.LpMinimize,
    )

    objective = pp.LpAffineExpression()
    # penalize number of intermediate splitters
    objective += pp.lpSum(
        100 * S[spl]
        for spl in Splitters[n_s_ipinned:len(Splitters) - n_s_opinned]
    )

    # search-biasing objective terms
    if not exact_counts:
        for ix, si in enumerate(Splitters):
            # penalize "backflow"
            for jx, sj in zip(range(ix), Splitters):
                objective += (ix - jx) * Conn[si][sj]
            # penalize "jumps"
            for jx in range(ix + 1, len(Splitters)):
                sj = Splitters[jx]
                objective += (jx - ix) * Conn[si][sj]

    prob += objective

    print(f'Solver/Info: n_splitters in [{min_spls}, {max_spls}].')
    print(
        f'Solver/Info: {n_s_ipinned} input pinned; '
        f'{n_s_opinned} output pinned'
    )

    # ## CONSTRAINTS
    # # CHAPTER 0: FORMULATION RESTRICTIONS
    # 0.1: RESPECT PINS
    for ix in range(n_s_ipinned):
        prob += S[Splitters[ix]] == 1
    for jx in range(n_s_opinned):
        prob += S[Splitters[-jx - 1]] == 1

    # 0.2: UNPINNED SPLITTERS ARE ORDERED
    if not exact_counts:
        for u, v in zip(Splitters[n_s_ipinned:],
                        Splitters[n_s_ipinned + 1:n_s_opinned]):
            prob += S[u] >= S[v]

    # 0.3: MINIMUM NUMBER OF SPLITTER
    # each input must have at least ceil(lg(N)) between it and any output
    # this creates a trivial lower bound of (M // 2)ceil(lg(N)) splitters
    prob += pp.lpSum(S[u] for u in Splitters) >= min_spls

    # 0.4: SPLITTER SEPARATION THEOREM
    # Theorem 3. If N > 4, an input splitter cannot be connected to an output
    # splitter.
    # Proof. The fraction of item i on output o i -> s0 -> s1 -> o is
    # at least 1 / 4. Balancing requires that it be 1 / N < 1 / 4.
    # Therefore such a subgraph cannot exist.
    # NOTE the N > 2 splitter separation theorm is implied by the splitter
    # pinning restrictions.
    if N > 4:
        print('Solver/Info: invoking N > 4 separation theorem.')
        for si, so in product(Splitters[:n_s_ipinned],
                              Splitters[-n_s_opinned:]):
            prob += Conn[si][so] == 0

    # Extension 3.1. If N > 8, a chain i -> s0 -> s1 -> s2 -> o cannot exist.
    # NOTE this theorem can be expressed linearly because of fixed inputs
    # and outputs.
    if N > 8:
        print('Solver/Info: invoking N > 8 separation theorem.')
        CO0_base = np.zeros(len(Splitters), dtype=np.uint8)
        CO0_base[-n_s_opinned:] = 1
        # CO0[s] = 1 if s -> output for any output
        # this is a constatn
        CO0 = ndarray_to_dicts(CO0_base, (Splitters, ))

        CI0_base = np.zeros(len(Splitters), dtype=np.uint8)
        CI0_base[:n_s_ipinned] = 1
        # CI0[s] = 1 if input -> s for any input
        # this is constant
        CI0 = ndarray_to_dicts(CI0_base, (Splitters, ))

        # CO1[s] == 1 <=> s -> s0 for any s0 in CO0
        CO1 = {
            u: pp.lpSum(Conn[u][v] * CO0[v] for v in Splitters)
            for u in Splitters
        }

        # CI1[s] == 1 <=> s0 -> s for any s0 in CI0
        CI1 = {
            v: pp.lpSum(Conn[u][v] * CI0[u] for u in Splitters)
            for v in Splitters
        }

        # we have NOT (Conn[u, v] and CI0[u] == 1 and CO1[v] == 1) AND
        # NOT (Conn[u, v] and CI1[u] == 1 and CO0[v] == 1)
        for u, v in product(Splitters, Splitters):
            prob += Conn[u][v] + CI0[u] + CO1[v] <= 2
            prob += Conn[u][v] + CI1[u] + CO0[v] <= 2

    # NOTE higher separation theorems are non-linear.

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

    return imap, omap, adjmat, labels


def draw_solution(
        imap: np.ndarray,
        omap: np.ndarray,
        splmat: np.ndarray,
        labels=None,
        graphname='graph.png',
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
    full_adjmat = np.zeros(
        (n_inps + n_spls + n_outs, ) * 2,
        dtype=np.uint8,
    )
    full_adjmat[:n_inps, n_inps:-n_outs] = imap
    full_adjmat[n_inps:-n_outs, n_inps:-n_outs] = splmat
    full_adjmat[n_inps:-n_outs, -n_outs:] = omap

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
        '--solver', type=str, default='coin', help='specify the solver'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='draw a reversed graph of the solution'
    )

    args = parser.parse_args(sys.argv[1:])

    graphname = (
        f'bal_{args.ni}-{args.no}-{"x" if args.exact else "le"}-'
        f'{args.maxs}'
    )
    graphname = str(in_sol_dir(SOL_SUBDIR + graphname))

    print(f'Solving the optimal {args.ni} -> {args.no} balancer...')
    try:
        imap, omap, adjmat, labels = solve_balancers(
            args.ni,
            args.no,
            max_spls=args.maxs,
            min_spls=args.mins,
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
    )
    print(f'solution saved to {graphname}.png')

    np.save(f'{graphname}.npy', adjmat)
    print(f'adjmat saved to {graphname}.npy')


if __name__ == '__main__':
    main()
