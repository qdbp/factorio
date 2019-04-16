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
    IllSpecified, Infeasible, dicts_to_ndarray, get_lparr_value, get_solver,
    in_sol_dir, lparray, ndarray_to_dicts, number, numprod
)

SOL_SUBDIR = './balancers/'


def lowerbound_splitters(M: int, N: int) -> int:
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
        raise IllSpecified(
            'The problem formulation only allows for fanout designs. '
            'Note that an optimal N -> M solution can be reversed to give '
            'what you want.'
        )

    min_spls = max(lowerbound_splitters(M, N), min_spls)
    if max_spls < min_spls:
        raise Infeasible(
            f'Balancing {M} -> {N} requires at least {min_spls}, which is '
            'lower than the max given'
        )

    # special case needed for fixed omaps
    if N == 2:
        max_spls = 1

    # number of pinned input splitters, disjoint from output splitters
    n_s_ipinned = ceil(M / 2)

    # number of pinned output splitters, disjoint from input splitters
    # if N > 2, outputs can't be connected straight to an input splitter.
    # This is the first separation theorem (see below)
    n_s_opinned = ceil(N / 2) if N > 2 else 0

    # we must have at least the input and output pinned splitters
    # this can be better than the previous lower bound for some M, N
    min_spls = max(min_spls, n_s_ipinned + n_s_opinned)

    Inps = [f'i{ix}' for ix in range(M)]
    Outs = [f'o{ix}' for ix in range(N)]
    Splitters = [f's{ix}' for ix in range(max_spls)]

    # splitters enabled
    if not exact_counts:
        S = lparray.create("S", (Splitters, ), 0, 1, pp.LpBinary)
    else:
        S = np.ones(len(Splitters), dtype=np.uint8)

    # capacity matrix
    # Conn[i, j] = 1 <=> directed capacity of 1 exists from i to j
    Conn = lparray.create("C", (Splitters, Splitters), 0, 1, pp.LpBinary)
    # internal flows
    # Fs = pp.LpVariable.dicts("Fs", (Splitters, Splitters, Inps))
    Fs = lparray.create("Fs", (Splitters, Splitters, Inps))

    # Theorem 1. We can pack all inputs without loss of generality.
    # i.e. We can have inputs 0, 1 -> splitter 0, 1, 2 -> splitter 2, etc.
    # Proof:
    # suppose we have a working balancer with input splitter, s0, s1, s.t.
    #   a->[s]<-i1, b->[t]<-i2 where i1, i2 are inputs
    # We know that by the balancer property i1, i2, a, and b are split
    # equally among all outputs. Swapping in connections a and i2 preserves
    # this. Therefore the sysem a->[s]<-b, i1->[s]<-i2 is also a balancer with
    # the same internal topology.
    Imap = np.zeros((M, len(Splitters)), dtype=np.uint8)
    for ix in range(M):
        Imap[ix, ix // 2] = 1

    # Theorem 2. We can pack all outputs without loss of generality.
    # Proof. Suppos we have a balancer with output splitters s0, s1, such that
    #   a<-[s0]->o0, b<-[s1]->o1 where o0, o1 are output lines.
    # Balancing and equal splitting imply ->a == ->o0 == ->01 == ->b
    # Therefore the system given by swapping the outputs on the two splitters
    # o0<-s0->o1, a<-s1->b is an equivalent balancer.
    Omap = np.zeros((len(Splitters), N), dtype=np.uint8)
    for ix in range(N):
        Omap[len(Splitters) - 1 - ix // 2, ix] = 1

    print(Imap)
    print(Omap)

    # input flows
    Fi = lparray.create("Fi", (Inps, Splitters, Inps))
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    # output flows
    Fo = lparray.create("Fo", (Splitters, Outs, Inps))

    # ## OBJECTIVE
    prob = pp.LpProblem(
        name="solve_balancer",
        sense=pp.LpMinimize,
    )

    objective = pp.LpAffineExpression()
    # penalize number of intermediate splitters
    if not exact_counts:
        objective += 100 * S.sumit()

    # search-biasing objective terms
    if not exact_counts:
        for si in number(Splitters):
            # penalize "backflow"
            for sj in range(ix):
                objective += (si - sj) * Conn[si, sj]
            # penalize "jumps"
            for sj in range(ix + 1, len(Splitters)):
                objective += (sj - si) * Conn[si, sj]

    prob += objective

    if not exact_counts:
        print(f'Solver/Info: n_splitters in [{min_spls}, {max_spls}].')
    else:
        print(f'Solver/Info: solving with exactly {max_spls} splitters')
    print(
        f'Solver/Info: {n_s_ipinned} input pinned; '
        f'{n_s_opinned} output pinned'
    )

    # ## CONSTRAINTS
    # # CHAPTER 0: FORMULATION RESTRICTIONS
    # 0.1: RESPECT PINS
    if not exact_counts:
        (S[:n_s_ipinned] == 1).constrain(prob)
        if n_s_opinned > 0:
            (S[-n_s_opinned:] == 1).constrain(prob)

    # 0.2: UNPINNED SPLITTERS ARE ORDERED
    if not exact_counts:
        for u, v in zip(
                range(n_s_ipinned),
                range(n_s_ipinned + 1, n_s_opinned),
        ):
            prob += S[u] >= S[v]

    # 0.3: MINIMUM NUMBER OF SPLITTER
    # each input must have at least ceil(lg(N)) between it and any output
    # this creates a trivial lower bound of (M // 2)ceil(lg(N)) splitters
    if not exact_counts:
        (S.sum() >= min_spls).constrain(prob)

    # 0.4: SPLITTER SEPARATION THEOREMS
    # Theorem 3.
    # If N > 4, an input splitter cannot be connected to an output splitter.
    # Proof.
    # The fraction of item i on output o under a subgraph like
    # i -> s0 -> s1 -> o is at least 1 / 4. Balancing requires that it
    # be 1 / N < 1 / 4.  Therefore such a subgraph cannot exist.
    # NOTE the N > 2 splitter separation theorm is implied by the splitter
    # pinning restrictions.
    if N > 4:
        print('Solver/Info: invoking N > 4 separation theorem.')
        (Conn[:n_s_ipinned, -n_s_opinned:] == 0).constrain(prob)

    # Extension 3.1.
    # If N > 8, a chain i -> s0 -> s1 -> s2 -> o cannot exist.
    # Proof. As above.
    # NOTE this theorem can be expressed linearly because of fixed input
    # and output maps.
    # FIXME currently broken: the CO1 and CI1 sums can produce more
    # than 1, thus excluding valid solutions in the constraint
    if N > 80000:
        print('Solver/Info: invoking N > 8 separation theorem.')
        # CO0[s] = 1 if s -> output for any output
        # this is a constatn
        CO0 = np.zeros(len(Splitters), dtype=np.uint8)
        CO0[-n_s_opinned:] = 1
        # CI0[s] = 1 if input -> s for any input
        # this is constant
        CI0 = np.zeros(len(Splitters), dtype=np.uint8)
        CI0[:n_s_ipinned] = 1

        # CO1[v] == 1 <=> exists v in Splitters : C[u, v] == 1 and CO0[v] == 1
        CO1 = (Conn * CO0[None, :]).sum(axis=1)
        # CI1[v] == 1 <=> exists u in Splitters : C[u, v] == 1 and CI0[u] == 1
        CI1 = (Conn * CI0[:, None]).sum(axis=0)
        # we have NOT (Conn[u, v] and CI0[u] == 1 and CO1[v] == 1)
        (CI0[:, None] + Conn + CO1[None, :] <= 2).constrain(prob)
        # and NOT (Conn[u, v] and CI1[u] == 1 and CO0[v] == 1)
        (CI1[:, None] + Conn + CO0[None, :] <= 2).constrain(prob)

    # NOTE higher separation theorems are non-linear.
    # NOTE it can be approximated (lower-bound on the number of constraints)
    # by "higher order pinning" similar to that implicit in the lowerbound
    # calculation

    # # CHAPTER 1: ADJACENCY RESTRICTIONS
    # 1.1 INPUTS WELL CONNECTED
    # 1.1.1 each input goes into exactly one splitter
    # 1.1.2 each splitter receives from at most two inputs
    # NOTE these are automatic with input pinning

    #  1.2 OUTPUTS UNIQUELY CONNECTED
    #  1.2.1 each output receives from exactly one splitter
    #  1.2.2 each splitter goes to at most two outputs
    # NOTE these are automatic with output pinning

    out_from_spls = Omap.sum(axis=1)
    inp_into_spls = Imap.sum(axis=0)

    # 1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
    (Conn.sum(axis=1) + out_from_spls <= 2 * S).constrain(prob, 'MaxOuts')
    (Conn.sum(axis=1) + out_from_spls >= S).constrain(prob, 'MinOuts')

    # 1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
    (Conn.sum(axis=0) + inp_into_spls <= 2 * S).constrain(prob, 'MaxIns')
    (Conn.sum(axis=0) + inp_into_spls >= S).constrain(prob, 'MinIns')

    # 1.5 NO SELF LOOPS
    (np.diag(Conn) == 0).constrain(prob, 'NoSelfLoops')

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    (Fs.sum(axis=2) <= Conn).constrain(prob, 'FlowCap')
    # for ux, vx in numprod(Splitters, Splitters):
    #     prob += Fs[ux, vx].sumit() <= Conn[ux, vx]

    # 2.2 INFLOW EDGE CONDITIONS
    # forall i,v,t: F[i, v, t] == Imap[i, v] * 1[i == t]
    (Fi == Imap[:, :, None] * np.eye(M)[:, None, :]).constrain(prob, 'InEdge')

    # 2.3 OUTFLOW EDGE CONDITIONS
    # forall v,o,t: Fo[v, o, t] == Omap[v, o] / N
    (N * Fo == Omap[..., None]).constrain(prob, 'OutEdge')

    # 2.4 PROPER FLOW
    (Fs >= 0).constrain(prob, 'ProperFlowS')
    (Fi >= 0).constrain(prob, 'ProperFlowI')
    (Fo >= 0).constrain(prob, 'ProperFlowO')

    # 2.5 INCOMPRESSIBILITY
    inflows = Fs.sum(axis=0) + Fi.sum(axis=0)  # (spls, types)
    outflows = Fs.sum(axis=1) + Fo.sum(axis=1)  # (spls, types)
    (inflows == outflows).constrain(prob, 'Incompressibility')

    # 2.6 EQUAL SPLITTING
    # forall w: 2 * Fs[s, w, t] <= inflow[s, t]
    (2 * Fs <= inflows[:, None, :]).constrain(prob, 'SplittingS')
    # forall w: 2 * Fo[s, o, t] <= inflow[s, t]
    (2 * Fo <= inflows[:, None, :]).constrain(prob, 'SplittingO')

    # ## SOLVING
    # #
    prob.solve(solver=get_solver(which=solver))

    if 'Infeasible' in pp.LpStatus[prob.status]:
        raise Infeasible

    keep_rows = np.where(get_lparr_value(S) > 0)[0]

    adjmat = get_lparr_value(Conn)[np.ix_(keep_rows, keep_rows)]
    labels = np.array(Splitters)[keep_rows]

    return Imap[:, keep_rows], Omap[keep_rows], adjmat, labels


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
