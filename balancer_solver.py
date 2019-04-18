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

    # special case
    if M == 1 and N == 1:
        return 0

    H = ceil(np.log2(N))
    W = ceil(M / 2)

    def min_by_height(h: int):
        return max(W, ceil(N / (2**(H - h))))

    return sum(min_by_height(h) for h in range(H))
    # return H * W


def solve_balancers(
        M: int,
        N: int,
        max_spls: int,
        min_spls: int = 0,
        max_backedges: int = None,
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

    # special case needed for fixed Omap.
    if N == 2:
        max_spls = 1

    # number of pinned input splitters, disjoint from output splitters
    n_s_ipinned_tot = n_s_ipinned_0 = ceil(M / 2)

    # number of pinned output splitters, disjoint from input splitters
    # if N > 2, outputs can't be connected straight to an input splitter.
    # This is the first separation theorem (see below)
    n_s_opinned_tot = n_s_opinned_0 = ceil(N / 2) if N > 2 else 0

    # we must have at least the input and output pinned splitters
    # this can be better than the previous lower bound for some M, N
    min_spls = max(min_spls, n_s_ipinned_0 + n_s_opinned_0)

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

    # input flows
    Fi = lparray.create("Fi", (Inps, Splitters, Inps))
    # output map: Omap[u, t] == 1 <=> splitter u flows into output t
    # output flows
    Fo = lparray.create("Fo", (Splitters, Outs, Inps))

    prob = pp.LpProblem(
        name="solve_balancer",
        sense=pp.LpMinimize,
    )

    if not exact_counts:
        print(f'Solver/Info: n_splitters in [{min_spls}, {max_spls}].')
    else:
        print(f'Solver/Info: solving with exactly {max_spls} splitters')
    print(
        f'Solver/Info: {n_s_ipinned_0} input pinned; '
        f'{n_s_opinned_0} output pinned'
    )
    if max_backedges is not None:
        print(f'Solver/Info: {max_backedges} max backedges')

    # ## CONSTRAINTS
    # # CHAPTER 0: FORMULATION RESTRICTIONS
    # 0.1: MINIMUM NUMBER OF SPLITTERS
    # each input must have at least ceil(lg(N)) between it and any output
    # this creates a trivial lower bound of (M // 2)ceil(lg(N)) splitters
    if not exact_counts:
        (S.sum() >= min_spls).constrain(prob, 'MinSplitters')

    # 0.2: SPLITTER SEPARATION THEOREMS
    # We adopt the following formalism:
    # Define I0 = {s | s is ipinned}, O0 = {s | s is opinned}
    # I1 = {s | exists u in I0: C[u, s] == 1 and s not in I0}
    # O1 = {s | exists w in O0: C[s, w] == 1 and s not in O0}
    # We have trivially that I0, I1, ... are pairwise disjoint, same for On
    # for N > 2, we have I0 disjoint O0, hence we have separate pinsets
    #
    # Theorem 3.
    # If N > 4, an input splitter cannot be connected to an output splitter.
    # Proof.
    # The fraction of item i on output o under a subgraph like
    # i -> s0 -> s1 -> o is at least 1 / 4. Balancing requires that it
    # be 1 / N < 1 / 4.  Therefore such a subgraph cannot exist.
    #
    # Therefore, N > 4 => Im disjoint On if m + n < 2. Thus any destination
    # nodes of I0 (by definition in I1) cannot flow into the output, since any
    # nodes flowing into the output are by definition in O0.
    if N > 4:
        print('Solver/Info: invoking N > 4 separation theorem.')
        print(f'Solver/Info: {n_s_opinned_0 * n_s_ipinned_0} entries zeroed.')

        (Conn[:n_s_ipinned_0, -n_s_opinned_0:] == 0).constrain(prob, "Sep4")

    # Extension 3.1.
    # If N > 8, a chain i -> s0 -> s1 -> s2 -> o cannot exist.
    # Proof. As above.
    #
    # From this it follows Im disjoint On if m + n < 3.
    #
    # I believe a linear constraint for this theorem exactly does not exist.
    # However, we can allocate lower bounds on the size of blocks on I1 and I2
    # |I1| >= I0 -- since the flow cannot be compressed.
    # |O1| >= max(|I0|, ceil(|O0| / 2)) -- since flow cannot be
    # compressed and splitters fan out at most by 2
    if N > 8:
        n_s_ipinned_1 = n_s_ipinned_0
        n_s_opinned_1 = max(n_s_ipinned_1, ceil(n_s_opinned_0 / 2))

        n_s_ipinned_tot += n_s_ipinned_1
        n_s_opinned_tot += n_s_opinned_1

        print('Solver/Info: invoking N > 8 separation theorem:')
        print(f'Solver/Info: {n_s_ipinned_1} I1 pinned.')
        print(f'Solver/Info: {n_s_opinned_1} O1 pinned.')
        print(
            f'Solver/Info: {n_s_ipinned_1 * n_s_opinned_0 * 2} entries zeroed.'
        )

        # lower bound constraint: each element in the I1 lower bound
        # must receive at least one connection from an element of I0
        (Conn[:n_s_ipinned_0, n_s_ipinned_0:n_s_ipinned_tot].sum(axis=0) >=
         1).constrain(prob, "I1def")

        # lower bound constraint: each element in the O1 lower bound
        # must send at least one connection to an element of O0
        (
            Conn[-n_s_opinned_tot:-n_s_opinned_0, -n_s_opinned_0:].sum(axis=1)
            >= 1
        ).constrain(prob, "O1def")

        # I1 disjoint O1 (lower bound)
        # NOTE implied by separate indices

        # I2 disjoint 00
        (Conn[n_s_ipinned_0:n_s_ipinned_tot, -n_s_opinned_0:] == 0
         ).constrain(prob, "Sep8a")

        # I0 disjoint O2
        (Conn[:n_s_ipinned_0, -n_s_opinned_tot:n_s_opinned_0] == 0
         ).constrain(prob, "Sep8b")

    # 0.3: ORDERED UNPINNED SPLITTERS
    if not exact_counts:
        for u, v in zip(
                range(n_s_ipinned_tot),
                range(n_s_ipinned_tot + 1, len(Splitters) - n_s_opinned_tot),
        ):
            prob += S[u] >= S[v]

    # 0.4: RESPECT PINS
    if not exact_counts:
        (S[:n_s_ipinned_tot] == 1).constrain(prob)
        if n_s_opinned_tot > 0:
            (S[-n_s_opinned_tot:] == 1).constrain(prob)

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

    # Theorem 4:
    # Every solution has a linearization in which no node has two backedges.
    #
    # Lemma 4.a: Every useful node must have a descendant in the output set.
    # Proof. Either the node is part of a cycle with no inputs, in which case
    # it is disallowed; or it is part of a cycle with one input and no outputs
    # which violates incompressibility; or it has a descendant output.
    #
    # Proof of theorem.
    # Consider an arbitary linear order of the graph G with outputs at the end.
    # Split the list of nodes at the lowest index k such that every node with
    # index > k has at least one forward edge. Call these halves Gl and Gh.
    # This is trivially possible since some nodes must be connected to the
    # outputs. By construction the node at k has two backedges. By lemma 4.a
    # we can trace either one of these to find a node at j < k with a child in
    # Gh. (Noting that for any node != k, (has child < k, has child in Gh) are
    # collectively exhaustive). Now swap nodes at k and j. We note now that
    # |Gl| has shrunk by at least 1 since the new node at k, picked to have a
    # child in Gh, has at most one backedge. We repeat this procedure until
    # |Gl| == 0.
    #
    # 1.6 BACKEDGE LIMITS
    if max_backedges is not None:
        mask = np.tril(np.ones(max_spls, max_spls), k=-1)
        ((Conn * mask).sum(axis=1) <= 1).constrain(prob, "LinearizationL")
        # this is redundant with the former, but let's baby the solver
        ((Conn * mask.T).sum(axis=1) >= 1).constrain(prob, "LinearizationU")

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    (Fs.sum(axis=2) <= Conn).constrain(prob, 'FlowCap')

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

    # ## OBJECTIVE
    # #
    objective = pp.LpAffineExpression()

    # penalize number of intermediate splitters
    if not exact_counts:
        objective += 1000 * S.sumit()

    # search-biasing objective terms
    # XXX ideally these will be subsumed into the layout problem
    if not exact_counts:
        for si in number(Splitters):
            # penalize "backjumps"
            for sj in range(ix):
                objective += (si - sj) * Conn[si, sj]
            # penalize "jumps"
            for sj in range(ix + 1, len(Splitters)):
                objective += (sj - si) * Conn[si, sj]

    prob += objective

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
    parser.add_argument(
        '--maxb',
        type=int,
        default=None,
        help='max number of backedges',
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
        f'bal_{args.ni}-{args.no}'
        f'{"-ge + str(args.mins)" if args.exact else ""}'
        f'{"-mbe + str(args.maxb)" if args.maxb else ""}'
    )
    graphname = str(in_sol_dir(SOL_SUBDIR + graphname))

    print(f'Solving the optimal {args.ni} -> {args.no} balancer...')
    try:
        imap, omap, adjmat, labels = solve_balancers(
            args.ni,
            args.no,
            max_spls=args.maxs,
            min_spls=args.mins,
            max_backedges=args.maxb,
            exact_counts=args.exact,
            solver=args.solver,
        )
    except (Infeasible, IllSpecified) as e:
        print(f'No feasible solution within the given splitter limits:')
        print(str(e))
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
