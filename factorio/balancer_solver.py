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
from scipy.linalg import toeplitz

from pulp_lparray import lparray, number

from .solver_core import IllSpecified, Infeasible, get_solver, in_sol_dir

SOL_SUBDIR = "./balancers/"


def lowerbound_splitters(M: int, N: int) -> int:
    """
    Guesses a lower bound on the number of splitters needed for an M-N balancer
    """

    # special case
    if M == 1 and N == 1:
        return 0

    H = ceil(np.log2(N))
    W = ceil(M / 2)

    def min_by_height(h: int):
        return max(W, ceil(N / (2 ** (H - h))))

    return sum(min_by_height(h) for h in range(H))


def solve_balancers(
    *,
    M: int,
    N: int,
    max_spls: int,
    min_spls: int,
    solver: pp.LpSolver,
    debug=False,
    exact_counts=False,
) -> np.ndarray:
    """
    Attempt to find a splitter-count optimal M -> N balancer for Factorio.

    More efficient than the naive expanded matrix approach, using "implicit"
    input and output flows with inherent restrictions for the splitter problem.

    Further optimized by restricting the search to N >= M and allowing only
    22 and 12 splitters.
    """

    if M > N:
        raise IllSpecified(
            "The problem formulation only allows for fanout designs. "
            "Note that an optimal N -> M solution can be reversed to give "
            "what you want."
        )

    if N == 1:
        return (
            np.zeros((1, 0)),
            np.zeros((0, 1)),
            np.zeros((0, 0)),
            np.zeros(0),
            np.zeros(0),
            list(),
            True,
        )

    min_spls = max(lowerbound_splitters(M, N), min_spls)
    if max_spls < min_spls:
        raise Infeasible(
            f"Balancing {M} -> {N} requires at least {min_spls}, which is "
            "lower than the max given"
        )

    # special case needed for fixed Omap.
    if N == 2:
        max_spls = 1

    # number of pinned input splitters, disjoint from output splitters
    nsitot = nsi0 = ceil(M / 2)

    # number of pinned output splitters, disjoint from input splitters
    # if N > 2, outputs can't be connected straight to an input splitter.
    # This is the first separation theorem (see below)
    nsotot = nso0 = ceil(N / 2) if N > 2 else 0

    # we must have at least the input and output pinned splitters
    # this can be better than the previous lower bound for some M, N
    min_spls = max(min_spls, nsi0 + nso0)

    if not exact_counts:
        print(f"Solver/Info: n_splitters in [{min_spls}, {max_spls}].")
    else:
        print(f"Solver/Info: solving with exactly {max_spls} splitters")
    print(f"Solver/Info: {nsi0} input pinned; {nso0} output pinned")

    Inps = [f"i{ix}" for ix in range(M)]
    Outs = [f"o{ix}" for ix in range(N)]
    Splitters = [f"s{ix}" for ix in range(max_spls)]

    # splitters enabled
    if not exact_counts:
        S = lparray.create("S", (Splitters,), 0, 1, pp.LpBinary)
    else:
        S = np.ones(len(Splitters), dtype=np.uint8)

    # capacity matrix
    # Conn[i, j] = 1 <=> directed capacity of 1 exists from i to j
    Conn = lparray.create("C", (Splitters, Splitters), 0, 1, pp.LpBinary)
    # internal flows
    # Fs = pp.LpVariable.dicts("Fs", (Splitters, Splitters, Inps))
    Fs = lparray.create(
        "Fs", (Splitters, Splitters, Inps), 0.0, 1.0, pp.LpContinuous
    )

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
    Fi = lparray.create(
        "Fi", (Inps, Splitters, Inps), 0.0, 1.0, pp.LpContinuous
    )
    # output flows
    Fo = lparray.create(
        "Fo", (Splitters, Outs, Inps), 0.0, 1.0, pp.LpContinuous
    )

    H = ceil(np.log2(N))
    # TODO H + 2 is arbitrary, find a proper bound
    MAX_RANK = H + 2

    Ro = lparray.create("Ro", (Splitters,), 0, MAX_RANK, pp.LpInteger)
    Ri = lparray.create("Ri", (Splitters,), 0, MAX_RANK, pp.LpInteger)

    prob = pp.LpProblem(name="solve_balancer", sense=pp.LpMinimize)

    # Ro gives the "out rank" (orank) of a node. The out rank is defined as the
    # shortest distance from the node to one of the outputs, measured in
    # intermediate nodes.
    # The edge nodes have rank 0 by definition
    # NOTE we do not constrain other nodes to allow non-existing splitters
    # to take rank 0, which is required (I strongly suspect based on testing)
    # for correct feasibility.
    (Ro[-nso0:] == 0).constrain(prob, "ORankEdge")
    (Ri[:nsi0] == 0).constrain(prob, "IRankEdge")

    # The orank of a node is 1 + the min orank of its descendants
    # We need to exclude the impredicative case of Rx[u] = 1 + min( ..., Rx[u]
    # + xxx, ...), or there is no feasible region
    # NOTE as much as I love to be clever, this is much clearer than freaky
    # array tricks.

    for u in number(Splitters)[:-nso0]:
        ixarr = np.ones(max_spls, dtype=bool)
        ixarr[u] = False

        # 0 <= dest cost <= 3 * MAX_RANK
        dest_cost = (Ro + MAX_RANK * (2 - Conn[u] - S))[ixarr]
        min_cost = dest_cost.lp_int_min(
            prob,
            f"ORankMin_{u}",
            lb=0,
            # from the max on dest_cost,
            # 0 <= min_cost <= 3 * MAX_RANK
            ub=3 * MAX_RANK,
        )
        # NOTE the rank of disabled splitters is unconstrained
        (Ro[u : u + 1] >= 1 + min_cost - 4 * MAX_RANK * (1 - S[u])).constrain(
            prob, f"ORankDef_{u}a"
        )
        (Ro[u : u + 1] <= 1 + min_cost + 4 * MAX_RANK * (1 - S[u])).constrain(
            prob, f"ORankDef_{u}b"
        )

    for u in number(Splitters)[nsi0:]:
        ixarr = np.ones(max_spls, dtype=bool)
        ixarr[u] = False

        source_cost = (Ri + MAX_RANK * (2 - Conn[:, u] - S))[ixarr]
        min_cost = source_cost.lp_int_min(
            prob, f"IRankMin_{u}", lb=0, ub=3 * MAX_RANK
        )
        (Ri[u : u + 1] >= 1 + min_cost - 4 * MAX_RANK * (1 - S[u])).constrain(
            prob, f"IRankDef_{u}a"
        )
        (Ri[u : u + 1] <= 1 + min_cost + 4 * MAX_RANK * (1 - S[u])).constrain(
            prob, f"IRankDef_{u}b"
        )

    # ## CONSTRAINTS
    # # CHAPTER 0: CONNECTIVITY RESTRICTIONS
    # 0.0: MINIMUM NUMBER OF SPLITTERS
    # each input must have at least ceil(lg(N)) between it and any output
    # this creates a trivial lower bound of (M // 2)ceil(lg(N)) splitters
    if not exact_counts:
        (S.sum() >= min_spls).constrain(prob, "MinSplitters")

    # 0.1: SPLITTER SEPARATION THEOREM
    # Theorem 3.
    # If N > 2 ** k, a chain like the following cannot exist:
    # i -> s1 -> ... -> sk -> o
    # Proof.
    # The fraction of item i on output o must be 1 / N < 1 / (2**k).
    # But since each splitter at most halves the output, the outflow of sk
    # of type i must be at least 1 / 2**k. Therefore such a subgraph cannot
    # exist.
    # 0.1.0 SEPARATION THEOREM
    (Ro + Ri >= H - 1).constrain(prob, "SepTheorem")

    # We can constrain the problem by forcing the output ranks to be ordered.
    # NOTE I don't believe we can always force both iranks and oranks to be
    # ordered simultaneously...
    # 0.2 ORANK ORDERING OF FREE SPLITTERS
    (Ro[nsi0 + 1 :] <= Ro[nsi0:-1]).constrain(prob, "RankOrdering")

    # 0.3: RESPECT PINS
    if not exact_counts:
        (S[:nsitot] == 1).constrain(prob, "InPins")
        if nsotot > 0:
            (S[-nsotot:] == 1).constrain(prob, "OutPins")

        # 0.4 UNPINNED SPLITTERS ARE ACTIVATED IN ORDER
        Sl = S[nsitot : max_spls - nsotot - 1]
        Su = S[nsitot + 1 : max_spls - nsotot]
        (Sl >= Su).constrain(prob, "UnpinnedSplitterOrder")

        print(f"Solver/Info: {len(Sl) + 1} unpinned splitters.")

        free_locked = max(0, min_spls - nsitot - nsotot)
        if free_locked > 0:
            print(f"Solver/Info: {free_locked} unpinned splitters are locked")
            (Sl[:free_locked] == 1).constrain(prob, "FreeLocked")

    out_from_spls = Omap.sum(axis=1)
    inp_into_spls = Imap.sum(axis=0)

    # 0.5 ENABLED SPLITTER OUPUTS WELL CONNECTED
    (Conn.sum(axis=1) + out_from_spls <= 2 * S).constrain(prob, "MaxOuts")
    (Conn.sum(axis=1) + out_from_spls >= S).constrain(prob, "MinOuts")

    # 0.6 ENABLED SPLITTER INPUTS WELL CONNECTED
    (Conn.sum(axis=0) + inp_into_spls <= 2 * S).constrain(prob, "MaxIns")
    (Conn.sum(axis=0) + inp_into_spls >= S).constrain(prob, "MinIns")

    # 0.7 NO SELF LOOPS
    (np.diag(Conn) == 0).constrain(prob, "NoSelfLoops")

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
    # NOTE empirically, it appears the presolversolver infers this from
    # the output rank ordering condition
    # XXX HEURISTIC
    # ((Conn * np.tril(np.ones((max_spls, max_spls)), -1)).sum() <= 3).constrain(
    #     prob, "MAXBACKEDGES HEURISTIC"
    # )
    # (Ro <= H - 1).constrain(prob, "MAXORANK HEURISTIC")

    # # CHAPTER 1: FLOW CONSTRAINTS
    # 1.0 RESPECT FLOW CAP
    (Fs.sum(axis=2) <= Conn).constrain(prob, "FlowCap")

    # 1.1 INFLOW EDGE CONDITIONS
    # forall i,v,t: F[i, v, t] == Imap[i, v] * 1[i == t]
    (Fi == Imap[:, :, None] * np.eye(M)[:, None, :]).constrain(prob, "InEdge")

    # 1.2 OUTFLOW EDGE CONDITIONS
    # forall v,o,t: Fo[v, o, t] == Omap[v, o] / N
    (N * Fo == Omap[..., None]).constrain(prob, "OutEdge")

    # 1.3 PROPER FLOW
    (Fs >= 0).constrain(prob, "ProperFlowS")
    (Fi >= 0).constrain(prob, "ProperFlowI")
    (Fo >= 0).constrain(prob, "ProperFlowO")

    # 1.4 INCOMPRESSIBILITY
    inflows = Fs.sum(axis=0) + Fi.sum(axis=0)  # (spls, types)
    outflows = Fs.sum(axis=1) + Fo.sum(axis=1)  # (spls, types)
    (inflows == outflows).constrain(prob, "Incompressibility")

    # 1.5 EQUAL SPLITTING
    # forall s,w,t: 2 * Fs[s, w, t] <= inflow[s, t]
    (2 * Fs <= inflows[:, None, :]).constrain(prob, "SplittingS")
    # forall s,o,t: 2 * Fo[s, o, t] <= inflow[s, t]
    (2 * Fo <= inflows[:, None, :]).constrain(prob, "SplittingO")

    # 1.6 FLOW THINNING
    # ∀ t, w:
    # Ro[u] <= k => Fs[u, w, t] <= (2**k / N)
    # By the same argument as Theorem 3.
    def impose_flow_thinning(rank):
        # xp >= 1 iff Ro <= rank
        limit = 2 ** rank
        xp, _ = (-Ro + rank + 1).abs_decompose(
            prob, f"FlowThin{rank}Abs", 0, MAX_RANK, pp.LpInteger
        )
        ro_le_rank = xp.logical_clip(prob, f"FlowThin{rank}Lclip")
        (
            N * Fs[:, :, :]
            <= limit + (N - limit) * (1 - ro_le_rank)[:, None, None]
        ).constrain(prob, f"FlowThin{rank}")

    for orank in range(1, H):
        impose_flow_thinning(orank)

    # 1.6 FLOW COVERING THEOREMS
    # 1.6.0 Ri <= 0 FLOW COVERING
    # FsMax = Fs.lp_real_max(prob, "FsMax", axis=(1, 2), lb=0.0, ub=1.0, bigM=1)

    # TODO debug this
    # def impose_flow_covering(rank):
    #     minmax = 2 ** (rank + 1)
    #     # xp >= 1 <=> Ri <= rank
    #     xp, _ = (-Ri + rank + 1).abs_decompose(
    #         prob, f"FsMaxRankMask{rank}Abs", 0, MAX_RANK, pp.LpInteger
    #     )
    #     ri_le_rank = xp.logical_clip(prob, f"FsMaxRankMask{rank}Lclip")
    #     (minmax * FsMax >= ri_le_rank).constrain(prob, f"FsMax{rank}")

    # if M > 2:
    #     impose_flow_covering(0)
    # if M > 4:
    # impose_flow_covering(1)
    # if M > 8:
    # impose_flow_covering(2)

    # ## OBJECTIVE
    # penalize number of intermediate splitters
    objective = pp.LpAffineExpression()
    if not exact_counts:
        pass
        # objective += S.sumit()
    # NOTE it was found that various "search biasing" objectives seriously slow
    # down the time to a splitter-optimal solution
    # XXX this may not be the case with exact_counts
    # else:
    # objective += (Conn * np.tril(toeplitz(np.arange(max_spls)))).sumit()
    # objective += (Conn * toeplitz(np.arange(max_spls))).sumit()
    # objective += (Conn * toeplitz(np.arange(max_spls))).sumit()
    prob += objective

    # ## SOLVING
    # #
    prob.solve(solver)
    if "Infeasible" == pp.LpStatus[prob.status]:
        raise Infeasible
    if "Not Solved" == pp.LpStatus[prob.status]:
        raise KeyboardInterrupt

    optimal = "Optimal" == pp.LpStatus[prob.status]

    if not exact_counts:
        keep_rows = np.where(S.values > 0)[0]
    else:
        keep_rows = np.arange(max_spls, dtype=np.int32)

    adjmat = Conn.values[np.ix_(keep_rows, keep_rows)]
    labels = [*map(lambda x: f"s{x}", range(len(adjmat)))]

    orank = Ro.values[keep_rows].astype(np.uint8)
    irank = Ri.values[keep_rows].astype(np.uint8)

    return (
        Imap[:, keep_rows],
        Omap[keep_rows],
        adjmat,
        irank,
        orank,
        labels,
        optimal,
    )


def draw_solution(
    imap: np.ndarray,
    omap: np.ndarray,
    splmat: np.ndarray,
    labels=None,
    graphname="graph.png",
    orank=None,
    irank=None,
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
    full_adjmat = np.zeros((n_inps + n_spls + n_outs,) * 2, dtype=np.uint8)
    full_adjmat[:n_inps, n_inps:-n_outs] = imap
    full_adjmat[n_inps:-n_outs, n_inps:-n_outs] = splmat
    full_adjmat[n_inps:-n_outs, -n_outs:] = omap

    g = nx.convert_matrix.from_numpy_array(full_adjmat, create_using=nx.DiGraph)

    orank_labels = (
        [f":o{r}" for r in orank] if orank is not None else [""] * n_spls
    )
    irank_labels = (
        [f"i{r}:" for r in irank] if irank is not None else [""] * n_spls
    )

    if labels is not None:
        g = nx.relabel_nodes(
            g,
            {
                ix + n_inps: irank_labels[ix] + label + orank_labels[ix]
                for ix, label in enumerate(labels)
            },
        )

    values = ["red"] * n_inps + ["black"] * n_spls + ["blue"] * n_outs

    for ndx, (_, attrs) in enumerate(g.nodes(data=True)):
        attrs["fillcolor"] = attrs["color"] = values[ndx]

    A = to_agraph(g)
    A.layout("dot")
    A.draw(graphname + ".png")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("M", type=int, help="number of inputs")
    parser.add_argument("N", type=int, help="number of outputs")
    parser.add_argument(
        "--maxs",
        type=int,
        default=None,
        help="max number of splitters to consider",
    )
    parser.add_argument(
        "--mins",
        type=int,
        default=0,
        help="minimum number of splitters to consider",
    )
    parser.add_argument(
        "--maxb", type=int, default=None, help="max number of backedges"
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="splitter counts are treated as exact instead of as upper bounds",
    )
    parser.add_argument(
        "--solver", type=str, default="gurobi", help="specify the solver"
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="draw a reversed graph of the solution",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress logging by the invoked solver",
    )

    args = parser.parse_args()
    do_iterate = False

    print(f"Solving the optimal {args.M} -> {args.N} balancer...")
    if args.maxs is None:
        print("No max given: iterating from minimum")
        args.maxs = max(args.mins, lowerbound_splitters(args.M, args.N))
        args.mins = args.maxs
        do_iterate = True

    solver = get_solver(which=args.solver, verbose=not args.quiet)

    while True:
        try:
            (
                imap,
                omap,
                adjmat,
                irank,
                orank,
                labels,
                optimal,
            ) = solve_balancers(
                M=args.M,
                N=args.N,
                max_spls=args.maxs,
                min_spls=args.mins,
                solver=solver,
                exact_counts=args.exact or do_iterate,
            )
        except (Infeasible, IllSpecified) as e:
            print(f"No feasible solution within the given splitter limits:")
            print(str(e))
            if not do_iterate:
                return
            else:
                args.maxs += 1
                args.mins += 1
                print(f"Trying again with exactly {args.maxs} splitters")
                continue
        break

    graphname = (
        f"bal_{args.M}-{args.N}"
        f'{("-ge" + str(args.maxs)) if args.exact else ""}'
        f'{("-mbe" + str(args.maxb)) if args.maxb else ""}'
        f'{"-subopt" if not optimal else ""}'
    )
    graphname = str(in_sol_dir(SOL_SUBDIR + graphname))

    if optimal:
        print(f"Solver: found optimal solution with {len(adjmat)} splitters")
    else:
        print(f"Solver: found SUBOPTIMAL solution with {len(adjmat)} splitters")

    draw_solution(
        imap,
        omap,
        adjmat,
        labels=labels,
        graphname=graphname,
        orank=orank,
        irank=irank,
    )
    print(f"solution saved to {graphname}.png")

    np.save(f"{graphname}.npy", adjmat)
    print(f"adjmat saved to {graphname}.npy")


if __name__ == "__main__":
    main()
