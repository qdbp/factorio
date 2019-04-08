# Dicovering the optimal M x N balancers, phase 1: max flow
# Evgeny Naumov, 2019
from __future__ import annotations

import typing as ty
from itertools import product
from tempfile import NamedTemporaryFile

import numpy as np
import pulp as pp


def dicts_to_ndarray(
        dicts,
        index_sets: ty.Tuple[ty.Sized, ...],
        dtype=np.float32,
) -> np.ndarray:

    shape = tuple(len(ixset) for ixset in index_sets)

    def _rworker(plane: np.ndarray, d: ty.Dict, index_sets):
        if len(index_sets) > 1:
            for ex, elem in enumerate(index_sets[0]):
                _rworker(plane[ex, ...], d[elem], index_sets[1:])
        else:
            for ex, elem in enumerate(index_sets[0]):
                plane[ex] = pp.value(d[elem])

    out = np.zeros(shape, dtype=dtype)
    _rworker(out, dicts, index_sets)

    return out


def solve_balancers(
        n_inps: int, n_outs: int, max_splitters: int = 50
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
    Spls = [f's{ix}' for ix in range(max_splitters)]

    Nodes = Inps + Spls + Outs

    # flows
    F = pp.LpVariable.dicts("F", (Nodes, Nodes, Inps))

    # splitters enabled
    S = pp.LpVariable.dicts("S", (Spls, ), 0, 1, pp.LpBinary)

    # capacity matrix
    # c[i, j] = 1 <=> directed capacity of 1 exists from i to j
    # XXX binary makes all splitters have to have two distinct inputs and
    # outputs. This should be fine, but making this {0, 1, 2} integer is more
    # general -- at the cost of input/output/splitter distinction complications
    # in the restrictions.
    C = pp.LpVariable.dicts("C", (Nodes, Nodes), 0, 1, pp.LpBinary)

    # it starts out gently enough...
    prob += pp.lpSum(S[spl] for spl in Spls)

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
    for spl in Spls:
        #  1.3 ENABLED SPLITTER OUPUTS WELL CONNECTED
        prob += pp.lpSum(C[spl][v] for v in Nodes) == 2 * S[spl]
        #  1.4 ENABLED SPLITTER INPUTS WELL CONNECTED
        prob += pp.lpSum(C[u][spl] for u in Nodes) == 2 * S[spl]

    # # CHAPTER 2: GENERIC MAX-FLOW PROBLEM RESTRICTIONS
    # 2.1 RESPECT FLOW CAP
    for u, v in product(Nodes, Nodes):
        prob += pp.lpSum(F[u][v][i] for i in Inps) <= C[u][v]

    # 2.2 INCOMPRESSIBLE PARTIAL FLOWS FOR INNER NODES
    for spl, i in product(Spls, Inps):
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
    for inp, out in product(Inps, Outs):
        prob += n_inps * pp.lpSum(F[v][out][inp] for v in Nodes) == pp.lpSum(
            F[v][out][j] for v in Nodes for j in Inps
        )

    # 3.3 SPLITTERS EQUALIZE ALL TYPES
    for spl in Spls:
        # n_spl_outs = pp.lpSum(C[spl][w] for w in Nodes)
        for inp in Inps:
            inflow = pp.lpSum(F[u][spl][inp] for u in Nodes)
            for dst in Nodes:
                # XXX to keep things linear we must hardcode distinct inputs
                # and distinct outputs UNLESS we have TODO separte indexer sets
                # for 2->1 and 1->2 wired splitters
                # afaict this is the only way to encode the splitter condition
                # (together with flow preservation and capacity respect)
                prob += 2 * F[spl][dst][i] <= inflow

    with NamedTemporaryFile(suffix='.lp') as f:
        fn = f.name
        prob.writeLP(fn)
        prob.solve()

    if 'Infeasible' in pp.LpStatus[prob.status]:
        print('INFEASIBLE')
        return None

    print('flows:')
    for inp in Inps:
        print(f'flow for {inp}')
        print(dicts_to_ndarray(F, (Nodes, Nodes, [inp])).squeeze())

    print('done')
    return dicts_to_ndarray(C, (Nodes, Nodes))


if __name__ == '__main__':

    # should give the trivial one-splitter solution, fingers crossed!
    adj_matrix = solve_balancers(2, 2, 1)
    print(adj_matrix)
