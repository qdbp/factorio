from __future__ import annotations

import heapq as heap
import typing as ty
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from itertools import combinations
from random import choice, sample

import networkx as nx
import numpy as np
import numpy.linalg as nlg
import numpy.random as npr
import pygraphviz as pgv
import tqdm
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from numpy.linalg import matrix_power as mpow

np.set_printoptions(precision=2, floatmode="fixed", suppress=True)


def investigate(mat, n_inps, n_outs):

    assert mat.shape[0] == mat.shape[1]
    assert n_inps + n_outs <= mat.shape[0]

    pw = mat
    for i in range(10):
        pw = mat @ pw
        print(pw.sum(axis=0))

    a_inp = np.zeros(mat.shape[0])
    a_inp[0:n_inps] = 1


def extend_splvec(
    splvecs: np.ndarray, input_map: ty.List[int], n_outputs: int
) -> np.ndarray:

    n_inps = len(input_map)

    final_mat = np.zeros(
        (n_inps + splvecs.shape[0] + n_outputs, n_inps + splvecs.shape[1])
    )

    final_mat[-n_outputs:, -n_outputs:] = np.eye(n_outputs)
    final_mat[n_inps:-n_outputs, n_inps:] = splvecs
    for inx, inp in enumerate(input_map):
        if inp >= splvecs.shape[0]:
            print(f"Mapping input {inx} straight to an output!")
        inp += n_inps
        final_mat[inx, inp] = 1

    return final_mat


class Splitter:
    __slots__ = ("i1", "i2", "o1", "o2")
    i1: int
    i2: ty.Optional[int]
    o1: int
    o2: int

    def __init__(self, *, i1: int, i2: ty.Optional[int], o1: int, o2: int):
        self.i1 = i1
        self.i2 = i2
        self.o1 = o1
        self.o2 = o2

        assert o2 > o1

    # def __post_init__(self):
    #     assert 1 <= len(self.i) <= 2
    #     assert 1 <= len(self.o) <= 2
    #     assert not self.i & self.o

    def __hash__(self):
        return (
            1_000_000 * self.i1
            + 10_000 * (self.i2 or 0)
            + 100 * self.o1
            + self.o2
        )

    def __eq__(self, other):
        return hash(self) == hash(other)


class SplState:
    _Ms: ty.Dict[ty.Tuple[int, int], ty.Tuple[np.ndarray, np.ndarray]] = {}
    __slots__ = ("n_points", "free_outputs", "spls")

    def __init__(
        self,
        *,
        n_points: int,
        free_outputs: ty.FrozenSet[int],
        spls: ty.FrozenSet[Splitter],
    ):
        self.n_points = n_points
        self.free_outputs = free_outputs
        self.spls = spls

    @classmethod
    def genesis(cls, m: int) -> SplState:
        return cls(
            n_points=m, free_outputs=frozenset(range(m)), spls=frozenset()
        )

    def _move_add_xi2o_spl(self, i1: int, i2: int = None) -> SplState:

        o1 = self.n_points
        o2 = self.n_points + 1

        assert i1 in self.free_outputs

        if i2 is not None:
            assert i2 in self.free_outputs

        new_spl = Splitter(i1=i1, i2=i2, o1=o1, o2=o2)
        new_free_outputs = self.free_outputs - {i1, i2}  # type: ignore

        return SplState(
            n_points=self.n_points + 2,
            free_outputs=new_free_outputs | {o1, o2},
            spls=self.spls | {new_spl},
        )

    def _move_add_backflow(
        self, i_spl: Splitter, o_spl: Splitter, bind: int
    ) -> SplState:

        assert i_spl in self.spls
        assert o_spl in self.spls
        assert i_spl.i2 is None

        bind_output = o_spl.o1 if bind == 0 else o_spl.o2
        assert bind_output in self.free_outputs

        new_free_outputs = self.free_outputs - {bind_output}

        new_i_spl = Splitter(
            i1=i_spl.i1, i2=bind_output, o1=i_spl.o1, o2=i_spl.o2
        )

        new_spls = self.spls - {i_spl}
        new_spls = new_spls | {new_i_spl}

        assert len(new_spls) == len(self.spls)

        return SplState(
            n_points=self.n_points, free_outputs=new_free_outputs, spls=new_spls
        )

    def spawn(self) -> ty.Generator[SplState, None, None]:
        """
        Generates all legal child states of this splitter.
        """

        # make only the smallest output of a splitter legal for
        # cross-connections, sucking out some symmetry
        illegal_free_outputs = set()
        for spl in self.spls:
            if spl.o1 in self.free_outputs:
                illegal_free_outputs.add(spl.o2)

        legal_free_outputs = self.free_outputs - illegal_free_outputs

        for i in legal_free_outputs:
            yield self._move_add_xi2o_spl(i)

        for i, j in combinations(legal_free_outputs, 2):
            # j > i  && o2 > o1 is an invariant
            if any(i == spl.o1 and j == spl.o2 for spl in self.spls):
                continue
            yield self._move_add_xi2o_spl(i, j)

        for i_spl in self.spls:
            if i_spl.i2 is not None:
                continue
            for o_spl in self.spls:
                if o_spl == i_spl:
                    continue
                if o_spl.o1 in legal_free_outputs:
                    yield self._move_add_backflow(i_spl, o_spl, 0)
                elif o_spl.o2 in legal_free_outputs:
                    yield self._move_add_backflow(i_spl, o_spl, 1)

    def _grab_M(self, n_inputs):
        try:
            return self._Ms[(n_inputs, self.n_points)]
        except KeyError:
            M = np.zeros((n_inputs, self.n_points, self.n_points))
            b = np.zeros((n_inputs, self.n_points))

            M = np.zeros((n_inputs, self.n_points, self.n_points))
            b = np.zeros((n_inputs, self.n_points))
            M[:, :n_inputs, :n_inputs] = np.eye(n_inputs)
            np.fill_diagonal(b, 1)

            self._Ms[(n_inputs, self.n_points)] = (M, b)

        return M, b

    def flow(self, n_inputs):
        """
        Assumes the first n input numbers are the balancer inputs, as enforced
        by `genesis`.
        """

        # solve for all input types in parallel -- this is the first dimension
        # the last two dimensions are a regular linear system for each flow!

        # every splitter adds two free outputs, so we should always
        # have a perfectly well-determined system! We assert this
        assert self.n_points - n_inputs == 2 * len(self.spls)

        M, b = self._grab_M(n_inputs)
        M[:, n_inputs:, :] = 0

        # M = np.zeros((n_inputs, self.n_points, self.n_points))
        # b = np.zeros((n_inputs, self.n_points))
        # M[:, :n_inputs, :n_inputs] = np.eye(n_inputs)
        # np.fill_diagonal(b, 1)

        # both outputs are assumed flowing - if they're in the free set
        # they flow as external output belts
        # f_o1 - 0.5 (f_i1 + f_i2) == 0
        # f_o2 - 0.5 (f_i1 + f_i2) == 0
        for sx, spl in enumerate(self.spls):
            assert not {spl.o1, spl.o2} & {spl.i1, spl.i2}
            base = n_inputs + 2 * sx

            M[:, base, spl.o1] = 1
            M[:, base + 1, spl.o2] = 1

            M[:, base : base + 2, spl.i1] = -1 / 2
            if spl.i2 is not None:
                M[:, base : base + 2, spl.i2] = -1 / 2

        # print(M[0])

        #         print((2 * M[0]).astype(np.int32))
        #         print(b[0].T)
        #
        return nlg.solve(M, b)

    def loss(self, flow, n_inputs, n_outputs):
        if len(self.free_outputs) < n_outputs:
            return n_inputs * n_inputs * 2

        target = np.ones((n_inputs, n_outputs)) / n_outputs
        matched = 1 - np.isclose(flow[:, -n_outputs:], target)
        return matched.sum()

    def heurloss(self, out_target):
        return 20 * len(self.spls) + 5 * np.abs(
            len(self.free_outputs) - out_target
        )

    @property
    def graph(self):
        inputs = {
            i
            for i in range(self.n_points)
            if all(i != spl.o1 and i != spl.o2 for spl in self.spls)
        }

        G = nx.DiGraph()
        to_contract = []
        for i in range(self.n_points):
            if i in self.free_outputs & inputs:
                G.add_node(i, color="purple")
            elif i in self.free_outputs:
                G.add_node(i, color="red")
            elif i in inputs:
                G.add_node(i, color="blue")
            else:
                G.add_node(i, color="black", node_size=1)

        for sx, spl in enumerate(self.spls):
            snode = f"s{sx}"
            G.add_node(snode, color="black", fillcolor="black", shape="square")

            G.add_edge(spl.i1, snode)
            if spl.i2 is not None:
                G.add_edge(spl.i2, snode)

            if spl.i1 not in inputs:
                to_contract.append((spl.i1, snode))
            if spl.i2 is not None and spl.i2 not in inputs:
                to_contract.append((spl.i2, snode))

            G.add_edge(snode, spl.o1)
            G.add_edge(snode, spl.o2)

        for u, v in to_contract:
            G = nx.contracted_nodes(G, v, u, self_loops=False)

        return G

    def draw_graph(self, name="splittertest"):
        G = self.graph
        A = to_agraph(G)
        A.layout("dot")
        A.draw(f"{name}.png")

    def __lt__(self, other):
        # random sorting
        return hash(self) < hash(other)

    def __hash__(self):
        return hash((self.n_points, self.free_outputs, self.spls))

    def __eq__(self, other):
        return hash(self) == hash(other)


def iterative_search(genesis, n_inputs, n_outputs):

    queue = [(0, genesis)]
    seen: ty.Set[SplState] = set()
    best_loss = 1e15

    p = tqdm.tqdm()

    while True:
        p.update()
        rank, cur = heap.heappop(queue)
        seen.add(cur)
        flow = cur.flow(n_inputs)
        loss = cur.loss(flow, n_inputs, n_outputs)
        if loss < best_loss:
            print(loss)
        best_loss = min(loss, best_loss)

        # we found the one!
        if loss == 0:
            print(f"Found result after searching {len(seen)} candidates")
            p.close()
            return cur
        for child in cur.spawn():
            if child in seen:
                continue
            rank = loss + child.heurloss(n_outputs)
            heap.heappush(queue, (rank, child))


def main():
    from argparse import ArgumentParser
    import sys

    parser = ArgumentParser()
    parser.add_argument("ni", type=int, help="n_inputs")
    parser.add_argument("no", type=int, help="n_outputs")

    args = parser.parse_args(sys.argv[1:])

    out = iterative_search(SplState.genesis(args.ni), args.ni, args.no)
    out.draw_graph(name=f"iterative_{args.ni}-{args.no}")
    print(out)
    flow = out.flow(args.ni)
    print(flow)
    print(out.loss(flow, args.ni, args.no))


if __name__ == "__main__":
    main()
