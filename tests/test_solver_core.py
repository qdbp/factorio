import numpy as np
import numpy.random as npr
import pulp as pp

import factorio.solver_core as core


def test_logical_clip():

    prob = pp.LpProblem('logical_clip', pp.LpMinimize)
    X = core.lparray.create_anon('arr', (5, 5), 0, 5, pp.LpInteger)
    (X.sum(axis=1) >= 5).constrain(prob, 'colsum')
    (X.sum(axis=0) >= 5).constrain(prob, 'rowsum')

    lclip = X.logical_clip(prob, 'lclip')

    bern = npr.binomial(3, .5, size=(5, 5))

    prob += (X * bern).sumit()
    prob.solve()

    assert X.values.max() > 1
    assert lclip.values.max() <= 1
    assert lclip.values.max() >= 0


def test_int_max():
    '''
    "The Rook Problem", with maxes.
    '''

    prob = pp.LpProblem('int_max', pp.LpMaximize)
    X = core.lparray.create_anon('arr', (8, 8), 0, 1, pp.LpBinary)
    (X.sum(axis=1) == 1).constrain(prob, 'colsum')
    (X.sum(axis=0) == 1).constrain(prob, 'rowsum')

    colmax = X.lp_bin_max(prob, 'colmax', axis=0)
    rowmax = X.lp_bin_max(prob, 'rowmax', axis=1)

    prob += colmax.sumit() + rowmax.sumit()

    prob.solve()
    assert prob.objective == 16


def test_abs():

    N = 20

    prob = pp.LpProblem('wavechaser', pp.LpMaximize)
    X = core.lparray.create_anon('arr', (N, ), -1, 1, pp.LpInteger)
    wave = 2 * npr.binomial(1, 0.5, size=(N, )) - 1

    xp, xm = X.abs(prob, 'abs')
    xabs = xp + xm

    prob += (wave * X).sumit()
    prob.solve()

    print(prob)

    print(X.values)

    assert prob.objective == N
    assert xabs.values.sum() == N
