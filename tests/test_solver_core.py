import pulp as pp
import solver_core as core
import numpy.random as npr


def test_logical_clip(self):

    prob = pp.prob('logical_clip', pp.LpMinimize)

    X = core.lparray.create_anon('arr', (5, 5), 0, 5, pp.LpInteger)

    (X.sum(axis=1) <= 15).constrain(prob, 'rowsum')
    (X.sum(axis=0) <= 15).constrain(prob, 'rowsum')

    lclip = X.logical_clip(prob)

    bern = npr.binomial(3, size=(5, 5))
    prob += X @ bern

    prob.solve()

    assert lclip.values.max() == 1
    assert lclip.values.max() == 0
