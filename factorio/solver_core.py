from __future__ import annotations

import os
import typing as ty
from itertools import product
from pathlib import Path
from typing_extensions import Literal

import numpy as np
import pulp as pp

SOLDIR = Path("./solutions/")


class Infeasible(Exception):
    pass


class IllSpecified(Exception):
    pass


def in_sol_dir(fn: str) -> Path:
    SOLDIR.mkdir(exist_ok=True)
    return SOLDIR.joinpath(fn)


SOLVER_FALLBACK = ["gurobi", "coin", "glpk"]


def get_solver(
    which=SOLVER_FALLBACK[0], threads: int = None, _fallback=None, verbose=True
):
    threads = threads or os.cpu_count() or 1
    if which == "coin":
        solver = pp.solvers.COIN_CMD(msg=int(verbose), threads=threads)
    elif which == "coinmp":
        solver = pp.solvers.COINMP_DLL()
    elif which == "glpk":
        solver = pp.solvers.GLPK()
    elif which == "gurobi":
        solver = pp.solvers.GUROBI(
            threads=threads,
            display_interval=60,
            mipgap=1e-6,
            outputflag=int(verbose),
        )
    else:
        solver = None

    if solver is not None and solver.available():
        return solver

    if _fallback is None:
        _fallback = SOLVER_FALLBACK.copy()
    _fallback.remove(which)
    if not _fallback:
        print("Could not get a solver!")
        return None
    print(
        f"Solver: {which} solver is not available. "
        f"Falling back on {_fallback[0]}"
    )
    return get_solver(which=_fallback[0], threads=threads, _fallback=_fallback)
