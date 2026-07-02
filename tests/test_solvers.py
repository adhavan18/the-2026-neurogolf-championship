"""Integration tests for solvers + faithful scorer on known tasks.

These require the competition JSON in ./data (see README); they skip cleanly if
the data isn't present so the pure builder/data tests still run everywhere.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurogolf.data import DATA_DIR, load_task  # noqa: E402
from neurogolf.pipeline import solve_task  # noqa: E402


def _has(num: int) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, f"task{num:03d}.json"))


def test_colormap_task016_gather():
    if not _has(16):
        return  # data not present; skip
    result, model = solve_task(load_task(16))
    assert result.solved and model is not None
    assert result.solver == "colormap"
    # Gather solution: 10 params, zero memory.
    assert result.cost == 10
    assert result.points > 22.0


def test_linear_conv_task053():
    if not _has(53):
        return  # data not present; skip
    result, model = solve_task(load_task(53))
    assert result.solved and model is not None
    assert result.solver == "linear_conv"
    assert result.memory == 0  # single conv node -> no intermediate tensors
    assert result.points > 15.0
