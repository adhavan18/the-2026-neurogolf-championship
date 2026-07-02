"""NeuroGolf 2026 — tools for synthesizing minimal ONNX networks for ARC-AGI tasks.

The competition asks for one ONNX network per ARC-AGI task that reproduces the
task's grid transformation *exactly* on every example, while minimising
``cost = parameters + memory_footprint_bytes``.  Each task scores
``max(1, 25 - ln(cost))`` (a zero-cost network scores a full 25).

Sub-modules:
  * :mod:`neurogolf.data`      — load tasks; encode/decode grids <-> tensors.
  * :mod:`neurogolf.builders`  — construct small ONNX graphs (identity, gather,
                                 conv, ...).
  * :mod:`neurogolf.scoring`   — faithful offline scoring via the vendored
                                 official ``neurogolf_utils``.
  * :mod:`neurogolf.solvers`   — pluggable per-task solvers.
  * :mod:`neurogolf.pipeline`  — run solvers over the task set and package a
                                 submission.
"""

from .data import CHANNELS, GRID, HEIGHT, WIDTH, Task, iter_tasks, load_task

__all__ = [
    "CHANNELS",
    "GRID",
    "HEIGHT",
    "WIDTH",
    "Task",
    "iter_tasks",
    "load_task",
]
