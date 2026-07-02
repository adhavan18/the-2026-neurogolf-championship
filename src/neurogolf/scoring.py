"""Faithful offline scoring of a candidate ONNX network.

This mirrors ``neurogolf_utils.verify_network`` step for step, but returns a
structured :class:`ScoreResult` instead of printing / drawing.  It *reuses* the
vendored official functions (``sanitize_model``, ``verify_subset``,
``score_network``, ``check_network``) so that the numbers we compute locally are
the same ones Kaggle will compute.

The one wrinkle handled here: ONNX Runtime's profiler writes a trace file into
the current working directory, and ``score_network`` needs that trace to measure
intermediate-tensor memory.  We therefore run each scoring in a throwaway temp
directory and clean it up afterwards.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import onnx

from .data import Task

# Make the vendored official module importable.
_VENDOR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "neurogolf_utils"))
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

_FILESIZE_LIMIT = 1.44 * 1024 * 1024


@dataclass
class ScoreResult:
    """Outcome of scoring one network against one task."""

    correct: bool
    params: Optional[int]
    memory: Optional[int]
    passed: int
    failed: int
    points: float
    error: str = ""

    @property
    def cost(self) -> Optional[int]:
        if self.params is None or self.memory is None:
            return None
        return self.params + self.memory

    @property
    def valid(self) -> bool:
        """True if the network is well-formed, correct, and measurable."""
        return (
            self.correct
            and self.params is not None
            and self.memory is not None
            and self.params >= 0
            and self.memory >= 0
        )


def _points(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


@contextlib.contextmanager
def _in_tmpdir():
    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="neurogolf_score_")
    try:
        os.chdir(tmp)
        yield tmp
    finally:
        os.chdir(prev)
        for name in os.listdir(tmp):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(tmp, name))
        with contextlib.suppress(OSError):
            os.rmdir(tmp)


def score_model(model: onnx.ModelProto, task: Task, task_num: Optional[int] = None) -> ScoreResult:
    """Score ``model`` against ``task``; returns a :class:`ScoreResult`.

    ``correct`` requires exact reproduction across ``train + test + arc-gen``.
    ``points`` is only meaningful when :pyattr:`ScoreResult.valid` is True.
    """
    import onnxruntime  # required dependency
    import neurogolf_utils as nu  # vendored official module

    num = task_num if task_num is not None else task.num
    examples = task.as_examples()

    with _in_tmpdir():
        filename = f"task{num:03d}.onnx"
        onnx.save(model, filename)

        if os.path.getsize(filename) > _FILESIZE_LIMIT:
            return ScoreResult(False, None, None, 0, 0, 0.0, error="filesize exceeds 1.44MB")

        try:
            sanitized = nu.sanitize_model(onnx.load(filename))
            if not sanitized:
                return ScoreResult(False, None, None, 0, 0, 0.0, error="sanitize rejected model")
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = f"{num:03d}"
            session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        except Exception as e:  # noqa: BLE001 - surface any load/build failure
            return ScoreResult(False, None, None, 0, 0, 0.0, error=f"load failure: {e}")

        agi_right, agi_wrong, _ = nu.verify_subset(session, examples["train"] + examples["test"])
        gen_right, gen_wrong, _ = nu.verify_subset(session, examples["arc-gen"])
        trace_path = session.end_profiling()

        memory, params = nu.score_network(sanitized, trace_path)

    passed = agi_right + gen_right
    failed = agi_wrong + gen_wrong
    correct = failed == 0

    if memory is None or params is None or memory < 0 or params < 0:
        return ScoreResult(correct, params, memory, passed, failed, 0.0, error="unmeasurable")

    points = _points(memory + params) if correct else 0.0
    return ScoreResult(correct, params, memory, passed, failed, points)
