"""Run the solvers over a set of tasks and assemble a submission.

For each task we collect every candidate the registered solvers propose, score
each with the *official* scorer, and keep the cheapest one that is genuinely
correct.  Correct networks are written as ``taskNNN.onnx`` into the output
directory; a ``manifest.json`` records the per-task result, and the whole
directory is zipped into ``submission.zip``.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import onnx

from .data import Task, iter_tasks
from .scoring import ScoreResult, score_model
from .solvers import REGISTRY, Candidate


@dataclass
class TaskResult:
    task: int
    solved: bool
    solver: str = ""
    detail: str = ""
    params: Optional[int] = None
    memory: Optional[int] = None
    cost: Optional[int] = None
    points: float = 0.0
    n_candidates: int = 0


def _candidates(task: Task) -> List[Candidate]:
    cands: List[Candidate] = []
    for solver in REGISTRY:
        try:
            cands.extend(solver(task))
        except Exception:  # noqa: BLE001 - a broken solver shouldn't kill the run
            continue
    # Cheapest estimate first, so ties resolve toward smaller networks.
    cands.sort(key=lambda c: c.est_cost)
    return cands


def solve_task(task: Task):
    """Return ``(TaskResult, best_model_or_None)`` for a single task."""
    best: Optional[ScoreResult] = None
    best_model: Optional[onnx.ModelProto] = None
    best_cand: Optional[Candidate] = None

    cands = _candidates(task)
    for cand in cands:
        res = score_model(cand.model, task, task.num)
        if not res.valid:
            continue
        if best is None or res.cost < best.cost:
            best, best_model, best_cand = res, cand.model, cand

    if best is None:
        return TaskResult(task.num, solved=False, n_candidates=len(cands)), None

    return (
        TaskResult(
            task=task.num,
            solved=True,
            solver=best_cand.solver,
            detail=best_cand.detail,
            params=best.params,
            memory=best.memory,
            cost=best.cost,
            points=best.points,
            n_candidates=len(cands),
        ),
        best_model,
    )


def run(
    nums: Optional[Sequence[int]] = None,
    out_dir: str = "submission",
    verbose: bool = True,
) -> List[TaskResult]:
    """Solve ``nums`` (or all tasks), write ONNX files + manifest into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    results: List[TaskResult] = []
    t0 = time.time()
    for task in iter_tasks(nums):
        result, model = solve_task(task)
        if model is not None:
            onnx.save(model, os.path.join(out_dir, f"task{task.num:03d}.onnx"))
        results.append(result)
        if verbose and result.solved:
            print(
                f"  task{task.num:03d}: {result.solver:11s} "
                f"cost={result.cost:<6} pts={result.points:6.3f} ({result.detail})"
            )
    if verbose:
        _summary(results, time.time() - t0)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return results


def package(out_dir: str = "submission", zip_path: str = "submission.zip") -> str:
    """Zip every ``taskNNN.onnx`` in ``out_dir`` into ``zip_path``."""
    onnx_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".onnx"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in onnx_files:
            zf.write(os.path.join(out_dir, name), arcname=name)
    return zip_path


def _summary(results: Sequence[TaskResult], elapsed: float) -> None:
    solved = [r for r in results if r.solved]
    total_pts = sum(r.points for r in solved)
    by_solver: dict[str, int] = {}
    for r in solved:
        by_solver[r.solver] = by_solver.get(r.solver, 0) + 1
    print()
    print(f"Solved {len(solved)}/{len(results)} tasks in {elapsed:.1f}s")
    print(f"Total local points: {total_pts:.3f}  (max possible per solved task = 25)")
    if by_solver:
        print("By solver: " + ", ".join(f"{k}={v}" for k, v in sorted(by_solver.items())))
