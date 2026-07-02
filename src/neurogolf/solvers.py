"""Pluggable per-task solvers.

A *solver* takes a :class:`~neurogolf.data.Task` and yields zero or more
:class:`Candidate` networks.  Solvers are deliberately optimistic: they *propose*
networks, and the pipeline (:mod:`neurogolf.pipeline`) is responsible for
verifying each candidate with the official scorer and keeping the cheapest one
that is actually correct.  This keeps solvers simple — they never have to reason
about every scoring edge case, they just have to be right often enough to be
worth trying.

Currently implemented (the reliably-automatable families; see README for why the
rest of ARC-AGI needs bespoke per-task network design):

* :func:`solve_identity`     — ``output == input`` on every pair (cost 0).
* :func:`solve_colormap`     — a global color remap (Gather, cost 10; or 1x1
                               conv, cost 100, when colors merge).
* :func:`solve_linear_conv`  — a single ``k x k`` conv + threshold, fit by a
                               hard-margin perceptron, for genuinely local
                               shape-preserving rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional

import numpy as np
import onnx

from . import builders as B
from .data import CHANNELS, Task

# ----------------------------------------------------------------------------- #
# Candidate + registry
# ----------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """A proposed network plus a human-readable description of how it was built."""

    model: onnx.ModelProto
    solver: str
    detail: str = ""
    # Optimistic cost estimate used only to order candidates before verification.
    est_cost: int = 0


Solver = Callable[[Task], Iterator[Candidate]]
REGISTRY: List[Solver] = []


def register(fn: Solver) -> Solver:
    REGISTRY.append(fn)
    return fn


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #


def _same_shape_pairs(task: Task) -> bool:
    pairs = task.scored_pairs
    return bool(pairs) and all(p.same_shape for p in pairs)


def _derive_colormap(task: Task) -> Optional[Dict[int, int]]:
    """Return a consistent color map ``in_color -> out_color``, or None.

    Requires every scored pair to be the same shape and the per-cell mapping to
    be a well-defined function of the input color alone.
    """
    if not _same_shape_pairs(task):
        return None
    mapping: Dict[int, int] = {}
    for p in task.scored_pairs:
        gi, go = p.input, p.output
        for r in range(len(gi)):
            row_i, row_o = gi[r], go[r]
            for c in range(len(row_i)):
                a, b = row_i[c], row_o[c]
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    return mapping


def _colors_present(task: Task) -> set[int]:
    seen: set[int] = set()
    for p in task.scored_pairs:
        for row in p.input:
            seen.update(row)
    return seen


# ----------------------------------------------------------------------------- #
# Solvers
# ----------------------------------------------------------------------------- #


@register
def solve_identity(task: Task) -> Iterator[Candidate]:
    """Yield an Identity network when the output equals the input everywhere."""
    pairs = task.scored_pairs
    if pairs and all(p.input == p.output for p in pairs):
        yield Candidate(B.make_identity(), "identity", "output == input", est_cost=0)


@register
def solve_colormap(task: Task) -> Iterator[Candidate]:
    """Yield color-map networks (Gather first, 1x1 conv as a robust fallback)."""
    mapping = _derive_colormap(task)
    if mapping is None:
        return
    if all(a == b for a, b in mapping.items()):
        return  # pure identity is handled by solve_identity

    # Gather candidate: source each output channel from one input channel.
    # For output channels that are never produced, source an input channel that
    # is never present (guaranteed all-zero); fall back to identity index.
    inv: Dict[int, int] = {}
    injective = True
    for src, dst in mapping.items():
        if dst in inv and inv[dst] != src:
            injective = False
            break
        inv[dst] = src
    if injective:
        present = _colors_present(task)
        absent = sorted(set(range(CHANNELS)) - present)
        zero_src = absent[0] if absent else None
        idx = list(range(CHANNELS))
        for o in range(CHANNELS):
            if o in inv:
                idx[o] = inv[o]
            elif zero_src is not None:
                idx[o] = zero_src
        yield Candidate(
            B.make_gather_colormap(idx), "colormap", f"gather idx={idx}", est_cost=CHANNELS
        )

    # 1x1 conv candidate: always correct for any color map (handles merges).
    yield Candidate(
        B.make_colormap_conv(mapping),
        "colormap",
        f"1x1 conv map={dict(sorted(mapping.items()))}",
        est_cost=CHANNELS * CHANNELS,
    )


def _fit_perceptron(X: np.ndarray, y: np.ndarray, iters: int = 80):
    """One-vs-rest hard-margin perceptron with integer weights.

    Returns ``(W[C,F], b[C], converged)``.  On convergence the sign scheme
    satisfies exactly the competition's threshold: score>0 for the target
    channel and <=0 for every other channel, at every training cell.
    """
    n, f = X.shape
    W = np.zeros((CHANNELS, f), dtype=np.float64)
    b = np.zeros(CHANNELS, dtype=np.float64)
    Y = np.full((CHANNELS, n), -1.0)
    for o in range(CHANNELS):
        Y[o, y == o] = 1.0
    converged = False
    for _ in range(iters):
        scores = (X @ W.T + b).T  # [C, n]
        pos_wrong = (Y > 0) & (scores <= 0)
        neg_wrong = (Y < 0) & (scores > 0)
        wrong = pos_wrong | neg_wrong
        if not wrong.any():
            converged = True
            break
        for o in range(CHANNELS):
            idx = wrong[o]
            if idx.any():
                W[o] += (Y[o, idx][:, None] * X[idx]).sum(0)
                b[o] += Y[o, idx].sum()
    return W, b, converged


# Sentinel target: this output cell must be all-clear (every channel <= 0).
_CLEAR = -1


def _conv_samples(task: Task, k: int):
    """Build unique (patch, target) samples for a k x k conv over the *full* grid.

    Crucially this includes the border ring *outside* the grid that a k>1 kernel
    can reach: those cells must decode to clear (all channels <= 0).  A single
    all-zero patch is added to force bias <= 0 so that far-away cells stay clear
    too.  Samples are de-duplicated; if one patch demands two different targets
    the rule isn't a function of the k x k neighbourhood and the fit will simply
    fail to converge (which is the correct outcome).
    """
    if not _same_shape_pairs(task):
        return None
    off = k // 2
    xs, ys = [], []
    for p in task.scored_pairs:
        gi, go = p.input, p.output
        h, w = len(gi), len(gi[0])
        # Build the actual 30x30 input tensor ONNX sees (grid top-left, else 0),
        # then pad by `off` so every cell's k x k neighbourhood is well defined.
        full = np.zeros((CHANNELS, 30, 30), dtype=np.float32)
        for r in range(h):
            for c in range(w):
                full[gi[r][c], r, c] = 1.0
        pad = np.zeros((CHANNELS, 30 + 2 * off, 30 + 2 * off), dtype=np.float32)
        pad[:, off:off + 30, off:off + 30] = full
        # Constrain every output cell whose k x k neighbourhood can touch the grid.
        rmax, cmax = min(30, h + off), min(30, w + off)
        for r in range(rmax):
            for c in range(cmax):
                xs.append(pad[:, r:r + k, c:c + k].reshape(-1))
                ys.append(int(go[r][c]) if (r < h and c < w) else _CLEAR)
    if not xs:
        return None
    xs.append(np.zeros(CHANNELS * k * k, dtype=np.float32))  # force bias <= 0
    ys.append(_CLEAR)
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    # De-duplicate (patch, target) rows.
    key = np.concatenate([X, y[:, None].astype(np.float32)], axis=1)
    key = np.unique(key, axis=0)
    return key[:, :-1].copy(), key[:, -1].astype(np.int64)


def _numpy_exact(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> bool:
    """True iff, for every sample, the target channel scores >0 and all others <=0.

    ``y == _CLEAR`` means *no* channel may score >0.
    """
    scores = X @ W.T + b  # [n, C]
    want_pos = (np.arange(CHANNELS)[None, :] == y[:, None])  # [n, C]
    ok_pos = (scores > 0) == want_pos
    return bool(ok_pos.all())


def solve_linear_conv(task: Task, kernels=(1, 3, 5), iters: int = 80) -> Iterator[Candidate]:
    """Yield a single-conv network for each kernel size that fits the task exactly.

    Only applicable to same-shape tasks whose rule is a *local, linearly
    separable* function of a ``k x k`` color neighbourhood.  Weights are integers
    from a perceptron; a bias channel is added only if it is nonzero.
    """
    for k in kernels:
        built = _conv_samples(task, k)
        if built is None:
            return
        X, y = built
        W, b, converged = _fit_perceptron(X, y, iters=iters)
        if not converged or not _numpy_exact(X, y, W, b):
            continue
        weight = W.reshape(CHANNELS, CHANNELS, k, k).astype(np.float32)
        bias = None if np.allclose(b, 0.0) else b.astype(np.float32)
        est = CHANNELS * CHANNELS * k * k + (CHANNELS if bias is not None else 0)
        yield Candidate(
            B.make_conv(weight, bias, kernel_size=k),
            "linear_conv",
            f"k={k}" + ("" if bias is None else " +bias"),
            est_cost=est,
        )
        return  # smallest converged kernel is cheapest; stop


# Register the conv fitter.  K in {1,3} keeps the automated sweep fast; larger
# kernels rarely help and can be run separately via solve_linear_conv(..., (5,)).
def _conv_solver(task: Task) -> Iterator[Candidate]:
    return solve_linear_conv(task, kernels=(1, 3))


register(_conv_solver)
