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
from numpy.lib.stride_tricks import sliding_window_view

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


def _fit_perceptron(X: np.ndarray, y: np.ndarray, iters: int = 80, margin: float = 0.0):
    """One-vs-rest perceptron with integer weights and an optional margin.

    Returns ``(W[C,F], b[C], converged)``.  On convergence the sign scheme
    satisfies exactly the competition's threshold: score>0 for the target
    channel and <=0 for every other channel, at every training cell — and, with
    ``margin > 0``, by at least that margin, which pushes the decision boundary
    away from the training points.  This matters for the *private* hold-out:
    zero-margin solutions fit the public examples exactly but sit razor-close
    to them, and the first slightly-novel private grid flips a cell.
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
        pos_wrong = (Y > 0) & (scores <= margin)
        neg_wrong = (Y < 0) & (scores > -margin)
        wrong = pos_wrong | neg_wrong
        if not wrong.any():
            converged = True
            break
        for o in range(CHANNELS):
            idx = wrong[o]
            if idx.any():
                W[o] += (Y[o, idx][:, None] * X[idx]).sum(0)
                b[o] += Y[o, idx].sum()
    if margin > 0.0 and not converged:
        # Margin unreachable in the budget: fall back to plain separation so we
        # never lose a task that the zero-margin fit could solve.
        return _fit_perceptron(X, y, iters=iters, margin=0.0)
    return W, b, converged


# Sentinel target: this output cell must be all-clear (every channel <= 0).
_CLEAR = -1


def _conv_samples(task: Task, k: int):
    """Build unique (patch, target) samples for a k x k conv over the *full* grid.

    Vectorized via ``sliding_window_view`` (im2col).  Crucially this includes
    the border ring *outside* the grid that a k>1 kernel can reach: those cells
    must decode to clear (all channels <= 0).  A single all-zero patch is added
    to force bias <= 0 so that far-away cells stay clear too.  Samples are
    de-duplicated; if one patch demands two different targets the rule isn't a
    function of the k x k neighbourhood and the fit will simply fail to
    converge (which is the correct outcome).
    """
    if not _same_shape_pairs(task):
        return None
    off = k // 2
    xs, ys = [], []
    for p in task.scored_pairs:
        gi, go = p.input, p.output
        h, w = len(gi), len(gi[0])
        gi_a = np.asarray(gi)
        go_a = np.asarray(go)
        # Build the actual 30x30 input tensor ONNX sees (grid top-left, else 0),
        # then pad by `off` so every cell's k x k neighbourhood is well defined.
        full = np.zeros((CHANNELS, 30, 30), dtype=np.float32)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        full[gi_a, rr, cc] = 1.0
        pad = full if off == 0 else np.pad(full, ((0, 0), (off, off), (off, off)))
        win = sliding_window_view(pad, (k, k), axis=(1, 2))  # [C, 30, 30, k, k]
        patches = np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4)).reshape(
            900, CHANNELS * k * k
        )
        tgt = np.full((30, 30), _CLEAR, dtype=np.int64)
        tgt[:h, :w] = go_a
        # Constrain every output cell whose k x k neighbourhood can touch the grid.
        rmax, cmax = min(30, h + off), min(30, w + off)
        mask = np.zeros((30, 30), dtype=bool)
        mask[:rmax, :cmax] = True
        m = mask.reshape(-1)
        xs.append(patches[m])
        ys.append(tgt.reshape(-1)[m])
    if not xs:
        return None
    xs.append(np.zeros((1, CHANNELS * k * k), dtype=np.float32))  # force bias <= 0
    ys.append(np.asarray([_CLEAR], dtype=np.int64))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
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


def solve_linear_conv(task: Task, kernels=(1, 3, 5, 7), iters: int = 300) -> Iterator[Candidate]:
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
        W, b, converged = _fit_perceptron(X, y, iters=iters, margin=2.0)
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


# Register the conv fitter.  The vectorized sampler makes the full k sweep
# cheap; the cheapest (smallest) converging kernel wins.  Tasks it solves are
# remembered so the (much slower) two-layer solver can skip them.
_LINEAR_SOLVED: set[int] = set()


def _conv_solver(task: Task) -> Iterator[Candidate]:
    for cand in solve_linear_conv(task, kernels=(1, 3, 5, 7)):
        _LINEAR_SOLVED.add(task.num)
        yield cand


register(_conv_solver)


# ----------------------------------------------------------------------------- #
# Two-layer conv solver (Conv -> ReLU -> Conv), fit by gradient descent
# ----------------------------------------------------------------------------- #


def _two_layer_windows(X: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """Slice K x K patches (K = k1+k2-1) into the k2 x k2 grid of k1 x k1 sub-patches.

    Returns ``P[n, Q, F1]`` with ``Q = k2*k2`` hidden positions (row-major, so
    position ``q = u*k2 + v`` aligns with conv2 weight index ``[.., u, v]``) and
    ``F1 = CHANNELS*k1*k1`` first-layer features.
    """
    n = X.shape[0]
    K = k1 + k2 - 1
    Xp = X.reshape(n, CHANNELS, K, K)
    win = sliding_window_view(Xp, (k1, k1), axis=(2, 3))  # [n, C, k2, k2, k1, k1]
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(
        n, k2 * k2, CHANNELS * k1 * k1
    )


def _conv2_forward(P, W1, b1, W2, b2):
    pre = P @ W1.T + b1  # [n, Q, H]
    h = np.maximum(pre, 0.0)
    s = np.einsum("nqj,ojq->no", h, W2) + b2  # [n, C]
    return pre, h, s


def _fit_conv2_gd(P, T, H, seed, iters=1200, lr=0.05, margin=1.0):
    """Adam on a hinge loss: want ``T*s >= margin`` everywhere.  Returns params + ok."""
    rng = np.random.default_rng(seed)
    n, Q, F1 = P.shape
    W1 = rng.normal(0.0, 0.5, (H, F1))
    b1 = np.zeros(H)
    W2 = rng.normal(0.0, 0.5, (CHANNELS, H, Q))
    b2 = np.zeros(CHANNELS)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        pre, h, s = _conv2_forward(P, *params)
        viol = (margin - T * s) > 0
        if not viol.any():
            return params, True
        gs = np.where(viol, -T, 0.0) / n
        gW2 = np.einsum("no,nqj->ojq", gs, h)
        gb2 = gs.sum(0)
        gh = np.einsum("no,ojq->nqj", gs, params[2])
        gpre = gh * (pre > 0)
        gW1 = np.einsum("nqj,nqf->jf", gpre, P)
        gb1 = gpre.sum((0, 1))
        grads = [gW1, gb1, gW2, gb2]
        step = lr if t < iters // 2 else lr * 0.3
        for i, g in enumerate(grads):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            mhat = m[i] / (1 - beta1**t)
            vhat = v[i] / (1 - beta2**t)
            params[i] -= step * mhat / (np.sqrt(vhat) + eps)
    # No full margin; still usable if strictly sign-correct with float32 headroom.
    _, _, s = _conv2_forward(P, *params)
    ok = bool(((s > 0) == (T > 0)).all() and np.abs(s).min() > 1e-3)
    return params, ok


def _targets_pm1(y: np.ndarray) -> np.ndarray:
    """[n, C] matrix of +1 (this channel must fire) / -1 (must stay <= 0)."""
    T = np.full((y.shape[0], CHANNELS), -1.0)
    hit = y >= 0
    T[np.nonzero(hit)[0], y[hit]] = 1.0
    return T


def _fit_conv2_task(X, y, k1, k2, H, seeds=(0, 1), cap=4000):
    """Active-set GD fit; returns float32 (W1, b1, W2, b2) or None.

    Trains on a subsample, checks exactness on *all* unique samples, folds the
    violators back in, and repeats.  Success requires strict sign-correctness
    on every sample (the pipeline still runs the official scorer afterwards).
    """
    P_full = _two_layer_windows(X, k1, k2)
    T_full = _targets_pm1(y)
    n = P_full.shape[0]
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        if n <= cap:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, cap, replace=False)
        for _ in range(6):
            params, ok = _fit_conv2_gd(P_full[idx], T_full[idx], H, seed)
            if not ok:
                break
            _, _, s = _conv2_forward(P_full, *params)
            bad = ~(((s > 0) == (T_full > 0)).all(axis=1) & (np.abs(s).min(axis=1) > 1e-3))
            if not bad.any():
                W1, b1, W2, b2 = params
                w1 = W1.reshape(H, CHANNELS, k1, k1).astype(np.float32)
                w2 = W2.reshape(CHANNELS, H, k2, k2).astype(np.float32)
                # Re-check in float32 (what ONNX will compute with).
                P32 = P_full.astype(np.float32)
                _, _, s32 = _conv2_forward(
                    P32,
                    W1.astype(np.float32),
                    b1.astype(np.float32),
                    W2.astype(np.float32),
                    b2.astype(np.float32),
                )
                if not ((s32 > 0) == (T_full > 0)).all():
                    break
                return w1, b1.astype(np.float32), w2, b2.astype(np.float32)
            extra = np.nonzero(bad)[0]
            if len(extra) > 2000:
                extra = rng.choice(extra, 2000, replace=False)
            idx = np.unique(np.concatenate([idx, extra]))
        # next seed
    return None


# Configurations tried in cost order: memory is 2*H*3600 bytes, so small H first.
_CONV2_CONFIGS = ((3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 3, 4))


def solve_conv2(task: Task, configs=_CONV2_CONFIGS) -> Iterator[Candidate]:
    """Two-layer Conv->ReLU->Conv for same-shape tasks the linear fitter missed."""
    if task.num in _LINEAR_SOLVED or not _same_shape_pairs(task):
        return
    cache: Dict[int, tuple] = {}
    for k1, k2, H in configs:
        K = k1 + k2 - 1
        if K not in cache:
            built = _conv_samples(task, K)
            if built is None:
                return
            cache[K] = built
        X, y = cache[K]
        fitted = _fit_conv2_task(X, y, k1, k2, H)
        if fitted is None:
            continue
        w1, b1, w2, b2 = fitted
        est = w1.size + b1.size + w2.size + b2.size + 2 * H * 3600
        yield Candidate(
            B.make_conv2(w1, b1, w2, b2),
            "conv2",
            f"k1={k1} k2={k2} H={H}",
            est_cost=est,
        )
        return  # first (cheapest) success wins


register(solve_conv2)


# ----------------------------------------------------------------------------- #
# Constant-output-shape solver (reduce features -> linear head)
# ----------------------------------------------------------------------------- #


def _reduce_feats(grid, blocks) -> np.ndarray:
    """Concatenated per-color reduce features for one input grid (matches ONNX)."""
    h, w = len(grid), len(grid[0])
    g = np.asarray(grid)
    oht = np.zeros((CHANNELS, 30, 30), dtype=np.float32)
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    oht[g, rr, cc] = 1.0
    parts = []
    for bname in blocks:
        if bname == "rs":
            parts.append(oht.sum(2).ravel())
        elif bname == "cs":
            parts.append(oht.sum(1).ravel())
        elif bname == "rm":
            parts.append(oht.max(2).ravel())
        elif bname == "cm":
            parts.append(oht.max(1).ravel())
        else:
            raise ValueError(bname)
    return np.concatenate(parts)


# Feature subsets tried smallest-first, so the first fit is also the cheapest.
_REDUCE_SUBSETS = (
    ("rs",), ("cs",), ("rm",), ("cm",),
    ("rs", "cs"), ("rm", "cm"),
    ("rs", "cs", "rm", "cm"),
)


@register
def solve_const_shape(task: Task) -> Iterator[Candidate]:
    """Fixed-output-shape tasks: per-cell linear classifiers over reduce features.

    Applies when every scored pair has the *same* small output shape (oh x ow,
    at most 36 cells) that differs from the input shape.  Row/col sums and
    occupancies per color feed one 10-class perceptron per output cell; the
    whole head is a single MatMul.
    """
    pairs = task.scored_pairs
    if not pairs or _same_shape_pairs(task):
        return
    oh, ow = pairs[0].out_shape
    if oh * ow == 0 or oh * ow > 36:
        return
    if any(p.out_shape != (oh, ow) for p in pairs):
        return
    tgt = np.array([p.output for p in pairs])  # [N, oh, ow]
    for blocks in _REDUCE_SUBSETS:
        feats = np.stack([_reduce_feats(p.input, blocks) for p in pairs])
        F = feats.shape[1]
        Wc = np.zeros((oh, ow, CHANNELS, F))
        Bc = np.zeros((oh, ow, CHANNELS))
        ok = True
        for r in range(oh):
            for c in range(ow):
                W, b, converged = _fit_perceptron(feats, tgt[:, r, c], iters=600, margin=5.0)
                if not converged:
                    ok = False
                    break
                Wc[r, c] = W
                Bc[r, c] = b
            if not ok:
                break
        if not ok:
            continue
        oc = CHANNELS * oh * ow
        yield Candidate(
            B.make_reduce_head(Wc, Bc, oh, ow, blocks),
            "const_shape",
            f"{oh}x{ow} blocks={','.join(blocks)}",
            est_cost=F * oc + oc + 6,
        )
        return  # smallest feature set wins


# ----------------------------------------------------------------------------- #
# Cell-mapping solver (fixed input & output dims; GatherND)
# ----------------------------------------------------------------------------- #

# Cap the number of pairs used for *detection* (the official scorer still
# verifies against every pair, so a wrong subsampled mapping is just rejected).
_CELLMAP_DETECT_PAIRS = 64


def _cellmap_scan(in_f: np.ndarray, out_f: np.ndarray, hw: int, ohw: int):
    """Assign to every output cell a source input cell + per-cell color map.

    Two passes: the first only accepts *injective* maps (one GatherND table);
    cells left over get a second pass that also accepts color merges (the
    builder emits one extra table per merge rank, so merges are strictly more
    expensive and must not shadow an injective source elsewhere).
    """
    src = np.full(ohw, -1)
    cmap: list = [None] * ohw
    remaining = set(range(ohw))
    for allow_merges in (False, True):
        for s in range(hw):
            if not remaining:
                break
            a = in_f[:, s]
            todo = list(remaining)
            code = a[:, None] * 10 + out_f[:, todo]
            for j, dst in enumerate(todo):
                u = np.unique(code[:, j])
                ins = u // 10
                if len(ins) != len(np.unique(ins)):
                    continue  # output color not a function of this input cell
                if not allow_merges and len(np.unique(u % 10)) != len(u):
                    continue  # merge: defer to the second pass
                src[dst] = s
                cmap[dst] = {int(i): int(o) for i, o in zip(ins, u % 10)}
                remaining.discard(dst)
        if not remaining:
            break
    if remaining:
        return None
    return src, cmap


def _cellmap_consistent(in_f, out_f, src, cmap) -> bool:
    """Cheap numpy check that the detected mapping reproduces *every* pair."""
    for dst in range(len(src)):
        lut = np.full(10, -1, dtype=np.int64)
        for i, o in cmap[dst].items():
            lut[i] = o
        if not np.array_equal(lut[in_f[:, src[dst]]], out_f[:, dst]):
            return False
    return True


def _detect_cellmap(task: Task):
    """Detect ``output[r,c] = f_rc(input[src_rc])`` with per-cell color maps.

    Requires every scored pair to share fixed input dims (h, w) and fixed
    output dims (oh, ow).  Detection first runs on a subsample of pairs for
    speed; the result is verified on all pairs and, on mismatch, detection is
    redone with every pair.  Returns ``(h, w, oh, ow, src, cmap)`` or None.
    """
    pairs = task.scored_pairs
    if not pairs:
        return None
    di = {p.in_shape for p in pairs}
    do = {p.out_shape for p in pairs}
    if len(di) != 1 or len(do) != 1:
        return None
    (h, w), (oh, ow) = di.pop(), do.pop()
    if h * w == 0 or oh * ow == 0 or h * w > 500:
        return None
    if max(h, w) > 29 or max(oh, ow) > 29:
        return None  # dead-index / Pad tricks need grids strictly inside 30x30
    n = len(pairs)
    in_f = np.array([p.input for p in pairs]).reshape(n, -1)
    out_f = np.array([p.output for p in pairs]).reshape(n, -1)
    hw, ohw = h * w, oh * ow
    if n > _CELLMAP_DETECT_PAIRS:
        step = n / _CELLMAP_DETECT_PAIRS
        sel = [int(i * step) for i in range(_CELLMAP_DETECT_PAIRS)]
        found = _cellmap_scan(in_f[sel], out_f[sel], hw, ohw)
        if found is None:
            return None  # a subsample failure implies a full failure
        if not _cellmap_consistent(in_f, out_f, *found):
            found = _cellmap_scan(in_f, out_f, hw, ohw)  # escalate to all pairs
    else:
        found = _cellmap_scan(in_f, out_f, hw, ohw)
    if found is None:
        return None
    return h, w, oh, ow, found[0], found[1]


@register
def solve_cellmap(task: Task) -> Iterator[Candidate]:
    """GatherND network for fixed-dims tasks where each output cell copies
    (through a per-cell color map, merges included) one fixed input cell."""
    detected = _detect_cellmap(task)
    if detected is None:
        return
    h, w, oh, ow, src, cmap = detected
    yield Candidate(
        B.make_cellmap_gather(src, cmap, w, oh, ow),
        "cellmap",
        f"{h}x{w}->{oh}x{ow}",
        est_cost=CHANNELS * oh * ow * 4 + 12,
    )


# ----------------------------------------------------------------------------- #
# Pixel-upscale solver (grouped ConvTranspose)
# ----------------------------------------------------------------------------- #

_UPSCALE_FACTORS = ((2, 2), (3, 3), (2, 1), (1, 2), (3, 1), (1, 3))


@register
def solve_upscale(task: Task) -> Iterator[Candidate]:
    """Pure pixel magnification: output = each input pixel duplicated sr x sc."""
    pairs = task.scored_pairs
    if not pairs:
        return
    for sr, sc in _UPSCALE_FACTORS:
        ch, cw = 30 // sr, 30 // sc
        ok = True
        for p in pairs:
            h, w = p.in_shape
            if h > ch or w > cw or p.out_shape != (h * sr, w * sc):
                ok = False
                break
            gi = np.asarray(p.input)
            if not np.array_equal(np.kron(gi, np.ones((sr, sc), dtype=gi.dtype)),
                                  np.asarray(p.output)):
                ok = False
                break
        if ok:
            yield Candidate(
                B.make_pixel_upscale(sr, sc),
                "upscale",
                f"{sr}x{sc}",
                est_cost=CHANNELS * sr * sc + 10 * ch * cw * 4,
            )
            return


# ----------------------------------------------------------------------------- #
# Flat linear-head solver (fixed input & output dims; whole-grid classifier)
# ----------------------------------------------------------------------------- #

# Keep heads comfortably under the 1.44MB file cap (float32 weights).
_FLAT_HEAD_MAX_PARAMS = 300_000


@register
def solve_flat_head(task: Task) -> Iterator[Candidate]:
    """Fixed-dims tasks: one 10-class linear classifier per output cell over the
    *entire* flattened one-hot input grid.

    Strictly more expressive than cell-mapping (any linearly separable function
    of the whole grid) but at much higher parameter cost, so it is registered
    last: the pipeline prefers cheaper correct candidates.  Row/col profile
    features are linear in the flat one-hot, so flat + bias spans the same
    hypothesis class as flat + profiles.
    """
    pairs = task.scored_pairs
    if not pairs:
        return
    (h, w), (oh, ow) = pairs[0].in_shape, pairs[0].out_shape
    if any(p.in_shape != (h, w) or p.out_shape != (oh, ow) for p in pairs):
        return
    if h * w > 120 or oh * ow > 120:
        return
    f, oc = CHANNELS * h * w, CHANNELS * oh * ow
    if f * oc > _FLAT_HEAD_MAX_PARAMS:
        return
    onehot = np.zeros((len(pairs), CHANNELS, h, w), dtype=np.float32)
    for i, p in enumerate(pairs):
        g = np.asarray(p.input)
        rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        onehot[i, g, rr, cc] = 1.0
    feats = onehot.reshape(len(pairs), f)
    tgt = np.array([p.output for p in pairs])  # [N, oh, ow]
    weight = np.zeros((f, oc), dtype=np.float32)
    bias = np.zeros(oc, dtype=np.float32)
    for r in range(oh):
        for c in range(ow):
            W, b, converged = _fit_perceptron(feats, tgt[:, r, c], iters=600, margin=5.0)
            if not converged:
                return
            for o in range(CHANNELS):
                j = o * oh * ow + r * ow + c
                weight[:, j] = W[o]
                bias[j] = b[o]
    yield Candidate(
        B.make_flat_head(weight, bias, h, w, oh, ow),
        "flat_head",
        f"{h}x{w}->{oh}x{ow}",
        est_cost=f * oc + oc + 6,
    )


# ----------------------------------------------------------------------------- #
# Fixed-window crop solver (zero parameters)
# ----------------------------------------------------------------------------- #


@register
def solve_fixed_crop(task: Task) -> Iterator[Candidate]:
    """Tasks whose output is a fixed window ``input[a:a+oh, c:c+ow]``.

    Output dims must be constant across pairs; the offset ``(a, c)`` is searched
    over every placement that fits inside all inputs.  The network is two Pad
    nodes with zero parameters, so this beats any other solver when applicable.
    """
    pairs = task.scored_pairs
    if not pairs:
        return
    oh, ow = pairs[0].out_shape
    if oh * ow == 0 or any(p.out_shape != (oh, ow) for p in pairs):
        return
    min_h = min(p.in_shape[0] for p in pairs)
    min_w = min(p.in_shape[1] for p in pairs)
    if min_h < oh or min_w < ow:
        return
    cands = [(a, c) for a in range(min_h - oh + 1) for c in range(min_w - ow + 1)]
    for p in pairs:
        gi, go = np.asarray(p.input), np.asarray(p.output)
        cands = [(a, c) for (a, c) in cands if np.array_equal(gi[a:a + oh, c:c + ow], go)]
        if not cands:
            return
    a, c = cands[0]
    yield Candidate(
        B.make_fixed_crop(a, c, oh, ow),
        "fixed_crop",
        f"window {oh}x{ow} at ({a},{c})",
        est_cost=40 * oh * ow,
    )
