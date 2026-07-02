"""Builders for small ONNX graphs that map the ``input`` tensor to ``output``.

Every builder returns an ``onnx.ModelProto`` whose single graph input is named
``input`` and single output ``output``, both of static shape ``[1, 10, 30, 30]``
and dtype float32 — the naming the official scorer requires (and which it
excludes from the memory footprint).

Cost reminders (see :mod:`neurogolf.scoring`):

* ``params`` counts every element of every initializer / ``Constant`` value.
* ``memory`` counts every *intermediate* tensor's byte footprint; the ``input``
  and ``output`` tensors are exempt.  A single-node graph therefore has zero
  memory cost.
* Only ops in the default (``ai.onnx``) domain are allowed; ``Loop``, ``Scan``,
  ``NonZero``, ``Unique``, ``Compress`` (and functions/subgraphs) are banned.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper

from .data import CHANNELS, GRID

_DTYPE = TensorProto.FLOAT
_IR_VERSION = 10
_OPSET = [helper.make_opsetid("", 10)]


def _finalize(nodes, initializers, name: str = "graph", opset: int = 10) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", _DTYPE, list(GRID))
    y = helper.make_tensor_value_info("output", _DTYPE, list(GRID))
    graph = helper.make_graph(nodes, name, [x], [y], initializers)
    opsets = _OPSET if opset == 10 else [helper.make_opsetid("", opset)]
    return helper.make_model(graph, ir_version=_IR_VERSION, opset_imports=opsets)


def make_identity() -> onnx.ModelProto:
    """``output = input``.  Zero params, zero memory -> full 25 points.

    Solves any task whose output grid equals its input grid on every example.
    """
    node = helper.make_node("Identity", ["input"], ["output"])
    return _finalize([node], [])


def make_gather_colormap(indices: Sequence[int]) -> onnx.ModelProto:
    """Permute/select color channels: ``output[:, o] = input[:, indices[o]]``.

    ``indices`` has length 10 (one source channel per output channel), giving a
    parameter cost of 10.  Use this for any color map where each output color is
    sourced from at most one input color (i.e. no color merges).
    """
    if len(indices) != CHANNELS:
        raise ValueError(f"indices must have length {CHANNELS}, got {len(indices)}")
    idx = helper.make_tensor("gather_idx", TensorProto.INT64, [CHANNELS], list(indices))
    node = helper.make_node("Gather", ["input", "gather_idx"], ["output"], axis=1)
    return _finalize([node], [idx])


def make_conv(
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    kernel_size: Optional[int] = None,
) -> onnx.ModelProto:
    """Single ``Conv`` with ``same`` padding: ``output = conv(input, weight[, bias])``.

    ``weight`` has shape ``[10, 10, k, k]``.  Parameter cost is ``10*10*k*k``
    (+10 if a bias is supplied).  ``same`` padding keeps the spatial size at
    30x30 so the graph stays single-node (zero memory).
    """
    weight = np.asarray(weight, dtype=np.float32)
    out_c, in_c, kh, kw = weight.shape
    if (out_c, in_c) != (CHANNELS, CHANNELS) or kh != kw:
        raise ValueError(f"weight must be [10,10,k,k], got {weight.shape}")
    k = kh if kernel_size is None else kernel_size
    pad = k // 2
    inits = [helper.make_tensor("W", _DTYPE, list(weight.shape), weight.reshape(-1).tolist())]
    inputs = ["input", "W"]
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (CHANNELS,):
            raise ValueError(f"bias must be [10], got {bias.shape}")
        inits.append(helper.make_tensor("B", _DTYPE, [CHANNELS], bias.tolist()))
        inputs.append("B")
    node = helper.make_node(
        "Conv", inputs, ["output"], kernel_shape=[k, k], pads=[pad, pad, pad, pad]
    )
    return _finalize([node], inits)


def make_conv2(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> onnx.ModelProto:
    """Two-layer conv net: ``output = conv2(relu(conv1(input)))``.

    ``w1`` is ``[H, 10, k1, k1]``, ``w2`` is ``[10, H, k2, k2]``; both convs use
    ``same`` padding so the spatial size stays 30x30.  The two intermediate
    tensors (pre-activation and ReLU output, each ``[1, H, 30, 30]`` float32)
    cost ``2 * H * 3600`` bytes of memory, so keep ``H`` small.
    """
    w1 = np.asarray(w1, dtype=np.float32)
    w2 = np.asarray(w2, dtype=np.float32)
    hidden, in_c, k1, k1b = w1.shape
    out_c, hidden2, k2, k2b = w2.shape
    if in_c != CHANNELS or out_c != CHANNELS or hidden != hidden2 or k1 != k1b or k2 != k2b:
        raise ValueError(f"bad shapes: w1={w1.shape} w2={w2.shape}")
    b1 = np.asarray(b1, dtype=np.float32).reshape(hidden)
    b2 = np.asarray(b2, dtype=np.float32).reshape(CHANNELS)
    p1, p2 = k1 // 2, k2 // 2
    inits = [
        helper.make_tensor("W1", _DTYPE, list(w1.shape), w1.reshape(-1).tolist()),
        helper.make_tensor("B1", _DTYPE, [hidden], b1.tolist()),
        helper.make_tensor("W2", _DTYPE, list(w2.shape), w2.reshape(-1).tolist()),
        helper.make_tensor("B2", _DTYPE, [CHANNELS], b2.tolist()),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["input", "W1", "B1"], ["h_pre"],
            kernel_shape=[k1, k1], pads=[p1, p1, p1, p1],
        ),
        helper.make_node("Relu", ["h_pre"], ["h"]),
        helper.make_node(
            "Conv", ["h", "W2", "B2"], ["output"],
            kernel_shape=[k2, k2], pads=[p2, p2, p2, p2],
        ),
    ]
    return _finalize(nodes, inits)


# Feature blocks for :func:`make_reduce_head`: name -> (ONNX reduce op, axis).
# Each produces a [1, 10, 30] tensor (per-color row/col sums or occupancies).
REDUCE_BLOCKS = {
    "rs": ("ReduceSum", 3),
    "cs": ("ReduceSum", 2),
    "rm": ("ReduceMax", 3),
    "cm": ("ReduceMax", 2),
}


def make_reduce_head(
    weight: np.ndarray,
    bias: np.ndarray,
    oh: int,
    ow: int,
    blocks: Sequence[str],
) -> onnx.ModelProto:
    """Reduce features -> linear head -> fixed ``oh x ow`` output grid.

    For tasks whose output is always the same small shape regardless of input
    size.  ``weight`` is ``[oh, ow, 10, F]`` and ``bias`` ``[oh, ow, 10]`` where
    ``F = 300 * len(blocks)``: per output cell, a 10-class linear classifier
    over the concatenated reduce features.  The head is a single MatMul+Add to
    ``[1, 10*oh*ow]``, reshaped to ``[1, 10, oh, ow]`` and padded to 30x30 with
    constant ``-1`` so the out-of-grid cells threshold to clear.
    """
    F = weight.shape[-1]
    if F != 300 * len(blocks):
        raise ValueError(f"F={F} does not match blocks={blocks}")
    oc = CHANNELS * oh * ow
    w_big = np.zeros((F, oc), dtype=np.float32)
    b_big = np.zeros(oc, dtype=np.float32)
    for r in range(oh):
        for c in range(ow):
            for o in range(CHANNELS):
                j = o * oh * ow + r * ow + c
                w_big[:, j] = weight[r, c, o]
                b_big[j] = bias[r, c, o]
    nodes, inits, outs = [], [], []
    for bname in blocks:
        op, ax = REDUCE_BLOCKS[bname]
        nodes.append(helper.make_node(op, ["input"], [bname], axes=[ax], keepdims=0))
        outs.append(bname)
    if len(outs) > 1:
        nodes.append(helper.make_node("Concat", outs, ["cat"], axis=1))
        src = "cat"
    else:
        src = outs[0]
    inits.append(helper.make_tensor("shape_flat", TensorProto.INT64, [2], [1, F]))
    nodes.append(helper.make_node("Reshape", [src, "shape_flat"], ["feats"]))
    inits.append(helper.make_tensor("Wbig", _DTYPE, [F, oc], w_big.reshape(-1).tolist()))
    inits.append(helper.make_tensor("Bbig", _DTYPE, [oc], b_big.tolist()))
    nodes.append(helper.make_node("MatMul", ["feats", "Wbig"], ["mm"]))
    nodes.append(helper.make_node("Add", ["mm", "Bbig"], ["logits"]))
    inits.append(helper.make_tensor("shape_out", TensorProto.INT64, [4], [1, CHANNELS, oh, ow]))
    nodes.append(helper.make_node("Reshape", ["logits", "shape_out"], ["small"]))
    nodes.append(
        helper.make_node(
            "Pad", ["small"], ["output"], mode="constant", value=-1.0,
            pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow],
        )
    )
    return _finalize(nodes, inits)


def make_cellmap_gather(
    src: np.ndarray,
    cmap: Sequence[Dict[int, int]],
    w: int,
    oh: int,
    ow: int,
) -> onnx.ModelProto:
    """Fixed cell-to-cell mapping via GatherND (for fixed input/output dims).

    ``src[d]`` is the flat input-cell index feeding output cell ``d`` (row-major
    over ``oh x ow``); ``cmap[d]`` maps input color -> output color for that
    cell.  Index tensor ``idx[10, oh, ow, 4]`` holds the coordinate
    ``(0, cin, r', c')`` whose one-hot value lands in output channel ``o`` at
    ``(r, c)``; channels a cell never produces read the dead coordinate
    ``(0, 0, 29, 29)``, which is guaranteed all-zero because the (fixed) input
    grid is smaller than 30x30.

    Color *merges* (several input colors mapping to one output color) are
    handled with one extra index table per merge rank, summed with ``Add``: at
    most one of the gathered one-hots is 1 per cell, so the sum stays exact.
    Requires opset >= 11 for GatherND (13 used; verified compatible with the
    official scorer).
    """
    tables: list[np.ndarray] = []

    def _table(k: int) -> np.ndarray:
        while len(tables) <= k:
            t = np.zeros((CHANNELS, oh, ow, 4), dtype=np.int64)
            t[..., 2] = 29
            t[..., 3] = 29  # dead coordinate: beyond any grid < 30x30
            tables.append(t)
        return tables[k]

    _table(0)
    for d in range(oh * ow):
        r, c = divmod(d, ow)
        rs, cs = divmod(int(src[d]), w)
        by_out: Dict[int, list] = {}
        for cin, o in cmap[d].items():
            by_out.setdefault(o, []).append(cin)
        for o, cins in by_out.items():
            for k, cin in enumerate(cins):
                _table(k)[o, r, c] = [0, cin, rs, cs]
    inits = [
        helper.make_tensor("oshape", TensorProto.INT64, [4], [1, CHANNELS, oh, ow]),
        helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow]),
    ]
    nodes = []
    for k, t in enumerate(tables):
        inits.append(
            helper.make_tensor(f"idx{k}", TensorProto.INT64, [CHANNELS, oh, ow, 4], t.reshape(-1).tolist())
        )
        nodes.append(helper.make_node("GatherND", ["input", f"idx{k}"], [f"g{k}"]))
    acc = "g0"
    for k in range(1, len(tables)):
        nodes.append(helper.make_node("Add", [acc, f"g{k}"], [f"sum{k}"]))
        acc = f"sum{k}"
    nodes.append(helper.make_node("Reshape", [acc, "oshape"], ["small"]))
    nodes.append(helper.make_node("Pad", ["small", "pads"], ["output"], mode="constant"))
    return _finalize(nodes, inits, opset=13)


def make_pixel_upscale(sr: int, sc: int) -> onnx.ModelProto:
    """Pixel magnification by ``(sr, sc)`` via grouped ConvTranspose.

    Crops the input to ``[1, 10, 30//sr, 30//sc]`` with negative Pad (safe when
    every input grid fits in the crop), then a per-channel all-ones
    ``ConvTranspose`` with stride ``(sr, sc)`` duplicates each pixel into an
    ``sr x sc`` block, landing exactly on 30x30.
    """
    ch, cw = 30 // sr, 30 // sc
    weight = np.ones((CHANNELS, 1, sr, sc), dtype=np.float32)
    inits = [
        helper.make_tensor("W", _DTYPE, [CHANNELS, 1, sr, sc], weight.reshape(-1).tolist())
    ]
    nodes = [
        helper.make_node(
            "Pad", ["input"], ["crop"], mode="constant", value=0.0,
            pads=[0, 0, 0, 0, 0, 0, ch - 30, cw - 30],
        ),
        helper.make_node(
            "ConvTranspose", ["crop", "W"], ["output"], strides=[sr, sc], group=CHANNELS
        ),
    ]
    return _finalize(nodes, inits)


def make_flat_head(
    weight: np.ndarray,
    bias: np.ndarray,
    h: int,
    w: int,
    oh: int,
    ow: int,
) -> onnx.ModelProto:
    """Linear head over the flattened one-hot grid, for fixed input/output dims.

    ``weight`` is ``[10*h*w, 10*oh*ow]`` and ``bias`` ``[10*oh*ow]``: a 10-class
    linear classifier per output cell over the full (cropped) input grid.  This
    expresses any transform where each output cell is a linearly separable
    function of the whole input — a strict superset of cell-mapping, at higher
    parameter cost, so register it after the cheaper solvers.

    Graph: Pad-crop input to ``[1,10,h,w]`` -> Reshape ``[1, 10hw]`` -> MatMul +
    Add -> Reshape ``[1,10,oh,ow]`` -> Pad to 30x30 with ``-1`` (clear).
    """
    f = CHANNELS * h * w
    oc = CHANNELS * oh * ow
    weight = np.asarray(weight, dtype=np.float32).reshape(f, oc)
    bias = np.asarray(bias, dtype=np.float32).reshape(oc)
    inits = [
        helper.make_tensor("shape_flat", TensorProto.INT64, [2], [1, f]),
        helper.make_tensor("W", _DTYPE, [f, oc], weight.reshape(-1).tolist()),
        helper.make_tensor("B", _DTYPE, [oc], bias.tolist()),
        helper.make_tensor("shape_out", TensorProto.INT64, [4], [1, CHANNELS, oh, ow]),
    ]
    nodes = [
        helper.make_node(
            "Pad", ["input"], ["crop"], mode="constant", value=0.0,
            pads=[0, 0, 0, 0, 0, 0, h - 30, w - 30],
        ),
        helper.make_node("Reshape", ["crop", "shape_flat"], ["feats"]),
        helper.make_node("MatMul", ["feats", "W"], ["mm"]),
        helper.make_node("Add", ["mm", "B"], ["logits"]),
        helper.make_node("Reshape", ["logits", "shape_out"], ["small"]),
        helper.make_node(
            "Pad", ["small"], ["output"], mode="constant", value=-1.0,
            pads=[0, 0, 0, 0, 0, 0, 30 - oh, 30 - ow],
        ),
    ]
    return _finalize(nodes, inits)


def make_colormap_conv(mapping: Dict[int, int]) -> onnx.ModelProto:
    """Color map as a 1x1 conv (cost 100).  Handles color *merges* (many->one).

    ``mapping[a] = b`` means every cell of color ``a`` becomes color ``b``.
    Colors absent from ``mapping`` are treated as identity (``a -> a``).
    """
    weight = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    full = {c: c for c in range(CHANNELS)}
    full.update(mapping)
    for src, dst in full.items():
        weight[dst, src, 0, 0] = 1.0
    return make_conv(weight)
