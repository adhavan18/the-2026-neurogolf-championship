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


def _finalize(nodes, initializers, name: str = "graph") -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", _DTYPE, list(GRID))
    y = helper.make_tensor_value_info("output", _DTYPE, list(GRID))
    graph = helper.make_graph(nodes, name, [x], [y], initializers)
    return helper.make_model(graph, ir_version=_IR_VERSION, opset_imports=_OPSET)


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
