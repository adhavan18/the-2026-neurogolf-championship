"""Tests for the ONNX builders: shapes, ops, and static-shape constraints."""

import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurogolf import builders as B  # noqa: E402
from neurogolf.data import GRID  # noqa: E402

_BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def _check(model: onnx.ModelProto):
    onnx.checker.check_model(model, full_check=True)
    g = model.graph
    assert [t.name for t in g.input] == ["input"]
    assert [t.name for t in g.output] == ["output"]
    for op in g.node:
        assert op.op_type.upper() not in _BANNED
        assert "Sequence" not in op.op_type
    # Statically-defined I/O shapes.
    for t in list(g.input) + list(g.output):
        dims = [d.dim_value for d in t.type.tensor_type.shape.dim]
        assert dims == list(GRID)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def test_identity():
    _check(B.make_identity())


def test_gather_colormap():
    _check(B.make_gather_colormap(list(range(10))))


def test_gather_colormap_bad_length():
    try:
        B.make_gather_colormap([0, 1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong index length")


def test_conv_shapes():
    for k in (1, 3, 5):
        w = np.zeros((10, 10, k, k), dtype=np.float32)
        _check(B.make_conv(w, kernel_size=k))
        _check(B.make_conv(w, bias=np.zeros(10, dtype=np.float32), kernel_size=k))


def test_colormap_conv():
    _check(B.make_colormap_conv({1: 2, 3: 4}))


def test_conv2():
    w1 = np.zeros((4, 10, 3, 3), dtype=np.float32)
    w2 = np.zeros((10, 4, 1, 1), dtype=np.float32)
    _check(B.make_conv2(w1, np.zeros(4), w2, np.zeros(10)))


def test_reduce_head():
    oh, ow = 3, 3
    weight = np.zeros((oh, ow, 10, 300), dtype=np.float32)
    bias = np.zeros((oh, ow, 10), dtype=np.float32)
    _check(B.make_reduce_head(weight, bias, oh, ow, ["rs"]))


def test_cellmap_gather():
    oh, ow, w = 2, 2, 2
    src = np.array([0, 1, 2, 3])
    cmap = [{0: 0, 1: 1} for _ in range(4)]
    _check(B.make_cellmap_gather(src, cmap, w, oh, ow))


def test_pixel_upscale():
    _check(B.make_pixel_upscale(2, 2))
    _check(B.make_pixel_upscale(1, 2))


def test_flat_head():
    h, w, oh, ow = 3, 3, 2, 2
    weight = np.zeros((10 * h * w, 10 * oh * ow), dtype=np.float32)
    bias = np.zeros(10 * oh * ow, dtype=np.float32)
    _check(B.make_flat_head(weight, bias, h, w, oh, ow))
