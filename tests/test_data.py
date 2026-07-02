"""Tests for grid encoding/decoding round-trips."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurogolf.data import GRID, decode_grid, encode_grid  # noqa: E402


def test_encode_shape_and_onehot():
    grid = [[0, 1, 2], [3, 4, 5]]
    t = encode_grid(grid)
    assert t.shape == GRID
    # exactly one channel hot per in-grid cell
    assert t[0, :, :2, :3].sum() == 6
    assert t[0, 1, 0, 1] == 1.0  # cell (0,1) is color 1
    # everything outside the 2x3 grid is clear
    assert t[0, :, 2:, :].sum() == 0
    assert t[0, :, :, 3:].sum() == 0


def test_decode_roundtrip():
    grid = [[0, 1, 9], [5, 5, 0], [2, 3, 4]]
    t = encode_grid(grid)
    # emulate the (x > 0) threshold the scorer applies
    thresholded = (t > 0.0).astype(np.float32)
    assert decode_grid(thresholded) == grid


def test_decode_trims_trailing_clear():
    # A grid whose last column/row are color 0 must survive (0 != clear).
    grid = [[0, 0], [0, 0]]
    t = (encode_grid(grid) > 0).astype(np.float32)
    assert decode_grid(t) == grid
