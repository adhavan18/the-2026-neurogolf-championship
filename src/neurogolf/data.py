"""Load ARC-AGI/ARC-GEN tasks and convert grids to/from the competition tensor.

Grid encoding (must match the official ``neurogolf_utils``):

* A *grid* is a list-of-lists of ints in ``0..9`` (colors), at most ``30x30``.
* It is embedded, **top-left anchored**, into a ``[1, 10, 30, 30]`` float32
  tensor with a one-hot channel per color.  Cells outside the grid's border are
  all-zero across the 10 channels ("clear"/zero-hot).
* A network's raw output is thresholded ``(x > 0.0)`` and must exactly equal the
  one-hot encoding of the expected output grid, for *every* example.

The top-left anchoring with variable grid sizes is the crux of the competition:
a single static op over the full 30x30 tensor cannot, in general, implement even
a simple geometric transform (flip/rotate) because it would move content off the
top-left origin.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Sequence

import numpy as np

BATCH, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID = (BATCH, CHANNELS, HEIGHT, WIDTH)

Grid = List[List[int]]

# Default location for the task JSON files (see .gitignore / README).
DATA_DIR = os.environ.get(
    "NEUROGOLF_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "data"),
)


@dataclass
class Pair:
    """A single input/output grid pair."""

    input: Grid
    output: Grid

    @property
    def in_shape(self) -> tuple[int, int]:
        return (len(self.input), len(self.input[0]) if self.input else 0)

    @property
    def out_shape(self) -> tuple[int, int]:
        return (len(self.output), len(self.output[0]) if self.output else 0)

    @property
    def same_shape(self) -> bool:
        return self.in_shape == self.out_shape

    @property
    def in_bounds(self) -> bool:
        """Both grids fit within 30x30 (larger grids are ignored by scoring)."""
        return max(self.in_shape) <= 30 and max(self.out_shape) <= 30


@dataclass
class Task:
    """One ARC-AGI task: its number and its example subsets."""

    num: int
    train: List[Pair]
    test: List[Pair]
    arc_gen: List[Pair]

    @property
    def all_pairs(self) -> List[Pair]:
        return self.train + self.test + self.arc_gen

    @property
    def scored_pairs(self) -> List[Pair]:
        """Pairs actually used for scoring (those that fit within 30x30)."""
        return [p for p in self.all_pairs if p.in_bounds]

    def as_examples(self) -> dict:
        """Return the ``{"train","test","arc-gen"}`` dict the official utils expect."""
        return {
            "train": [{"input": p.input, "output": p.output} for p in self.train],
            "test": [{"input": p.input, "output": p.output} for p in self.test],
            "arc-gen": [{"input": p.input, "output": p.output} for p in self.arc_gen],
        }


def _pairs(raw: Sequence[dict]) -> List[Pair]:
    return [Pair(input=p["input"], output=p["output"]) for p in raw]


def load_task(num: int, data_dir: str = DATA_DIR) -> Task:
    """Load ``taskNNN.json`` from ``data_dir`` and return a :class:`Task`."""
    path = os.path.join(data_dir, f"task{num:03d}.json")
    with open(path) as f:
        raw = json.load(f)
    return Task(
        num=num,
        train=_pairs(raw.get("train", [])),
        test=_pairs(raw.get("test", [])),
        arc_gen=_pairs(raw.get("arc-gen", [])),
    )


def available_task_nums(data_dir: str = DATA_DIR) -> List[int]:
    """Return the sorted task numbers for which a JSON file exists in ``data_dir``."""
    nums = []
    for path in glob.glob(os.path.join(data_dir, "task*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            nums.append(int(stem.replace("task", "")))
        except ValueError:
            continue
    return sorted(nums)


def iter_tasks(nums: Sequence[int] | None = None, data_dir: str = DATA_DIR) -> Iterator[Task]:
    """Yield :class:`Task` objects for ``nums`` (or all available)."""
    if nums is None:
        nums = available_task_nums(data_dir)
    for n in nums:
        yield load_task(n, data_dir)


def encode_grid(grid: Grid) -> np.ndarray:
    """Encode a color grid to a ``[1, 10, 30, 30]`` one-hot float32 tensor."""
    tensor = np.zeros(GRID, dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            tensor[0, color, r, c] = 1.0
    return tensor


def decode_grid(tensor: np.ndarray) -> Grid:
    """Decode a thresholded ``[1, 10, H, W]`` tensor back to a color grid.

    Mirrors the official ``convert_from_numpy``: trailing all-clear rows/cols are
    trimmed; a cell with >1 active channel decodes to 11, an all-clear interior
    cell to 10 (both are "error" sentinels used only for visualisation).
    """
    _, channels, height, width = tensor.shape
    grid: Grid = []
    for row in range(height):
        cells = []
        for col in range(width):
            active = [c for c in range(channels) if tensor[0, c, row, col] == 1]
            cells.append(active[0] if len(active) == 1 else (11 if active else 10))
        while cells and cells[-1] == 10:
            cells.pop()
        grid.append(cells)
    while grid and not grid[-1]:
        grid.pop()
    return grid


def target_tensor(grid: Grid) -> np.ndarray:
    """One-hot target the network output must match after ``(x > 0)`` thresholding."""
    return encode_grid(grid)
