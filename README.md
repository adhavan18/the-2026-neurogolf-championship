# The 2026 NeuroGolf Championship

Tooling for the Kaggle competition
[*The 2026 NeuroGolf Championship*](https://www.kaggle.com/competitions/neurogolf-2026):
synthesize the **smallest possible ONNX network** for each of the 400
ARC-AGI-1 tasks, such that the network reproduces the task's grid transformation
*exactly* while minimising `cost = parameters + memory_footprint_bytes`.

Per-task score is `max(1, 25 - ln(cost))` (a zero-cost network scores a full
**25**; the theoretical max over all 400 tasks is 10 000).

This repo provides an end-to-end, **verified** pipeline — data loading, ONNX
network builders, a scorer that reuses the *official* scoring code, pluggable
solvers, and submission packaging — plus a baseline set of automated solvers.
It is a foundation to build bespoke per-task networks on, not a finished
leaderboard entry (see [Scope & roadmap](#scope--roadmap)).

## How scoring works (the important details)

Understanding these decides what is even expressible:

- **Encoding.** Each grid (colors `0..9`, up to `30x30`) is embedded
  **top-left-anchored** into a `[1, 10, 30, 30]` float32 tensor, one-hot over 10
  color channels; cells outside the grid border are all-zero ("clear").
- **Correctness.** The network's raw output is thresholded `(x > 0.0)` and must
  **exactly** equal the one-hot target on *every* example in
  `train + test + arc-gen` — and on a private hold-out set. One wrong cell fails
  the whole task.
- **Cost.** `params` counts every element of every initializer / `Constant`.
  `memory` sums the byte footprint of every *intermediate* tensor — but the
  tensors literally named `input` and `output` are **exempt**. So a single-node
  graph (e.g. one `Conv`) has **zero** memory cost; its cost is just its params.
- **Constraints.** Static shapes only; ops must be in the default `ai.onnx`
  domain; `Loop`, `Scan`, `NonZero`, `Unique`, `Compress`, functions and
  subgraphs are banned; each file ≤ 1.44 MB.

The **top-left anchoring with variable grid sizes** is the crux of the
competition: a single static op over the full 30×30 tensor generally *cannot*
implement even a flip/rotate, because it would shift content off the origin.

## What's here

```
src/neurogolf/
  data.py      # load tasks; encode/decode grids <-> [1,10,30,30] tensors
  builders.py  # ONNX graphs: identity, gather-colormap, conv (+bias), 1x1 colormap
  scoring.py   # faithful offline scoring — reuses vendored official neurogolf_utils
  solvers.py   # pluggable solvers (identity, colormap, linear-conv fitter)
  pipeline.py  # run solvers over tasks, keep cheapest *verified* net, package zip
scripts/
  build_submission.py   # CLI: solve -> write ONNX -> submission.zip
vendor/neurogolf_utils/ # official competition module (Apache-2.0), used for scoring
tests/                  # builder/data unit tests + solver integration tests
```

**Design principle:** solvers only *propose* candidate networks; the pipeline
verifies each one with the official scorer and keeps the cheapest that is
actually correct. Adding a new transformation family is just adding a solver
that yields `Candidate`s — it can be optimistic and even occasionally wrong.

## Setup

```bash
pip install -r requirements.txt
# Place the competition task JSON in ./data (task001.json ... task400.json).
# The files are ~97 MB total and are .gitignored. Override the location with
# $NEUROGOLF_DATA if you keep them elsewhere.
```

## Usage

```bash
# Build a full submission (solves every task in ./data):
python scripts/build_submission.py

# Just a few tasks:
python scripts/build_submission.py --tasks 16 53 276

# Outputs: submission/taskNNN.onnx (+ manifest.json) and submission.zip
```

Run the tests:

```bash
PYTHONPATH=src python -m pytest tests -q
```

## Baseline results

Running `build_submission.py` over all 400 tasks with the current solvers,
scored by the vendored official scorer:

| Solver        | Tasks | Cost each | Points each | Subtotal |
|---------------|------:|----------:|------------:|---------:|
| `colormap`    |     4 |        10 |      22.697 |   90.790 |
| `linear_conv` |    15 |       910 |      18.187 |  272.798 |
| **Total**     |  **19** |         — |           — | **363.588** |

- **Solved:** 19 / 400 tasks; **local score ≈ 363.6** / 10 000.
- **Color-maps** (`task016, 276, 309, 337`) use a 10-parameter `Gather` that
  permutes/selects color channels — the cheapest correct form for a global
  color remap.
- **Linear convs** (`task053, 073, 095, 127, 147, 171, 230, 258, 266, 272, 282,
  283, 294, 317, 331`) are a single 3×3 `Conv` (+bias) whose integer weights are
  fit by a hard-margin perceptron; each is a genuinely local, linearly-separable
  rule. Memory cost is 0 (single node), so cost = params = 910.

Every network in `submission.zip` passes the official verifier on
`train + test + arc-gen`. This is a **verified baseline**, not a competitive
entry — see below for why most tasks need bespoke networks and where to go next.

## Scope & roadmap

The empirical reality (confirmed by analysis in this repo and by the public
leaderboard): only a small family of tasks is *automatically* solvable by a
single static op. Across the 400 tasks there are **no** pure identity/constant
tasks, **4** clean global color-maps, and a single linear `k×k` conv reproduces
only ~1 in 40 of the same-shape tasks. The remaining ~380 tasks need **bespoke,
multi-op networks designed per task** — which is exactly why this is a
months-long competition.

The automated solvers here therefore form a **verified baseline and a
platform**, not a leaderboard-topping entry. Natural next steps, in rough order
of expected payoff:

1. **Multi-layer conv nets (Conv→ReLU→Conv).** A single linear conv can't do
   non-linear local logic (fill 3×3 holes, denoise, dilate/erode, outline,
   local majority). Two conv layers with a ReLU can — this is the biggest lever
   for shape-preserving *local* rules, and stays cheap (memory only counts the
   one hidden activation).
2. **Size-aware geometric transforms.** Flip / rotate / transpose are 0-param in
   principle but blocked by top-left anchoring for variable sizes. Worth
   researching static constructions (e.g. per-size handling, or ops that are
   invariant to the clear border).
3. **Per-family hand-crafted solvers** for the common ARC motifs present in the
   same-shape set (borders/frames, gravity, symmetry completion, flood fill,
   connect-the-dots), each expressed with the cheapest static op sequence.
4. **Cost minimisation of solved tasks** — e.g. pruning conv kernels, preferring
   `Gather` over `Conv`, and shrinking `Constant`s.

Because every candidate is checked by the official scorer before it's accepted,
new solvers can be added incrementally with confidence that the submission stays
correct.

## Attribution

`vendor/neurogolf_utils/neurogolf_utils.py` is the official competition module
(© 2026 Google LLC, Apache-2.0); see `vendor/neurogolf_utils/README.md`. It is
used unmodified so local scores match Kaggle's.
