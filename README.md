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
  builders.py  # ONNX graph constructors (see solver table below)
  scoring.py   # faithful offline scoring — reuses vendored official neurogolf_utils
  solvers.py   # pluggable solver registry (nine families, see below)
  pipeline.py  # run solvers over tasks, keep cheapest *verified* net, package zip
scripts/
  build_submission.py   # CLI: solve -> write ONNX -> submission.zip
  merge_shards.py       # combine parallel shard runs into one submission
vendor/neurogolf_utils/ # official competition module (Apache-2.0), used for scoring
tests/                  # builder/data unit tests + solver integration tests
```

### Solver families

| Solver | Applies to | Network | Typical cost |
|---|---|---|---|
| `identity` | output == input | `Identity` | 0 |
| `colormap` | global color remap | `Gather` over channels | 10 |
| `fixed_crop` | output = fixed input window | 2 × `Pad` (crop + re-pad) | mem only |
| `linear_conv` | local linearly separable rules | 1 × `Conv` (k ≤ 7, +bias) | 100–4 910 |
| `conv2` | non-linear local rules | `Conv→ReLU→Conv` (GD-fit) | ~1–15k |
| `cellmap` | fixed dims, per-cell source+colormap (merges OK) | `GatherND`(+`Add`) → `Pad` | ~1–60k |
| `upscale` | pure pixel magnification | `Pad` crop + grouped `ConvTranspose` | ~9k |
| `const_shape` | fixed small output, any input size | `Reduce*` → `MatMul` head | ~5–230k |
| `flat_head` | fixed dims, whole-grid linear rules | flatten → `MatMul` head | ~19–300k |

Solvers only *propose* candidates; the pipeline scores each with the official
scorer and keeps the cheapest correct one, so overlapping families are safe.

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

## Results

Running the full pipeline over all 400 tasks (four parallel shards merged with
`scripts/merge_shards.py`), scored by the vendored official scorer:

| Solver        | Tasks | Points |
|---------------|------:|-------:|
| `linear_conv` |    30 | 537.82 |
| `flat_head`   |    32 | 447.22 |
| `cellmap`     |    19 | 316.43 |
| `const_shape` |    12 | 178.71 |
| `colormap`    |     4 |  90.79 |
| `fixed_crop`  |     2 |  39.04 |
| `upscale`     |     2 |  32.57 |
| **Total**     | **101** | **1642.59** |

- **Solved: 101 / 400 tasks, local score ≈ 1642.6** / 10 000 (per-task results
  in `results/baseline_manifest.json`).
- Every network in `submission.zip` passes the official verifier on
  `train + test + arc-gen`; the private hold-out remains the usual caveat.
- `conv2` (Conv→ReLU→Conv, gradient-fit) solved **zero** tasks in this sweep —
  its GD fitting rarely reaches *exact* solutions. Improving that fit (integer
  rounding, better initialisation, logic-synthesis instead of GD) is the
  clearest next lever on the ~160 unsolved same-shape tasks.

## Scope & roadmap

The empirical reality (confirmed by analysis in this repo and by the public
leaderboard): across the 400 tasks there are **no** pure identity/constant
tasks and only a handful of single-op transforms. 190 tasks have fixed
input/output dims (where `cellmap`/`flat_head` operate); the rest mix variable
sizes with non-linear logic. The ~300 unsolved tasks need **bespoke, multi-op
networks designed per task** — which is exactly why this is a months-long
competition.

Next steps, in rough order of expected payoff:

1. **Make `conv2` actually converge.** Gradient descent with a margin loss
   solved zero tasks; exact solutions likely need integer weight search,
   boolean-logic synthesis over one-hot channels, or perceptron-style layerwise
   fitting. ~160 same-shape tasks are the prize.
2. **Richer fixed-dims heads.** `flat_head` is linear; adding a hidden ReLU
   layer (fit per output cell) would capture non-linear whole-grid rules while
   fixed dims keep everything static.
3. **Cost-golf solved tasks** — prune zero weight columns (feature `Gather`),
   quantize `flat_head` heads, replace `MatMul` heads with sparse ops. Each
   halving of cost is +0.69 pts/task.
4. **Size-aware geometric transforms** for variable-size tasks (flip/rotate
   blocked by top-left anchoring) — needs clever static constructions.
5. **Per-family hand-crafted solvers** for common ARC motifs (borders/frames,
   gravity, symmetry completion, flood fill).

Because every candidate is checked by the official scorer before it's accepted,
new solvers can be added incrementally with confidence that the submission stays
correct.

## Attribution

`vendor/neurogolf_utils/neurogolf_utils.py` is the official competition module
(© 2026 Google LLC, Apache-2.0); see `vendor/neurogolf_utils/README.md`. It is
used unmodified so local scores match Kaggle's.
