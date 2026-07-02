# Vendored `neurogolf_utils`

`neurogolf_utils.py` in this directory is the **official** utility module distributed
with the Kaggle competition
[*The 2026 NeuroGolf Championship*](https://www.kaggle.com/competitions/neurogolf-2026)
(dataset file `neurogolf_utils/neurogolf_utils.py`).

- **Copyright** 2026 Google LLC.
- **License:** Apache License 2.0 (see the header at the top of the file).
- **Version vendored:** the `2026-05-14` revision (per the module's `Version History`).

It is included here **unmodified** so that this repository can reproduce the
competition's *exact* scoring logic offline (parameter count, memory footprint,
banned-op checks, and functional-correctness verification). Our own code in
`src/neurogolf/scoring.py` imports the scoring/verification functions from this
module rather than re-implementing them, which guarantees our local score
matches Kaggle's.

If the organizers publish a newer revision, replace this file with the updated
version and re-run the pipeline; the `Version History` docstring inside the file
records what changed between revisions.
