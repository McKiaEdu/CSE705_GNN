# Depth, Oversmoothing, and Architecture: A Comparative Study of GNNs on Cora

Code for a comparative study of GCN, GraphSAGE, GAT, and GCNII on the Cora
citation network, measuring how representational collapse (oversmoothing)
changes with network depth (2, 4, 8, 16, 32 layers), and testing four
mitigations (residual connections, PairNorm, Jumping Knowledge, GCNII).

## Requirements

- Python 3.14 (the reported results were produced under 3.14.4; any recent
  Python 3.11+ should work, since nothing in this codebase is version-specific).
- Internet access on first run only, to download the Cora dataset. Every
  run after that uses the local cache under `data/Cora/`.

Install dependencies:

    pip install -r requirements.txt

This is the only install step. It covers the test suite and the notebook
kernel as well as the runtime, since reproducing the reported results involves
both.

All commands below are run from this directory (the repository root), not
from inside `src/`.

## Quick check: does the code run?

Before committing to anything longer, one command runs a single configuration
end to end and checks it against a known result:

    python src/smoke_test.py

This trains the depth-2 GCN at seed 0 through the same modules the full sweep
uses, then compares its test accuracy against the shipped record for that run.
It takes roughly 8 seconds on CPU and prints PASS or FAIL (exit status 0 or 1).
Nothing is written: `results/` is left untouched, so this cannot disturb the
shipped records. Pass `--save` to keep the record under `results/smoke/`, which
is a separate directory the report's aggregation does not read.

For the full test suite (120 tests, roughly two and a half minutes):

    pytest

## Reproducing the report's results

This submission ships the full set of results already produced
(`results/*.json`, 534 files, plus `results/embeddings/*.pt` for the
embedding-projection figure), so the figures and tables can be regenerated
without re-running the training sweep:

    python src/generate_report_figures.py

This reads every record under `results/`, `results/fidelity/`, and
`results/hpsearch/`, and (re)writes `figures/*.pdf` and `tables/*.md` /
`tables/*.tex`.

To reproduce the results from scratch instead (full training sweep, 534
runs across 5 depths x 4 architectures x mitigation combinations x 10
seeds), delete or rename `results/` and run:

    python src/run_sweep.py

This downloads Cora automatically on first use (via `data.LoadCora`,
cached under `data/Cora/`), then runs all six experiment arms in order
(F: hyperparameter search, A: unmitigated depth sweep, C: GCNII, E:
fidelity check, B: mitigation ablation, D: best-mitigation cross-architecture
check), writing one JSON record per run under `results/`. The sweep is
idempotent: it skips any run whose result file already exists, so it is
safe to re-run after an interruption. On CPU, the full sweep takes several
hours; `src/run_sweep.py` prints elapsed time and a per-arm progress log as
it goes.

To run the test suite:

    pytest

(`pytest.ini` points it at `src/tests/`; no extra configuration needed.)

## Code structure

    src/
    +-- data/          Cora loading and invariant checks
    +-- models/        GnnModel base class and the four architectures
    +-- metrics/        Dirichlet energy, MAD, comparable-band selection
    +-- mitigations/    residual, PairNorm, and Jumping Knowledge hooks/readout
    +-- train/          optimizer, training loop, results-record assembly
    +-- experiments/    declarative grid, model construction, filesystem writes
    +-- viz/            aggregation, plotting, table export
    +-- tests/          pytest suite (one test module per package above)
    +-- smoke_test.py           entry point: one run, verifies the pipeline in ~8s
    +-- run_sweep.py            entry point: runs the full 534-run sweep
    +-- generate_report_figures.py   entry point: rebuilds figures/ and tables/ from results/

    results/            one JSON record per run (results/, results/fidelity/,
                        results/hpsearch/), plus results/embeddings/*.pt for
                        the ten runs the embedding-projection figure uses
    requirements.txt        all dependencies, pinned (runtime, tests, notebooks)
    pytest.ini               points pytest at src/tests/

Module responsibilities, in build/dependency order:

| Module | Depends on | Owns |
|---|---|---|
| data | (none) | Loading Cora, validating its shape |
| models | data | The layer loop and the four architectures (GCN, GraphSAGE, GAT, GCNII) |
| metrics | data, models | Dirichlet energy, MAD, the comparable-layer band, the contraction slope |
| train | models, metrics | The optimizer, the training loop, assembling each run's results record |
| mitigations | models (two protocols only) | Residual, PairNorm, and Jumping Knowledge, attached by composition |
| experiments | every module above | The declarative run grid and all filesystem writes |
| viz | data, the records experiments writes | Aggregating results and generating every figure and table |

`experiments` is the only module that writes to disk; `viz` is never
imported by anything else and is the only module whose output (figures,
tables) a reader sees directly.

Two scripts sit outside this module graph as the two entry points:
`src/run_sweep.py` (produces `results/`) and
`src/generate_report_figures.py` (produces `figures/` and `tables/` from
`results/`).

## Results file naming

Each `results/*.json` file is named
`<architecture>_<mitigation>_d<depth>_s<seed>.json`, e.g.
`gcn_none_d32_s0.json` (unmitigated GCN, depth 32, seed 0) or
`gcn_pairnorm_d16_s3.json` (GCN with PairNorm, depth 16, seed 3).
`results/fidelity/` and `results/hpsearch/` follow the same convention for
their own arms.
