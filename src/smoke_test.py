"""One-run smoke test: does the pipeline execute and reproduce a known result?

The full sweep is 534 runs and hours of compute. This runs exactly one of them,
the depth-2 GCN at seed 0, end to end through the same modules the sweep uses,
and checks the accuracy against the record shipped in results/. It answers the
question a reader has before committing to the grid: does this code run, and
does it produce the number the report claims.

Not a substitute for the test suite (`pytest`), which checks properties this
cannot: metric correctness, seeding, schema shape, collapse behavior at depth.

Nothing is written by default. `RunOne` is pure: it builds, trains, and returns
a record, while every write lives in `RunSweep`, which this does not call. The
shipped `results/` cannot be touched by running this, which matters because arm
collisions silently overwriting result files were a real defect in this project
(D-030). Passing --save writes the record to its own subdirectory instead,
following the same convention that isolates results/fidelity and
results/hpsearch.

Usage:
    python src/smoke_test.py
    python src/smoke_test.py --save    # also write results/smoke/gcn_none_d2_s0.json

Exit status is 0 when the run completes and lands in the expected accuracy
band, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import replace

from data import LoadCora
from experiments import BuildGrid, RunOne
from metrics import OversmoothingMetrics

# the shipped record this run should reproduce
REFERENCE_RECORD = os.path.join("results", "gcn_none_d2_s0.json")

# --save writes here, never into results/ itself. LoadRecords globs
# "<dir>/*.json" non-recursively, so a subdirectory is invisible to the
# aggregation that builds the report's tables and cannot contaminate them.
SMOKE_OUTPUT_DIR = os.path.join("results", "smoke")

# a band, not an equality: BLAS and library versions differ across machines, so
# an exact match is not required, but anything outside this is a real failure
# rather than numerical drift. The lower bound sits far above Cora's 31.9%
# majority-class floor, so a model that failed to train cannot pass.
ACCURACY_LOWER_BOUND = 0.78
ACCURACY_UPPER_BOUND = 0.84


def SelectSmokeConfig():
    """The depth-2, seed-0 GCN entry of arm A.

    Pulled from the real grid rather than hand-specified, so the smoke test
    cannot drift from the sweep's own hyperparameters. Embedding saving is
    turned off: that flag is set for this configuration in arm A, and writing
    .pt files is not part of what the smoke test verifies.
    """
    for config in BuildGrid("A"):
        if config.convType == "gcn" and config.numLayers == 2 and config.seed == 0:
            return replace(config, saveEmbeddings=False)
    raise SystemExit("arm A does not contain a depth-2 seed-0 GCN; the grid has changed")


def ReadReferenceAccuracy() -> float | None:
    """Test accuracy from the shipped record, or None when it is absent."""
    if not os.path.exists(REFERENCE_RECORD):
        return None
    with open(REFERENCE_RECORD) as f:
        return json.load(f)["results"]["testAccuracy"]


def main() -> int:
    print("Smoke test: one run of the depth-2 GCN at seed 0.\n")

    config = SelectSmokeConfig()
    print(
        f"  configuration  {config.convType}, {config.numLayers} layers, "
        f"hiddenDim={config.hiddenDim}, seed={config.seed}"
    )
    print(
        f"                 lr={config.learningRate}, dropout={config.dropout}, "
        f"weightDecay={config.weightDecay}, patience={config.patience}"
    )

    startTime = time.perf_counter()
    data = LoadCora()
    metricsInstrument = OversmoothingMetrics(data.edge_index, data.num_nodes)
    record = RunOne(config, data, metricsInstrument)
    elapsedSeconds = time.perf_counter() - startTime

    accuracy = record["results"]["testAccuracy"]
    macroF1 = record["results"]["testMacroF1"]
    epochsRun = record["results"]["epochsRun"]

    print(f"\n  test accuracy  {accuracy:.4f}")
    print(f"  macro-F1       {macroF1:.4f}")
    print(f"  epochs run     {epochsRun}")
    print(f"  elapsed        {elapsedSeconds:.1f}s")

    reference = ReadReferenceAccuracy()
    if reference is None:
        print(f"\n  (no shipped record at {REFERENCE_RECORD} to compare against)")
    else:
        difference = abs(accuracy - reference)
        print(f"\n  shipped record {reference:.4f}  (difference {difference:.4f})")

    if "--save" in sys.argv:
        os.makedirs(SMOKE_OUTPUT_DIR, exist_ok=True)
        outputPath = os.path.join(SMOKE_OUTPUT_DIR, f"{record['runId']}.json")
        with open(outputPath, "w") as f:
            json.dump(record, f)
        print(f"  wrote          {outputPath}")
    else:
        print("\n  nothing written; results/ is untouched (--save to keep the record)")

    isInBand = ACCURACY_LOWER_BOUND <= accuracy <= ACCURACY_UPPER_BOUND
    if not isInBand:
        print(
            f"\nFAIL: accuracy {accuracy:.4f} is outside the expected band "
            f"[{ACCURACY_LOWER_BOUND}, {ACCURACY_UPPER_BOUND}]."
        )
        return 1

    print("\nPASS: the pipeline runs and reproduces the published two-layer baseline.")
    print("Next: `pytest` for the full test suite, or see README.txt to rerun the sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
