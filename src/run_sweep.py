"""Top-level orchestration driver for the full 534-run sweep.

Not a tested module component -- the "notebook or top-level script" experiments_spec.md's
Approach section anticipates calling RunSweep per arm (D-040). Ordering per
experiments_spec.md: F first (sets the shared hyperparameter defaults, D-029),
then A, C, E (independent of B), then B, then D once B is aggregated (D-041).

The arm-F and arm-B aggregation logic here is intentionally minimal -- just
enough to make the two sweep-ordering decisions (D-029's three-level rule,
D-041's depth-32 mean test accuracy rule). It is not viz/'s aggregation layer,
which does not exist yet and will supersede this for report figures.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import experiments.grid as grid_module
from experiments import BuildGrid, RunSweep
from experiments.runner import ResultPath

RESULTS_DIR = "results"
FIDELITY_DIR = os.path.join(RESULTS_DIR, "fidelity")
HPSEARCH_DIR = os.path.join(RESULTS_DIR, "hpsearch")

PUBLISHED_LR = 0.01
PUBLISHED_DROPOUT = 0.5
PUBLISHED_WEIGHT_DECAY = 5e-4


def _LoadRecords(directory: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if path.endswith(".failed.json"):
            continue
        with open(path) as f:
            records.append(json.load(f))
    return records


def _SelectArmFWinner() -> tuple[float, float, float]:
    """D-029's three-level rule: highest mean val accuracy; tie-break (within
    one std of the top) lower mean val loss; final fallback closest to the
    published baseline."""
    records = _LoadRecords(HPSEARCH_DIR)
    byCombo: dict[tuple[float, float, float], list[dict]] = {}
    for record in records:
        cfg = record["config"]
        key = (cfg["learningRate"], cfg["dropout"], cfg["weightDecay"])
        byCombo.setdefault(key, []).append(record)

    stats = []
    for combo, group in byCombo.items():
        valAccuracies = [r["results"]["valAccuracy"] for r in group]
        valLosses = [r["results"]["valLoss"] for r in group]
        meanAcc = statistics.mean(valAccuracies)
        stdAcc = statistics.stdev(valAccuracies) if len(valAccuracies) > 1 else 0.0
        meanLoss = statistics.mean(valLosses)
        stats.append({"combo": combo, "meanAcc": meanAcc, "stdAcc": stdAcc, "meanLoss": meanLoss})

    stats.sort(key=lambda s: s["meanAcc"], reverse=True)
    top = stats[0]
    tied = [s for s in stats if s["meanAcc"] >= top["meanAcc"] - top["stdAcc"]]
    print(f"arm F: {len(tied)}/{len(stats)} combos within 1 std of the top mean val accuracy ({top['meanAcc']:.4f})")

    tied.sort(key=lambda s: s["meanLoss"])
    bestLoss = tied[0]["meanLoss"]
    stillTied = [s for s in tied if s["meanLoss"] == bestLoss]

    if len(stillTied) > 1:
        stillTied.sort(
            key=lambda s: (
                abs(s["combo"][0] - PUBLISHED_LR)
                + abs(s["combo"][1] - PUBLISHED_DROPOUT)
                + abs(s["combo"][2] - PUBLISHED_WEIGHT_DECAY)
            )
        )

    winner = stillTied[0]["combo"]
    print(f"arm F winner: learningRate={winner[0]}, dropout={winner[1]}, weightDecay={winner[2]}")
    for s in stats:
        print(f"  combo={s['combo']} meanValAcc={s['meanAcc']:.4f} stdValAcc={s['stdAcc']:.4f} meanValLoss={s['meanLoss']:.4f}")
    return winner


def _SelectArmDMitigation() -> list[str]:
    """D-041: highest mean test accuracy at depth=32 among arm B's four arms."""
    records = [r for r in _LoadRecords(RESULTS_DIR) if r["config"]["numLayers"] == 32 and r["config"]["mitigations"]]
    byMitigation: dict[tuple[str, ...], list[float]] = {}
    for record in records:
        key = tuple(sorted(record["config"]["mitigations"]))
        byMitigation.setdefault(key, []).append(record["results"]["testAccuracy"])

    ranked = sorted(byMitigation.items(), key=lambda kv: statistics.mean(kv[1]), reverse=True)
    print("arm D selection (depth=32 mean test accuracy):")
    for mitigation, accuracies in ranked:
        print(f"  {mitigation}: mean={statistics.mean(accuracies):.4f} n={len(accuracies)}")

    winner = list(ranked[0][0])
    print(f"arm D winner mitigation: {winner}")
    return winner


def main() -> None:
    startTime = time.time()

    print("=== grid sanity check ===")
    for arm, expected in [("A", 150), ("B", 200), ("C", 50), ("E", 10), ("F", 24)]:
        count = len(BuildGrid(arm))
        assert count == expected, f"arm {arm}: expected {expected}, got {count}"
        print(f"arm {arm}: {count} configs OK")

    print("\n=== arm F: hyperparameter search (24 runs) ===")
    RunSweep(BuildGrid("F"), HPSEARCH_DIR, includeHyperparamsInPath=True)

    print("\n=== aggregating arm F ===")
    winningLr, winningDropout, winningWeightDecay = _SelectArmFWinner()
    grid_module.DEFAULT_LEARNING_RATE = winningLr
    grid_module.DEFAULT_DROPOUT = winningDropout
    grid_module.DEFAULT_WEIGHT_DECAY = winningWeightDecay

    print("\n=== arm A: depth sweep, unmitigated (150 runs) ===")
    RunSweep(BuildGrid("A"), RESULTS_DIR)

    print("\n=== arm C: GCNII across depth (50 runs) ===")
    RunSweep(BuildGrid("C"), RESULTS_DIR)

    print("\n=== arm E: fidelity arm (10 runs) ===")
    RunSweep(BuildGrid("E"), FIDELITY_DIR)

    print("\n=== arm B: mitigation ablation on GCN (200 runs) ===")
    RunSweep(BuildGrid("B"), RESULTS_DIR)

    print("\n=== aggregating arm B ===")
    armDMitigation = _SelectArmDMitigation()

    print("\n=== arm D: best mitigation across architectures (100 runs) ===")
    RunSweep(BuildGrid("D", armDMitigation=armDMitigation), RESULTS_DIR)

    elapsed = time.time() - startTime
    print(f"\n=== sweep complete in {elapsed / 3600:.2f} hours ===")


if __name__ == "__main__":
    main()
