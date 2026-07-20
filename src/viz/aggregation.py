"""LoadRecords, BuildTable, Aggregate, CheckCoverage, EnergyCurve.

The aggregation layer, strictly separated from plotting (D-033): returns tables,
never touches matplotlib. Mean/std are computed at read time and never written
back to results/ (C3).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Sequence

import pandas as pd

C3_TOP_LEVEL_KEYS = {
    "runId",
    "timestamp",
    "config",
    "bandIndices",
    "results",
    "trainingCurve",
    "epoch0Metrics",
    "checkpointMetrics",
    "finalMetrics",
    "trajectory",
    "environment",
}

CAPTURE_BLOCKS = ("epoch0Metrics", "checkpointMetrics", "finalMetrics")


def LoadRecords(resultsDir: str) -> list[dict]:
    """Reads and JSON-parses every *.json in resultsDir (non-recursive), skipping
    D-039 failure markers (*.failed.json -- not a C3 record), and raises on a
    file that IS meant to be a record but is missing required keys, rather than
    skipping it silently."""
    records = []
    for path in sorted(glob.glob(os.path.join(resultsDir, "*.json"))):
        if path.endswith(".failed.json"):
            continue
        with open(path) as f:
            record = json.load(f)
        missingKeys = C3_TOP_LEVEL_KEYS - record.keys()
        if missingKeys:
            raise ValueError(f"{path}: missing required C3 keys {sorted(missingKeys)}")
        records.append(record)
    return records


def BuildTable(records: list[dict]) -> pd.DataFrame:
    """One tidy row per run: config, results, and per-capture scalars (D-013's
    frobeniusSquared aside, everything a headline figure needs without going
    back to the raw per-layer arrays -- those are read via EnergyCurve).
    `mitigations` is stored as a tuple, not a list, so it stays hashable and
    group-by-able.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = dict(record["config"])
        row["mitigations"] = tuple(sorted(row["mitigations"]))
        row["runId"] = record["runId"]
        row.update(record["results"])
        for capture in CAPTURE_BLOCKS:
            block = record[capture]
            bandIndices = record["bandIndices"]
            prefix = capture[: -len("Metrics")]  # "epoch0" / "checkpoint" / "final"
            row[f"{prefix}ContractionSlope"] = block["contractionSlope"]
            row[f"{prefix}MadAtLastBand"] = block["mad"][bandIndices[-1]]
            row[f"{prefix}EnergyAtLastBand"] = block["dirichletEnergy"][bandIndices[-1]]
        rows.append(row)
    return pd.DataFrame(rows)


def Aggregate(table: pd.DataFrame, groupBy: list[str]) -> pd.DataFrame:
    """Mean, standard deviation, and count per group, over every numeric column
    not itself a group-by key. count is not decorative -- CheckCoverage is the
    guard, but a reader of this table alone can already see a short group."""
    grouped = table.groupby(groupBy)
    numericColumns = [c for c in table.select_dtypes(include="number").columns if c not in groupBy]
    means = grouped[numericColumns].mean().add_suffix("_mean")
    stds = grouped[numericColumns].std().add_suffix("_std")
    counts = grouped.size().rename("count")
    return pd.concat([means, stds, counts], axis=1).reset_index()


def CheckCoverage(table: pd.DataFrame, expected: Sequence[dict]) -> list[str]:
    """Configurations present in `expected` (identity: convType, mitigations,
    numLayers, seed) but absent from `table`. `expected` is a plain sequence of
    dicts, not a RunConfig -- viz must not import experiments/ (dependency
    direction), so the caller (which does have BuildGrid) supplies identities
    generically.
    """
    identityColumns = ["convType", "mitigations", "numLayers", "seed"]

    def _Identity(source: dict) -> tuple:
        return tuple(
            tuple(sorted(source[col])) if col == "mitigations" else source[col] for col in identityColumns
        )

    present = {_Identity(row) for row in table[identityColumns].to_dict("records")}
    missing = []
    for item in expected:
        identity = _Identity(item)
        if identity not in present:
            convType, mitigations, numLayers, seed = identity
            mitigationsStem = "+".join(mitigations) if mitigations else "none"
            missing.append(f"{convType}_{mitigationsStem}_d{numLayers}_s{seed}")
    return missing


def EnergyCurve(record: dict, capture: str) -> tuple[list[int], list[float]]:
    """(bandIndices, normalizedEnergy) for one record: per-dimension Dirichlet
    energy restricted to the band, normalized at the first band index (D-002 as
    amended). Values that come out nan/inf (e.g. a zero reference energy) are
    returned untouched -- a collapsed run shows a gap on the plot, not a
    fabricated point.
    """
    bandIndices = record["bandIndices"]
    energies = record[capture]["dirichletEnergy"]
    referenceEnergy = energies[bandIndices[0]]
    normalized = [energies[l] / referenceEnergy for l in bandIndices]
    return bandIndices, normalized
