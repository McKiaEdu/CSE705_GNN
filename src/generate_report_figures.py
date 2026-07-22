"""Generates figures/*.pdf and tables/*.md /*.tex from the real results/ sweep.

Not a tested module component: a one-shot driver, matching run_sweep.py's
role for the sweep itself. Regenerable from results/ at any time; the README
documents this command as the way to rebuild both directories.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch

from data import LoadCora
from experiments import BuildGrid
from viz import (
    Aggregate,
    BuildTable,
    CheckCoverage,
    ExportTable,
    LoadRecords,
    PlotAccuracyVsDepth,
    PlotEmbeddingProjection,
    PlotEnergyShift,
    PlotEnergyVsLayer,
    PlotLossCurves,
    PlotMadVsDepth,
    PlotMitigationAblation,
)

FIGURES_DIR = "figures"
TABLES_DIR = "tables"


def _LoadRecordById(directory: str, runId: str) -> dict:
    path = os.path.join(directory, f"{runId}.json")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("=== loading records ===")
    mainRecords = LoadRecords("results")
    fidelityRecords = LoadRecords("results/fidelity")
    print(f"results/: {len(mainRecords)} records; results/fidelity: {len(fidelityRecords)} records")

    mainTable = BuildTable(mainRecords)
    fidelityTable = BuildTable(fidelityRecords)

    print("\n=== coverage check ===")
    expectedArmA = [c.__dict__ for c in BuildGrid("A")]
    expectedArmB = [c.__dict__ for c in BuildGrid("B")]
    expectedArmC = [c.__dict__ for c in BuildGrid("C")]
    expectedArmD = [c.__dict__ for c in BuildGrid("D", armDMitigation=["jk"])]
    for armName, expected in [("A", expectedArmA), ("B", expectedArmB), ("C", expectedArmC), ("D", expectedArmD)]:
        missing = CheckCoverage(mainTable, expected)
        print(f"arm {armName}: {len(missing)} missing" + (f": {missing}" if missing else ""))

    print("\n=== figures ===")
    PlotAccuracyVsDepth(mainTable, os.path.join(FIGURES_DIR, "accuracy_vs_depth.pdf"))
    PlotMadVsDepth(mainTable, os.path.join(FIGURES_DIR, "mad_vs_depth.pdf"))
    PlotMitigationAblation(mainTable, os.path.join(FIGURES_DIR, "mitigation_ablation.pdf"))
    print("wrote accuracy_vs_depth.pdf, mad_vs_depth.pdf, mitigation_ablation.pdf")

    energyRecords = [
        _LoadRecordById("results", f"{convType}_none_d32_s0") for convType in ("gcn", "sage", "gat")
    ]
    PlotEnergyVsLayer(energyRecords, os.path.join(FIGURES_DIR, "energy_vs_layer_depth32.pdf"))
    print("wrote energy_vs_layer_depth32.pdf")

    baselineDeep = _LoadRecordById("results", "gcn_none_d32_s0")
    mitigatedDeep = _LoadRecordById("results", "gcn_jk_d32_s0")
    PlotLossCurves(
        [baselineDeep, mitigatedDeep],
        os.path.join(FIGURES_DIR, "loss_curves_depth32.pdf"),
        labels=["gcn baseline", "gcn + jk"],
    )
    print("wrote loss_curves_depth32.pdf")

    baselineGcnTable = mainTable[(mainTable["convType"] == "gcn") & (mainTable["mitigations"].apply(len) == 0)]
    PlotEnergyShift(baselineGcnTable, os.path.join(FIGURES_DIR, "energy_shift.pdf"))
    print("wrote energy_shift.pdf")

    cora = LoadCora()
    labels = cora.y.numpy()
    embeddingPaths = {
        "depth 2": "results/embeddings/gcn_none_d2_s0_l1.pt",
        "depth 32": "results/embeddings/gcn_none_d32_s0_l31.pt",
    }
    PlotEmbeddingProjection(embeddingPaths, labels, os.path.join(FIGURES_DIR, "embedding_projection.pdf"))
    print("wrote embedding_projection.pdf")

    print("\n=== tables ===")
    accuracyAgg = Aggregate(
        mainTable[mainTable["mitigations"].apply(len) == 0], ["convType", "numLayers"]
    ).sort_values(["convType", "numLayers"])
    accuracyCols = ["convType", "numLayers", "testAccuracy_mean", "testAccuracy_std", "count"]
    ExportTable(accuracyAgg[accuracyCols], os.path.join(TABLES_DIR, "accuracy_vs_depth.md"), "md")
    ExportTable(accuracyAgg[accuracyCols], os.path.join(TABLES_DIR, "accuracy_vs_depth.tex"), "tex")

    mitigationSubset = mainTable[
        ((mainTable["convType"] == "gcn") & (mainTable["mitigations"].apply(len) > 0))
        | (mainTable["convType"] == "gcnii")
    ]
    mitigationAgg = Aggregate(mitigationSubset, ["convType", "mitigations", "numLayers"]).sort_values(
        ["convType", "mitigations", "numLayers"]
    )
    mitigationCols = ["convType", "mitigations", "numLayers", "testAccuracy_mean", "testAccuracy_std", "count"]
    ExportTable(mitigationAgg[mitigationCols], os.path.join(TABLES_DIR, "mitigation_ablation.md"), "md")
    ExportTable(mitigationAgg[mitigationCols], os.path.join(TABLES_DIR, "mitigation_ablation.tex"), "tex")

    slopeAgg = Aggregate(mainTable[mainTable["mitigations"].apply(len) == 0], ["convType", "numLayers"]).sort_values(
        ["convType", "numLayers"]
    )
    slopeCols = ["convType", "numLayers", "checkpointContractionSlope_mean", "checkpointContractionSlope_std", "count"]
    ExportTable(slopeAgg[slopeCols], os.path.join(TABLES_DIR, "contraction_slope.md"), "md")
    ExportTable(slopeAgg[slopeCols], os.path.join(TABLES_DIR, "contraction_slope.tex"), "tex")

    fidelityAgg = Aggregate(fidelityTable, ["convType", "hiddenDim", "numLayers"])
    armATwoLayerGcn = mainTable[
        (mainTable["convType"] == "gcn") & (mainTable["mitigations"].apply(len) == 0) & (mainTable["numLayers"] == 2)
    ]
    armAAgg = Aggregate(armATwoLayerGcn, ["convType", "hiddenDim", "numLayers"])
    fidelityComparison = (
        __import__("pandas")
        .concat([fidelityAgg, armAAgg])
        .sort_values("hiddenDim")[["convType", "hiddenDim", "numLayers", "testAccuracy_mean", "testAccuracy_std", "count"]]
    )
    ExportTable(fidelityComparison, os.path.join(TABLES_DIR, "fidelity_comparison.md"), "md")
    ExportTable(fidelityComparison, os.path.join(TABLES_DIR, "fidelity_comparison.tex"), "tex")

    print("wrote accuracy_vs_depth, mitigation_ablation, contraction_slope, fidelity_comparison (md + tex)")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
