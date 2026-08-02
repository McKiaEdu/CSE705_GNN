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

import pandas as pd
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
    PlotArmDDepthCurve,
    PlotMitigationAblation,
)

FIGURES_DIR = "figures"
TABLES_DIR = "tables"

# depth-2 Cora accuracies as published, entered from the papers rather than
# recomputed, so Section 6.1 compares against a figure with a citation attached
PUBLISHED_DEPTH2_ACCURACY: dict[str, dict[str, object]] = {
    "gcn": {"published": 81.5, "publishedStd": None, "source": "kipf2017semi"},
    "sage": {"published": None, "publishedStd": None, "source": "hamilton2017inductive"},
    "gat": {"published": 83.0, "publishedStd": 0.7, "source": "velickovic2018graph"},
}

ARCHITECTURE_LABELS: dict[str, str] = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}

SWEEP_DEPTHS: tuple[int, ...] = (2, 4, 8, 16, 32)


def _LoadRecordById(directory: str, runId: str) -> dict:
    path = os.path.join(directory, f"{runId}.json")
    with open(path) as f:
        return json.load(f)


def _UnmitigatedSubset(table: pd.DataFrame, convType: str) -> pd.DataFrame:
    return table[(table["convType"] == convType) & (table["mitigations"].apply(len) == 0)]


def _MitigatedSubset(table: pd.DataFrame, convType: str, mitigation: str) -> pd.DataFrame:
    matchesMitigation = table["mitigations"].apply(lambda m: tuple(m) == (mitigation,))
    return table[(table["convType"] == convType) & matchesMitigation]


def BuildBaselineComparison(table: pd.DataFrame) -> pd.DataFrame:
    """Depth-2 measured accuracy per architecture against its own paper's figure.

    Arm A's depth-2 subset already holds every measured number, so this reads
    the same records the rest of Section 6 does rather than requiring new runs.
    The published column is a literal from PUBLISHED_DEPTH2_ACCURACY.
    """
    rows: list[dict[str, object]] = []
    for convType, label in ARCHITECTURE_LABELS.items():
        depth2 = _UnmitigatedSubset(table, convType)
        depth2 = depth2[depth2["numLayers"] == 2]
        measuredMean = 100.0 * depth2["testAccuracy"].mean()
        measuredStd = 100.0 * depth2["testAccuracy"].std()
        reference = PUBLISHED_DEPTH2_ACCURACY[convType]
        publishedMean = reference["published"]
        # GraphSAGE has no published Cora figure to compare against, so the
        # difference column stays empty rather than being filled with a zero
        difference = None if publishedMean is None else measuredMean - float(publishedMean)
        rows.append(
            {
                "architecture": label,
                "measured (%)": round(measuredMean, 2),
                "std": round(measuredStd, 2),
                "published (%)": publishedMean,
                "published std": reference["publishedStd"],
                "difference (pp)": None if difference is None else round(difference, 2),
                "count": int(len(depth2)),
            }
        )
    return pd.DataFrame(rows)


def BuildArmDDepth32(table: pd.DataFrame) -> pd.DataFrame:
    """Depth-32 unmitigated against +JK, per architecture.

    GCN's mitigated row comes from arm B and GraphSAGE's and GAT's from arm D;
    all three are the same `jk` mitigation, so they tabulate together.
    """
    rows: list[dict[str, object]] = []
    for convType, label in ARCHITECTURE_LABELS.items():
        unmitigated = _UnmitigatedSubset(table, convType)
        unmitigated = unmitigated[unmitigated["numLayers"] == 32]
        mitigated = _MitigatedSubset(table, convType, "jk")
        mitigated = mitigated[mitigated["numLayers"] == 32]
        unmitigatedMean = 100.0 * unmitigated["testAccuracy"].mean()
        mitigatedMean = 100.0 * mitigated["testAccuracy"].mean()
        rows.append(
            {
                "architecture": label,
                "unmitigated (%)": round(unmitigatedMean, 2),
                "unmitigated std": round(100.0 * unmitigated["testAccuracy"].std(), 2),
                "+JK (%)": round(mitigatedMean, 2),
                "+JK std": round(100.0 * mitigated["testAccuracy"].std(), 2),
                "difference (pp)": round(mitigatedMean - unmitigatedMean, 2),
                "unmitigated macro-F1": round(unmitigated["testMacroF1"].mean(), 4),
                "+JK macro-F1": round(mitigated["testMacroF1"].mean(), 4),
                "count": int(len(mitigated)),
            }
        )
    return pd.DataFrame(rows)


def BuildArmDDepthCurve(table: pd.DataFrame) -> pd.DataFrame:
    """Test accuracy against depth for +JK and unmitigated, all three architectures.

    Reports the whole sweep rather than depth 32 alone, since the finding is
    that GraphSAGE+JK falls with depth where GCN+JK and GAT+JK hold.
    """
    rows: list[dict[str, object]] = []
    for convType, label in ARCHITECTURE_LABELS.items():
        for depth in SWEEP_DEPTHS:
            unmitigated = _UnmitigatedSubset(table, convType)
            unmitigated = unmitigated[unmitigated["numLayers"] == depth]
            mitigated = _MitigatedSubset(table, convType, "jk")
            mitigated = mitigated[mitigated["numLayers"] == depth]
            rows.append(
                {
                    "architecture": label,
                    "depth": depth,
                    "unmitigated (%)": round(100.0 * unmitigated["testAccuracy"].mean(), 2),
                    "unmitigated std": round(100.0 * unmitigated["testAccuracy"].std(), 2),
                    "+JK (%)": round(100.0 * mitigated["testAccuracy"].mean(), 2),
                    "+JK std": round(100.0 * mitigated["testAccuracy"].std(), 2),
                    "+JK macro-F1": round(mitigated["testMacroF1"].mean(), 4),
                    "count": int(len(mitigated)),
                }
            )
    return pd.DataFrame(rows)


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
    PlotArmDDepthCurve(mainTable, os.path.join(FIGURES_DIR, "armd_depth_curve.pdf"))
    print("wrote accuracy_vs_depth.pdf, mad_vs_depth.pdf, mitigation_ablation.pdf,")
    print("      armd_depth_curve.pdf")

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
        pd.concat([fidelityAgg, armAAgg])
        .sort_values("hiddenDim")[["convType", "hiddenDim", "numLayers", "testAccuracy_mean", "testAccuracy_std", "count"]]
    )
    ExportTable(fidelityComparison, os.path.join(TABLES_DIR, "fidelity_comparison.md"), "md")
    ExportTable(fidelityComparison, os.path.join(TABLES_DIR, "fidelity_comparison.tex"), "tex")

    baselineComparison = BuildBaselineComparison(mainTable)
    ExportTable(baselineComparison, os.path.join(TABLES_DIR, "baseline_comparison.md"), "md")
    ExportTable(baselineComparison, os.path.join(TABLES_DIR, "baseline_comparison.tex"), "tex")

    armDDepth32 = BuildArmDDepth32(mainTable)
    ExportTable(armDDepth32, os.path.join(TABLES_DIR, "armd_depth32.md"), "md")
    ExportTable(armDDepth32, os.path.join(TABLES_DIR, "armd_depth32.tex"), "tex")

    armDDepthCurve = BuildArmDDepthCurve(mainTable)
    ExportTable(armDDepthCurve, os.path.join(TABLES_DIR, "armd_depth_curve.md"), "md")
    ExportTable(armDDepthCurve, os.path.join(TABLES_DIR, "armd_depth_curve.tex"), "tex")

    hpsearchTable = BuildTable(LoadRecords("results/hpsearch"))
    hpsearchAgg = Aggregate(hpsearchTable, ["learningRate", "dropout", "weightDecay"]).sort_values(
        "valAccuracy_mean", ascending=False
    )
    hpsearchSummaryCols = ["learningRate", "dropout", "weightDecay", "valAccuracy_mean", "valAccuracy_std"]
    ExportTable(hpsearchAgg[hpsearchSummaryCols], os.path.join(TABLES_DIR, "hpsearch_summary.md"), "md")
    ExportTable(hpsearchAgg[hpsearchSummaryCols], os.path.join(TABLES_DIR, "hpsearch_summary.tex"), "tex")

    print("wrote accuracy_vs_depth, mitigation_ablation, contraction_slope, fidelity_comparison,")
    print("      baseline_comparison, armd_depth32, armd_depth_curve, hpsearch_summary (md + tex)")

    print("\n=== appendix tables ===")
    hpsearchCols = ["learningRate", "dropout", "weightDecay", "seed", "valAccuracy", "valLoss"]
    hpsearchFull = hpsearchTable[hpsearchCols].sort_values(
        ["learningRate", "dropout", "weightDecay", "seed"]
    )
    ExportTable(hpsearchFull, os.path.join(TABLES_DIR, "hpsearch_per_seed.md"), "md")
    ExportTable(hpsearchFull, os.path.join(TABLES_DIR, "hpsearch_per_seed.tex"), "tex")

    depthSweepCols = ["convType", "numLayers", "seed", "testAccuracy", "testMacroF1"]
    depthSweepFull = mainTable[
        (mainTable["mitigations"].apply(len) == 0) & (mainTable["convType"].isin(["gcn", "sage", "gat"]))
    ][depthSweepCols].sort_values(["convType", "numLayers", "seed"])
    ExportTable(depthSweepFull, os.path.join(TABLES_DIR, "depth_sweep_per_seed.md"), "md")
    ExportTable(depthSweepFull, os.path.join(TABLES_DIR, "depth_sweep_per_seed.tex"), "tex")

    # matches metrics.oversmoothing._MIN_ENERGY_FOR_LOG, the floor FitContractionSlope applies
    energyFloor = 1e-12
    perLayerRows = []
    for seed in range(10):
        record = _LoadRecordById("results", f"gcn_none_d32_s{seed}")
        bandIndices = record["bandIndices"]
        energies = record["checkpointMetrics"]["dirichletEnergy"]
        for layer in bandIndices:
            energy = energies[layer]
            perLayerRows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "dirichletEnergy": energy,
                    "belowFloor": energy < energyFloor,
                }
            )
    perLayerTable = pd.DataFrame(perLayerRows)
    # scientific notation: dirichletEnergy spans ~30 orders of magnitude in
    # this table (Section 6.5), so a fixed decimal count would round every
    # collapsed layer to "0.0000" and erase the thing the table shows
    ExportTable(
        perLayerTable, os.path.join(TABLES_DIR, "depth32_gcn_per_layer_energy.md"), "md", floatFormat=".2e"
    )
    ExportTable(
        perLayerTable, os.path.join(TABLES_DIR, "depth32_gcn_per_layer_energy.tex"), "tex", floatFormat=".2e"
    )

    print("wrote hpsearch_per_seed, depth_sweep_per_seed, depth32_gcn_per_layer_energy (md + tex)")

    belowFloorRows = []
    for seed, group in perLayerTable.groupby("seed"):
        atOrAboveFloor = group.loc[~group["belowFloor"], "layer"]
        energyAtLayer1 = group.loc[group["layer"] == 1, "dirichletEnergy"].iloc[0]
        belowFloorRows.append(
            {
                "seed": seed,
                "below-floor layers (of 31)": int(group["belowFloor"].sum()),
                "first layer at or above floor": int(atOrAboveFloor.min()),
                "E_1": energyAtLayer1,
            }
        )
    belowFloorSummary = pd.DataFrame(belowFloorRows).sort_values("seed")
    ExportTable(
        belowFloorSummary, os.path.join(TABLES_DIR, "depth32_gcn_below_floor_summary.md"), "md", floatFormat=".1e"
    )
    ExportTable(
        belowFloorSummary, os.path.join(TABLES_DIR, "depth32_gcn_below_floor_summary.tex"), "tex", floatFormat=".1e"
    )
    print("wrote depth32_gcn_below_floor_summary (md + tex)")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
