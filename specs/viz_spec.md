# viz — aggregation, report figures, and embedding projection

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed   <!-- "defended" = passed the walk-through -->

## Purpose

Turns `results/*.json` into the figures and tables the report and the article use. Reads
records, aggregates across seeds, and renders. It is the last component in the build
order and the only one that produces anything a reader sees directly.

Realizes D-031 (persisted embeddings for designated runs), D-032 (t-SNE only), and
D-033 (aggregation separated from plotting).

## Approach (agreed)

**Two layers, strictly separated: aggregation returns tables, plotting consumes them.**
`LoadRecords` reads the JSON files, `BuildTable` flattens them to one tidy row per run,
`Aggregate` groups and computes mean and standard deviation, and only then does a
`Plot*` function touch matplotlib.

The separation is not stylistic. The report needs *numbers in tables* as much as it needs
figures — the results section quotes accuracies with standard deviations, and the
mitigation comparison is more readable as a table than as five overlaid curves. It also
means the defense gate can inspect an aggregate without rendering anything, and a figure
that looks wrong can be diagnosed by reading the table it came from.

**Mean and standard deviation are computed at read time and never stored** (C3). A
records directory is the single source of truth; no derived file sits beside it that can
go stale.

**The comparable band is read from `bandIndices`, never recomputed.** `metrics` derived
it from tensor widths (D-001 C1), and it differs between `LastLayerReadout` runs
(`1..L-1`) and JK runs (`1..L`). A plotting function that assumed `1..L-1` would silently
truncate every JK curve by one layer — the exact error the derived band exists to
prevent.

**The normalized energy curve is formed here, not stored.** Records hold raw
per-dimension energy; `viz` computes `(E_l / dim h_l) / (E_1 / dim h_1)` over the band per
D-002 as amended. Keeping the ratio out of the record means the reference layer can be
changed without a rerun.

**t-SNE only; UMAP is not used.** The proposal says "t-SNE / UMAP," which is an either/or
rather than a requirement for both. UMAP adds `umap-learn` and `numba` for a second view
of the same embeddings, and two projections that disagree slightly is a question to answer
in the video for no analytical gain. One method, with `random_state` fixed and perplexity
stated, is more defensible than two that cannot both be fully explained.

**matplotlib only, sized for single-column IEEE from the start.** The article is five
pages, so every figure must be legible at roughly 3.5 inches wide. That is a constraint on
font size, tick density, and how many series one axes can carry — decided now rather than
discovered during layout. No seaborn: it would add a dependency to change defaults that
can be set directly.

## Interface & contract touchpoints

Functions/methods use PascalCase; variables use camelCase; PyG `Data` fields keep their
library names (`edge_index`, `train_mask`).

- `LoadRecords(resultsDir: str) -> list[dict]` — defined here; reads and JSON-parses
  every record, validating each against the C3 key set and raising on a malformed file
  rather than skipping it silently.
- `BuildTable(records: list[dict]) -> DataFrame` — defined here; one tidy row per run,
  columns flattened from `config` and `results` plus `contractionSlope` per capture.
- `Aggregate(table: DataFrame, groupBy: list[str]) -> DataFrame` — defined here; returns
  mean, standard deviation, and count per group. The count column is not decorative: it
  is how a missing run is detected.
- `EnergyCurve(record: dict, capture: str) -> tuple[list[int], list[float]]` — defined
  here; returns `(bandIndices, normalizedEnergy)` for one record, normalizing at the
  first band index.
- `PlotAccuracyVsDepth`, `PlotEnergyVsLayer`, `PlotMadVsDepth`,
  `PlotMitigationAblation`, `PlotLossCurves`, `PlotEnergyShift`,
  `PlotEmbeddingProjection` — defined here; each takes a table (or, for the last, a
  loaded tensor) and an output path, and writes one figure file.
- `ExportTable(table: DataFrame, path: str, fmt: str) -> None` — defined here; writes
  markdown or LaTeX for direct inclusion in the report.
- consumes: `results/*.json` per C3; `results/embeddings/*.pt` for the designated figure
  runs per D-031; `data.y` from `data/` for colouring the projection.
- produces: figure files under `figures/` and table files under `tables/`. Nothing
  downstream consumes them programmatically.

Contract properties this component must honor:

- `bandIndices` is read from the record, never recomputed.
- Aggregates are computed at read time and never written back into `results/`.
- `viz` imports `data/` only for labels. It does not import `models/`, `train/`, or
  `experiments/`, and never constructs a model.

## Implementation plan

- `LoadRecords` / `BuildTable` / `Aggregate` — defined here; reused by every plot
  function and by the report's table generation.
- `EnergyCurve` — defined here. Reads `dirichletEnergy` and `bandIndices` from the named
  capture block, restricts to the band, divides by the value at the first band index.
  Returns `nan` values untouched rather than dropping them, so a collapsed run shows a
  gap instead of a fabricated point.
- Figure functions, one per report figure:
  1. `PlotAccuracyVsDepth` — arm A, grouped `(convType, numLayers)`, three series with
     error bars. The study's headline.
  2. `PlotEnergyVsLayer` — normalized energy against layer index, log y-axis, one panel
     per depth or one series per architecture at fixed depth. Reads
     `checkpointMetrics`.
  3. `PlotMadVsDepth` — MAD at the last band index, grouped `(convType, numLayers)`.
  4. `PlotMitigationAblation` — arm B plus arm C (GCNII), accuracy against depth, one
     series per mitigation arm. Paired with a table, since five series on one axes at
     single-column width is at the edge of legible.
  5. `PlotLossCurves` — `trainingCurve` at depth 32, baseline versus mitigated. This is
     the D-005 diagnostic: it is what distinguishes "never trained" from "trained then
     oversmoothed."
  6. `PlotEnergyShift` — `epoch0Metrics` versus `checkpointMetrics` versus
     `finalMetrics`, three points per depth. Tests the recorded prediction in C5.
  7. `PlotEmbeddingProjection` — t-SNE of the saved embeddings, coloured by `data.y`,
     shallow versus deep side by side. The proposal's qualitative analysis.
- `PlotEmbeddingProjection` detail: `sklearn.manifold.TSNE` with `random_state=0`,
  `perplexity=30`, both recorded in the figure caption. The projection is fitted
  separately per panel — a shared fit across depths would impose one neighbourhood
  structure on both and manufacture the very difference the figure is meant to show.
- `CheckCoverage(table, expected) -> list[str]` — defined here; returns the
  configurations present in the grid but absent from `results/`. Called before any figure
  is produced, so a hole in the sweep surfaces as a message rather than as a curve that
  quietly averages over fewer seeds.

Vectorization is not a concern here; the work is table operations and rendering.

## Dependencies

- Depends on: `data/` (labels only), `results/*.json`, `matplotlib`, `pandas`,
  `scikit-learn` (t-SNE), `torch` (loading saved embeddings).
- **Dependency scope note.** D-021 declares `scikit-learn` test-only, and `data_spec` and
  `metrics_spec` declare no scikit-learn dependency. Those statements are scoped to the
  numerical modules. `viz` uses `scikit-learn` and `pandas` at runtime; `data`, `models`,
  `metrics`, `train`, and `experiments` remain free of both. `requirements.txt` records
  them as viz-scope, so the reproducibility-critical path stays minimal.
- Consumed by: nothing. This is the leaf of the dependency graph.
- Build order: last.

## Assumptions & constraints

Every item here is provisional until confirmed.

- Python / version: Python 3.12.13.
- Libraries / role: `matplotlib` (rendering), `pandas` (tables), `scikit-learn`
  (t-SNE only), `torch` (loading `.pt` embeddings).
- Rendering: the `Agg` backend, so figures render headless and reproducibly. No
  interactive display anywhere in the module.
- Figure sizing: 3.5 inches wide by default, matching IEEE single-column. Font sizes set
  explicitly rather than inherited from the matplotlib default, which is tuned for a
  larger canvas.
- Confirmed assumptions:
  - Aggregation groups only within one hyperparameter configuration; the coordinated
    search (D-029) means there is exactly one, but the grouping asserts it rather than
    assuming it.
  - Every figure names which capture block it reads, in the caption.
  - Colour choices are colourblind-safe and the figures remain readable in greyscale,
    since printed submission is possible.

## Outputs & artifacts

- `figures/*.pdf` — vector format, so the article can scale them without resampling.
  One file per figure function.
- `tables/*.md` and `tables/*.tex` — the accuracy table, the mitigation ablation table,
  the contraction-slope table, and the fidelity comparison (arm E versus arm A at
  depth 2).
- Both directories are regenerable from `results/` and are gitignored except for a
  `.gitkeep`; the README documents the command that rebuilds them.

## Test plan

- **Aggregation correctness:** on synthetic records with known values, assert `Aggregate`
  returns the expected mean, standard deviation, and count per group.
- **Coverage detection:** delete one record from a synthetic results directory and assert
  `CheckCoverage` names exactly that configuration. This is the guard against a figure
  silently averaging nine seeds and reporting it as ten.
- **Band is read, not recomputed:** construct a synthetic JK record whose `bandIndices`
  is `1..L` and assert `EnergyCurve` returns L points, not L-1. The regression test for
  the JK truncation failure.
- **Normalization reference:** assert the first value of every normalized energy curve is
  exactly 1.0.
- **`nan` propagation:** a record whose `contractionSlope` is `nan` (the L=2
  `LastLayerReadout` case) produces a gap in the slope plot, not a zero.
- **t-SNE determinism:** two runs with `random_state=0` on identical input produce
  identical coordinates.
- **Headless rendering:** every figure function runs to completion and writes a non-empty
  file under the `Agg` backend with no display available.
- **Single-column legibility:** assert each figure's width is 3.5 inches and its smallest
  font size is at least 8 points. Mechanical, but it catches the figure that has to be
  regenerated the night before submission.

## Report / novelty note

This module produces every artifact a reader sees, so its contribution is the report's
evidence rather than a mechanism. Three points:

1. **Figure 5 (loss curves at depth 32) is the one that makes the central claim
   defensible.** Without it, "deep GNNs oversmooth" is indistinguishable from "our deep
   models failed to train." It is the least visually striking figure in the set and the
   most load-bearing.
2. **The energy-versus-layer figure is where the study's diagnostic apparatus is
   visible** — per-layer, normalized, log-axis, band-restricted. That combination is what
   makes GCN, GraphSAGE, and GAT curves overlayable, and each element traces to a
   recorded decision.
3. **The t-SNE panels are qualitative and should be labelled as such.** They illustrate
   collapse; they do not measure it. Dirichlet energy and MAD measure it. Saying so
   explicitly is the difference between an illustration and an overclaim.

## Open questions

- **How many t-SNE panels.** Currently four saved embeddings (GCN baseline and mitigated,
  depths 2 and 32, seed 0). Whether the figure shows two panels or four affects
  legibility at single-column width. Decide when the first draft is rendered.
- **Whether `contractionSlope` deserves a figure or only a table.** It is one number per
  configuration, so a table may serve better than a plot — but a slope-versus-depth curve
  would show whether the decay rate itself changes with depth, which is a sharper claim
  than a single rate. Undecided.
- **Greyscale legibility with five mitigation series.** Colourblind-safe is achievable;
  five distinguishable greyscale line styles at 3.5 inches is harder. May force splitting
  the ablation into two panels.
- **Fit quality for the slope.** Flagged in `metrics_spec` as possibly warranting an
  `r^2` alongside the slope. If it is added there, this module should surface it, since a
  poorly fitted slope presented as a single number is exactly the kind of thing a grader
  probes.
