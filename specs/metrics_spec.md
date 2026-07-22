# metrics — Dirichlet energy, MAD, and the contraction slope

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed   <!-- "defended" = passed the walk-through -->

## Purpose

Turn the `layerEmbeddings` list produced by `models` into the per-layer scalars the
study reports: Dirichlet energy and Mean Average Distance at every index, plus the
fitted contraction slope over the metrically comparable band. This component is the
measuring instrument — it holds the fixed metric graph, decides which layer indices are
comparable, and produces the only numbers in the repo that quantify oversmoothing.

Realizes D-003 (per-dimension energy, slope over l >= 1), D-004 (augmented metric
graph fixed across architectures), and the reported-quantity clause of D-002 as edited.

## Approach (agreed)

**One instrument object, constructed once per run, holding the metric graph as state.**
`metrics` receives the raw `edge_index`, augments it internally to `G~ = (A + I)`, and
precomputes `D~^{-1/2}` once. Every subsequent per-layer call is a pure function of the
embedding tensor against that fixed operator.

The decisions this carries, and what each rejects:

**The metric graph is fixed to `G~` for all architectures (D-004).** Not matched to the
operator each architecture propagates with. For GAT that matching is ill-defined — its
propagation graph is re-weighted by learned attention that differs per layer and
throughout training, so `E_l` and `E_{l+1}` would be measured against different graphs
and the fitted slope would compare quantities with different units. A model could also
lower measured energy purely by concentrating attention, with representations unchanged,
making the metric partly measure itself. The metric graph is an instrument, not part of
the model.

**Energy is per-dimension and normalized at layer 1, not layer 0 (D-002, D-003).**
Energy sums over feature dimensions, so `E_0` at 1433 dims and `E_1` at `hiddenDim` are
not on a common footing; dividing by the feature count removes the dimension-count
effect. The reference layer is 1 rather than 0 because per-dimension division corrects
dimension COUNT but not representation KIND — `h_0` is sparse binary row-normalized
bag-of-words, `h_1` is dense post-`W_0` post-ReLU — so the step from `E_0` to `E_1`
mixes aggregation smoothing with the arbitrary scale of the learned input projection.

**No Frobenius normalization of the reported curve; record `||H||_F^2` instead.**
Cai & Wang's bound is `E_{l+1} <= (1 - lambda)^2 ||W||^2 E_l`. Weight-norm growth is
inside the mechanism the bound describes, not noise sitting on top of it. A mitigation
that restores deep performance partly by letting `||W||` grow is succeeding by a route
the theory predicts, and dividing out representation magnitude would hide exactly that,
making residual and GCNII look weaker than they are.
*Rejected — reporting `E_l / ||H_l||_F^2` as the headline curve:* removes the one term
that distinguishes a mitigation acting on the operator from one acting on the weights.
*Mitigated:* `||H_l||_F^2` is one scalar per layer, free to compute alongside the
energy, and is recorded in every results record. The Frobenius-normalized curve is
therefore derivable at aggregation time from stored data, with no rerun, if a reviewer
asks whether the effect is only scale.

**Both metrics are computed over the full node set, no mask.** Oversmoothing is a claim
about the learned representation, not about the labeled nodes, and neither metric uses
labels. Restricting to `train_mask` would leave 140 of 2708 nodes; the induced subgraph
is close to edgeless, so the energy sum would be dominated by which few edges happened
to survive rather than by smoothing.
*Rejected — masking to the train or test split:* introduces split-dependent sampling
noise into a metric that needs none, and would make the energy curve incomparable to
Cai & Wang's whole-graph bound.

**MAD only; MADGap is out of scope.** Deli Chen et al. (arXiv:1909.03211) define MAD
from the cosine-distance matrix over node pairs and then extend it to MADGap by
splitting nodes into neighboring and remote sets by hop distance and taking the gap.
MADGap is validated in that paper as a *predictor of model performance*, which is not
the claim this study makes; the proposal commits to MAD and does not mention MADGap;
and the remote/neighbor masks require a hop-distance computation this component
otherwise has no reason to hold.
*Rejected — computing MADGap as a bonus figure:* a second smoothing number with a
different interpretation, added late, invites the report to conflate "representations
collapsed" with "the model will do badly," which are exactly the two things the depth
sweep is trying to keep separate.

**The comparable band is derived from tensor widths, not hardcoded to `1..L-1`.** D-001
C1 states the band as `1..L-1` on the assumption that index `L` is the 7-dim logit
space. That assumption holds for the `LastLayerReadout` configurations and fails for
Jumping Knowledge, where the final conv emits `hiddenDim` and the readout produces the
logits separately — under JK, index `L` is a hidden representation and belongs in the
band. Hardcoding `1..L-1` would silently discard one layer from every JK run and make
the JK slope fitted over a different range than its baseline's.
*Rejected — passing the readout type into `metrics`:* couples the measuring instrument
to the model's configuration for information it can read directly off the tensors.

## Interface & contract touchpoints

Functions/methods use PascalCase; variables use camelCase; PyG `Data` fields keep their
library names (`edge_index`, `train_mask`).

- `class OversmoothingMetrics(edgeIndex: Tensor, numNodes: int)` — defined here.
  Constructed once per run. Asserts on entry that `edgeIndex` contains no self-loops,
  then builds and stores the augmented index and `D~^{-1/2}`.
- `ComputeAll(layerEmbeddings: Sequence[Tensor]) -> LayerMetrics` — defined here; the
  single entry point `train` calls at each of the three capture points.
- `class LayerMetrics` — defined here; a frozen dataclass, the component's output type:
  - `dirichletEnergy: list[float]` — per-dimension, length `L + 1`
  - `mad: list[float]` — length `L + 1`
  - `frobeniusSquared: list[float]` — length `L + 1`
  - `bandIndices: list[int]` — the comparable band, derived
  - `contractionSlope: float` — `log rho`, fitted over `bandIndices`
- `DirichletEnergy(h: Tensor) -> float` — defined here; per-dimension energy of one
  layer against the stored operator.
- `MeanAverageDistance(h: Tensor) -> float` — defined here.
- `SelectComparableBand(layerEmbeddings: Sequence[Tensor]) -> list[int]` — defined here.
- `FitContractionSlope(energies: Sequence[float], bandIndices: Sequence[int]) -> float`
  — defined here.
- consumes: `layerEmbeddings` from `models` per D-001 C1 — length `L + 1`, index 0 is
  raw `X`, detached by `train` at the capture site per D-011. Also the raw
  `edge_index` from `data`, self-loop-free per the `data` spec.
- produces: the `dirichletEnergy` and `mad` arrays written into `epoch0Metrics`,
  `checkpointMetrics`, and `finalMetrics` in the C3 results record.

**Flagged contract addition.** C3 currently defines `dirichletEnergy` and `mad` inside
each capture block. This component additionally produces `frobeniusSquared`,
`bandIndices`, and `contractionSlope`. The first two are per-layer arrays and belong
alongside the existing two in each capture block; `contractionSlope` is a per-capture
scalar. This is an additive schema change and needs a D-001 changelog line before
`train` is implemented.

**Flagged contract exception.** C1's clause "the final entry equals the returned logits"
holds only under `LastLayerReadout`. It is false under Jumping Knowledge, and possibly
under GCNII depending on how the output projection is counted. This component does not
rely on the clause — it derives the band from widths — but the clause itself is wrong as
written and should be narrowed. See Open questions.

## Implementation plan

- `class OversmoothingMetrics` — defined here; reused by `train/` and by `notebooks/`
  for figure generation. Holds `augmentedEdgeIndex: Tensor [2, E + N]` and
  `invSqrtDegree: Tensor [N]` as buffers, built once in the constructor.
- `BuildAugmentedOperator(edgeIndex: Tensor, numNodes: int) -> tuple[Tensor, Tensor]` —
  defined here; asserts self-loop-free input via
  `torch_geometric.utils.contains_self_loops`, calls `add_self_loops`, computes
  `d~ = degree(...)` and returns `(augmentedEdgeIndex, d~^{-1/2})`.
- `DirichletEnergy(h)` — defined here. Computes

  `E(H) = 0.5 * sum over (i,j) in E~ of || h_i / sqrt(d~_i) - h_j / sqrt(d~_j) ||^2`

  which equals `trace(H^T Δ~ H)`, then divides by `h.shape[1]` for the per-dimension
  form of D-003. Vectorized as: scale rows by `invSqrtDegree`, gather source and target
  rows by `augmentedEdgeIndex`, take the squared row-norm of the difference, sum, halve.
  No Python loop over nodes or edges. Self-loop terms contribute exactly zero to the sum
  but change every `d~_i`, which is the point of D-004.
- `MeanAverageDistance(h)` — defined here. L2-normalizes rows, forms
  `D = 1 - Hn @ Hn.T` as a single dense `[N, N]` matmul, then reduces per the paper's
  convention (see Open questions on the non-zero denominator). No pair subsampling: at
  `N = 2708` the matrix is ~29 MB transient and there are only three captures per run.
- `SelectComparableBand(layerEmbeddings)` — defined here. Returns every index `l >= 1`
  whose width equals the modal width across indices `1..L`. Index 0 is excluded
  unconditionally (representation kind, per D-003). Asserts the returned indices are
  contiguous and raises otherwise, so an unexpected architecture cannot silently produce
  a gapped band that `FitContractionSlope` would fit across.
- `FitContractionSlope(energies, bandIndices)` — defined here. Least-squares fit of
  `log E_l` against `l` over `bandIndices` only; returns the slope. Returns `nan` when
  the band has fewer than two points, which is the `L = 2` case flagged in C1 — the
  baselines do not yield a slope and this must not silently return 0.
- `ComputeAll(layerEmbeddings)` — defined here; loops over the `L + 1` tensors (a loop
  over layers, not nodes), calls the three per-layer functions, derives the band, fits
  the slope, returns `LayerMetrics`.

Normalization at layer 1 is **not** applied here. `ComputeAll` returns raw
per-dimension energies and the results record stores those. The `E_l / E_1` ratio is
formed at aggregation time by the plotting code, so a record remains faithful to what
was computed and the reference layer can be changed without a rerun.

## Dependencies

- Depends on: `data/` for the raw self-loop-free `edge_index`; `models/` for the
  `layerEmbeddings` contract; `torch`, `torch_geometric.utils` (`add_self_loops`,
  `contains_self_loops`, `degree`).
- Does NOT depend on `train/`, `mitigations/`, or the model object. It sees tensors and
  a graph, never a configuration. This is what makes it a fixed instrument.
- Consumed by: `train/` (called at the three capture points of C5), `viz/` and
  `notebooks/` (figure generation from stored records).
- Build order: after `data/` and `models/`, before `train/`.

## Assumptions & constraints

Every item here is provisional until confirmed.

- Python / version: Python 3.12.13 (carried from the data spec).
- Libraries / role: `torch`, `torch_geometric` 2.8.0 (core). No `scipy`, no
  `scikit-learn` — every reduction is a torch tensor op.
- Compute / runtime: negligible. Energy is a sum over ~10.5k + 2708 edges. MAD is one
  `[2708, hiddenDim] x [hiddenDim, 2708]` matmul, ~29 MB transient, at most 33 layers
  times 3 captures per run. This resolves C5's original MAD-cost concern: no
  subsampling is needed and the checkpoint and non-checkpoint captures compute the
  identical quantity.
- Device: device-agnostic. Metrics run on whatever device the embeddings arrive on;
  `train` moves them to CPU at the capture site per D-011.
- Numerics: float32 throughout, matching the model. MAD row normalization clamps the
  norm denominator to avoid division by zero on all-zero rows, which ReLU will produce
  at depth.
- Confirmed assumptions:
  - Embeddings arrive detached; this component never calls `backward` and never needs
    gradients.
  - `edge_index` is undirected and each undirected edge appears as two directed
    entries. The `0.5 *` factor in the energy is exactly this double-count correction.
  - The metric graph is built once per run, not per capture and not per layer.

## Outputs & artifacts

`metrics` writes no files. It returns in-memory `LayerMetrics` objects that `train`
serializes into the results record.

- Per capture point, written into `epoch0Metrics` / `checkpointMetrics` /
  `finalMetrics` in `results/<convType>_<mitigations>_d<depth>_s<seed>.json`:
  `dirichletEnergy [L+1]`, `mad [L+1]`, `frobeniusSquared [L+1]`, `bandIndices`,
  `contractionSlope`.
- Full-length arrays are stored even though only `bandIndices` are comparable, per C3.
  The band restriction is applied by the plotting code.

## Test plan

- **Collapse floor:** an embedding with all rows identical gives Dirichlet energy `0`
  to floating-point tolerance and MAD `0`. This is the oversmoothed limit and the
  metric must hit it exactly, not approximately-but-biased.
- **Non-degenerate:** a random Gaussian embedding gives clearly positive values for
  both.
- **Energy against a dense reference:** on a small random graph (N = 20), assert the
  vectorized edge-list energy equals `trace(H^T Δ~ H)` computed with a dense
  `Δ~ = I - D~^{-1/2} A~ D~^{-1/2}`, to within `1e-5`. This is the correctness test
  for the whole component — everything else is bookkeeping around this quantity.
- **Scaling behavior, both metrics:** for `c > 0`, `DirichletEnergy(c * H)` equals
  `c^2 * DirichletEnergy(H)`, and `MeanAverageDistance(c * H)` equals
  `MeanAverageDistance(H)` unchanged. This is the concrete check of the invariance
  asymmetry D-002 relies on.
- **Self-loop guard:** constructing `OversmoothingMetrics` with an already-augmented
  `edge_index` raises, rather than silently double-augmenting.
- **Band derivation under JK:** build a JK-configured model at `L = 4`, assert
  `SelectComparableBand` returns `[1, 2, 3, 4]`; build the same depth with
  `LastLayerReadout`, assert it returns `[1, 2, 3]`. This is the regression test for
  the hardcoded-band failure.
- **Slope on a synthetic decay:** feed energies `E_l = E_1 * r^(l-1)` for known `r`;
  assert the fitted slope recovers `log r` to within `1e-6`.
- **Slope refuses to guess:** at `L = 2` the band has one point; assert
  `FitContractionSlope` returns `nan`, not `0.0`.
- **Depth qualitative gate:** a 32-layer unmitigated GCN at the checkpoint shows
  monotonically decreasing per-dimension energy across the band, and a residual variant
  at the same depth shows a shallower fitted slope. Qualitative, not an exact-value
  assertion.

## Report / novelty note

This component produces the study's central figure — normalized per-dimension energy
against layer index, overlaid across architectures — and the one-number summary
(`contractionSlope`) that lets the mitigation ablation be stated as a table rather than
a wall of curves.

Three points worth a sentence each in Results or Discussion, all of them methodological
rather than reportage:

1. **The metric graph is fixed, and for GAT that is a stated cost.** The fitted `rho`
   for GAT measures decay of GAT's embeddings with respect to Cora's structure, not
   with respect to GAT's own attention-weighted graph — legitimate and necessary for
   cross-architecture comparison, but not the quantity Cai & Wang's theorem bounds for
   GAT.
2. **Energy and MAD disagree by construction, and the disagreement is informative.**
   MAD is cosine-based and therefore blind to representation magnitude; energy is
   quadratic and is not. A configuration where MAD collapses while energy holds up is
   one where representations spread in magnitude but align in direction — which is a
   real distinction between mitigation mechanisms, not a measurement inconsistency.
3. **The comparable band is not `1..L-1` for every configuration.** Deriving it from
   tensor widths rather than hardcoding it is a small correctness point that a grader
   reading the code can verify, and it is the kind of detail that separates a harness
   that was reasoned about from one that was assembled.

## Open questions

- **MAD's reduction convention.** Deli Chen et al. average over *non-zero* entries of
  the distance matrix rather than all entries, at both the row-mean and the final-mean
  stage. This must be read off the paper (arXiv:1909.03211, the MAD subsection) and
  matched exactly before the reduction is written. It is not cosmetic: at depth, ReLU
  produces zero rows whose cosine distance is undefined or zero throughout, and the two
  conventions diverge precisely in the regime the study is about.
- **`hiddenDim` value.** Still unfixed at 16 vs 64, carried from `models_spec`. It does
  not block this component's design — per-dimension division already removes the
  dimension-count effect and the Frobenius decision removes the scale dependence from
  the question — but it must be fixed before the sweep, and the band width assertion in
  the test plan is written against whatever value is chosen.
- **C1's "final entry equals the logits" clause.** False under Jumping Knowledge. This
  component is robust to it, but the clause is a contract statement other components may
  rely on and should be narrowed to "under `LastLayerReadout`." Needs a D-001 changelog
  line.
- **GCNII's input projection and index 0.** If GCNII adds a linear projection before the
  conv stack, is `layerEmbeddings[0]` the raw 1433-dim `X` or the projected
  `hiddenDim` tensor? If the latter, GCNII's index 0 is a different representation kind
  from every other architecture's, and the `E_0` anchor is not comparable across
  architectures. Belongs to `models` but lands here; resolve with the GCNII `x0`
  question already open in `models_spec`.
- **`contractionSlope` fit quality.** The fit currently returns only the slope. If the
  log-energy curve is visibly non-linear for some configuration, a single slope
  misrepresents it. Consider also recording `r^2` so a bad fit is detectable at
  aggregation time rather than invisible. Cheap to add; not yet decided.
