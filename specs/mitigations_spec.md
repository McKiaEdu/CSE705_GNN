# mitigations — residual, PairNorm, and Jumping Knowledge as injected objects

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed   <!-- "defended" = passed the walk-through -->

## Purpose

Implements the three oversmoothing mitigations that attach to a model by composition:
residual connections and PairNorm as `LayerHook` objects, and Jumping Knowledge as a
`Readout`. These are the treatment arms of the mitigation ablation — the components
whose presence or absence is the only thing that differs between a baseline run and a
mitigated one.

GCNII is NOT here. It is a fourth `convType` in `models` per D-008, because its update
rule is written on the normalized adjacency and there is no meaningful "GCNII applied
to GAT."

Realizes D-006 (composition, two protocols), D-007 (hook order), D-016 (readout
declares whether the final layer emits logits), and D-024 through D-026 below.

## Approach (agreed)

**This module implements protocols it does not own.** `LayerHook` and `Readout` are
declared in `models`; `mitigations` imports them and `models` imports nothing from here.
The dependency runs one way, which is what lets a model be constructed with any
combination of hooks without `models` knowing the set of mitigations that exists.

The three mechanisms, and what each rejects:

**Residual is plain identity addition, with an explicit no-op on width mismatch.**
`h + hPrev`, which is exactly what the `LayerHook` protocol's `hPrev` argument was
declared for. Widths match on intermediate layers only: layer 1 maps `1433 -> hiddenDim`
and, under `LastLayerReadout`, layer L maps `hiddenDim -> outDim`, so neither can take
an identity residual. The hook returns `h` unchanged in those cases.
*Rejected — a learned projection to reconcile mismatched widths.* Adds parameters whose
count depends on depth, confounding the capacity of the residual arm with the depth axis
— the same objection that settled D-019 against first-layer-only weight decay. It is
also not what Kipf & Welling did in the Appendix B depth study this arm is compared
against.
*The mismatch skip is explicit and tested, not incidental.* A hook that silently returns
`h` whenever shapes disagree would let a misconfigured run report a residual arm in which
the residual never fired, with nothing erroring.

**PairNorm uses the scale-individual variant (PN-SI) at fixed scale.** The authors'
reference implementation states that plain PN behaves badly with a symmetric normalized
adjacency — their paper's experiments used a row-normalized adjacency — and recommends
PN-SI or PN-SCS for GCN and GAT. `GCNConv` is symmetrically normalized, so this study
sits in exactly the flagged case. A later note in the same source says a bug was fixed
and plain PN should now work, which leaves the guidance unresolved; PN-SI is the choice
that does not depend on which note is current.
*Scale `s = 1.0`, fixed.* The scale is data-dependent and the authors tune it per
dataset. The coordinated hyperparameter search holds one configuration fixed across all
architectures and depths, so `s` is not tuned here. This is a stated limitation, not an
oversight: a tuned PairNorm would be compared against untuned baselines.
*Rejected — plain PN.* Documented by the authors as interacting badly with the
normalization this study's propagation actually uses.
*Rejected — sweeping `s`.* A full extra axis answering a question the report does not
ask, against the Aug 3 deadline.

**Jumping Knowledge aggregates by max pooling, not concatenation.** Concatenation
produces `T x hiddenDim`, so the readout's final linear layer would be `32 x 64 -> 7` at
depth 32 and `2 x 64 -> 7` at depth 2. The JK arm's parameter count would then be a
function of depth, and a depth curve for JK would confound the mitigation with a
capacity increase. Max pooling holds the readout input at `hiddenDim` at every depth.
*Rejected — concatenation.* Depth-varying capacity, as above. Worth noting that the JK
paper's own citation-network experiments selected depths from {2, 3}; the concatenation
variant was never exercised at the depths this study runs.
*Rejected — LSTM-attention.* Fixed-size and therefore not subject to the capacity
objection, but it introduces a bidirectional LSTM that must be defended line by line in
the report's source-code section and the video, for a mechanism the study is not asking
about.
*Aggregation range is indices 1..L, not 0..L.* Index 0 is the raw 1433-dimensional
input and cannot be max-pooled against 64-dimensional hidden states. The JK paper takes
the first linear-transformed representation into account, which in our indexing is
index 1.

## Interface & contract touchpoints

Functions/methods use PascalCase; variables use camelCase; PyG `Data` fields keep their
library names (`edge_index`, `train_mask`).

- `class ResidualHook` — defined here; implements `LayerHook`.
  `Apply(h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor` returns `h + hPrev`
  when `h.shape == hPrev.shape`, else `h` unchanged.
- `class PairNormHook(scale: float = 1.0)` — defined here; implements `LayerHook`.
  `Apply(...)` returns the PN-SI normalized tensor; `hPrev` and `edgeIndex` are ignored.
  Shape-preserving.
- `class JkReadout(mode: str = "max", hiddenDim: int, outDim: int)` — defined here;
  implements `Readout`. `FinalLayerIsLogits = False` per D-016.
  `Apply(layerEmbeddings: list[Tensor]) -> Tensor` aggregates indices `1..L` and returns
  `logits [N, outDim]`.
- consumes: the `LayerHook` and `Readout` protocol declarations from `models`. Nothing
  else. This module does not import `GnnModel`, `train`, or `metrics`.
- produces: hook and readout instances that `experiments` injects into `GnnModel` at
  construction.

Contract properties this component must honor:

- **`LayerHook.Apply` is shape-preserving.** Both hooks return a tensor of the same
  shape they received. `metrics` derives the comparable band from tensor widths, so a
  hook that changed width would silently alter the band.
- **`JkReadout.FinalLayerIsLogits` is `False`** (D-016). Consequently the final conv
  emits `hiddenDim` rather than `outDim`, the final layer IS activated and dropped out
  like every other, and `layerEmbeddings[L]` is a hidden state rather than the logits.
  The comparable band for a JK run is therefore `1..L`.
- **The readout's linear layer is not a message-passing layer.** `Linear(hiddenDim ->
  outDim)` inside `JkReadout` does not count toward `numLayers`, so the "L layers = L
  hops" assumption in `models_spec` holds unchanged for JK runs.
- **Hook order is residual then PairNorm** (D-007), applied in list order by the caller.
  Neither hook enforces this; `experiments` constructs the list.
- **`mitigations` appears in the results record as a sorted list of hook/readout names**
  (C3). `["jk"]` denotes the JK readout, not a hook.

## Implementation plan

- `class ResidualHook` — defined here; reused by `experiments/`. Stateless, no
  parameters, no `nn.Module` state beyond the base class. The width check is
  `h.shape == hPrev.shape`, evaluated per call rather than cached, since the same hook
  instance is applied at every layer and only some layers match.
- `class PairNormHook(scale)` — defined here; reused by `experiments/`. PN-SI computes,
  per the paper's two-step center-and-scale procedure:
  1. Center: subtract the column mean over nodes, `h = h - h.mean(dim=0, keepdim=True)`.
  2. Scale, individually per node: divide each row by its own L2 norm and multiply by
     `scale`. The row-norm denominator is clamped away from zero, since ReLU produces
     all-zero rows at depth and an unclamped division would produce `nan` in exactly the
     regime this study measures.
  Vectorized: two tensor reductions, no loop over nodes.
- `class JkReadout(mode, hiddenDim, outDim)` — defined here; reused by `experiments/`.
  Holds `torch_geometric.nn.JumpingKnowledge(mode="max")` and
  `torch.nn.Linear(hiddenDim, outDim)`. `Apply` slices `layerEmbeddings[1:]`, passes the
  list to the aggregator, and applies the linear layer. Asserts on entry that every
  sliced tensor shares one width, so a misconfigured model fails loudly rather than
  broadcasting.
- `MitigationNames(layerHooks, readout) -> list[str]` — defined here; returns the sorted
  canonical name list `experiments` writes into the results record, so the naming exists
  in one place rather than being reconstructed at the call site.

Vectorization: no Python loops over nodes or edges anywhere. `JkReadout` loops over the
layer list, which is `numLayers` iterations and is what the aggregator consumes.

## Dependencies

- Depends on: `models/` for the two protocol declarations only; `torch`;
  `torch_geometric.nn.JumpingKnowledge`.
- Must NOT depend on: `GnnModel` itself, `train/`, `metrics/`, `experiments/`. The
  dependency inversion in D-006 requires this direction and must not be flipped.
- Consumed by: `experiments/` (constructs and injects), and indirectly by every run in
  the ablation arms.
- Build order: after `models/`, `metrics/`, and `train/`. The null-object default means
  the harness runs and the baselines reproduce before this module exists, which is why
  it sits late in the order despite being conceptually central.

## Assumptions & constraints

Every item here is provisional until confirmed.

- Python / version: Python 3.14.4 (matches `data_spec.md`'s corrected pin).
- Libraries / role: `torch`, `torch_geometric` 2.8.0 (core — `JumpingKnowledge`).
- Compute / runtime: negligible. Both hooks are elementwise or per-row reductions on a
  `[2708, 64]` tensor; the JK aggregation is a max over at most 32 such tensors.
- `hiddenDim = 64` (D-023), total width across heads.
- Confirmed assumptions:
  - Hooks are applied inside the layer loop immediately after the conv and before the
    activation, per `models_spec`'s data flow.
  - Both hooks are shape-preserving, so a mitigated run and its baseline produce
    `layerEmbeddings` lists of identical shapes and the energy curves are directly
    comparable.
  - PairNorm's scale is fixed at 1.0 across every configuration and is not tuned.
  - The ablation arms are `[]`, `["residual"]`, `["pairnorm"]`, `["jk"]`,
    `["pairnorm","residual"]` per C3. Residual + JK and PairNorm + JK are not run.

## Outputs & artifacts

`mitigations` writes no files and produces no records. It produces objects:

- `ResidualHook`, `PairNormHook` instances — injected as `layerHooks`.
- `JkReadout` instance — injected as `readout`, replacing `LastLayerReadout`.
- The canonical name list that `experiments` writes to `config.mitigations` and uses to
  build the results filename.

## Test plan

- **Shape preservation:** for each hook, on a `[2708, 64]` input, assert the output
  shape equals the input shape exactly.
- **Residual fires where widths match:** with `h` and `hPrev` both `[2708, 64]`, assert
  the output equals `h + hPrev` elementwise.
- **Residual is a no-op where widths differ:** with `h` `[2708, 64]` and `hPrev`
  `[2708, 1433]`, assert the output is `h` unchanged, bitwise. This is the regression
  test for the silent-skip failure — a residual arm whose residual never fired.
- **Residual fires on the expected layer count:** in a depth-8 model under
  `LastLayerReadout`, instrument the hook and assert it performs the addition on exactly
  6 layers (2 through 7), skipping layer 1 and layer 8.
- **PairNorm invariant:** after `PairNormHook.Apply`, assert the column means are zero
  to floating-point tolerance, and that every row's L2 norm equals `scale`. These are the
  two steps of the mechanism and each is separately checkable.
- **PairNorm survives zero rows:** with an input containing all-zero rows, assert the
  output contains no `nan` or `inf`. This is the depth-32 case, not a corner case.
- **PairNorm reduces energy dispersion:** on a random input, assert the Dirichlet energy
  computed by `metrics` after PairNorm differs from that before, so the hook is
  demonstrably in the measured path per D-010's "after the hooks" tap decision.
- **JK output width is depth-invariant:** build JK models at depths 2, 8, and 32 and
  assert the readout's linear layer has identical parameter count in all three. This is
  the regression test for the concatenation rejection — under `cat` this test fails.
- **JK aggregates the right range:** assert `Apply` consumes `layerEmbeddings[1:]` and
  never touches index 0, by passing a deliberately mis-shaped index 0 and confirming no
  error is raised.
- **JK readout declares itself:** assert `JkReadout.FinalLayerIsLogits is False` and
  `LastLayerReadout.FinalLayerIsLogits is True`.
- **Gradient reaches every layer under JK:** covered by the existing `models` test; noted
  here because this module supplies the readout that makes it meaningful.
- **Name canonicalization:** `MitigationNames` returns sorted lists, so
  `[PairNormHook, ResidualHook]` and `[ResidualHook, PairNormHook]` both yield
  `["pairnorm", "residual"]` and group to one key at aggregation time.

## Report / novelty note

This module supplies the mitigation ablation, which is the study's second headline
result after the depth sweep itself. Three points worth stating explicitly:

1. **The mitigations are composed, not inherited, and the baseline runs the identical
   code path.** This is what licenses attributing a measured difference to the
   mitigation rather than to a divergence in control flow. It is a methodological claim,
   and it is checkable by a grader reading the layer loop.
2. **JK is run with max pooling for a stated reason, not by default.** Concatenation
   would have made the JK arm's capacity grow with depth, which would have inflated its
   apparent effectiveness at exactly the depths where the study claims mitigations help.
   Naming that is a small piece of genuine analysis.
3. **PairNorm is untuned, deliberately.** Its scale is data-dependent and the authors
   tune it per dataset; holding it fixed keeps the comparison against untuned baselines
   honest, at the cost of possibly understating PairNorm. State the direction of the
   bias rather than leaving it for a reader to infer.

## Open questions

- **GCNII's `alpha` and `lambda`.** Resolved (D-042): `alpha = 0.1`, `theta = 0.5`,
  matching Chen et al.'s published Cora configuration (arXiv:2007.02133, Table 6).
  **Terminology note:** the paper's `lambda` is PyG's `GCN2Conv(theta=...)` keyword —
  the code and D-042 both use `theta`, not `lambda`; this spec's own prose above ("its
  `alpha` and `lambda`") should be read as the paper's naming, not the constructor's.
- **PairNorm's rescaling denominator.** Resolved (D-025): PN-SI as specified — center by
  subtracting the column mean over nodes, then divide each row by its own L2 norm and
  multiply by the fixed scale, confirmed against the paper rather than inferred from the
  variant name.
- **Whether `["jk"]` belongs in `config.mitigations` at all.** Resolved (D-028): kept,
  for the filename-collision reason given there, with a write-time assertion
  (`("jk" in mitigations) == (readout == "jk")`) so the two fields cannot silently
  disagree.
- **Residual on the first layer via projection.** Still rejected, still not built into
  the ablation, and no one-off diagnostic run has been made. Remains open — F-005 does
  show the residual arm at depth 32 underperforming even the unmitigated baseline
  (19.25% vs 24.07%), which is exactly the result that would make this diagnostic worth
  running if time permits.
