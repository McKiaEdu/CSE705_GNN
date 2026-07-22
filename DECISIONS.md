# Decisions — Cora GNN Oversmoothing Study

Record of settled decisions. Each entry states the decision, the reasoning, the
alternatives rejected, and anything still open.

**Convention.** Entries are **edited in place** when a decision is corrected or
refined. The superseded wording does not survive as a separate entry — it moves into
that entry's *Rejected alternatives* section together with the reason it changed, so
the reasoning is preserved while the file continues to state exactly one current
answer per question. Git holds the diff. A new entry is appended only for a question
the log had not previously answered.

D-001 is the interface contract and grows by clause. Because a contract change must
be *noticed* and not merely correct, every edit to a D-001 clause is additionally
recorded as one line in the D-001 changelog at the end of that entry. Later entries
are independent and stand alone.


## D-001 — Interface contract

The shared surface between Kiarash (models, train, experiments) and Yiheng
(metrics, viz). Fixed once set; any change is a flagged contract change and both
sides update. Clauses are appended as they are settled.

### C1 — Model forward signature

`Forward(x: Tensor, edgeIndex: Tensor) -> tuple[Tensor, list[Tensor]]`
returns `(logits [N, C], layerEmbeddings)`.

`layerEmbeddings` has length L+1: index 0 is the raw input X, index l is the
output of layer l. The final entry equals the returned logits ONLY when
`readout.FinalLayerIsLogits` is true (D-016) — i.e. under `LastLayerReadout`.
Under Jumping Knowledge, index L is a hidden representation at hiddenDim and the
logits are a separate tensor built by the readout from the whole stack. Consumers
must not assume the identity.

`models` applies no filtering to the list. The choice of which entries are
metrically comparable belongs to `metrics`.

X is available to every consumer as `layerEmbeddings[0]` and is **not** passed
separately alongside the list. This is the single source of the layer-0
representation.

The comparable band is derived by `metrics` from tensor widths, not fixed by this
clause. Index 0 is excluded unconditionally: it is 1433-dim sparse binary, so
energy across that boundary confounds a dimensionality and representation-kind
change with a smoothing effect. Index L is excluded when it is the logit space —
a decision space rather than a representation space, which cross-entropy actively
contracts, so its energy tracks training quality rather than oversmoothing. Under
Jumping Knowledge index L is a hidden state and belongs in the band. The band is
therefore 1..L-1 under `LastLayerReadout` and 1..L under JK.

Consequence: under `LastLayerReadout` at L=2 the band is a single point and no
slope can be fitted. Under JK the same depth yields two. The depth sweep, not the
baselines, is where the energy curve is read.

Rejected — trimming to the band inside `models`: viz needs the logit layer for
t-SNE and X as a sanity anchor, so discarding at the source loses information
another consumer wants.

Settled in D-013 — no Frobenius normalization of the reported curve; `||H||_F^2`
is recorded per layer instead, so the normalized variant stays derivable at
aggregation time.

### C2 — Tap point within a layer

A layer computes: aggregate → linear → normalize → activate → dropout.

`layerEmbeddings[l]` is the **post-activation, pre-dropout** tensor — the
representation layer l+1 actually receives.

Reasoning: the metric must describe what propagates. A pre-activation tap
measures a tensor the network never uses as a representation, so any collapse
it reports is not the collapse that produced the accuracy drop. The ReLU
contraction is genuine model behavior and belongs in the measurement.

Pre-dropout because dropout is stochastic and mode-dependent; tapping after it
makes the metric depend on a random mask. Under C5 all captures are taken in
eval mode, where dropout is the identity and the distinction is moot — the
clause is specified anyway so the value does not silently depend on which mode
a future caller happened to be in.

Which layers receive an activation at all is governed by D-016: every layer whose
output is a hidden representation, which under JK includes the final layer.

Rejected — pre-activation tap: loses fidelity to the propagated representation.
Cost of the rejection: we cannot decompose decay into aggregation-driven vs.
nonlinearity-driven components. Recoverable later via a diagnostic run that
captures both; noted as a possible extension, not built into the contract.

### C3 — Results record schema

One JSON file per run: `results/<convType>_<mitigations>_d<depth>_s<seed>.json`
where `<mitigations>` is the sorted set joined by `+`, or `none` when empty.
Every report figure is an aggregation over these files; mean and std are computed
at aggregation time and never stored.

`train` assembles the complete record and returns it; `experiments` serializes it
to disk (D-022). There is exactly one writer.

{
  "runId": str,                    // filename stem, unique
  "timestamp": str,                // ISO 8601
  "config": {
    "convType": str,               // "gcn" | "sage" | "gat" | "gcnii"
    "numLayers": int,
    "mitigations": [str],          // sorted; [] for the baseline arm
    "readout": str,                // "lastLayer" | "jk"; determines banding, per D-016
    "hiddenDim": int,
    "dropout": float,
    "learningRate": float,
    "weightDecay": float,
    "maxEpochs": int,
    "patience": int,
    "seed": int
  },
  "bandIndices": [int],            // derived comparable band, per C1. Run-invariant:
                                   // widths do not change across captures, so this is
                                   // stored once rather than in each capture block.
  "results": {
    "testAccuracy": float,
    "testMacroF1": float,
    "valAccuracy": float,          // at the selected checkpoint
    "valLoss": float,              // at the selected checkpoint; the selection criterion
    "bestEpoch": int,              // epoch of minimum validation loss, per D-017
    "epochsRun": int               // < maxEpochs when early stopping fired
  },
  "trainingCurve": [               // per-epoch; closes D-005
    {"epoch": int, "trainLoss": float, "valLoss": float, "valAccuracy": float}
  ],
  "epoch0Metrics": {               // eval-mode capture before any gradient step, per C5
    "dirichletEnergy": [float],    // length L+1, index 0 = X; index L is the logits
                                   // only under LastLayerReadout (C1)
    "mad": [float],                // same indexing
    "frobeniusSquared": [float],   // same indexing, per D-013
    "contractionSlope": float      // log rho fitted over bandIndices; nan if < 2 points
  },
  "checkpointMetrics": {           // eval-mode capture at bestEpoch, per C5
    "dirichletEnergy": [float],
    "mad": [float],
    "frobeniusSquared": [float],
    "contractionSlope": float
  },
  "finalMetrics": {                // eval-mode capture at the last epoch run, per C5
    "dirichletEnergy": [float],
    "mad": [float],
    "frobeniusSquared": [float],
    "contractionSlope": float
  },
  "trajectory": [                  // nullable; normally empty. Extension point for a
                                   // dense-capture diagnostic run, see C5.
    {"epoch": int, "dirichletEnergy": [float], "mad": [float]}
  ],
  "environment": {
    "python": str, "torch": str, "torchGeometric": str, "device": str
  }
}

The three capture blocks are separate top-level keys rather than one array,
because each answers a different question and every figure names which one it
reads. `epoch0Metrics` isolates collapse attributable to the propagation
operator; `checkpointMetrics` is the state the reported accuracy comes from and
is the source for every headline energy figure; `finalMetrics` shows whether
early stopping fired meaningfully. An array would force positional indexing and
make a figure's provenance implicit.

`trainingCurve` records both `valLoss` and `valAccuracy` at every epoch even though
only loss drives selection (D-017). This makes the accuracy-selected alternative
derivable from stored data, so the claim that the choice of criterion is immaterial
can be checked rather than asserted.

`bandIndices` sits at the top level rather than inside each capture block because
it is a function of architecture, readout, and depth — none of which change during
a run. Duplicating it three times would create three places for it to disagree with
itself.

`trajectory` is retained as a nullable field even though D-012 withdrew per-epoch
capture. It is normally an empty list. Keeping it means a later dense-capture
diagnostic run needs no schema migration and no change to the aggregation code.

Every hyperparameter is recorded, not encoded in the filename alone. Aggregation
must filter to a fixed hyperparameter configuration before grouping, otherwise a
figure silently averages across settings — e.g. depth-2 runs at lr=0.01 and
depth-16 runs at lr=0.001 would produce a depth curve contaminated by a learning
rate effect, with nothing erroring.

`mitigations` is a list, not a string, so a combination arm needs no schema
migration. It is canonicalized (sorted, joined) only at aggregation time, to
produce a stable group key.

`dirichletEnergy`, `mad`, and `frobeniusSquared` are stored full length (L+1) even
though only the indices in the derived comparable band are metrically comparable
per C1. Storing the full arrays costs nothing and keeps the record faithful to what
the model produced; the band restriction is applied by the plotting code, which
reads `bandIndices` rather than recomputing it.

Figure group-bys this schema supports:
- accuracy vs. depth        — group (convType, numLayers), filter mitigations=[]
- energy vs. layer index    — one record, checkpointMetrics
- MAD vs. depth             — group (convType, numLayers)
- mitigation ablation       — group (mitigations, numLayers), filter convType
- slope vs. depth           — group (convType, numLayers), read contractionSlope
- loss curves by depth      — one record, trainingCurve; the D-005 diagnostic
- energy shift over training — one record, epoch0Metrics vs. checkpointMetrics
                               vs. finalMetrics (three points, not a curve)

Ablation arms (settled): [], ["residual"], ["pairnorm"], ["jk"],
["pairnorm","residual"]. GCNII is a convType, not a mitigation — it is a distinct
convolution, not a wrapper. Rejected: the full 2^3 cross; three of the eight arms
have no motivating question, and the source papers evaluate these mechanisms
singly rather than combined.

### C4 — PyG Data fields

Consumed as returned by `torch_geometric.datasets.Planetoid(name="Cora")`, using
the standard public split. Library field names are kept verbatim.

  x           Tensor [2708, 1433]  float, node features
  edge_index  Tensor [2, E]        long, COO, symmetrized by the loader
  y           Tensor [2708]        long, class index in 0..6
  train_mask  Tensor [2708]        bool, 140 True
  val_mask    Tensor [2708]        bool, 500 True
  test_mask   Tensor [2708]        bool, 1000 True

Transforms are a `data` spec decision, not a contract clause, with one exception
recorded here because it crosses modules: whether `NormalizeFeatures` is applied
changes the scale of X and therefore E(H^(0)). Whichever is chosen, it is applied
identically across every run in the sweep, and the choice is recorded in the
`data` spec (D-002: it is applied).

Self-loops and normalization are handled inside PyG's conv layers rather than at
the data level. This is a runtime verification rather than a spec question, and is
carried as a Claude Code check: confirm empirically that the conv layers are the
only place self-loops are added and that no double normalization occurs, since
`data` ships the raw graph and `metrics` augments its own copy independently
(D-004).

### C5 — Capture timing

Embeddings are captured by an explicit eval-mode forward pass, never reused
from the training forward. Three captures per run:

1. **Epoch 0** — before any gradient step, on the initialized weights.
2. **Selected checkpoint** — the epoch of minimum validation loss (D-017). Those
   weights are restored after the loop and the capture is taken from them. Early
   stopping determines when the loop *halts*; it does not by itself determine which
   epoch is selected, and `bestEpoch` and `epochsRun` will differ by roughly
   `patience` on any run that early-stops.
3. **Final epoch** — the last epoch actually run, captured BEFORE the checkpoint
   weights are restored.

The reasoning for each of the three, and for rejecting per-epoch capture, is in
D-012 and is not duplicated here.

`layerEmbeddings` is not written to disk, with one enumerated exception. `metrics`
consumes it in memory and the results record stores only the reduced scalars, so a
capture costs on the order of a hundred floats rather than ~23 MB. The exception is
the designated figure runs of D-031 — ten configurations flagged in `experiments` —
which additionally save two checkpoint-capture layer tensors each to
`results/embeddings/`, because the report's t-SNE analysis has no other source for
them. Every other run writes scalars only.

Rejected — capture every k epochs during training. This was the original clause
and was withdrawn. It was written before its own MAD-cost concern was resolved:
MAD is pairwise over ~7.3M node pairs per layer, times up to 33 layers, times
every k epochs, which would have forced either a large k or pair subsampling for
the trajectory capture while keeping the full computation at the checkpoint. Two
variants of the same metric in one record is a worse outcome than fewer capture
points. At three captures the cost is negligible and no subsampling is needed.
The capability is retained without cost via the nullable `trajectory` field in C3.

Known cost of the rejection: the recorded prediction below loses its smooth
curve. Three points still resolve the sign of the change, which is the
prediction's actual content.

Recorded prediction (Kiarash, before the sweep): at a working depth (~4 layers),
Dirichlet energy falls over training epochs, driven by cross-entropy contracting
same-class nodes under Cora's homophily. Counter-mechanism: weight norms are free
to grow and energy scales with ||W||^2, which could push energy up. Which
dominates is the empirical question; if energy rises at working depths and falls
only at deep ones, that is a sharper result than "energy falls with depth."
Read from epoch0Metrics vs. checkpointMetrics vs. finalMetrics.

### D-001 changelog

Contract clauses change only with an entry here, so a change cannot be silent.

- 2026-07-19 — C5: per-k trajectory capture withdrawn; three captures (epoch 0,
  best-validation, final epoch). MAD cost was the driver. Full reasoning in D-012.
- 2026-07-19 — C3: `checkpointMetrics` split into `epoch0Metrics`,
  `checkpointMetrics`, `finalMetrics` to hold all three captures; `trajectory`
  redefined as nullable and normally empty. Consequence of the C5 change.
- 2026-07-19 — C1: added the explicit statement that X reaches consumers as
  `layerEmbeddings[0]` and is not passed separately. Clarification, not a change —
  it corrects a stale note in `spec/data_spec.md` that claimed the opposite.
- 2026-07-19 — C1: "the final entry equals the returned logits" narrowed to hold only
  under LastLayerReadout. False under Jumping Knowledge, where index L is a hidden
  representation at hiddenDim and the logits are a separate tensor. Consumers must not
  assume the identity; `metrics` derives the comparable band from tensor widths.
- 2026-07-19 — C1: the comparable band is no longer stated as the fixed range 1..L-1.
  It is derived by `metrics` from tensor widths and is 1..L-1 under LastLayerReadout,
  1..L under JK. Consequence of the clause above.
- 2026-07-19 — C1: the open item on Frobenius normalization closed by D-013.
- 2026-07-19 — C2: noted that which layers are activated is governed by D-016; the tap
  point itself is unchanged.
- 2026-07-19 — C3: each capture block gains `frobeniusSquared` [float] and
  `contractionSlope` (scalar), produced by `metrics` per D-013 and the derived-band
  rule in C1. `bandIndices` [int] added once at the top level, being run-invariant.
  `config.readout` added, since the readout determines banding per D-016. Additive;
  no migration needed for existing records.
- 2026-07-19 — C4: the self-loop / double-normalization item was pointing at the models
  spec, which is now written and does not settle it. Restated as a Claude Code runtime
  check, which is what it always was.
- 2026-07-19 — C3: `trainingCurve` added as a top-level per-epoch array
  (epoch, trainLoss, valLoss, valAccuracy), closing D-005's requirement for a training
  loss record. `results.valLoss` added, being the selection criterion under D-017.
  Note added that `train` assembles the record and `experiments` writes it (D-022).
- 2026-07-19 — C5: capture point 2 reworded. It previously read "the weight state early
  stopping selects," which conflated halting with selection. Early stopping halts the
  loop; the selected epoch is the minimum-validation-loss epoch per D-017, and the two
  differ by roughly `patience` on any early-stopped run. Also stated explicitly that
  the final-epoch capture is taken BEFORE checkpoint weights are restored.


## D-002 — Row-normalize Cora node features (`T.NormalizeFeatures`)

**Date:** 2026-07-19
**Component:** data
**Status:** settled

**Decision.** Apply `torch_geometric.transforms.NormalizeFeatures()` to the Cora
`Data` object as a fixed preprocessing step for every run in the study. It is not
an ablation axis.

**What it does.** Divides each node's binary bag-of-words row by its own row sum,
so each row sums to 1. Per-node, not global: it removes document length as a
determinant of a neighbor's influence in the first aggregation.

**Why (constraint).** Kipf & Welling (arXiv:1609.02907, Sec. 5.2) state that input
feature vectors are (row-)normalized in the published GCN setup. Our proposal
commits to reproducing ~81.5% Cora accuracy at 2 layers, so faithful reproduction
requires it. Verified by fetching the paper, not from memory.

**Why it is safe for the oversmoothing claim.** Two separate arguments, both needed:
1. *Preprocessing held fixed.* Row scaling is per-node, so it does NOT cancel in
   any energy ratio; MAD is invariant (cosine ignores positive per-node scaling)
   but Dirichlet energy is not. Safety comes from applying the same transform to
   every configuration, so it cannot confound any comparison we make. We never
   compare a normalized run against an unnormalized one.
2. *Report the ratio, not raw energy.* We report a normalized energy rather than
   raw `E_l`. Under a global rescale `h -> c*h`, `E` is quadratic so `E_l -> c^2 E_l`
   and the c^2 cancels in the ratio. Plotted log-linear, a global `c` shifts the
   line vertically (intercept) and leaves the slope untouched; the ratio removes
   the shift too. This is what makes GCN and GAT curves overlayable despite
   different activation scales.

**Reported quantity (consequence).** Per-layer per-dimension Dirichlet energy,
normalized at layer 1:

  `(E_l / dim h_l) / (E_1 / dim h_1)`,  for l in the derived comparable band,

plotted on a log y-axis. The denominator is `E_1`, not `E_0` — see the rejected
alternative below and D-003 for why. Cai & Wang (arXiv:2006.13318) give
`E_{l+1} <= (1-lambda)^2 ||W||^2 E_l`, i.e. exponential decay in l, so log-linear
is the natural axis and the fitted slope `log rho` is the one-number summary of a
configuration's oversmoothing rate. `E_0` is still recorded and plotted as an
informational input-space anchor, but it is neither the denominator nor part of
the slope fit.

**Rejected alternatives.**
- *No normalization.* Breaks fidelity to the published baseline for no gain, since
  the metric concern is handled by the two arguments above.
- *Normalization as an ablation axis.* Costs a full extra sweep to answer a
  question the report does not ask.
- *Separate subplots per architecture instead of the ratio.* Prevents overlay and
  still yields no comparable number.
- *Normalizing the energy curve by `E_0` (this entry's original wording).* Stated
  here initially, before D-003. `h_0` is a sparse binary row-normalized
  bag-of-words while `h_1` is dense, post-W, post-ReLU, so the step from `E_0` to
  `E_1` mixes aggregation smoothing with the arbitrary scale and shape of the
  learned projection `W_0`. Using `E_0` as the denominator contaminates every
  curve's first segment with that projection scale. Superseded by the `E_1`
  denominator above; the scale-invariance reasoning in argument 2 is unaffected
  and still holds, since it concerns the ratio form rather than which layer is the
  reference.

**Follow-on flagged.** Kipf & Welling Appendix B (depth 1-10, with/without
residual, best at 2-3 layers, training difficult past 7 without residuals) is
prior art for our depth sweep, but uses 5-fold CV with ALL labels, not the
140-label standard split. Cite as qualitative precedent only; the numbers are not
directly comparable to ours. Carry into the sweep spec.


## D-003 — Per-dimension Dirichlet energy; slope fitted over l >= 1

**Date:** 2026-07-19
**Component:** metrics (contract touchpoint with data, model)
**Status:** settled

**Decision.** Record per-dimension Dirichlet energy `E_l / dim(h_l)` for every
layer including l = 0, in every results record. Fit the contraction slope
`log rho` over l >= 1 only.

**Why per-dimension.** Energy sums over feature dimensions, so raw `E_0` (1433
dims) and `E_1` (hidden width, e.g. 16) are not on a common footing. Dividing by
the feature count removes the dimension-count effect.

**Why the slope excludes l = 0.** Per-dimension division corrects dimension COUNT
but not representation KIND. `h_0` is a sparse binary row-normalized bag-of-words;
`h_1` is dense, post-W, post-ReLU. The drop from `E_0` to `E_1` therefore mixes
aggregation smoothing with the arbitrary scale and shape of the learned projection
`W_0`. Layers l >= 1 share both width and representation kind, so the fitted decay
rate over that range measures smoothing rather than the input-projection artifact.

**Consequence.** `E_0` is reported (informative as the input-space baseline) but is
not load-bearing: it is not the denominator of the reported ratio and is excluded
from the slope fit. Where a normalized curve is plotted, it is normalized at l = 1.

**Rejected alternative.** Normalizing by `E_0` as the denominator — contaminates
every curve's first segment with the `W_0` projection scale.


## D-004 — Metric graph is the augmented graph, fixed across all architectures

**Date:** 2026-07-19
**Component:** metrics (contract; consumes data)
**Status:** settled

**Decision.** Dirichlet energy is computed over the augmented graph
`G~` (`A~ = A + I`) for GCN, GraphSAGE, and GAT alike. `metrics/` receives the raw
`edge_index` and augments internally (mechanism (b)), asserting on entry that the
input contains no self-loops so double-augmentation fails loudly. The augmented
index and `D~^{-1/2}` are precomputed once per run and held as state, not rebuilt
per layer.

**Principle.** The metric graph is a measuring instrument, not part of the model.
Fixing it makes embeddings the only thing that varies across architectures, which
is the comparison the study is built on.

**Why not "match the operator each architecture propagates with".**
- GAT: ill-defined. Its propagation graph is re-weighted by learned attention,
  differing per layer and throughout training. `E_l` and `E_{l+1}` would be measured
  against different graphs, so `rho` would compare quantities with different units.
  A model could also lower measured energy purely by concentrating attention, with
  representations unchanged — the metric would partly measure itself.
- GraphSAGE: well-defined (fixed neighborhoods, no self-loops) and would return raw
  `G`. Overridden for uniformity, not correctness; raw and augmented differ only by
  `d_i -> d_i + 1`, i.e. constant-factor-equivalent quadratic forms.

**Why augmented rather than raw, given near-equivalence.** Cai & Wang
(arXiv:2006.13318, Sec. 2) state their contraction bound over the augmented
normalized Laplacian `Δ~`. Our slope-fitting argument inherits from that bound, so
`G~` keeps the reported `rho` connected to the cited theory. It is the one
non-arbitrary choice available.

**Known cost, to be stated in the report.** For GAT, the fitted `rho` measures decay
of GAT's embeddings with respect to CORA's structure, not with respect to GAT's own
attention-weighted graph. Legitimate, and necessary for cross-architecture
comparison, but not the quantity Cai & Wang's theorem bounds for GAT. One sentence
in the discussion.

**Note.** `SAGEConv` is believed not to add self-loops internally (GraphSAGE keeps
the root term as a separate weight). Verify in Claude Code rather than assume; it
does not change this decision either way.


## D-005 — Results schema must disambiguate optimization failure from oversmoothing

**Date:** 2026-07-19
**Component:** train / metrics (contract)
**Status:** settled — resolved in the train spec

**Problem.** At depth 32 on Cora, both "never trained" and "trained then
oversmoothed" produce test accuracy near the majority-class rate (~30%). Accuracy
alone cannot separate them, and the report's claim is specifically the second.

**Required in the results record.**
- Per-epoch training loss (or at minimum final train loss + whether it descended
  from ~ln 7 = 1.95).
- Per-layer energy at epoch 0 (pre-training) alongside post-training, so collapse
  attributable to the propagation operator is separable from collapse attributable
  to learned weights.

**Resolution.** Both requirements are now met by C3. `trainingCurve` records
`trainLoss`, `valLoss`, and `valAccuracy` at every epoch, so whether the loss
descended from ~1.95 is read directly off the record. `epoch0Metrics` records
per-layer energy on the initialized weights.

**Third requirement added during the train spec.** The stopping rule must not
manufacture the failure it is meant to diagnose. A tight patience can halt a deep
run before its loss begins to descend, producing a flat `trainingCurve` that looks
like an untrainable model but is an artifact of the harness. D-018 sets a generous
patience held constant across configurations for exactly this reason.

**Basis.** Kipf & Welling Appendix B report training becoming DIFFICULT past 7
layers without residuals on Cora — an optimization statement, not a wall-clock one.
Compute is not the binding constraint at this graph size.


## D-006 — models: mitigations attach by composition, not inheritance

**Decision.** `GnnModel` (base) owns the layer loop; per-architecture subclasses supply
only the conv layer. Mitigations are injected at construction as fields and called at
named points in the loop. `models` imports nothing from `mitigations`; both depend on
two protocols:

- `LayerHook.Apply(h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor`
  shape-preserving, called once per layer immediately after the conv.
- `Readout.Apply(layerEmbeddings: list[Tensor]) -> Tensor`
  called once after the loop, consumes the whole stack.

**Why two protocols, not one list.** A hook sees one layer and preserves shape; a
readout consumes the stack and determines the final representation. Merging them
forces a union type and a runtime branch. Keeping them separate makes Jumping
Knowledge's effect on the return value visible in the type system.

**Rejected — mitigation as subclass of architecture.** A mitigation is a modification
applied to a model, not a kind of model, so inheritance is the wrong axis. Concretely:
4 mitigations x 3 architectures is 12 classes before any combination, and
residual + PairNorm together has no honest class name.

**Rejected — mitigation methods on the base class.** Moves the coupling up a level
rather than removing it: the base would have to know the full set of mitigations, and
each addition edits the base plus its dispatch logic.


## D-007 — models: layerHooks is ordered; residual precedes PairNorm

**Decision.** `layerHooks` is an ordered sequence, applied in list order. The default
order when both are present is residual, then PairNorm.

**Why.** Residual is part of the update rule — it defines what H^(l+1) is. PairNorm is
a normalization of the resulting representation, and its invariant is constant total
pairwise squared distance across nodes. Applying PairNorm first and then adding an
un-normalized H^(l) leaves the sum un-normalized, defeating the invariant.

**Consequence.** Order is explicit and configurable rather than baked into the loop, so
the reverse order is runnable as an ablation if we want to demonstrate the point.


## D-008 — models: GCNII is a convType, not a mitigation

**Decision.** GCNII is a fourth entry in the conv factory alongside gcn, sage, gat —
not a LayerHook.

**Why.** Its update is written on the normalized adjacency specifically, with an
initial-residual term in H^(0) and an identity-mapped weight
((1-beta_l) I + beta_l W^(l)). It is an architecture, not a modification: there is no
meaningful "GCNII applied to GAT." PyG resolves it the same way, shipping GCN2Conv as
a conv layer.

**Consequence — uniform conv signature.** GCNII requires H^(0) inside every layer, so
all conv types are called as (x, edgeIndex, x0); non-GCNII convs ignore x0. This avoids
a branch in the loop at the cost of one unused argument for three of four conv types.

**Consequence for the report.** GCNII appears in the mitigation comparison as a
mitigation in the experimental sense while being an architecture in the code sense.
This must be stated explicitly in the writeup or it reads as an inconsistency.


## D-009 — models: no-mitigation case uses null objects, not branching

**Decision.** An unmitigated model is constructed with `layerHooks = []` and
`readout = LastLayerReadout()`, which returns `layerEmbeddings[-1]`. The loop contains
no "are there mitigations?" test.

**Why.** The baseline and every mitigated variant then execute the identical code path,
so a difference in results is attributable to the mitigation rather than to a
divergence in control flow. This matters because the depth sweep's central claim is a
comparison between those exact configurations.


## D-010 — models: layerEmbeddings tap point

**Decision.** `layerEmbeddings[l]` records the hidden state after the conv, after all
layerHooks, and after the activation — but BEFORE dropout. Under `LastLayerReadout` the
final layer applies no activation and no dropout, so `layerEmbeddings[L]` equals the
returned logits, consistent with D-001 C1. Under Jumping Knowledge the final layer IS
activated and dropped out like every other, per D-016.

**After the hooks.** Residual and PairNorm are the treatment under test. Tapping before
them would make every mitigated run produce the same energy curve as its unmitigated
baseline, and the mitigation ablation would measure nothing.

**After the activation.** ReLU is non-expansive with respect to Dirichlet energy —
zeroing negative entries can only pull node representations together — so the
post-activation value is not purely a product of the aggregation operator. It is
nevertheless the correct tap for two reasons: the same activation is applied in every
configuration, so it cannot confound a comparison across architectures or depths (the
same argument that made NormalizeFeatures safe); and the post-activation tensor is what
the next layer actually receives, so it is the representation the model works with.

**Before dropout.** Three reasons, each sufficient on its own:
1. Dropout is stochastic, so the recorded energy would carry a mask-dependent noise
   floor unrelated to smoothing.
2. Dropout is inactive in eval mode, so identical weights would yield different curves
   depending on train/eval state — a metric whose meaning depends on a mode flag the
   metrics consumer does not control.
3. Inverted dropout rescales survivors by 1/(1-p) at train time, inflating raw
   Dirichlet energy by a constant. The E_l/E_1 ratio absorbs this; MAD does not respond
   identically, so the two metrics would disagree for a reason that is an artifact.

**Consequence.** The recorded tensor is not bitwise identical to the tensor passed to
layer l+1 during training. This is deliberate and must be noted in the code comment so
it does not read as a bug.


## D-011 — models: layerEmbeddings are returned attached; the caller detaches

**Decision.** `Forward` returns `layerEmbeddings` still attached to the autograd graph.
Detaching happens at the capture site in `train`, not inside `models`.

**Why.** Under Jumping Knowledge the readout consumes `layerEmbeddings` to produce the
logits, so the loss backpropagates through them. Detaching in `Forward` would silently
zero the gradient to every layer except the last — JK would train incorrectly without
raising an error.

**Rejected — a detach flag on Forward.** Puts a training-mode concern inside the model
and creates a configuration in which JK is silently broken.

**Consequence.** `train` performs `h.detach().cpu()` only at the three capture points of
C5, each of which is a separate eval-mode forward under `torch.no_grad()`. Ordinary
training steps discard `layerEmbeddings` without materializing a copy.


## D-012 — train/models: which weights the energy curve is read from

**Decision.** `layerEmbeddings` are captured at exactly three points per run: epoch 0
before any gradient step, the selected checkpoint, and the final epoch. Not every epoch.

**Epoch 0.** Distinguishes structural oversmoothing from training-induced collapse at
depth 32, per the data-spec requirement to separate optimization failure from
oversmoothing. Collapse present at initialization is a property of the operator, not of
training.

**Selected checkpoint.** The reported test accuracy comes from these weights, so the
energy curve must too. Otherwise a figure pairing accuracy decay with energy decay is
comparing two different models.

**Final epoch.** Its divergence from the checkpoint curve indicates whether early
stopping fired meaningfully — relevant when claiming a deep configuration failed rather
than was merely stopped early.

**Rejected — per-epoch capture.** Storage and I/O cost scales with epochs x seeds x
depths x architectures, and no claim in the proposal requires per-epoch resolution.


## D-013 — metrics: no Frobenius normalization of the reported curve; record ||H||_F^2

**Date:** 2026-07-19
**Component:** metrics
**Status:** settled

**Decision.** The reported energy curve is per-dimension energy normalized at layer 1.
It is NOT additionally divided by `||H_l||_F^2`. `metrics` records `frobeniusSquared`
per layer in every capture block so the normalized variant is derivable at aggregation
time without a rerun.

**Why.** Cai & Wang's bound is `E_{l+1} <= (1-lambda)^2 ||W||^2 E_l`. Weight-norm growth
is inside the mechanism the bound describes, not noise on top of it. A mitigation that
restores deep performance partly by letting `||W||` grow succeeds by a route the theory
predicts; dividing out representation magnitude would hide exactly that and understate
residual and GCNII.

**Rejected — `E_l / ||H_l||_F^2` as the headline curve.** Removes the one term
distinguishing a mitigation acting on the operator from one acting on the weights.

**Closes** the open item carried in D-001 C1.


## D-014 — metrics: both metrics computed over the full node set, no mask

**Date:** 2026-07-19
**Component:** metrics
**Status:** settled

**Decision.** Dirichlet energy and MAD are computed over all 2708 nodes. No split mask
is applied to either.

**Why.** Neither metric uses labels, and oversmoothing is a claim about the learned
representation rather than about the labeled nodes. The subgraph induced by
`train_mask` is 140 of 2708 nodes and close to edgeless, so a masked energy sum would
be dominated by which few edges survived rather than by smoothing.

**Rejected — masking to train or test.** Adds split-dependent sampling noise to a metric
that needs none, and breaks comparability with Cai & Wang's whole-graph bound.


## D-015 — metrics: MAD only; MADGap is out of scope

**Date:** 2026-07-19
**Component:** metrics
**Status:** settled

**Decision.** Global all-pairs MAD from the cosine-distance matrix. MADGap is not
computed.

**Why.** Deli Chen et al. (arXiv:1909.03211) validate MADGap as a predictor of model
performance, splitting nodes into neighboring and remote sets by hop distance. That is
not the claim this study makes; the proposal commits to MAD and does not mention
MADGap; and the remote/neighbor masks require a hop-distance computation `metrics`
otherwise has no reason to hold.

**Rejected — MADGap as a bonus figure.** A second smoothing number with a different
interpretation invites the report to conflate "representations collapsed" with "the
model will do badly" — the two things the depth sweep exists to keep apart.

**Resolved by D-035** (2026-07-20): the exact reduction convention was read off
the paper directly (Eq. 1-4) rather than assumed — averaging over non-zero
entries at both the row-mean and final-mean stage, confirmed by fetching
arXiv:1909.03211 itself. See D-035 for the full formula and the two
implementation choices (diagonal exclusion, zero-denominator clamping) the
paper leaves unstated.


## D-016 — models: the readout declares whether the final layer emits logits

**Date:** 2026-07-19
**Component:** models (contract touchpoint with metrics)
**Status:** settled

**Decision.** The `Readout` protocol carries a boolean property `FinalLayerIsLogits`.
`LastLayerReadout` sets it True: the last conv maps hiddenDim -> outDim, receives no
activation and no dropout, and its output IS the logits. A Jumping Knowledge readout
sets it False: the last conv maps hiddenDim -> hiddenDim, is activated and dropped out
like every other layer, and the readout produces the logits from the whole stack.

**Why.** Two failures follow from treating the last layer identically in both cases.
(a) Width: under JK the last conv would emit 7 dims, so the readout would assemble
logits from a stack whose final entry is already a decision space. (b) Activation: JK
puts index L inside the metrically comparable band, so an un-activated index L would
be the only pre-activation tensor in a band of post-activation ones, and the fitted
contraction slope would cross a discontinuity that is not smoothing.

**Why a property rather than a branch.** An `isinstance(readout, LastLayerReadout)`
test in the loop reintroduces exactly the control-flow divergence D-009 exists to
prevent, and would mean the baseline and the JK arm no longer execute one code path.
A property is data the null object carries, so the loop stays branch-free on type.

**Rejected — a constructor flag on GnnModel.** Puts a readout-derived fact in the
model's signature, where it can be set inconsistently with the readout actually
passed. Two sources of truth for one question.

**Consequence.** The readout name is recorded in the results record
(`config.readout`), since it determines whether index L is a logit or a hidden state
and therefore how `metrics` bands the run.


## D-017 — train: early stopping and checkpoint selection both use validation loss

**Date:** 2026-07-19
**Component:** train
**Status:** settled

**Decision.** Training halts when validation loss has not improved for `patience`
consecutive epochs. The reported checkpoint is the epoch of MINIMUM validation loss,
whose weights are restored after the loop. One signal drives both, so `bestEpoch` is
unambiguous. `valAccuracy` is recorded at that epoch but is not the selection
criterion.

**Stopping and selection are different operations.** Kipf & Welling
(arXiv:1609.02907, Sec. 5.2, verified by fetching) specify only a stopping rule —
maximum 200 epochs, Adam at learning rate 0.01, and a window of 10 in which training
halts if validation loss does not decrease. A stopping rule says when to leave the
loop; it does not say which weights to keep. This entry supplies the missing half.
`bestEpoch` and `epochsRun` will differ by roughly `patience` on any early-stopped run,
and that difference is meaningful rather than a bug.

**Deviation from the proposal, stated deliberately.** The submitted proposal (Sec. 2)
says "early stopping on validation accuracy." This entry refines that to validation
loss, for two reasons. First, fidelity: the ~81.5% reproduction claim in D-002 rests on
matching Kipf & Welling's protocol, and their criterion is loss. Second, resolution:
accuracy on 500 validation nodes moves in steps of 0.002, so a plateau in accuracy can
mask a still-descending loss and trigger patience early.
*Mitigation:* `trainingCurve` records BOTH `valLoss` and `valAccuracy` at every epoch
(C3), so the accuracy-selected alternative is derivable from stored data. The report
states the refinement and shows from the recorded curves whether the choice changes any
conclusion, rather than asserting that it does not.

**Rejected — selecting on validation accuracy while stopping on loss.** Two criteria
means `bestEpoch` can fall outside the window the stopping rule examined, so the
reported number would come from weights the stopping rule never endorsed.

**Rejected — following the proposal's wording unchanged.** Internally consistent, but
weakens the one place the study claims fidelity to a published result, and silently
adopts a coarser signal.


## D-018 — train: generous patience, held constant across every configuration

**Date:** 2026-07-19
**Component:** train
**Status:** settled

**Decision.** `patience = 100`, `maxEpochs = 1000`, identical for every architecture,
depth, mitigation arm, and seed.

**Why generous.** D-005 exists to separate "never trained" from "trained then
oversmoothed." A tight patience defeats that. A 32-layer GCN may sit on a plateau for
many epochs before its loss begins to descend, or never descend; at Kipf & Welling's
window of 10 the run halts around epoch 12 and records a flat `trainingCurve` that is
indistinguishable from genuine untrainability. The harness would have manufactured the
result the study is trying to diagnose. A generous patience cannot manufacture failure
— it can only cost compute, and at Cora's size compute is not the binding constraint.

**Why constant.** Holding the stopping rule fixed across depth is what makes the depth
comparison valid. Tuning patience per depth would confound the depth effect with a
stopping-rule effect, exactly as C3's warning about per-depth learning rates describes.

**Why `maxEpochs = 1000`.** It is a ceiling, not an expected cost: `patience` binds
first, so a converging run halts in the low hundreds and a non-converging deep run halts
around epoch 100. The ceiling is set high enough that it is never the thing that ended a
run, which keeps `epochsRun == maxEpochs` a meaningful signal if it ever occurs.

**Deviation from Kipf & Welling, stated in the report.** Their window of 10 and cap of
200 are appropriate for the two-layer model they studied and inappropriate for a depth
sweep to 32. The depth-2 reproduction must still land at ~81% under the loosened
setting; if it does not, this is the first parameter to check.


## D-019 — train: uniform weight decay on all layers

**Date:** 2026-07-19
**Component:** train
**Status:** settled

**Decision.** Weight decay of `5e-4` applied uniformly to every layer's parameters via
the Adam optimizer. Learning rate `0.01`, dropout `0.5`, carried from Kipf & Welling.

**Why uniform rather than first-layer-only.** Kipf & Welling apply L2 regularization to
the first layer only. That is a sensible choice for a two-layer model and has no
principled extension to 32 layers, where "the first layer" regularizes roughly a
thirtieth of the parameters — a materially different intervention wearing the same name.
Uniform decay is the honest generalization and is what the depth axis requires: the
regularization applied must not itself vary with depth.

**Consequence for the baseline.** This slightly changes the depth-2 reproduction
relative to the published number. If the smoke test lands below ~81%, weight-decay scope
is the first thing to check, before learning rate or initialization.

**Rejected — first-layer-only to match the paper exactly.** Would make the strength of
regularization an implicit function of depth, confounding the one axis the study varies.


## D-020 — train: the checkpoint capture restores weights rather than caching embeddings

**Date:** 2026-07-19
**Component:** train
**Status:** settled

**Decision.** The best `state_dict` is deep-copied in memory whenever validation loss
improves. After the loop ends, the final-epoch capture is taken FIRST on the current
weights, then the best `state_dict` is restored and the checkpoint capture is taken.
Test-set evaluation happens after the restore.

**Why not cache embeddings at each improvement.** Caching would hold `L+1` tensors of
`[2708, hiddenDim]` and recompute nothing, but it captures during training-mode forward
passes, which C5 forbids — captures must come from an explicit eval-mode pass so dropout
is inactive. Caching a `state_dict` is smaller than caching embeddings at depth 32 and
keeps every capture on the same footing.

**Ordering matters and is easy to get wrong.** Capturing the final epoch after restoring
the checkpoint would silently make `finalMetrics` a duplicate of `checkpointMetrics`,
and the early-stopping diagnostic in D-012 would report no divergence on every run. The
test plan asserts the two differ whenever `bestEpoch != epochsRun`.


## D-021 — train: macro-F1 implemented in-repo; scikit-learn is test-only

**Date:** 2026-07-19
**Component:** train
**Status:** settled

**Decision.** Macro-F1 is implemented as a vectorized torch function inside `train`.
`scikit-learn` is not a runtime dependency and does not appear in `requirements.txt`; it
appears in `requirements-dev.txt` and is used in exactly one test, which compares the
in-repo value against sklearn's on fixed input.

**Why.** The course grades how much of the code the team wrote versus took from prior
work. Macro-F1 over seven classes is roughly ten lines of tensor operations — per-class
true positives, false positives, false negatives via `bincount` on a class-index pair,
then an unweighted mean of per-class F1. It is defensible in the "explanation of the
source code" section in a way an imported call is not.

**Why the sklearn test nonetheless.** Own implementation is the goal; unverified own
implementation is not. One equality assertion against a reference removes the risk
without adding a runtime dependency.

**Note — scope.** The test-only scope covers the numerical modules: `data`, `models`,
`metrics`, `train`, and `experiments` have no scikit-learn dependency at runtime or
otherwise, and `data_spec` and `metrics_spec` are correct as written. `viz` sits
outside that scope and uses scikit-learn at runtime for t-SNE (D-032), so
`requirements.txt` records it as a viz-scope dependency. The reproducibility-critical
path — everything that produces a number — remains free of it.


## D-022 — train: assembles the results record; experiments writes it

**Date:** 2026-07-19
**Component:** train / experiments
**Status:** settled

**Decision.** `TrainRun` returns the complete results record as a dict conforming to
C3 — including the model configuration block, which `models` supplies to it.
`experiments` serializes that dict to
`results/<convType>_<mitigations>_d<depth>_s<seed>.json`. `train` never touches disk.

**Why.** A `train` that does not write is testable without a filesystem fixture, and a
single writer means the filename convention exists in exactly one place. If `train`
wrote its own file, the sweep and the single-run path would each need the naming logic
and could drift.

**Consequence.** `SetSeed(seed)` lives in `train` and is called at the top of `TrainRun`;
`experiments` passes the seed rather than seeding on its own. No `utils/` module is
created to hold one function.

**Note.** `models_spec` previously described the configuration record as something
`train` "embeds in each `results/...json`." Corrected to "embeds in the record it
returns." The same spec restated the results filename in a form that omitted the
`<mitigations>` segment. Rather than correcting it there, the filename was removed
from `models_spec` entirely and replaced with a pointer to C3, so the convention
exists in one place.


## D-023 — experiments: hiddenDim = 64, total width across heads

**Date:** 2026-07-19
**Component:** experiments (blocks models, metrics, train)
**Status:** settled

**Decision.** `hiddenDim = 64` for every architecture, depth, mitigation arm, and seed.
It denotes TOTAL width of the hidden representation, not per-head width. GAT uses 8
attention heads of `hiddenDim / 8 = 8` features each, held at 8 heads across all
depths; under `LastLayerReadout` its final layer is a single head emitting 7, matching
Velickovic et al. (arXiv:1710.10903, Sec. 3.3, verified by fetching).

**Why not 16.** No single value matches every reference: Kipf & Welling publish GCN at
16 hidden units, Velickovic et al. publish GAT at 8 heads x 8 = 64 total. Capacity
parity across architectures is required for the depth comparison to be interpretable,
so one number governs all three and some deviation is unavoidable. At 16 with 8 heads,
GAT receives 2 dimensions per head — attention over 2-dimensional keys is close to
degenerate, and GAT would underperform for a reason unrelated to depth, placing a
confound in the study's headline figure. At 64 the only cost is that GCN runs wider
than its published width.

**Secondary reason.** The energy measurement has more dynamic range at 64: collapse
toward a rank-1 subspace is a larger measurable drop from 64 dimensions than from 16,
and the mitigation ablation needs that range to resolve differences.

**Fidelity, recovered rather than sacrificed.** A separate one-off arm runs the 2-layer
GCN at `hiddenDim = 16` over the same 10 seeds and is reported as the published-baseline
reproduction. The sweep itself is 64 throughout. Cost is minutes of compute; the
deviation becomes a measured quantity rather than a stated caveat.

**Closes** the GAT head-handling open question in `models_spec` and the `hiddenDim`
open item in `models_spec`, `metrics_spec`, and `train_spec`.

**Related deviations, held deliberately.** GAT publishes ELU and dropout 0.6; this study
uses ReLU and dropout 0.5 for every architecture. Activation uniformity is load-bearing
rather than cosmetic: D-010's justification for the post-activation tap rests on the
same activation being applied everywhere, and ELU is not non-expansive on Dirichlet
energy in the way ReLU is, so a mixed-activation study would attribute part of GAT's
energy curve to its nonlinearity rather than its aggregation. Stated in the report as a
deviation with this reason.


## D-024 — mitigations: residual is plain identity, no-op on width mismatch

**Date:** 2026-07-19
**Component:** mitigations
**Status:** settled

**Decision.** `ResidualHook.Apply` returns `h + hPrev` when the shapes match and `h`
unchanged when they do not. No learned projection. Under `LastLayerReadout` at depth L,
the residual therefore fires on layers 2 through L-1 only: layer 1 maps 1433 -> hiddenDim
and layer L maps hiddenDim -> outDim, and neither admits an identity residual.

**Why no projection.** A learned projection to reconcile mismatched widths adds
parameters at the boundary layers, and it is not what Kipf & Welling did in the
Appendix B depth study this arm is compared against. The comparison would then be
against a different intervention wearing the same name.

**Why the skip must be explicit.** A hook that silently returns `h` on any shape
disagreement would let a misconfigured run report a residual arm in which the residual
never fired, with nothing erroring. The test plan asserts the addition occurs on exactly
L-2 layers at a known depth.

**Known cost.** If the residual arm shows no benefit at depth, the absent layer-1
residual is a candidate explanation. Recoverable as a one-off diagnostic; not built into
the ablation.


## D-025 — mitigations: PairNorm uses PN-SI at fixed scale 1.0

**Date:** 2026-07-19
**Component:** mitigations
**Status:** settled

**Decision.** The scale-individual variant (PN-SI), scale `s = 1.0`, held fixed across
every architecture, depth, and seed. Two steps: center by subtracting the column mean
over nodes, then divide each row by its own L2 norm and multiply by `s`.

**Why PN-SI rather than plain PN.** The authors' reference implementation states that
plain PN behaves badly with a symmetric normalized adjacency, their paper's experiments
having used a row-normalized adjacency, and recommends PN-SI or PN-SCS for GCN and GAT.
`GCNConv` is symmetrically normalized, so this study sits in the flagged case. A later
note in the same source says a bug was fixed and plain PN should now work, leaving the
guidance unresolved; PN-SI does not depend on which note is current.

**Why the scale is not tuned.** `s` is data-dependent and the authors tune it per
dataset. The coordinated hyperparameter search holds one configuration fixed across all
configurations, so tuning `s` would compare a tuned PairNorm against untuned baselines.
Stated in the report as a limitation with its direction: this can only understate
PairNorm, never overstate it.

**Implementation guard.** The row-norm denominator is clamped away from zero. ReLU
produces all-zero rows at depth, and an unclamped division would yield `nan` in exactly
the regime the study measures.

**Rejected — sweeping `s`.** A full extra axis answering a question the report does not
ask.


## D-026 — mitigations: Jumping Knowledge aggregates by max pooling over indices 1..L

**Date:** 2026-07-19
**Component:** mitigations (contract touchpoint with models, metrics)
**Status:** settled

**Decision.** `JkReadout` uses `JumpingKnowledge(mode="max")` followed by
`Linear(hiddenDim -> outDim)`, aggregating `layerEmbeddings[1:]`.

**Why max rather than concatenation.** Concatenation produces `T x hiddenDim`, so the
readout's linear layer would be `2 x 64 -> 7` at depth 2 and `32 x 64 -> 7` at depth 32.
The JK arm's parameter count would grow with depth, and a JK depth curve would confound
the mitigation with a capacity increase — precisely at the depths where the study claims
mitigations help. Max pooling holds the readout input at `hiddenDim` at every depth. Note
that the JK paper's own citation-network experiments selected depths from {2, 3}; the
concatenation variant was never exercised at this study's depths.

**Rejected — LSTM-attention.** Fixed-size, so not subject to the capacity objection, but
it introduces a bidirectional LSTM to defend line by line in the source-code section and
the video, for a mechanism the study does not ask about.

**Why indices 1..L and not 0..L.** Index 0 is the raw 1433-dimensional input and cannot
be max-pooled against hiddenDim-wide hidden states. The JK paper aggregates the first
linear-transformed representation, which in our indexing is index 1.

**Consequences.** `FinalLayerIsLogits = False` (D-016), so under JK the final conv emits
`hiddenDim`, is activated and dropped out like every other layer, and the comparable band
is `1..L`. The readout's `Linear` is not a message-passing layer and does not count
toward `numLayers`, so "L layers = L hops" holds unchanged.


## D-027 — experiments: depth grid {2, 4, 8, 16, 32}, seeds 0..9

**Date:** 2026-07-19
**Component:** experiments
**Status:** settled

**Decision.** Five depths, log-spaced: 2, 4, 8, 16, 32. Ten seeds, the literal integers
0 through 9, recorded in the record and the README.

**Why log-spaced.** Cai & Wang's contraction bound makes energy decay exponential in
depth, so log-spaced depths are evenly spaced on the log-energy axis every headline
figure uses.

**Rejected — adding 24.** Buys resolution between two points where the curve has already
collapsed, and breaks the log spacing. The fine structure of the energy curve comes from
the layer index within a run, not from more depth points.

**Rejected — seeds from a meta-RNG.** A literal list is one fewer piece of state a
reproduction attempt has to recover.


## D-028 — experiments: "jk" stays in config.mitigations, with a consistency assertion

**Date:** 2026-07-19
**Component:** experiments (contract touchpoint with models)
**Status:** settled

**Decision.** Jumping Knowledge appears both in `config.mitigations` as `["jk"]` and in
`config.readout` as `"jk"`. The redundancy is accepted. `experiments` asserts
`("jk" in mitigations) == (readout == "jk")` before writing any record.

**Why the redundancy is kept.** The results filename is built from the mitigation list,
and `mitigations = []` renders as `none`. Dropping `"jk"` would collide the JK arm with
the baseline at `gcn_none_d8_s0.json`. Removing it would require adding the readout to
the filename — a contract change to C3 for no benefit.

**Why an assertion rather than derivation.** Deriving one field from the other would put
the naming logic in two modules. One assertion at the single write point is cheaper and
fails loudly.

**Test.** `ResultPath` values are unique across the whole grid; with `"jk"` dropped this
test fails on the collision, so the decision is checkable rather than remembered.


## D-029 — experiments: coordinated hyperparameter search, tuned once and frozen

**Date:** 2026-07-19
**Component:** experiments
**Status:** settled

**Decision.** Eight configurations on GCN at depth 2 over three seeds — learning rate
{0.005, 0.01} x dropout {0.5, 0.6} x weight decay {5e-4, 1e-3} — selected on mean
validation accuracy. The winner is frozen and used by every architecture, depth,
mitigation arm, and seed in the study.

**Why tuned at depth 2 and held fixed.** Tuning per depth would confound the depth effect
with a hyperparameter effect, which is the exact failure C3's group-by warning describes.
The search exists to justify the fixed configuration, not to find an optimum.

**Known cost.** GAT publishes at learning rate 0.005 and dropout 0.6 on Cora, GCNII at
dropout 0.6 with far larger convolutional weight decay. A single shared configuration
cannot match all three published setups. Stated in the report as a limitation, with the
direction named: architectures whose published settings differ most from the winner are
the ones most likely to be understated.

**Rejected — per-architecture or per-depth search.** Multiplies the grid and destroys the
comparison the study is built on.

**Resolved by `FINDINGS.md` F-007** (2026-07-21): no, not cleanly — the 8
configurations span only 1.14 accuracy points, comparable to individual
per-config seed-to-seed std (0.12-0.42 points), and the top 2 configs differ
by only 0.47 points, barely outside the winner's own 0.42-point std. The
winner nonetheless matches the published baseline exactly, so the fallback
criterion this entry names would have selected it regardless of whether the
primary ranking is trustworthy.


## D-030 — experiments: idempotent sweep, atomic writes, single writer

**Date:** 2026-07-19
**Component:** experiments
**Status:** settled

**Decision.** `RunSweep` skips any configuration whose results file already exists unless
`force=True`. Records are written to a temporary path and renamed, so a file is either
complete or absent. `experiments` is the only module that writes to `results/`, and
`ResultPath` is the only expression of the C3 filename convention.

**Why.** At 534 runs on a single machine across a two-week window, the sweep will be
interrupted. Restarting from zero each time is the failure mode that consumes the
schedule; a half-written JSON that parses is worse, because it corrupts an aggregate
silently.

**Grid totals, asserted in the test plan.** A 150, B 200, C 50, D 100, E 10, F 24 —
534 runs. An off-by-one in an enumeration is invisible until the aggregation has a hole.

**Resolved by D-039** (2026-07-20): record and continue, with a
`<runId>.failed.json` failure marker in `results/` so the gap is visible at
aggregation rather than resembling a run that was never scheduled. Decided
before the first full sweep, as required here.


## D-031 — viz/experiments: embeddings persisted for ten designated figure runs

**Date:** 2026-07-19
**Component:** viz / experiments (contract change to D-001 C5)
**Status:** settled — FLAGGED CONTRACT CHANGE

**Problem.** C5 stated that `layerEmbeddings` is never written to disk, and no component
saves trained weights. After a sweep completes, the `[2708, 64]` embedding matrices the
proposal's qualitative analysis depends on therefore exist nowhere, and `viz` cannot
produce the shallow-versus-deep t-SNE figure from `results/*.json`. This was a gap, not a
disagreement — nothing in the log was wrong, but a required deliverable had no input.

**Decision.** `RunConfig` gains `saveEmbeddings: bool = False`. It is set True for ten
configurations: GCN, seed 0, at depths 2 and 32, for each of the five mitigation arms
([], ["residual"], ["pairnorm"], ["jk"], ["pairnorm","residual"]). Each flagged run saves the checkpoint-capture tensors at the first and last band
indices to `results/embeddings/<runId>_l<index>.pt`. Those indices coincide at depth 2
under `LastLayerReadout`, whose band is the single point `1..L-1`, so those four runs
save one tensor each; the depth-2 JK run has a two-point band `1..L` and saves two.
Sixteen tensors in total, roughly 11 MB.

**Why these ten.** Depths 2 and 32 are the contrast the figure makes. Seed 0 only,
because the projection is qualitative and averaging over seeds is meaningless for it. All
five arms rather than only the winner, because the winner is not known until arm B
aggregates, and re-running flagged configurations afterward would cost more than the
storage saves.

**Why not all runs.** A full depth-32 run's `layerEmbeddings` is roughly 23 MB; 534 runs
would produce gigabytes. Ten runs at two tensors each is roughly 14 MB.

**No change to the run count.** These are existing configurations in arms A and B with a
flag set, not new runs. The grid remains 534.

**Storage.** `results/embeddings/` is gitignored; the README documents that the sweep
regenerates it.


## D-032 — viz: t-SNE only; UMAP is not used

**Date:** 2026-07-19
**Component:** viz
**Status:** settled

**Decision.** Embedding projection uses `sklearn.manifold.TSNE` with `random_state = 0`
and `perplexity = 30`, both stated in the figure caption. UMAP is not used.

**Why.** The proposal says "t-SNE / UMAP," which is an either/or rather than a
requirement for both. UMAP adds `umap-learn` and `numba` for a second view of the same
embeddings, and two projections that disagree slightly is a question to answer in the
video for no analytical gain. One method whose parameters are stated and fixed is more
defensible than two that cannot both be explained in full.

**Fitted separately per panel.** A shared fit across depths would impose one
neighbourhood structure on both and manufacture the difference the figure exists to show.

**Framing constraint for the report.** The projection illustrates collapse; it does not
measure it. Dirichlet energy and MAD measure it. The figure is labelled qualitative, and
the text says so, because a t-SNE panel presented as evidence of a quantity is an
overclaim a grader will catch.


## D-033 — viz: aggregation separated from plotting; matplotlib at single-column width

**Date:** 2026-07-19
**Component:** viz
**Status:** settled

**Decision.** Two layers. `LoadRecords` -> `BuildTable` -> `Aggregate` return tables;
`Plot*` functions consume tables and touch matplotlib. Mean and standard deviation are
computed at read time and never written back (C3). Figures are 3.5 inches wide, matching
IEEE single-column, with font sizes set explicitly and rendered under the `Agg` backend.

**Why the separation.** The report needs numbers in tables as much as figures — the
results section quotes accuracies with standard deviations, and the mitigation comparison
reads better as a table than as five overlaid curves at single-column width. The defense
gate can then inspect an aggregate without rendering, and a figure that looks wrong is
diagnosed by reading the table behind it.

**Why sizing is decided now.** The article is five pages. Legibility at 3.5 inches
constrains font size, tick density, and how many series one axes can carry. Discovering
that during layout means regenerating figures against a deadline.

**`bandIndices` is read from the record, never recomputed.** It is `1..L-1` under
`LastLayerReadout` and `1..L` under JK (D-001 C1). A plotting function assuming the
former would silently truncate every JK curve by one layer.

**Rejected — seaborn.** A dependency to change defaults that can be set directly.

**Rejected — caching aggregates to disk.** A derived file beside `results/` that can go
stale, for a computation that takes seconds.


## D-034 — models: GCNII resolves via uncounted input/output projections; numLayers stays a hop count

**Date:** 2026-07-20
**Component:** models
**Status:** settled

**Decision.** `GcniiModel` wraps exactly `numLayers` `GCN2Conv` layers at constant
`hiddenDim` width, preceded by an uncounted `Linear(inDim, hiddenDim)` input
projection and, under `LastLayerReadout`, followed by an uncounted
`Linear(hiddenDim, outDim)` output projection that replaces the final layer's
activation. `numLayers = L` therefore means `L` real `GCN2Conv` hops, matching
"`L` layers = `L` hops" for every other architecture (models_spec.md's confirmed
depth assumption). `x0`, the initial-residual term `GCN2Conv` requires at every
layer, is the input projection's activated-and-dropped-out output, held fixed
across all `L` hops — NOT raw `X`. `layerEmbeddings[0]` remains raw `X` regardless,
per D-001 C1's changelog; the projected `x0` and the input projection's output are
never tapped into `layerEmbeddings`.

**Why.** `GCN2Conv` requires `x` and `x_0` at equal width — `channels` is a single
int in PyG's implementation (verified by reading `GCN2Conv.__init__`/`forward`
directly) — so, unlike `GCNConv`/`SAGEConv`/`GATConv`, it cannot itself perform the
`1433 -> hiddenDim` or `hiddenDim -> outDim` boundary maps the generic per-layer
loop assumes every conv type can do. This mirrors the original GCNII paper's own
architecture: input FC -> K GCNII layers -> output FC, with reported depth = K.

**Rejected — counting the input projection as layer 1.** GCNII at nominal depth
`L` would then perform only `L - 1` real hops, breaking hop-comparability with
`gcn`/`sage`/`gat` at the same `L` — a confound in the study's primary axis, worse
than the cost below.

**Consequence for the `LastLayerReadout` band.** Index `numLayers` is excluded
from the metrically comparable band under `LastLayerReadout` regardless of
architecture (C1: "index L is excluded when it is the logit space"), so
substituting the output projection for the final activation at that one index
does not touch the band; `layerEmbeddings[1 .. numLayers - 1]` are all raw
`GCN2Conv` outputs at `hiddenDim` width, so width homogeneity across the band
still holds.

**Known cost, to be stated in the report.** Two extra parameterized linear layers
(`1433 -> 64` and `64 -> 7`) that `gcn`/`sage`/`gat` do not carry. Bounded, does not
scale with depth. Direction named: this can only overstate GCNII's capacity
relative to its peers, in GCNII's favor, at every depth.

**Resolved by D-042** (2026-07-21): `alpha=0.1`, `theta=0.5`, matching Chen et
al.'s published Cora configuration. `GcniiModel` still takes them as required
constructor arguments rather than baking in a default at the model level —
D-042 is the experiments-level decision that supplies the actual values.


## D-035 — metrics: MAD reduction convention, verified against the source paper

**Date:** 2026-07-20
**Component:** metrics
**Status:** settled — resolved by reading arXiv:1909.03211 directly (Chen, Lin, Li,
Li, Zhou, Sun, "Measuring and Relieving the Over-smoothing Problem for Graph Neural
Networks From the Topological View," AAAI 2020), Eq. 1-4, not from memory.

**Decision.** MAD is computed exactly per the paper's Eq. 1-4, restricted to the
global variant (`M^tgt` all-ones), since MADGap is out of scope (D-015):

1. `D_ij = 1 - cosine_similarity(H_i, H_j)` for all `i, j` (Eq. 1).
2. Row average `D̄_i = (sum_j D_ij) / |{j : D_ij > 0}|` — averaged over NON-ZERO
   entries of the row only (Eq. 3).
3. `MAD = (sum_i D̄_i) / |{i : D̄_i > 0}|` — averaged over rows with a non-zero
   average only (Eq. 4).

**Two implementation choices the paper leaves unstated, both needed to make the
formula well-defined and to satisfy metrics_spec.md's collapse-floor test (all rows
identical -> MAD exactly 0):**

- **Diagonal (self-pairs, `i = j`) excluded from `D` before the row-average
  reduction.** Mathematically inert for any row with positive norm —
  `cosine_similarity(H_i, H_i) = 1` always, so `D_ii = 0` and Eq. 3's non-zero
  filter already drops it — but load-bearing for an all-zero row: the row-norm
  clamp needed elsewhere (metrics_spec.md's numerics note, for ReLU-dead nodes at
  depth) makes a zero row's self-cosine-similarity compute as `0`, not the true
  self-similarity of `1`, which would otherwise leave a spurious `D_ii = 1` that
  survives the non-zero filter and contaminates that node's own row average.
  Consistent with the paper's own framing of MAD as distance "from nodes to
  OTHER nodes."
- **Both reduction denominators (Eq. 3's per-row count, Eq. 4's global count) are
  clamped to return exactly 0, not NaN, when the count is 0.** Needed for the
  collapse floor itself (all rows identical -> every `D_ij = 0` -> every count is
  0) and consistently for any node whose row fully collapses.

**Why this matters, not cosmetic.** At depth, ReLU produces exact zero rows and
near-collapsed representations produce exact zero pairwise distances — precisely
the regime the depth sweep is about. Averaging over ALL entries instead (a
reading the paper explicitly rejects by specifying the non-zero filter) would
treat those exact-zero distances as informative near-collapse signal rather than
filtering them, diverging from Eq. 3/4's stated convention exactly where it
matters most.


## D-036 — metrics: on Cora, the Dirichlet-energy null space is sqrt-degree-weighted, not literal collapse

**Date:** 2026-07-20
**Component:** metrics
**Status:** settled — discovered empirically (verified numerically), resolved
as a test-design decision. Not a `FINDINGS.md` entry: this is a stable
mathematical property of a fixed operator, not an evolving empirical result
about model behavior — see `FINDINGS.md`'s preamble for why the two files
draw that line differently.

**Finding.** `Δ~ = I - D~^{-1/2} A~ D~^{-1/2}` (the augmented symmetric-normalized
Laplacian D-004 fixes as the metric graph) has a null space spanned by
`sqrt(d~_i)`, not by the constant vector — standard for the *symmetric*-normalized
Laplacian, as opposed to the random-walk-normalized one, whose null space is
literally constant. On a regular graph the two coincide (every `d~_i` is equal,
so a constant vector is a special case of sqrt-degree-proportional). Cora is far
from regular (degree 2 to 169, mean ~4.9), so on the graph this study actually
uses, identical node representations do NOT give zero Dirichlet energy.

**Verified numerically** on Cora's real graph: identical rows -> energy 2774.4;
rows proportional to `sqrt(d~_i)` -> energy 4.4e-11 (floating-point zero).

**Consequence for metrics_spec.md's test plan.** The "collapse floor" test
("an embedding with all rows identical gives Dirichlet energy 0") is true only on
a regular graph. The test suite covers both: the literal identical-rows case on a
synthetic regular (cycle) graph, and the actual Cora null space
(sqrt-degree-proportional rows) on Cora itself. `MeanAverageDistance` is exactly 0
in both cases regardless, since cosine distance ignores positive per-node scaling
(D-002's argument 1) — a sqrt-degree-proportional embedding is still literal
representation collapse in direction, just not in magnitude.

**Consequence for the report.** A real deep GCN that "fully collapses" under this
metric is not converging to literally identical node representations — it is
converging toward a representation proportional to `sqrt(degree)`, so hub nodes
retain larger-magnitude representations even at total collapse. Worth one sentence
in the report's methodology or discussion, since it is a non-obvious property of
the exact quantity `contractionSlope` summarizes, not merely a unit-test detail.


## D-037 — metrics: FitContractionSlope floors energy at 1e-12 before taking log

**Date:** 2026-07-20
**Component:** metrics
**Status:** settled — discovered empirically while running metrics_spec.md's
depth qualitative gate test (not hypothetical), resolved as a code-level
decision. Not a `FINDINGS.md` entry: the floor and its cost are a permanent
property of `FitContractionSlope`'s implementation, not an evolving empirical
result about model behavior.

**Problem.** A genuinely collapsed deep GCN (32 layers, unmitigated, 200 epochs)
produces per-dimension Dirichlet energy that is exact-`0.0` in float32 at some
band layers. `log(0)` is undefined, so `FitContractionSlope` raised instead of
returning a slope, on the very configuration the depth sweep most needs a slope
for.

**Decision.** `FitContractionSlope` floors each energy at `1e-12` before taking
the log: `log(max(E_l, 1e-12))`. metrics_spec.md's implementation plan does not
mention this case; it is filled in here the same way D-035's MAD zero-denominator
clamps were — an unstated numerical-stability detail needed to make the formula
well-defined, not a scientific tradeoff.

**Known cost.** Caps how negative the fitted slope can read for a fully collapsed
run — the reported `contractionSlope` understates the true decay rate once energy
underflows below the floor. Direction named: this can only make a fully collapsed
configuration's slope look shallower than it truly is, never the reverse, so it
does not manufacture an oversmoothing effect that is not there. This applies
symmetrically to a positive (growing) slope, not only the decaying case the
original wording illustrated: capping the magnitude of change pulls a reported
slope toward zero regardless of which direction it runs.

**Quantified, 2026-07-21** (prompted by a direct question about whether this
floor interacts with D-038's epoch0-vs-checkpoint gate): for
`gcn_none_d32_s0`, the floor is not a rare edge case for this configuration —
it binds on a substantial fraction of the band in both captures. `epoch0Metrics`:
6 of 31 band layers (the deepest ones) sit below `1e-12`; reported slope
`-0.7579` vs. the true unfloored slope `-0.8508`. `checkpointMetrics`: **22 of
31** band layers are below the floor (the shallow end this time, since energy
starts near-zero and grows); reported slope `+0.8622` vs. true unfloored
`+1.1435`. Both reported values understate the true magnitude of change, per
the known-cost direction above — the checkpoint blow-up F-001/F-002 report is
if anything *more* extreme than the `contractionSlope` field alone would
suggest. See `FINDINGS.md` F-001/F-002, which report the energy ratio directly
(not through `contractionSlope`) for exactly this reason.


## D-038 — metrics: depth qualitative gate test is gated on epoch0Metrics, not checkpointMetrics

**Date:** 2026-07-20
**Component:** metrics
**Status:** settled

**Decision.** metrics_spec.md's "depth qualitative gate" test plan item
(a 32-layer unmitigated GCN shows monotonically decreasing per-dimension energy)
is asserted against `epoch0Metrics`, not `checkpointMetrics`. At epoch 0
(before any gradient step, on the initialized weights) the monotonic decay
holds cleanly and is well-defined. At the checkpoint (post-training, under this study's
settled hyperparameters), it does not — see `FINDINGS.md` F-002 for the
epoch-0/checkpoint contrast this rests on, and F-001 for the broader
architecture-dependent picture.

**Why.** Asserting the property at the checkpoint would be a flaky or false
test given what training actually does under this project's hyperparameters;
re-deriving the training-dynamics explanation inline in a `metrics` test would
also be out of that component's scope. Gating on epoch 0 keeps the test
meaningful, stable, and scoped to what `metrics` alone is responsible for.

**Note.** This entry originally also carried an empirical narrative about
training dynamics at depth 32. That content has been moved to `FINDINGS.md` —
the architecture-dependent finding and its correction to F-001, the inferred
GCN mechanism specifically to F-003 — since it is an evolving empirical
result, not a project decision, and does not belong in this log per this
file's own stated scope. Not restated here, to keep one source of truth per
fact.


## D-039 — experiments: failure handling — record and continue, with a failure marker

**Date:** 2026-07-20
**Component:** experiments
**Status:** settled

**Decision.** `RunSweep` catches any exception raised while running a single
configuration, writes a failure marker to `results/<runId>.failed.json`
(containing the error message and the config) instead of a results file, prints
it, and continues to the next configuration. The completion summary reports a
failed count alongside executed and skipped.

**Why.** At 534 runs on a single machine over roughly two weeks, one bad run (a
`nan` loss at depth 32, say) aborting the whole sweep is the exact failure mode
D-030's idempotent/resumable design already exists to avoid on the success path.
A visible failure marker keeps the gap distinguishable at aggregation time from a
run that was simply never scheduled, per the open question's own framing.

**Rejected — abort on first failure.** Simpler, but loses all remaining
unattended progress after one bad run.


## D-040 — experiments: arms E and F write to their own subdirectories

**Date:** 2026-07-20
**Component:** experiments
**Status:** settled — bug found while running experiments_spec.md's own "no
duplicate paths" test, not hypothetical

**Problem.** The C3 filename convention (`<convType>_<mitigations>_d<depth>_s<seed>.json`)
does not encode `hiddenDim` or the training hyperparameters. Arm E (fidelity,
`hiddenDim=16`) reuses the exact `(convType, mitigations, depth, seed)` space as
arm A's `gcn` depth-2 subset; arm F (the hyperparameter search, varying
`learningRate`/`dropout`/`weightDecay`) collapses all 8 per-seed hyperparameter
combinations onto one filename each. Enumerating the full 534-config grid and
computing `ResultPath` for every entry gives only 500 distinct paths: arm E's 10
files are exact duplicates of 10 of arm A's, and arm F's 24 configs map onto 3
filenames arm A already claims. Run into one flat `results/` directory as
experiments_spec.md's Outputs section literally describes, later arms would
silently overwrite earlier ones' result files — a direct threat to the study's
reported numbers, verified by hand arithmetic (150+200+50+100+0+0 = 500) before
being treated as settled.

**Decision.** Arm E writes to `results/fidelity/`, arm F to `results/hpsearch/`.
Arms A-D keep the flat `results/` layout exactly as C3 describes. The
orchestrating driver (not yet built; a notebook or top-level script per
experiments_spec.md's own "Approach") is responsible for calling
`RunSweep(BuildGrid("E"), "results/fidelity")` and
`RunSweep(BuildGrid("F"), "results/hpsearch")` separately from
`RunSweep(BuildGrid(arm), "results")` for arm in A-D.

**Refinement, same day: subdirectories alone do not fix arm F.** Isolating arm F
into its own subdirectory removes the collision WITH arm A, but does not resolve
arm F's OWN internal collision: all 24 of its configs share
`(convType="gcn", mitigations=[], numLayers=2)`, varying only `seed` (3 values)
and the 8 `(learningRate, dropout, weightDecay)` combinations — so 8 configs
collapse onto each of the 3 seed-only filenames even within `hpsearch/` alone.
Confirmed by hand (150 arm-A paths, +0 new from E, +0 new from F = 513, not 534)
before being treated as settled, the same way the original 500-path finding was.

`ResultPath` gains a third parameter, `includeHyperparams: bool = False`
(additive, does not change any existing call's output). When `True`, it appends
`_lr{learningRate}_do{dropout}_wd{weightDecay}` to the stem, using the config's
own field values directly — not a comparison against `DEFAULT_LEARNING_RATE`
et al., which are mutated in place once arm F's winner is known (grid.py's own
module docstring says so) and would make the suffix's presence depend on
call-time global state, silently changing `ResultPath`'s output for a config
whose fields never changed. `RunSweep` gains the same passthrough parameter.
The driver calls `RunSweep(BuildGrid("F"), "results/hpsearch", includeHyperparamsInPath=True)`;
every other call keeps the default `False`.

**Why subdirectories plus a targeted filename extension, rather than extending
every filename.** The C3 filename STEM convention is untouched for arms A-D —
the smaller deviation stays smaller. Arms E and F are one-off / diagnostic arms
(the published-baseline reproduction and the hyperparameter search
respectively), not part of the headline cross-architecture aggregation C3's
"Figure group-bys this schema supports" list enumerates for arms A-D, so
extending only arm F's filenames does not affect any of those aggregations.

**Consequence for `viz/` and the README.** `viz/`'s `LoadRecords` (not yet
built) must glob `results/*.json` for arms A-D and, separately, `results/fidelity/*.json`
/ `results/hpsearch/*.json` when those arms' results are needed. The README's
description of the results layout must document all three directories, not one.


## D-041 — experiments: arm D's mitigation is selected by highest mean test accuracy at depth=32

**Date:** 2026-07-20
**Component:** experiments
**Status:** settled

**Decision.** Among arm B's four mitigation arms (`residual`, `pairnorm`, `jk`,
`pairnorm+residual`), the winner passed to `BuildGrid("D", armDMitigation=...)`
is whichever has the highest mean `testAccuracy` across its 10 seeds, evaluated
at `numLayers=32` only — not averaged across all five depths.

**Why depth=32 only, not averaged across depths.** experiments_spec.md fully
specifies arm F's three-level hyperparameter selection rule but leaves arm D's
"most effective" mitigation undefined. The mitigation ablation's own purpose is
restoring DEEP performance (D-005's framing, the depth sweep's central claim),
so the selection criterion should measure exactly that — which mechanism helps
most where oversmoothing is worst — rather than a broader average a shallow-depth
advantage (or disadvantage, e.g. residual/PairNorm overhead at depth 2) could
dilute or invert.

**Rejected — averaging test accuracy across all five depths.** A broader
"overall effectiveness" measure that does not target what the ablation exists
to test.


## D-042 — models: GCNII's alpha=0.1, theta=0.5, closing D-034's open item

**Date:** 2026-07-21
**Component:** models / experiments
**Status:** settled — a gap found on review: this value was already in use
throughout the sweep (`experiments/runner.py`'s `GCNII_ALPHA`/`GCNII_LAMBDA`
module constants) and cited in `FINDINGS.md` F-004, but had never actually
been logged as a decision. D-034's original text left this open and deferred
it to `experiments` (that wording has since been replaced in place by the
"Resolved by D-042" note now in D-034, per this file's edit-in-place
convention); this entry is that deferred decision, closing it.

**Decision.** `alpha=0.1`, `theta=0.5` (GCN2Conv's `theta` argument), matching
Chen et al.'s published Cora configuration (arXiv:2007.02133, Table 6, Cora
row: `layers: 64, α_ℓ: 0.1, lr: 0.01, hidden: 64, λ: 0.5, dropout: 0.6, L2c:
0.01, L2d: 0.0005`), verified by fetching the actual paper (D-034's original
verification), not from memory. Used for every GCNII run in the sweep (arm C,
depth 2–32, all 10 seeds).

**Why not re-tune via a coordinated search, the way D-029 tunes GCN's shared
hyperparameters.** GCNII is a single convType appearing only in arm C; a
dedicated hyperparameter search for it would be a second D-029-scale
undertaking for one architecture, and the published values are already a
verified, citable reference point rather than an arbitrary guess. This is a
narrower, cheaper justification than D-029's — stated as its own case, not
implied to meet the same bar.

**Known cost.** Chen et al.'s own Cora configuration uses `dropout=0.6` and a
much larger convolutional weight decay (`L2c=0.01`) than this study's uniform
`dropout=0.5`/`weightDecay=5e-4` (D-019). Using their `alpha`/`theta` while
not matching their dropout/weight-decay means GCNII runs under a hybrid
configuration — published architecture hyperparameters, this study's shared
training hyperparameters — not a full reproduction of their setup. Stated
here so `alpha`/`theta`'s citation to Chen et al. does not imply the whole
configuration matches theirs.