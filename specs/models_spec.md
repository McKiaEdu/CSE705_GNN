# models — GNN architectures and the depth-parameterized layer stack

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed   <!-- "defended" = passed the walk-through -->

## Purpose

Defines the model family under study: a single depth-parameterized layer stack whose
per-layer aggregation operator is supplied by an architecture subclass (GCN, GraphSAGE,
GAT, GCNII), and whose behavior is modified by externally injected mitigation objects.
This component produces the `(logits, layerEmbeddings)` pair that the entire study rests
on — `train` consumes the logits, `metrics` and `viz` consume the embeddings.

Realizes the class-design and tap-point decisions logged in `DECISIONS.md`
(composition over inheritance; ordered `layerHooks`; GCNII as a conv type; null-object
default; tap after activation and before dropout; attached return).

## Approach (agreed)

A base class owns the layer loop; subclasses supply only "what one conv layer is."
Mitigations attach by **composition**, not inheritance: they are injected at
construction as fields and invoked at two named points in the loop. `models` imports
nothing from `mitigations`; both depend on two protocols.

The reasoning, and what it rejects:

**Mitigation is a modification applied to a model, not a kind of model.** Inheritance
encodes "is-a," so `GcnResidual(Gcn)` is a category error for three of the four
mitigations — nothing about residual, PairNorm, or Jumping Knowledge mentions GCN.
Concretely, mitigation-as-subclass gives four mitigations times three architectures
before any combination, and `residual + PairNorm` together has no honest class name.

**Mitigation methods on the base class were also rejected.** That moves the coupling up
a level rather than removing it: the base would have to know the full set of
mitigations, and adding a fifth would edit both the base and its dispatch logic.

**Two protocols, not one hook list.** A `LayerHook` sees a single layer and preserves
shape; a `Readout` consumes the whole stack and determines the final representation.
Merging them forces a union type and a runtime branch. Keeping them separate makes
Jumping Knowledge's effect on the return value visible in the type system rather than
hidden behind a flag.

**GCNII is a conv type, not a mitigation.** Its update rule is written on the normalized
adjacency specifically, with an initial-residual term in `H^(0)` and an identity-mapped
weight `((1 - beta_l) I + beta_l W^(l))`. There is no meaningful "GCNII applied to GAT."
PyTorch Geometric resolves the same question the same way, shipping `GCN2Conv` as a conv
layer. Note for the report: GCNII is a *mitigation* in the experimental sense and an
*architecture* in the code sense, and that must be stated explicitly or it reads as an
inconsistency.

**The unmitigated case uses null objects.** `layerHooks = []` and
`readout = LastLayerReadout()` make the baseline execute the identical code path as
every mitigated variant, so a difference in results is attributable to the mitigation
rather than to a divergence in control flow. Since the depth sweep's central claim is a
comparison between exactly those configurations, this is not merely stylistic.

**The readout declares whether the final layer emits logits or a hidden state.**
Added in revision; see D-016 and Open questions. `Readout` carries a boolean property
`FinalLayerIsLogits`. `LastLayerReadout` sets it `True`: the last conv maps
`hiddenDim -> outDim`, receives no activation, and its output is the logits. A Jumping
Knowledge readout sets it `False`: the last conv maps `hiddenDim -> hiddenDim`, IS
activated like every other layer, and the readout produces the logits from the stack.

Without this, JK would break twice over. The last conv would emit the wrong width, and
`layerEmbeddings[L]` would be the only un-activated tensor in a band that — under JK —
includes it, so `metrics` would compare one pre-activation tensor against `L - 1`
post-activation ones. The property is polymorphic data on the null object rather than an
`isinstance` check, so D-009's single-code-path guarantee is preserved.

## Interface & contract touchpoints

Functions/methods use PascalCase; variables use camelCase; PyG `Data` fields keep their
library names (`edge_index`, `train_mask`).

- `class GnnModel(convType: str, numLayers: int, inDim: int, hiddenDim: int, outDim: int, dropout: float, layerHooks: Sequence[LayerHook], readout: Readout)` — defined here; abstract base owning the layer loop
- `Forward(x: Tensor, edgeIndex: Tensor) -> tuple[Tensor, list[Tensor]]` — defined here; returns `(logits [N, C], layerEmbeddings)` per the contract
- `class LayerHook(Protocol)` — declared here; `Apply(h: Tensor, hPrev: Tensor, edgeIndex: Tensor) -> Tensor`, shape-preserving, called once per layer immediately after the conv
- `class Readout(Protocol)` — declared here; `Apply(layerEmbeddings: list[Tensor]) -> Tensor`, called once after the loop; carries the boolean property `FinalLayerIsLogits`
- `class LastLayerReadout` — defined here; null-object default, `FinalLayerIsLogits = True`, returns `layerEmbeddings[-1]`
- consumes: PyG `Data` (`x [N, 1433]`, `edge_index [2, E]`, `y [N]`, `train_mask` / `val_mask` / `test_mask`)
- produces: `logits [N, 7]` consumed by `train/`; `layerEmbeddings` consumed by `metrics/` and `viz/`

Contract properties fixed by `DECISIONS.md` and not to be drifted:

- `layerEmbeddings` has length `numLayers + 1`. Index 0 is the raw input `X`; index `l`
  is the output of layer `l`.
- **`layerEmbeddings[numLayers]` equals the returned logits ONLY when
  `readout.FinalLayerIsLogits` is `True`.** Under Jumping Knowledge it is a hidden
  representation at `hiddenDim`, and the logits are a separate tensor the readout
  builds from the whole stack. D-001 C1 states this unconditionally and is wrong as
  written; see Open questions.
- `models` applies no filtering to `layerEmbeddings`. Selecting the metrically
  comparable band is the responsibility of `metrics`, which derives it from tensor
  widths rather than from a hardcoded index range — precisely because the band is
  `1 .. numLayers - 1` under `LastLayerReadout` and `1 .. numLayers` under JK.
- **Width homogeneity guarantee.** Every index `metrics` will place in the band carries
  the same width. `models` guarantees this by holding the intermediate width constant
  at every depth; whatever that width resolves to for a given architecture (see the GAT
  head question), it is identical across all intermediate layers.
- Each recorded tensor is taken after the conv, after all `layerHooks`, and after the
  activation, but **before** dropout.
- Returned tensors remain attached to the autograd graph. Detaching happens at the
  capture site in `train`.
- All conv types are invoked with the uniform signature `(x, edgeIndex, x0)`. GCNII
  requires `x0`; the other three ignore it.

## Implementation plan

- `class GnnModel(nn.Module)` — defined here; reused by `train/`, `experiments/`.
  Holds `convs: nn.ModuleList`, `layerHooks`, `readout`, `dropout`. Owns `Forward`.
- `BuildConv(convType: str, inDim: int, outDim: int, **kwargs) -> nn.Module` —
  defined here; factory dispatching on `convType` in `{"gcn", "sage", "gat", "gcnii"}`,
  wrapping the PyG conv in a thin adapter so every conv exposes the uniform
  `(x, edgeIndex, x0)` call signature.
- `class GcnModel(GnnModel)`, `class SageModel(GnnModel)`, `class GatModel(GnnModel)`,
  `class GcniiModel(GnnModel)` — defined here; each supplies only its conv construction
  and any architecture-specific width bookkeeping (notably GAT's head handling). None
  of them redefines `Forward`.
- `class LastLayerReadout` — defined here; reused as the default by `experiments/`.

The final conv's output width is `outDim` when `readout.FinalLayerIsLogits` is `True`
and `hiddenDim` otherwise. This is read once at construction, not per forward pass.

Data flow inside `Forward`:

1. `h = x`; `layerEmbeddings = [x]`; `x0 = x` (or its projection, see Open questions).
2. For each layer `l` in `1 .. numLayers`:
   a. `h = self.convs[l - 1](h, edgeIndex, x0)`
   b. for each hook in `self.layerHooks` (in list order): `h = hook.Apply(h, hPrev, edgeIndex)`
   c. `isFinalAndLogits = (l == numLayers) and self.readout.FinalLayerIsLogits`
      if not `isFinalAndLogits`: `h = activation(h)`
   d. `layerEmbeddings.append(h)`
   e. if not `isFinalAndLogits`: `h = dropout(h)`
3. `logits = self.readout.Apply(layerEmbeddings)`
4. return `(logits, layerEmbeddings)`

Note that step (d) sits between activation and dropout, which is the tap-point decision
made concrete. Under `LastLayerReadout`, step 3 is a no-op returning
`layerEmbeddings[-1]`, so `layerEmbeddings[numLayers]` and `logits` are the same tensor.
Under Jumping Knowledge every layer including the last is activated, the readout builds
the logits from the whole list, and the two are distinct tensors — which is also why the
list must remain attached.

Vectorization: no Python loops over nodes or edges anywhere. The only loop is over
layers, which is `numLayers` iterations and is inherently sequential. All aggregation is
delegated to PyG's sparse message-passing kernels.

## Dependencies

- Depends on: `data/` for the PyG `Data` object and its field names; `torch`,
  `torch_geometric.nn` for the conv primitives.
- Depends on the two protocol declarations (`LayerHook`, `Readout`), which are declared
  in this module so that `mitigations` can depend on `models` without `models`
  depending on `mitigations`. This is the direction of the dependency inversion and
  must not be flipped.
- Consumed by: `train/` (logits, and the capture of `layerEmbeddings`),
  `metrics/` (`layerEmbeddings`), `viz/` (`layerEmbeddings`), `experiments/`
  (construction).
- Build order (aligned to the settled component order): `data/`, then `models/`, then
  `metrics/`, then `train/`, then `mitigations/`, then `experiments/`, then `viz/`.
  `mitigations` comes after `train` because the null-object default lets the harness be
  built and the baselines run before any mitigation exists; the protocols it will
  implement are declared here, so nothing blocks on it.

## Assumptions & constraints

Every item here is provisional until confirmed.

- Python / version: Python 3.12.13 (carried from the data spec).
- Libraries / role: `torch`, `torch_geometric` 2.8.0 (core — `GCNConv`, `SAGEConv`,
  `GATConv`, `GCN2Conv`); `typing.Protocol` for the two hook interfaces.
- Compute / runtime: CPU wheels, device-agnostic code. Full-batch — the whole graph is
  forwarded per step, no neighbor sampling even for GraphSAGE, so that depth is the only
  varying factor across architectures.
- Activation: ReLU. Applied to every layer whose output is a hidden representation —
  i.e. all layers except the last, and including the last when
  `readout.FinalLayerIsLogits` is `False`.
- Dropout: applied wherever the activation is applied, after the tap.
- Confirmed assumptions:
  - `numLayers = L` means exactly `L` message-passing layers and therefore `L` hops.
    The width changes are folded into the first layer (`1433 -> hiddenDim`) and the last
    (`hiddenDim -> 7`, under `LastLayerReadout`); there are no extra projection layers.
    `numLayers = 2` is therefore the Kipf & Welling baseline.
  - `hiddenDim` is held constant across all intermediate layers at every depth.
  - Ordered `layerHooks`, default order residual then PairNorm. Residual is part of the
    update rule; PairNorm normalizes the resulting representation, and its invariant
    (constant total pairwise squared distance) is defeated if an un-normalized `H^(l)` is
    added afterward.

## Outputs & artifacts

`models` writes no files. It produces in-memory objects only:

- `logits [N, 7]` — consumed by the loss and by accuracy / macro-F1 in `train/`.
- `layerEmbeddings` — a list of `numLayers + 1` tensors, attached, consumed by `train/`
  at the three capture points of C5 and forwarded to `metrics/` and `viz/`.
- A model configuration record (`convType`, `numLayers`, `hiddenDim`, `dropout`, hook
  names in applied order, readout name) that `train/` embeds in the results record it
  returns, so a run is reconstructible from its record alone. `experiments/` serializes
  that record under the filename C3 defines (D-022); the convention is not restated
  here. The hook order must appear in the record, since it is a decision and not a
  default. The readout name must appear for the same reason — it determines whether
  index `numLayers` is a logit or a hidden state, which changes how `metrics` bands the
  run.

## Test plan

- **Shape contract:** for each `convType` and each depth in `{2, 4, 8, 16, 32}`, assert
  `logits.shape == (N, 7)`, `len(layerEmbeddings) == numLayers + 1`, and
  `layerEmbeddings[0] is x`.
- **Final-entry identity, conditioned on the readout:** under `LastLayerReadout`,
  `layerEmbeddings[-1]` is the same tensor object as `logits`. Under a JK readout,
  `layerEmbeddings[-1]` has width `hiddenDim`, `logits` has width 7, and the two are
  distinct objects. Both directions asserted — this is the regression test for the
  narrowed C1 clause.
- **Width homogeneity across the band:** assert that all of
  `layerEmbeddings[1 .. numLayers - 1]` share one width as each other, and that under a
  JK readout `layerEmbeddings[numLayers]` shares it too. Asserted as mutual equality
  rather than equality to `hiddenDim`, so the test does not presume an answer to the
  open GAT head question — if `hiddenDim` turns out to be per-head with concatenation,
  GAT's width is `numHeads * hiddenDim` and the test still correctly checks the property
  `metrics` actually depends on.
- **Activation consistency under JK:** with a JK readout, assert
  `layerEmbeddings[numLayers]` is non-negative everywhere, confirming the final layer was
  activated like the rest of the band. Under `LastLayerReadout`, assert the same tensor
  DOES contain negative values, confirming the logits were not passed through ReLU.
- **Null-object path identity:** a model built with `layerHooks = []` and
  `LastLayerReadout` must produce bitwise-identical logits to a reference forward pass
  with the hook machinery bypassed, under a fixed seed and `model.eval()`.
- **Gradient reaches every layer:** with a JK readout attached, backward from a dummy
  loss and assert every `conv` parameter has a non-`None`, non-zero gradient. This is the
  regression test for the attached-return decision — detaching would silently zero all
  but the last.
- **Tap point:** with `dropout = 0.9` and `model.train()`, assert that
  `layerEmbeddings[l]` differs from the tensor actually passed into layer `l + 1`,
  confirming the tap sits before dropout rather than after. (The earlier form of this
  test additionally asserted the recorded tensor contained no exact zeros; that clause
  was unsatisfiable, since ReLU produces exact zeros throughout and they cannot be
  distinguished from dropout zeros. Removed.)
- **Uniform conv signature:** assert every conv adapter accepts `(x, edgeIndex, x0)` and
  that passing a different `x0` changes the output for `gcnii` only.
- **Smoke test:** a 2-layer GCN with default hyperparameters reaches roughly 81 percent
  test accuracy on the standard Planetoid split, matching the published baseline. A
  32-layer unmitigated GCN collapses well below it. Both are qualitative gates, not
  exact-value assertions.

## Report / novelty note

This component supplies the "Implementation Details" and "Explanation of the Source
code" sections with their central design argument: mitigations are composed rather than
inherited, which is what makes the four-mitigation ablation tractable and what makes the
baseline and mitigated runs provably the same code path. That last point is a
methodological claim worth making explicitly in Results — it is the reason a measured
difference can be attributed to the mitigation.

The GCNII-as-architecture distinction is a second point for the Discussion: it shows the
taxonomy in the literature ("four mitigations") does not survive contact with
implementation, and being precise about that is a small piece of genuine analysis rather
than reportage.

A third, smaller point worth a sentence: Jumping Knowledge changes what the final layer
*is*, not merely how the output is assembled. Handling that with a property on the
readout rather than a special case in the loop is the concrete payoff of the
composition design, and it is checkable by a grader reading `Forward`.

## Open questions

- **D-001 C1 must be narrowed.** The clause "the final entry equals the returned logits"
  is false under Jumping Knowledge. It should read "under `LastLayerReadout`." Needs a
  contract edit plus a D-001 changelog line before `train` is implemented.
- **`hiddenDim` value.** Kipf & Welling use 16; earlier discussion in this project has
  used both 16 and 64. This must be fixed before the sweep, since the energy curve's
  comparable band is defined by it. Belongs to `experiments` but blocks `models` tests.
- **GAT head handling.** Unresolved: is `hiddenDim` per-head or total across heads, and
  does the final layer concatenate or average its heads? This has a fairness
  consequence — if `hiddenDim` is per-head, GAT carries `numHeads` times the width of
  GCN at the same nominal setting, and any depth comparison is confounded by capacity.
- **GCNII and `x0` width.** `x0` must match the hidden width for the initial-residual
  term to be added, but the raw `X` is 1433-dimensional. PyG's `GCN2Conv` assumes a
  prior linear projection to `hiddenDim`. If a projection layer is added, does it count
  toward `numLayers`? If it does, GCNII at depth `L` performs `L - 1` hops and is not
  depth-comparable to the others; if it does not, GCNII has one more parameterized layer
  than its peers. Neither option is free — decide and record which cost we accept.
  Secondary consequence flagged by `metrics`: if the projection exists,
  `layerEmbeddings[0]` for GCNII may be the projected tensor rather than raw `X`, making
  its `E_0` anchor a different representation kind from the other architectures'.
- **Depth grid.** The proposal states 2 to 32; the specific set (2, 4, 8, 16, 32 versus
  adding 24) is not yet fixed. Belongs to `experiments`.
- **Weight initialization.** Left to PyG defaults unless the depth-32 runs show
  initialization-dependent failure, which the epoch-0 capture is designed to detect.