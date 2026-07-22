# data — Cora loading, preprocessing, and split provisioning

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed

## Purpose

Load the Cora citation network via PyTorch Geometric's `Planetoid` dataset, apply the
one preprocessing transform the study fixes (row normalization of node features), and
hand a single validated `Data` object to every downstream component. This component
realizes D-002 (row-normalize Cora node features) and is the sole source of the graph,
features, labels, and split masks used anywhere in the repo.

## Approach (agreed)

Load `Planetoid(root, name='Cora', split='public', transform=T.NormalizeFeatures())`.
Four decisions were settled and are carried here so they are not relitigated:

1. **Row normalization is ON and fixed, not an ablation axis.** Kipf & Welling
   (arXiv:1609.02907, Sec. 5.2) state that input feature vectors are (row-)normalized
   in the published GCN setup, so faithful reproduction of the ~81.5% two-layer Cora
   baseline requires it. Verified by fetching the paper, not from memory.
   Safety for the oversmoothing measurement rests on two *separate* arguments, both
   needed. (a) Row scaling is per-node, so it does NOT cancel in an energy ratio —
   MAD is invariant (cosine ignores positive per-node scaling) but Dirichlet energy is
   not. Safety comes from applying the identical transform to every configuration, so
   it cannot confound any comparison we make; we never compare a normalized run
   against an unnormalized one. (b) We report a normalized energy rather than raw
   `E_l`; under a global rescale `h -> c*h`, energy is quadratic so `c^2` cancels in
   the ratio, which is what lets GCN and GAT curves overlay despite different
   activation scales. The reference layer for that ratio is `l = 1`, not `l = 0` —
   see D-002 and D-003.
   *Rejected:* no normalization (breaks baseline fidelity for no gain);
   normalization as an ablation axis (a full extra sweep answering a question the
   report does not ask).

2. **Public split, seeds vary initialization only.** `split='public'` gives the fixed
   Yang et al. (2016) masks — 140 train / 500 val / 1000 test. Across the 10 seeds the
   masks are identical, so the reported mean ± std is *training variance under a fixed
   split* (weight init, dropout masks), matching Kipf & Welling's headline protocol of
   100 runs with random weight initializations.
   *Rejected:* adding a random-splits arm. It would show the depth effect is not
   split-specific, but roughly doubles sweep cost against the Aug 3 deadline. The
   report instead states the protocol explicitly and cites K&W's own `rand. splits`
   row (Cora 80.1 ± 0.5) as evidence the two protocols agree closely.

3. **Self-loops are NOT added here.** `A~ = A + I` lives inside the GCN propagation
   rule; `GCNConv` adds self-loops internally (`add_self_loops=True` by default). This
   component ships the raw graph. Consequence for `metrics/`: see the contract note
   below — self-loops contribute zero to the energy sum directly, but change every
   degree `d_i`, so the choice of graph is not cosmetic.

4. **Undirectedness is asserted, not imposed.** We do not call `to_undirected`
   unconditionally. We assert `data.is_undirected()` at load and fail loudly if it does
   not hold, which makes PyG's inherited behavior explicit and checkable rather than
   silently assumed.

## Interface & contract touchpoints

- `LoadCora(root: str, normalizeFeatures: bool = True) -> Data` — defined here.
  The flag exists for the test plan only; every experiment calls it with the default.
- `AssertGraphInvariants(data: Data) -> None` — defined here; raises on violation.
- produces: PyG `Data` with `x [2708, 1433]` (float, row sums 1), `edge_index [2, E]`
  (raw, no self-loops), `y [2708]` (int, 7 classes), and boolean `train_mask`,
  `val_mask`, `test_mask` of sizes 140 / 500 / 1000.
- consumed by: `models/` (`x`, `edge_index`), `train/` (masks, `y`), `metrics/`
  (`edge_index`), `viz/` (`y` for coloring).
- **contract note (flagged, cross-owner).** `metrics/` computes Dirichlet energy over
  the **self-looped** graph, not the raw `edge_index`. Cai & Wang (arXiv:2006.13318,
  Sec. 2) define `A~ := A + I_N`, `D~ := D + I_N`, and the augmented normalized
  Laplacian `Δ~ := I_N − D~^{-1/2} A~ D~^{-1/2}`, with propagation operator
  `P := I_N − Δ~`; their contraction bound's eigenvalue is an eigenvalue of `Δ~`.
  Computing energy on the raw graph would use degrees from a different operator than
  the theory we cite, and than the one `GCNConv` actually propagates with. `metrics/`
  receives the RAW `edge_index` and augments internally, asserting on entry that the
  input is self-loop-free so double-augmentation fails loudly. The augmented graph is
  used for all three architectures, not matched per-architecture — see D-004.
- **contract note (corrected 2026-07-19).** An earlier version of this spec stated that
  `layerEmbeddings[0]` is `h^(1)` and that `metrics/` receives `data.x` separately as
  the layer-0 embedding. That was wrong and contradicted the contract. Per D-001 C1,
  `layerEmbeddings` has length L+1 and **index 0 is the raw input X**. `metrics/` reads
  the layer-0 representation from `layerEmbeddings[0]`; `data.x` is NOT passed a second
  time alongside the list. The list is therefore heterogeneous in width by design —
  index 0 is 1433-dim, indices 1..L-1 are `hiddenDim`, index L is 7-dim — and
  restricting to the comparable band is `metrics/`'s responsibility, not this
  component's.

## Implementation plan

- `LoadCora(root: str, normalizeFeatures: bool = True) -> Data` — defined here.
  Builds the `Planetoid` dataset with `transform=T.NormalizeFeatures()` when the flag
  is set, takes `dataset[0]`, calls `AssertGraphInvariants`, returns the `Data`.
- `AssertGraphInvariants(data: Data) -> None` — defined here. Checks node count,
  feature dimension, class count, `is_undirected()`, absence of self-loops, mask sizes,
  and pairwise mask disjointness. Fails loudly rather than warning.
- Vectorization: all checks are tensor reductions (`torch.bincount`, boolean mask
  `.sum()`, `contains_self_loops`). No Python loops over nodes or edges anywhere in
  this component.
- Seeding is NOT defined here — `SetSeed` belongs to the training harness, since this
  component's output is deterministic given a fixed split. Flagged in Open questions
  so the location is settled once rather than duplicated.

## Dependencies

- Depends on: `torch`, `torch_geometric` (`Planetoid`, `transforms`, `utils`). No
  internal modules — this is the root of the build order and must exist first.
- Consumed by: `models/`, `train/`, `metrics/`, `viz/`, `experiments/`.

## Assumptions & constraints

- Python / version: Python 3.14.4 (updated from an earlier 3.12.13 placeholder to
  match the actual `.venv`; all pinned packages in `requirements.txt`, including
  `numba`/`umap-learn`, import cleanly under 3.14.4, so there is no compatibility
  reason to pin an older interpreter).
- Libraries / role: `torch`, `torch_geometric==2.8.0` (core); no `scikit-learn`
  dependency in this component. Exact pins live in `requirements.txt`, which is the
  source of truth; the torch version is recorded there post-install rather than chosen
  in advance. `NormalizeFeatures` behavior on all-zero rows is version-dependent
  (see Test plan).
- Compute / runtime: negligible. Full-batch; the whole graph is held in memory and
  forwarded per step. Dataset downloads once to `root` and is cached.
- Confirmed assumptions: public split only (no random-splits arm); seeds vary
  initialization, not data; self-loops are the layer's responsibility; undirectedness
  is asserted rather than imposed.

## Outputs & artifacts

- No files written. This component's output is the in-memory `Data` object.
- The dataset cache under `root` (default `data/Cora/`) is a build artifact and is
  gitignored; the README documents that the first run downloads it.

## Test plan

- `data.num_nodes == 2708`, `data.x.shape == (2708, 1433)`,
  `int(data.y.max()) + 1 == 7`.
- `data.is_undirected()` is True.
- `torch_geometric.utils.contains_self_loops(data.edge_index)` is False (confirms the
  loader ships the raw graph, per decision 3).
- Mask sizes: `train_mask.sum() == 140`, `val_mask.sum() == 500`,
  `test_mask.sum() == 1000`; all three pairwise disjoint.
- Row normalization: with `normalizeFeatures=True`, every row sum is 1 **or** 0. The
  zero case is retained as a defensive branch in the assertion, but is not expected to
  fire on Cora: verified empirically that Cora contains zero all-zero bag-of-words rows
  (see Open questions — resolved), so `data.x` contains no `NaN`/`Inf` and the "sums to
  1" case is the only one actually exercised. The test still reports the zero-sum row
  count so a future dataset swap would surface the case rather than silently assume it
  away.
- Contrast check: loading with `normalizeFeatures=False` yields a binary `x` with
  integer-valued row sums, confirming the transform is what changes the values.
- Determinism: two `LoadCora` calls produce identical `x`, `edge_index`, and masks.

## Report / novelty note

Two contributions to the report. First, Section 4 (Implementation Details) can state
the preprocessing choice as a *reasoned* one rather than a default: we row-normalize
because the published baseline does, and we show the choice cannot contaminate the
oversmoothing measurement via the two-argument analysis above. Most course-level
treatments apply `NormalizeFeatures` without noticing it interacts with a
scale-sensitive metric. Second, the self-loop/energy contract point is a genuine
correctness subtlety — measuring energy on the raw graph while the model propagates on
the augmented one is an easy and invisible error, and naming it is worth a sentence in
the discussion.

## Open questions — resolved

- **Does Cora contain all-zero feature rows, and does PyG 2.8.0 clamp the row-sum
  denominator?** Resolved empirically (`src/tests/test_data.py`): Cora has **zero**
  all-zero rows, both before and after `NormalizeFeatures`, and `data.x` contains no
  `NaN`/`Inf`. The PyG-clamp behavior is therefore moot on this dataset — the "every
  row sums to 1" assertion holds unconditionally and does not need the "or 0" carve-out
  the Test plan below still states.
- **Where does `SetSeed` live?** Resolved: `train/harness.py`, called at the top of
  `TrainRun` before any model construction (D-022; see also `experiments_spec.md`'s
  `RunOne` note on why it must run before `BuildModel`, not just before training).
- **Does `SAGEConv` add self-loops internally?** Resolved: **no**. Verified directly by
  reading `SAGEConv.forward`'s source (`inspect.getsource`) — it calls
  `self.propagate(edge_index, ...)` with no `add_self_loops` anywhere in the method.
  GraphSAGE's root term is handled separately via `lin_r`, exactly as guessed in D-004's
  original note. This closes D-004's "believed not, unverified" note.