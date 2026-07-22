# train — training loop, early stopping, captures, and record assembly

<!-- Internal planning record for the handoff between the Claude.ai Project (which
decides) and Claude Code (which implements). NOT report prose. One file per
component, saved as spec/<component>_spec.md in the repo. -->

**Owner:** Kiarash
**Status:** spec agreed   <!-- "defended" = passed the walk-through -->

## Purpose

Runs one configuration end to end: seeds, trains full-batch with early stopping,
selects a checkpoint, takes the three eval-mode metric captures required by D-001 C5,
evaluates the test split, and assembles the complete C3 results record. It is the only
component that owns an optimizer or a training loop, and the only one that decides
which weights the reported numbers come from.

Realizes D-017 (loss-based stopping and selection), D-018 (generous constant patience),
D-019 (uniform weight decay), D-020 (restore-then-capture ordering), D-021 (in-repo
macro-F1), and D-022 (assembles the record, does not write it).

## Approach (agreed)

**One function per run, returning a record.** `TrainRun` takes a constructed model, the
`Data` object, a config, and a constructed `OversmoothingMetrics` instrument, and
returns a dict conforming to C3. It touches no files, holds no global state, and can be
called in a loop by `experiments` or once from a notebook.

The decisions this carries, and what each rejects:

**Validation loss drives both stopping and selection (D-017).** Training halts when
validation loss has not improved for `patience` consecutive epochs; the reported
checkpoint is the epoch of minimum validation loss. These are two operations, and Kipf
& Welling specify only the first — a maximum of 200 epochs, Adam at learning rate 0.01,
and a window of 10 in which training stops if validation loss does not decrease. A
stopping rule says when to leave the loop, not which weights to keep.
*Deviation from the proposal, deliberate.* The submitted proposal says early stopping on
validation *accuracy*. Loss is used instead, for fidelity to the reference protocol the
~81.5% reproduction claim rests on, and because accuracy over 500 validation nodes moves
in steps of 0.002, so an accuracy plateau can mask a still-descending loss. Both
quantities are recorded per epoch so the alternative is derivable and the report can
show rather than assert that the choice is immaterial.
*Rejected — stopping on loss but selecting on accuracy:* `bestEpoch` could then fall
outside the window the stopping rule examined.

**Patience is generous and constant (D-018).** `patience = 100`, `maxEpochs = 1000`,
identical for every architecture, depth, mitigation arm, and seed. A tight patience
defeats D-005: a 32-layer GCN may plateau for many epochs before its loss descends, and
at a window of 10 the run halts around epoch 12 with a flat curve indistinguishable from
genuine untrainability. The harness would have manufactured the failure the study exists
to diagnose. A generous patience can only cost compute, and compute is not binding at
Cora's size. Constant because holding the stopping rule fixed across depth is what makes
the depth comparison valid.

**Weight decay is uniform across layers (D-019).** Kipf & Welling regularize the first
layer only, which is reasonable for two layers and meaningless at 32, where it would
cover a thirtieth of the parameters. Uniform decay keeps the regularization from being
an implicit function of depth.

**Capture ordering: final first, then restore, then checkpoint (D-020).** The
final-epoch capture is taken on the weights the loop ended with; only then is the best
`state_dict` restored and the checkpoint capture taken. Reversing this makes
`finalMetrics` a duplicate of `checkpointMetrics` on every run, silently killing the
early-stopping diagnostic in D-012.

**Macro-F1 is written here, not imported (D-021).** Roughly ten lines of tensor
operations. `scikit-learn` is a test-only dependency used to verify it once.

## Interface & contract touchpoints

Functions/methods use PascalCase; variables use camelCase; PyG `Data` fields keep their
library names (`edge_index`, `train_mask`).

- `class TrainConfig` — defined here; frozen dataclass:
  `learningRate: float = 0.01`, `weightDecay: float = 5e-4`,
  `maxEpochs: int = 1000`, `patience: int = 100`, `seed: int`
- `TrainRun(model: GnnModel, data: Data, config: TrainConfig, metrics: OversmoothingMetrics) -> dict`
  — defined here; the single entry point. Returns a C3-conforming record.
- `SetSeed(seed: int) -> None` — defined here; seeds `torch`, `random`, and `numpy`,
  and is called at the top of `TrainRun`. Lives here rather than in a `utils/` module,
  per D-022.
- `EvaluateSplit(model: GnnModel, data: Data, mask: Tensor) -> tuple[float, float, float]`
  — defined here; returns `(loss, accuracy, macroF1)` in eval mode under `no_grad`.
- `MacroF1(predictions: Tensor, targets: Tensor, numClasses: int) -> float` — defined
  here; vectorized, no `scikit-learn` at runtime.
- `CaptureMetrics(model: GnnModel, data: Data, metrics: OversmoothingMetrics) -> dict`
  — defined here; one eval-mode `no_grad` forward, detaches `layerEmbeddings` to CPU per
  D-011, calls `metrics.ComputeAll`, returns the capture block.
- consumes: `GnnModel.Forward` per D-001 C1; `OversmoothingMetrics.ComputeAll` per the
  metrics spec; PyG `Data` fields per C4.
- produces: the C3 record dict, consumed by `experiments/` which serializes it.

Contract properties this component is responsible for honoring:

- Exactly three metric captures per run, each an explicit eval-mode forward, never
  reused from a training forward (C5).
- `layerEmbeddings` detached at the capture site, not inside `models` (D-011).
- `bandIndices` written once at the top level, taken from any capture since it is
  run-invariant (C3).
- `config.readout` recorded, since it determines banding (D-016).
- The filename form is NOT applied here — `train` returns `runId` as the stem and
  `experiments` builds the path (D-022).

## Implementation plan

- `TrainRun` — defined here; reused by `experiments/` and `notebooks/`. Flow:

  1. `SetSeed(config.seed)`.
  2. `optimizer = Adam(model.parameters(), lr=config.learningRate, weight_decay=config.weightDecay)`
     — uniform decay over all parameter groups, per D-019.
  3. `epoch0Metrics = CaptureMetrics(...)` — before any optimizer step.
  4. Loop `epoch` in `1 .. config.maxEpochs`:
     a. `model.train()`; `optimizer.zero_grad()`;
        `logits, _ = model.Forward(data.x, data.edge_index)`;
        `trainLoss = CrossEntropy(logits[data.train_mask], data.y[data.train_mask])`;
        `trainLoss.backward()`; `optimizer.step()`.
     b. `valLoss, valAccuracy, _ = EvaluateSplit(model, data, data.val_mask)`.
     c. Append `{epoch, trainLoss, valLoss, valAccuracy}` to `trainingCurve`.
     d. If `valLoss < bestValLoss`: record `bestEpoch = epoch`, `bestValLoss`,
        `bestState = deepcopy(model.state_dict())`, reset `epochsWithoutImprovement`.
        Else increment; if it reaches `config.patience`, break.
  5. `finalMetrics = CaptureMetrics(...)` — on the weights the loop ended with,
     BEFORE any restore. Ordering is load-bearing, per D-020.
  6. `model.load_state_dict(bestState)`.
  7. `checkpointMetrics = CaptureMetrics(...)`.
  8. `testLoss, testAccuracy, testMacroF1 = EvaluateSplit(model, data, data.test_mask)`;
     `_, valAccuracy, _ = EvaluateSplit(model, data, data.val_mask)`.
  9. Assemble and return the C3 record.

- `MacroF1` — defined here. Vectorized: form a combined index
  `targets * numClasses + predictions`, `torch.bincount` it to length
  `numClasses ** 2`, reshape to a `[numClasses, numClasses]` confusion matrix, then read
  per-class true positives off the diagonal and false positives / false negatives off the
  column and row sums. Per-class F1 with a zero-denominator guard, then an unweighted
  mean. No Python loop over classes or nodes.
- `CaptureMetrics` — defined here. `model.eval()`; under `torch.no_grad()` call
  `Forward`; `[h.detach().cpu() for h in layerEmbeddings]`; `metrics.ComputeAll(...)`;
  return the block plus `bandIndices` for the caller to hoist.
- `EvaluateSplit` — defined here; reused by step 4b and step 8. `model.eval()`, no_grad,
  cross-entropy and argmax on the masked rows only.
- `SetSeed` — defined here. Seeds `torch.manual_seed`, `random.seed`, `numpy.random.seed`.
  Deterministic-algorithm flags are NOT set by default; see Open questions.

Vectorization: no Python loops over nodes, edges, or classes. The only loops are over
epochs and over layers, both inherently sequential.

## Dependencies

- Depends on: `models/` (the `GnnModel` instance and its `Forward` contract),
  `metrics/` (a constructed `OversmoothingMetrics`), `data/` (the `Data` object),
  `torch` (Adam, cross-entropy, `state_dict`).
- Does NOT depend on `experiments/` or `viz/`, and does not import `mitigations/` —
  mitigation objects arrive already injected into the model.
- Consumed by: `experiments/` (which supplies the config and seed, and writes the
  returned record), `notebooks/` (single-run exploration).
- Build order: after `data/`, `models/`, and `metrics/`; before `mitigations/` and
  `experiments/`. The null-object default means the harness runs and the baselines are
  reproducible before any mitigation exists.

## Assumptions & constraints

Every item here is provisional until confirmed.

- Python / version: Python 3.14.4 (matches `data_spec.md`'s corrected pin).
- Libraries / role: `torch`, `torch_geometric` 2.8.0 (core). `scikit-learn` is
  **test-only** and belongs in `requirements-dev.txt`, not `requirements.txt`, per
  D-021 — this is stated because `data_spec` and `metrics_spec` both declare no
  scikit-learn dependency and the two claims must not read as inconsistent.
- Compute / runtime: CPU wheels, device-agnostic. Full-batch: the whole graph is
  forwarded per step, no mini-batching and no neighbor sampling even for GraphSAGE.
- Optimizer: Adam. Loss: cross-entropy on `train_mask` rows only.
- Initialization: PyG defaults, matching Kipf & Welling's stated choice for GCN. Not
  overridden — no depth-32 initialization-dependent failure was found (F-002's epoch-0
  captures show the expected clean collapse for GCN and GAT). **Revised**: "PyG defaults"
  is not uniformly Glorot — verified via `inspect.getsource` (F-002) that `GCNConv` and
  `GATConv` pass `weight_initializer='glorot'` explicitly, while `SAGEConv.lin_l`/`lin_r`
  pass none and fall back to PyTorch's `nn.Linear` default, `kaiming_uniform`. Left
  unoverridden regardless, since no failure triggered the stated condition, but this is a
  genuine cross-architecture difference worth naming in the report rather than assuming
  uniformity.
- Confirmed assumptions:
  - Validation loss is computed in eval mode, so dropout is inactive and the stopping
    signal is not stochastic beyond the weights themselves.
  - "Improvement" is a strict decrease in validation loss, with no minimum delta. See
    Open questions.
  - The three captures are the only eval-mode forwards that invoke `metrics`; the
    per-epoch validation pass computes loss and accuracy only, never energy or MAD.
  - `epochsRun` is the number of epochs actually executed, so `epochsRun < maxEpochs`
    indicates early stopping fired.

## Outputs & artifacts

`train` writes no files. It returns one in-memory dict per run, conforming to C3 in
full:

- `config` block — assembled from `TrainConfig` plus the model configuration record
  `models` supplies (`convType`, `numLayers`, `mitigations`, `readout`, `hiddenDim`,
  `dropout`).
- `bandIndices` — hoisted to the top level from any capture, per C3.
- `results` — `testAccuracy`, `testMacroF1`, `valAccuracy`, `valLoss`, `bestEpoch`,
  `epochsRun`.
- `trainingCurve` — one entry per executed epoch, closing D-005.
- `epoch0Metrics`, `checkpointMetrics`, `finalMetrics` — the three capture blocks.
- `trajectory` — empty list under the default configuration.
- `environment` — python, torch, torch-geometric versions and device string, read at
  runtime rather than hardcoded.

## Test plan

- **Smoke / baseline:** a 2-layer GCN at `hiddenDim` with default config reaches roughly
  81 percent test accuracy on the standard Planetoid split. Qualitative gate. If it
  lands low, check weight-decay scope (D-019) before learning rate or initialization.
- **Seed determinism:** two `TrainRun` calls with the same seed and config produce
  identical records except `timestamp`. This is the reproducibility claim the README
  makes, so it is asserted rather than assumed.
- **Macro-F1 correctness:** on fixed synthetic predictions and targets covering an
  unrepresented class, `MacroF1` equals
  `sklearn.metrics.f1_score(average='macro', zero_division=0)`. The unrepresented-class
  case is the one that separates a correct implementation from a plausible one.
- **Early stopping fires:** with `patience = 1`, assert `epochsRun < maxEpochs` and
  `bestEpoch <= epochsRun`.
- **Checkpoint restore is real:** on a run where `bestEpoch != epochsRun`, assert
  `finalMetrics` and `checkpointMetrics` differ. This is the regression test for the
  D-020 ordering — if the restore happened before the final capture, they would be
  identical on every run.
- **Epoch-0 capture precedes training:** assert the parameters at the time of the
  epoch-0 capture are bitwise identical to the freshly constructed model's, i.e. no
  optimizer step has occurred.
- **Capture count:** instrument `OversmoothingMetrics.ComputeAll` with a call counter
  and assert exactly three invocations per run. Guards against a future edit
  reintroducing per-epoch metric computation, which C5 rejected on cost grounds.
- **Curve length:** `len(trainingCurve) == results.epochsRun`, and every entry has all
  four keys.
- **Record conformance:** the returned dict validates against the C3 key set — all
  required keys present, capture blocks the same length `L+1`, `bandIndices` non-empty
  except where the band is degenerate.
- **Loss descends on a trainable config:** at depth 2, assert `trainLoss` at the final
  epoch is materially below `ln 7 ≈ 1.95`. This is the D-005 diagnostic exercised on a
  case where the answer is known, so a deep run's flat curve can be read as a finding
  rather than a bug.

## Report / novelty note

Two contributions.

**Section 4 (Implementation Details) gets a protocol statement that is defensible rather
than default.** The stopping rule, the selection rule, and the fact that they are
different operations; why patience is loosened relative to the reference; why weight
decay is uniform. Each is a stated deviation with a reason, which reads very differently
from an unexplained mismatch with the cited baseline.

**Section 6 (Results) gets the optimization-versus-oversmoothing separation.** The
`trainingCurve` plus `epoch0Metrics` are what let the report claim a deep configuration
*oversmoothed* rather than *failed to train* — and the D-018 reasoning is itself worth a
sentence, because a reader who knows the literature will wonder whether the deep failures
are a stopping-rule artifact. Answering that before it is asked is the difference between
a result and an assertion.

## Open questions

- **`hiddenDim` value.** Resolved: 64 (D-023), except arm E at 16.
- **Minimum improvement delta.** Still undecided and unimplemented: `harness.py`'s
  improvement check is a strict `valLoss < bestValLoss` with no `minDelta`, exactly as
  originally described. Not revisited, since generous patience (D-018) absorbs the cost
  this open question worried about.
- **Deterministic algorithms.** Still undecided and unimplemented:
  `torch.use_deterministic_algorithms` is not called anywhere in the harness. The
  reproducibility claim remains "same machine, same seed," not the stronger
  bitwise-across-machines claim.
- **Test-set loss.** Resolved: decided against a schema field. `EvaluateSplit` computes
  `testLoss` in `TrainRun` and it is explicitly discarded (`del testLoss`, with a comment
  citing this open question) rather than written into the record.
- **Per-architecture learning rate.** Resolved by the hyperparameter search (arm F,
  D-041/F-007): one shared configuration (`lr=0.01, dropout=0.5, weightDecay=5e-4`) is
  frozen across every architecture and depth. The GAT confound this question anticipated
  is accepted and stated as a limitation rather than re-tuned per architecture — F-007
  also found the three-seed search margins were comparable to seed noise, reinforcing
  that a per-architecture re-tune would not obviously separate configurations anyway.
