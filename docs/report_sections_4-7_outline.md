# Report Sections 4–7 — Outline, Content Checklist, and Sourcing Rules

Cora GNN Oversmoothing Study — SEP 740 / CSE 705, Summer 2026.
Internal planning document for the section drafts; not report prose.

---

## How to use this document

This covers Sections 4–7 of the detailed report. Each section lists its
subsections and the content that must appear in each. Items tagged
**[LOAD-BEARING]** are not optional — they carry a claim the report rests on, or
close a gap a grader will otherwise find.

These four sections differ from 1–3 in one important way: **1–3 are written from
published sources, 4–7 are written from this project's own artifacts.** That
changes the sourcing rule. Nothing in 4–7 should be written from memory of the
code or memory of the results.

### Where each section's content comes from

| Section | Written from | Not written from |
|---|---|---|
| §4 Implementation details | `spec/*_spec.md`, `DECISIONS.md` | the code, read informally |
| §5 Explanation of source code | the defense gates and walk-throughs | the source files alone |
| §6 Results and discussion | `FINDINGS.md`, `results/*.json`, generated figures | any number typed from chat or memory |
| §7 Future work | §6's open questions, `FINDINGS.md` open items | speculation |

### Three rules that apply throughout

1. **Every number traces.** A figure quoted in §6 comes from a `FINDINGS.md`
   entry or from a value computed out of `results/`. No number is transcribed
   from a conversation, a summary, or a remembered run.
2. **The measured/inferred distinction survives into the prose.**
   `FINDINGS.md` tags every clause as measured or inferred. §6 must preserve
   that: measured claims stated flat, inferred claims phrased as *consistent
   with*, never *because*. This is the single most attackable property of the
   results section and the one the study has already invested most in getting
   right.
3. **§4 and §5 are drafted per component, immediately after that component's
   defense gate** — not batched at the end. The understanding is loaded at that
   moment and will not be again.

### Ownership note

The proposal assigns Kiarash the architectures, harness, depth-sweep
infrastructure and integration; Yiheng the metrics, visualization, tuning, and
the applications/noise discussion, with both co-leading the analysis. §4 and §5
should be written by whoever defended the component in question — those sections
are precisely the ones the video requires each member to speak to. Confirm the
split before drafting, and make sure the report's stated role division matches
what actually happened, since a grader can compare it against the proposal.

---

## Section 4 — Implementation Details

**Purpose.** Describe what was built and why it was built that way, at the level
of design decisions rather than line-by-line code. §4 answers "what are the
moving parts and what choices govern them"; §5 answers "how does the code
express that."

**Source of truth.** The seven spec files and `DECISIONS.md`. Where a spec and
`DECISIONS.md` disagree, `DECISIONS.md` wins — and the disagreement should be
fixed in the spec, not silently resolved in the prose.

### 4.1 Environment and reproducibility

- [ ] Python version, PyTorch and PyTorch Geometric versions, device. These are
      recorded in every results record's `environment` block — read them off a
      record rather than from memory.
- [ ] Seeding: the literal seed list 0–9, seeded once per run at the top of the
      training routine.
- [ ] **[LOAD-BEARING]** The reproducibility path end to end: how the dataset is
      obtained, how the sweep is launched, where results land, how figures are
      regenerated from them. The course grades explicitly on whether the
      submitted code reproduces the report's results using the README's
      instructions, so §4 and the README must describe the same path.

### 4.2 Repository structure and module boundaries

- [ ] The module layout (`models/`, `train/`, `metrics/`, `mitigations/`,
      `viz/`, `experiments/`, `results/`, `notebooks/`) and what each owns.
- [ ] Each component defined once in its module and imported by the harness;
      notebooks used for exploration and figure generation only, never as a
      source of truth.
- [ ] **[LOAD-BEARING]** The interface contract as the shared surface that let
      two people work in parallel: models return logits plus the per-layer
      embedding stack; the results record schema; the PyG data fields. This is
      worth stating as an engineering decision, not just documenting — it is
      the reason the metrics work and the harness work could proceed
      independently.

### 4.3 Data pipeline

- [ ] Dataset loading via Planetoid with the standard public split; the fields
      consumed and their shapes.
- [ ] Feature row-normalization applied uniformly to every run, with the
      reproduction-fidelity reason (§3 defines what it does; §4 states that it
      is applied and why it is not an ablation axis).
- [ ] Where self-loops and normalization are handled — inside the convolution
      layers rather than at the data level — and the verification that no
      double normalization occurs.
- [ ] That the metrics module augments its own copy of the graph independently,
      and asserts on entry that the input has no self-loops so
      double-augmentation fails loudly rather than silently.

### 4.4 Model implementation

- [ ] The base class owning the layer loop, with per-architecture subclasses
      supplying only the convolution. Why composition rather than inheritance
      for mitigations: a mitigation is a modification applied to a model, not a
      kind of model.
- [ ] The two protocols — the per-layer, shape-preserving hook and the readout
      that consumes the whole embedding stack — and why they are separate types
      rather than one list.
- [ ] The null-object treatment of the unmitigated baseline (empty hook list,
      last-layer readout) so that baseline and mitigated runs execute an
      identical code path. **[LOAD-BEARING]** — this is what licenses
      attributing a difference in results to the mitigation rather than to a
      divergence in control flow, and the depth sweep's central comparison
      depends on it.
- [ ] The uniform convolution call signature, and the cost accepted for it (one
      unused argument for three of four convolution types).
- [ ] GCNII's resolution via uncounted input and output projections, and why
      `numLayers` stays a hop count across all four architectures.
      **[LOAD-BEARING]** — depth is the study's primary axis, so hop
      comparability across architectures is what makes the axis mean one thing.
- [ ] **[LOAD-BEARING]** The layer-embedding tap point: after the convolution,
      after the hooks, after the activation, before dropout — with the reasoning
      for each of those three positions. §6's per-layer curves are curves of
      *this* tensor, and the tap point determines what they measure.
- [ ] That embeddings are returned attached to the autograd graph and detached
      at the capture site, with the reason (Jumping Knowledge backpropagates
      through the stack, so detaching inside the model would silently break it).

### 4.5 Mitigation implementations

- [ ] Residual as plain identity addition with an explicit no-op on width
      mismatch; the consequence that it fires only on the interior layers.
- [ ] PairNorm: the variant used, the fixed scale, the two steps, and the
      zero-clamp guard needed because dead units produce all-zero rows at depth.
- [ ] Jumping Knowledge: max-pooling over the layer stack plus a linear readout,
      and why max rather than concatenation (concatenation would make the
      parameter count grow with depth, confounding the mitigation with a
      capacity increase at exactly the depths where the study claims it helps).
      **[LOAD-BEARING]** — a grader who knows the JK paper will ask why the
      standard concatenation variant was not used.
- [ ] Hook ordering when both residual and PairNorm are present, and why the
      order is not arbitrary.

### 4.6 Training and evaluation harness

- [ ] The optimizer, learning rate, dropout, weight decay, and where each came
      from.
- [ ] **[LOAD-BEARING]** Early stopping and checkpoint selection: both driven by
      validation loss; stopping determines when the loop halts, selection
      determines which weights are kept, and the two are different operations.
      §6 quotes `epochsRun` as evidence of trainability, which only means
      something if the reader knows what ended the run.
- [ ] **[LOAD-BEARING]** The generous, constant patience, and the reason it is
      generous: a tight stopping rule can halt a deep run before its loss begins
      to descend, manufacturing exactly the flat training curve the study uses
      to diagnose untrainability. The harness must not manufacture the failure
      it is meant to measure. This is a direct pre-emption of the strongest
      available attack on §6's SAGE/GAT result.
- [ ] Uniform weight decay across all layers, and why the published
      first-layer-only scheme has no principled extension to 32 layers.
- [ ] The three capture points and the order in which they are taken, including
      why the final-epoch capture is taken before the checkpoint weights are
      restored.
- [ ] Macro-F1 implemented in-repo rather than imported, with the verification
      test against a reference implementation. Worth stating explicitly: the
      course grades how much of the code the team wrote versus took from prior
      work.

### 4.7 Metrics implementation

- [ ] Dirichlet energy computed over the augmented graph, fixed across all
      architectures, with the metric graph treated as a measuring instrument
      rather than part of the model. **[LOAD-BEARING]** — and state the known
      cost for attention-based architectures, whose own propagation graph
      differs from the one the metric uses.
- [ ] Both metrics computed over the full node set with no split mask, and why.
- [ ] The MAD reduction convention as specified by the source paper, plus the
      two implementation choices the paper leaves unstated (diagonal exclusion,
      zero-denominator clamping) and why they matter specifically in the deep
      regime this study measures.
- [ ] **[LOAD-BEARING]** The comparable band: which layer indices are metrically
      comparable, why index 0 is excluded unconditionally, why the final index
      is excluded when it is the logit space, and how the band differs under
      Jumping Knowledge. Every per-layer figure in §6 is plotted over this band.
- [ ] **[LOAD-BEARING]** The energy floor applied before taking logarithms, and
      its quantified effect. This is not a footnote: the floor binds on a large
      fraction of the band at depth, and the direction of its bias must be
      named — it pulls a reported slope toward zero, so it can only understate
      the effect, never manufacture one.
- [ ] The decision not to divide the reported curve by representation magnitude,
      and why: weight-norm growth is inside the mechanism the theory describes,
      not noise on top of it.

### 4.8 Experiment grid and sweep infrastructure

- [ ] The six arms, what each answers, and the run counts that sum to the total.
- [ ] The depth grid and why it is log-spaced.
- [ ] The coordinated hyperparameter search: tuned once at shallow depth on one
      architecture, then frozen across everything. **[LOAD-BEARING]** — tuning
      per depth would confound the depth effect with a hyperparameter effect,
      which would invalidate the study's primary axis.
- [ ] The idempotent, resumable sweep with atomic writes and a single writer;
      the failure-marker convention so a missing result is distinguishable from
      a run that was never scheduled.
- [ ] The results-file naming convention and the subdirectory separation for the
      fidelity and hyperparameter-search arms, with the collision that motivated
      it. Worth including: it was found by the spec's own test, not in
      production, and it would otherwise have silently overwritten results.
- [ ] Which runs persist embeddings to disk and why only those.

### 4.9 Visualization and aggregation

- [ ] The separation of aggregation from plotting: loading and table-building
      return tables, plotting functions consume tables. Mean and standard
      deviation computed at read time, never stored.
- [ ] **[LOAD-BEARING]** That the band is read from each record rather than
      recomputed by the plotting code — a plotting function that assumes a fixed
      band silently truncates every Jumping Knowledge curve.
- [ ] The projection method used for the qualitative embedding figures, its
      fixed parameters, and that panels are fitted separately.
- [ ] Figure sizing for the article's column width, decided in advance.

### 4.10 Deviations from published setups, consolidated

**[LOAD-BEARING] — this subsection should exist as a single consolidated list,
not scattered across the section.** A grader comparing against the original
papers will find these; finding them already stated, each with its reason and
the direction of its effect named, converts a weakness into evidence of care.

- [ ] Hidden width chosen for capacity parity across architectures rather than
      matching any single published value; the fidelity arm run separately to
      recover the published-width number as a measured quantity.
- [ ] Uniform activation and dropout across architectures, with the reason it is
      load-bearing rather than cosmetic (the tap-point justification depends on
      the same activation being applied everywhere).
- [ ] Weight decay applied uniformly rather than to the first layer only.
- [ ] Stopping-rule window and epoch ceiling loosened relative to the published
      two-layer setup, with the reason.
- [ ] Checkpoint selection on validation loss where the proposal said accuracy —
      stated as a deliberate refinement, with the recorded note that the
      alternative is derivable from stored data and can be checked rather than
      asserted.
- [ ] One shared hyperparameter configuration across architectures whose
      published setups differ, with the direction named: architectures furthest
      from the shared configuration are the ones most likely to be understated.
- [ ] The mitigation configurations held at fixed, untuned values, with the
      direction named.
- [ ] GCNII's architecture hyperparameters taken from its own paper while its
      training hyperparameters come from this study's shared configuration — a
      hybrid, not a reproduction, and stated as such.
- [ ] The extra projection layers GCNII carries that the other architectures do
      not, with the direction named.

### 4.11 Placement decision to settle

The diagnostic apparatus — the three capture points, the comparable band, the
contraction slope — is currently listed in both this section and the §1–3
outline as a placement question. **Decide once.** Either §3 defines them as part
of the measurement theory and §4 describes how they are implemented, or §4 owns
them entirely. What is not acceptable is neither claiming them, leaving §6 to
define its own instruments in the middle of reporting results.

---

## Section 5 — Explanation of the Source Code

**Purpose.** The course asks for an explanation of the source code as a distinct
deliverable. This is not §4 restated and it is not a code listing. It is a
walkthrough of how the code expresses the design, at a level where a reader who
has the repository open can follow along.

**Source of truth.** The defense gates and walk-throughs. This is the section
that cannot be written without them — which is why the gates are on the critical
path and not optional overhead.

### 5.0 Framing

- [ ] **[LOAD-BEARING]** Write this section only for components that have been
      through a defense gate. A subsection written about code nobody walked
      line by line is exactly the passage that cannot be defended on video, and
      the code is separately reviewed by an AI tool under a zero-tolerance
      plagiarism policy. If a component has not been gated, gate it or say less
      about it.
- [ ] Decide the level of detail once and hold it: per-module, describing the
      key classes and functions and the data flowing between them, with short
      excerpts only where an excerpt makes a point that prose cannot.

### 5.1 Repository map and reading order

- [ ] A short orientation: which module to read first, what depends on what, and
      where the entry point is.
- [ ] The build order that the components were actually developed in, since it
      is also the dependency order.

### 5.2 The interface contract in code

- [ ] The forward signature and what it returns.
- [ ] The results record structure and which module assembles it versus which
      module writes it — exactly one writer.
- [ ] **[LOAD-BEARING]** The one place where the contract is expressed, for each
      clause, and why single-sourcing it matters (a convention duplicated in two
      modules drifts).

### 5.3 Per-module walkthroughs

One subsection per module, in dependency order: data, models, metrics, train,
mitigations, experiments, viz.

For each module:

- [ ] What it owns and what it deliberately does not.
- [ ] The main classes/functions, their signatures, and the shapes flowing
      through them.
- [ ] The one or two non-obvious implementation points — the places where a
      reader would otherwise ask "why is it written this way." These are
      typically the ones already recorded as decisions.
- [ ] Where vectorization was required and how it was achieved (no per-node or
      per-edge Python loops).
- [ ] **[LOAD-BEARING]** For `metrics` specifically: the numerical guards, and
      why each exists. Dead units at depth produce exact zeros, and both metrics
      have a reduction that is undefined on them. The guards are not
      housekeeping — they are the reason the metrics return a number at all in
      the regime the study is about.

### 5.4 Testing

- [ ] The test suite's role: it replaces the "runs top to bottom" guarantee a
      notebook would have given.
- [ ] The categories of test — numerical correctness against hand-computed or
      reference values, grid enumeration and uniqueness, shape and contract
      assertions, and the qualitative gates.
- [ ] **[LOAD-BEARING]** At least two tests that caught a real defect, described
      as such. The path-collision test and the collapse-floor test both found
      genuine problems before they reached the results. This is the most
      persuasive evidence available that the test suite is doing work rather
      than decorating the repository.
- [ ] The one test that gates on the pre-training capture rather than the
      trained one, and why — it connects directly to §6's central finding.

### 5.5 What the code does not do

- [ ] Honest statement of what is not instrumented: per-layer gradient norms and
      per-layer weight norms are not persisted anywhere in the results records.
      §6 and §7 both depend on this being stated, because it is the reason one
      of §6's mechanism claims stays inferred rather than measured.
      **[LOAD-BEARING]**

---

## Section 6 — Results and Discussion

**Purpose.** Report what was measured, and interpret it — with the boundary
between those two visible in the prose.

**Source of truth.** `FINDINGS.md` F-001 through F-007 are the skeleton of this
section. Each entry is already split into Measured and Inferred; keep that split
when it becomes prose. Numbers come from the findings entries or from
regenerated figures, never from memory.

**Before drafting:** confirm the figures and tables have been regenerated since
the corrections. Several were produced before the energy-ratio reading was
corrected and before the summary statistic changed.

### 6.0 Reporting conventions

- [ ] State up front that results are reported as mean and standard deviation
      across ten seeds, on the standard test split.
- [ ] **[LOAD-BEARING]** State which summary statistic is used for which
      quantity, and why. Accuracy is summarized by arithmetic mean; the energy
      ratio is a multiplicative, log-scale quantity spanning many orders of
      magnitude across seeds and is summarized by geometric mean and range. A
      reader who sees two different summary conventions without explanation will
      assume inconsistency.
- [ ] State which capture each figure reads from — pre-training,
      best-validation checkpoint, or final epoch. Every energy or MAD figure
      must name its capture, or its provenance is implicit.

### 6.1 Baseline reproduction

- [ ] The shallow-depth reproduction result against the published number, from
      the fidelity arm.
- [ ] The width comparison showing the study's chosen width does not materially
      change the shallow baseline. **[LOAD-BEARING]** — this is what licenses
      every subsequent deviation from the published width.
- [ ] The hyperparameter search result: the winner, and the honest statement
      that the search does not cleanly separate the top configurations at the
      seed count used. State that the winner also coincides with the published
      configuration, so the selection does not depend on trusting a ranking that
      close to noise.

### 6.2 Depth sweep: classification performance

- [ ] Accuracy and macro-F1 versus depth, per architecture, with variance bands.
- [ ] The monotone degradation for the three conventional architectures, and the
      exception.
- [ ] **[LOAD-BEARING]** The comparison against the majority-class floor, and
      the statement that it holds for every seed rather than only on average.
      This is the sharpest single number in the study.

### 6.3 Structural collapse at initialization

- [ ] The pre-training picture: what the operator composed with untrained
      weights does to both metrics as depth grows.
- [ ] **[LOAD-BEARING]** State clearly that this is the initialized network, not
      the propagation operator alone — the random weights' magnitudes are part
      of what drives the contraction.
- [ ] The architecture-dependent difference in *kind*: two architectures
      collapsing in both direction and magnitude, one collapsing in direction
      only.
- [ ] **[LOAD-BEARING]** The geometric explanation for that dissociation, tied
      back to §3.2's null space result. This is the point where the two-metric
      design pays off, and it should be presented as such.
- [ ] The single-hop anomaly, what was ruled out by direct check, and what
      remains open. Present the ruled-out candidate as a measured negative
      result, not as an omission.
- [ ] Consistency with the theoretical prediction stated qualitatively only —
      the decay is consistent with exponential contraction; no claim of
      numerical agreement with the bound.

### 6.4 The trained state, and what it does to the collapse signature

- [ ] The checkpoint picture, contrasted against the pre-training picture.
- [ ] **[LOAD-BEARING]** The correct reading of the energy ratio at depth: it is
      large because the early-band denominator has collapsed, not because
      nothing collapsed. Read correctly, the trained state shows early-layer
      collapse together with late-layer growth. Getting this backwards was a
      real error caught during analysis, and the corrected reading is what §6
      must present.
- [ ] The cross-seed evidence for early-layer collapse, including the one seed
      that does not follow the pattern — reported as an exception, not smoothed
      into the average.

### 6.5 Separating oversmoothing from optimization failure

- [ ] **[LOAD-BEARING]** This is the section's central methodological
      contribution and should be presented as one. Accuracy alone cannot
      distinguish "never trained" from "trained then collapsed"; the schema was
      designed from the start to separate them, and here is what that separation
      shows.
- [ ] The training-loss evidence: which architectures descend, which stay flat,
      and against what reference value.
- [ ] The independent stopping-behavior evidence, and why it is independent —
      it falls out of the early-stopping mechanism rather than out of the
      analysis, so it is not derived from either the loss values or the metrics.
- [ ] **[LOAD-BEARING]** The three-way convergence argument stated explicitly:
      three separately measured quantities point the same way, none computed
      from the others. Say why that is convergence rather than circularity.
- [ ] The resulting taxonomy: distinct depth-failure modes across the
      conventional architectures, named and distinguished.
- [ ] The positive-control argument: the metrics do fire cleanly where collapse
      is genuinely present, which is what rules out insensitivity as the
      explanation where they do not fire.

### 6.6 The architecture that does not degrade

- [ ] Accuracy rising with depth, against the trend.
- [ ] Its contraction behavior contrasted against the architecture that blows
      up.
- [ ] The mechanism stated as *consistent with* the architectural design, not as
      demonstrated — the per-layer weight and gradient evidence that would
      demonstrate it was not collected.

### 6.7 Mitigation ablation

- [ ] The full ablation table at depth, all arms including the baseline.
- [ ] The clear winner and the margin.
- [ ] The arm that does not beat the baseline, reported with its variance and
      with the honest statement that the seed count does not support a confident
      "this hurts" claim.
- [ ] The negative interaction between two mitigations that each help
      individually — a real result worth its own sentence.
- [ ] **[LOAD-BEARING]** The design note that anticipated the residual result
      before it was measured, and the diagnostic that would test it. A design
      predicting its own finding is worth stating; so is the fact that the
      prediction is confirmed as *worth investigating* rather than confirmed as
      correct.
- [ ] Whether the winning mitigation generalizes to the other architectures.

### 6.8 Qualitative embedding analysis

- [ ] The projection figures at shallow versus deep settings.
- [ ] **[LOAD-BEARING]** Framing constraint: the projection *illustrates*
      collapse, it does not measure it. The metrics measure it. Label the figure
      qualitative and say so in the text — a projection presented as evidence of
      a quantity is an overclaim a grader will catch.
- [ ] Consistency (or otherwise) between what the projection shows and what the
      metrics report, discussed rather than assumed.

### 6.9 Threats to validity and limitations

**[LOAD-BEARING] as a subsection.** Every deviation listed in §4.10 has a
consequence, and this is where the consequences are owned. For each: name it,
name the direction of its effect, and say whether it could have manufactured the
finding or only understated it.

- [ ] The metric graph does not match the propagation graph for the
      attention-based architecture.
- [ ] The energy floor understates the magnitude of change at depth.
- [ ] Mitigations run at untuned, fixed configurations — can only understate
      them.
- [ ] One shared hyperparameter configuration across architectures with
      differing published setups.
- [ ] The residual mechanism cannot fire on the boundary layers.
- [ ] The extra parameters one architecture carries.
- [ ] Ten seeds; where that is and is not enough (three seeds in the
      hyperparameter arm specifically).
- [ ] Single dataset, single split — what does not generalize.
- [ ] **[LOAD-BEARING]** The findings that remain inferred rather than measured,
      named as such, with what instrumentation would settle them.
- [ ] The full-collapse geometry caveat: a "fully collapsed" representation
      under this metric is not literally identical across nodes.

### 6.10 Synthesis

- [ ] Pull the parts together into the claim the report actually defends.
- [ ] **[LOAD-BEARING]** Return explicitly to the thesis stated in §1 and
      confirm what the evidence supports and what it does not. The proposal
      anticipated progressive collapse with depth; the report should close the
      loop on that honestly, presenting the narrower and more precise finding as
      the contribution it is.
- [ ] Applications revisited: what the depth-failure taxonomy implies for the
      real-world settings named in §1.

---

## Section 7 — Recommendations for Future Work

**Purpose.** Say what should be done next, grounded in what this study actually
left open. The strongest version of this section is short and specific, drawn
from §6's own open questions rather than from generic suggestions.

### 7.1 Questions this study raised and did not answer

- [ ] The unexplained single-hop behavior of one architecture, with the two
      candidate explanations that remain unisolated and the specific
      experiment — an aggregation variant and an initialization variant, tested
      separately — that would isolate them. **[LOAD-BEARING]** — this is the
      most concrete open question the study produced.
- [ ] The mitigated arms received none of the per-run scrutiny the unmitigated
      baseline received; the negative interaction between two mitigations is
      measured but unexplained.
- [ ] The one seed that behaves differently from the other nine, noted as a
      correlation and not investigated.

### 7.2 Instrumentation gaps

- [ ] **[LOAD-BEARING]** Per-layer gradient norms are the single measurement
      that would convert this study's main mechanism claim from inferred to
      demonstrated. Say exactly what would need to be recorded and what it would
      settle.
- [ ] Persisted per-layer weight norms across the sweep, which would upgrade a
      single-run observation to a characterized pattern.
- [ ] Dense-capture trajectories over training, for which the schema already
      reserves a field — the capability exists and was deliberately deferred.
- [ ] A training-accuracy field, which would allow the generalization-gap
      question to be answered rather than set aside.

### 7.3 Extensions to the experimental design

- [ ] Depth resolution between the points where the transition happens.
- [ ] Per-architecture hyperparameter tuning as a separate study, and what it
      would and would not tell you that the shared configuration cannot.
- [ ] Additional datasets, particularly ones differing in homophily and in
      degree distribution — the second matters directly given the null-space
      result.
- [ ] Decomposing the contraction into its aggregation-driven and
      nonlinearity-driven parts, which the current tap point cannot separate.

### 7.4 Scale and noise

- [ ] Picks up the discussion opened in §2.5. What the depth findings imply for
      graphs too large for full-batch training.
- [ ] Structural noise and whether the diagnostic apparatus transfers to it.
- [ ] The improvement proposal the project description asks for — depth-aware
      architecture design with normalization — stated here as a recommendation,
      with the study's own evidence behind it.

### 7.5 Transfer to physics-informed graph learning

- [ ] The motivation stated in §1 revisited: domains where depth is not
      optional.
- [ ] **[LOAD-BEARING]** Keep the honest framing the proposal already
      committed to. Cora is small and tightly clustered and will not transfer
      domain-specific findings to much larger meshes. What transfers is the
      diagnostic methodology and the qualitative behavior of the mitigations —
      a reusable measurement apparatus, not the numbers. Overclaiming here
      undoes the care taken everywhere else.

---

## Not sections — separate deliverables

These are required but are not part of the seven-section report, and each has
its own constraints. Listed here so they are not forgotten in the section
planning.

- **The article.** Five pages in conference format plus one page for references
  only. It is a *compression* of the detailed report, so it is written by
  cutting the detailed report down, not by drafting independently. Figure sizing
  for the column width was already decided; the constraint is text, not figures.
- **The README.** Written last, then tested by following it from a clean clone.
  It must describe the same reproduction path §4.1 describes. The course grades
  explicitly on whether the submitted code reproduces the report's results using
  these instructions.
- **The video.** Fifteen to twenty-five minutes, with every member presenting
  their own contribution. Script from the report's structure. Rehearse the
  components most likely to be probed — the metrics and the models — since those
  carry the claims.
