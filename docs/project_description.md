# Project Description — Cora GNN Oversmoothing Study

Internal reference for the repo: what we are building and what the course requires.
This is a source of record for requirements — re-read it (not memory) when confirming
what a deliverable must contain.

## Course

SEP 740 / CSE 705 — Deep Learning (Dr. Anwar Mirza), W Booth School of Engineering,
McMaster University, Summer 2026. Group project (2–5 students). Group: Kiarash Ara
(lead / integration owner), Yiheng Wu.

## Topic (course project: Graph Representation Learning with GNNs on Cora)

Develop a graph representation learning system using graph neural networks on the Cora
citation network. Papers are nodes, citation links are edges; each node has a
bag-of-words feature vector and one of seven topic labels. The task is semi-supervised
node classification: preprocess the graph, implement GNN models, train with graph
convolutions / message passing, tune hyperparameters, and evaluate with accuracy and
F1. The project also asks to visualize node embeddings, analyze performance
qualitatively, discuss real-world applications (recommendation systems, social-network
analysis, drug discovery), and address the challenges of large-scale and noisy graphs
with proposed improvements.

## Our angle (past-vanilla scope)

We go beyond a vanilla GCN baseline: we compare GCN, GraphSAGE, and GAT as network
depth increases, measure oversmoothing (representation collapse) with per-layer
Dirichlet energy and MAD, and test mitigations (residual/skip, PairNorm,
Jumping-Knowledge, GCNII). Depth is the comparison axis; oversmoothing is the measured
phenomenon. The diagnostic tooling is reusable for physics-informed GNN work (McPINN),
but the course project stands on its own.

Methods in brief: three architectures via PyTorch Geometric; depth sweep 2–32; metrics
accuracy + macro-F1 (micro-F1 equals accuracy for single-label multiclass) plus
Dirichlet energy and MAD; embedding visualization (t-SNE / UMAP); 10 seeds, mean ± std;
standard Planetoid split.

## Deliverables (final project)

1. Final report, two parts:
a. Detailed report (Word / LaTeX), sections: (1) Introduction, (2) Problem
Statement / Review / Background, (3) Theory and Datasets, (4) Implementation
Details, (5) Explanation of the Source code, (6) Results and Discussion,
(7) Recommendations for Future work.
b. Article-style report: 5 pages IEEE conference style (text, figures, appendices),
plus one additional page allowed for references only.
2. Recorded video presentation: 15–25 minutes; all members present and showing their
contribution.
3. Project code: GitHub repo or zip with a root README.txt giving how to download the
dataset and run the code to replicate the report's results. Code must be clean,
commented, easy to read, and executable without errors. Code is reviewed with an
AI-powered tool; zero-tolerance plagiarism policy.

(Report 1, the proposal, is submitted separately and is 10% of the project marks.)

## Grading factors (no line-item rubric)

The final project (90% of the marks) is judged on:

1. Technical quality of the project.
2. Significance — a real-world problem versus a toy problem, and the impact of the work.
3. Novelty — how novel the approach is; common versus relatively unexplored.
4. Code — how much was written by the team versus taken from prior work, and whether
the submitted code reproduces the report's results using the README instructions.

## Deadlines

* Proposal: submitted (originally June 10, extended to early July).
* Final project (report + article + video + code): Week 13 — August 3, 2026.

