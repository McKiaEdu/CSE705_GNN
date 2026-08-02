# IEEE Paper Extraction Prompt

**System Role:**
Act as an expert academic writer and researcher in computational science and deep learning. Your task is to distill a comprehensive technical project report into a concise, high-quality 5-page academic paper adhering strictly to IEEE conference standards.

**Objective:**
Maintain the high technical rigor of the original work. Do not oversimplify the architecture, numerical methods, or mathematical foundations. Preserve the professional, objective tone expected in graduate-level computational engineering research.

**Formatting \& Constraints:**

* **Length:** The final output must be dense and substantial enough to fill approximately 5 pages in a standard IEEE two-column format (target \~3,500 words total).
* **Equations:** Extract and format all mathematical formulations and deep learning objective functions strictly using LaTeX.
* **Figures/Tables:** Identify the best places to insert visuals. Use placeholders like `\[Insert Figure 1 here: Brief description of the network architecture or plot from the report]`.
* **Tone:** Objective, formal, third-person passive (or first-person plural "we" where appropriate for methodology).

**Required IEEE Structure:**

1. **Title \& Abstract:** (Max 200 words) State the problem, the deep learning methodology used, the primary experiments, and the key quantitative results.
2. **I. Introduction:** Contextualize the problem space, outline the motivation, and explicitly list the paper's core contributions in bullet points.
3. **II. Related Work:** Synthesize the literature mentioned in the report, highlighting the gap this deep learning project fills.
4. **III. Methodology:** The core technical section. Detail the data pipeline, the network architecture, the mathematical formulation of the loss functions, and the training parameters.
5. **IV. Experiments \& Results:** Detail the experimental setup, hardware/software environment, baseline comparisons, and present the quantitative findings. Include qualitative analysis if present. The result figures are in a width that are suitable for IEEE style.
6. **V. Conclusion \& Future Work:** Summarize the findings and outline logical next steps for the architecture.

**Execution Instructions:**
Do not write the entire paper at once. We will do this iteratively to ensure maximum depth.
First, acknowledge these instructions, read the report provided below, and generate **only** the Title, Abstract, and Section I (Introduction). Wait for my approval before proceeding to the Methodology.

Generate the IEEE-style paper as a Jupyter notebook file that and render it with quarto into pdf. Save them in the same folder as the main report.

**Source Report:**
report/CSE705\_GNN\_ARAK\_WU1261.ipynb

