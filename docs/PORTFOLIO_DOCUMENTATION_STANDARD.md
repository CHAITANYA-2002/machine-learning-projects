# ML Portfolio Documentation Standard

Every project walkthrough and README in this repository will follow this standard.
It is deliberately evidence-led: a reader should understand what was built, why
the design is defensible, what was measured, what failed, and what remains
unverified without opening the source code.

## Required narrative structure

1. **The problem and its hardest constraint** — lead with the issue that shapes every later decision: class imbalance, temporal leakage, missing data, compute limits, evaluation bias, or reliability risk.
2. **System at a glance** — project inventory, architecture diagram, data flow, exact artefact counts, and a one-screen description of the runnable path.
3. **Data contract** — origin, schema, collection period, ownership/licensing boundary, missingness, representativeness, and the exact local layout expected.
4. **Design decisions** — explain why every consequential choice was made, what alternative was rejected, and what would make the choice invalid.
5. **Workflow and dependencies** — ordered stages, dependencies, commands, expected runtime, generated artefacts, and safe restart points.
6. **Evaluation and evidence** — exact split strategy, metrics that fit the task, measured values, uncertainty where possible, and plots generated from actual outputs.
7. **Honesty instrumentation** — explicitly state which misleading metrics are excluded, what checks guard leakage or invalid data, and which results cannot be claimed.
8. **Failures and fixes** — retain the real mistakes that changed the implementation. A fix is credible only when its prior failure and regression guard are named.
9. **Current state** — completed, verified, partially verified, blocked, and open items in a compact table.
10. **Limitations and responsible use** — distinguish data limits from model limits, rule out inappropriate uses, and define what would be needed for a production-grade next step.
11. **Appendices** — constants, commands, file layout, data-source links, and reproducibility details.

## Presentation standard

- README and `docs/index.html` carry the same substantive narrative; HTML adds a polished reading layout rather than becoming a thin summary.
- Every diagram must clarify a real relationship: data flow, model boundary, evaluation split, error path, or decision trade-off. Decorative diagrams are excluded.
- Every number is tied to an executed artefact or explicitly marked unavailable.
- Figures are generated from the project workflow whenever source data is present; screenshots or aspirational scores are never used as evidence.
- Preserve upstream/vendor documentation separately and add a project-specific guide rather than overwriting valuable original material.
- Avoid generic claims such as “high accuracy,” “robust,” or “production-ready” unless the accompanying evidence and scope are stated.

## Verification labels

| Label | Meaning |
|---|---|
| **Verified** | Code executed against the stated data; relevant tests and outputs passed. |
| **Implementation verified** | Unit/integration tests pass, but the original full dataset or model asset is unavailable. |
| **Documented, blocked** | Original work is preserved and its recovery path is documented; a missing external input prevents execution. |
| **Open** | A known next step has not yet been implemented or measured. |
