# Sparse-DFT route for a project AGENTS.md

可将以下路由加入使用 Sparse-DFT 的项目说明；无需覆盖用户已有学术写作或 DFT_Skills 规范。

```markdown
## Sparse-DFT execution and case studies

- When Sparse-DFT is requested, use sparse-dft-driver as the software execution adapter.
- For research design, use dft-workflow-orchestrator when available. It owns the integrated research plan, claims, stage DAG and research spine; sparse-dft-driver owns Sparse-DFT calls and result handoff.
- Record Sparse-DFT calculation stages as kind=dft, engine=Sparse-DFT. Use sparse-dft-driver CLI/API/Python/native/tool routes; do not run those stages through VASP-specific scripts.
- Choose one relevant executable case from sparse-dft-driver/case-studies/INDEX.md. Use its actual task ladder, inputs, analysis and completion conditions.
- Reuse identifiable project accuracy evidence and retain its source; archived declarations do not validate new capabilities. Do not repeat software precision tests as a routine prerequisite. Check actual new-run convergence and outputs; preserve source flags and external validation provenance separately.
- Standalone sparse-dft-driver must complete the same computational route without requiring companion skills. Derived analyses use project-local scripts with explicit formulas and raw-output provenance.
- Send actual result artifacts, conditions, units, source hashes and evidence locators to Academic_Writing_Skills when writing is requested. Literature DOI/page checks remain in sci-literature-evidence; original calculation records do not receive fabricated citations.
- Keep outputs outside skill directories, preserve existing claim/stage IDs and never overwrite another session's active files.
```
