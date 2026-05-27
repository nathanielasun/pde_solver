# Planning Documents

All project planning lives in this directory. Use **`MASTER_ROADMAP.md`** as the single
authoritative plan for ongoing and future work.

## Primary document

| File | Purpose |
| --- | --- |
| [`MASTER_ROADMAP.md`](MASTER_ROADMAP.md) | Consolidated roadmap: current state, priority tracks, work packages, acceptance criteria |

## Source / reference documents

These files record original intent and detailed specifications. They are kept for
historical context; new work should be tracked in `MASTER_ROADMAP.md`.

| File | Scope |
| --- | --- |
| [`PDE_EXPANSION_PLANS.md`](PDE_EXPANSION_PLANS.md) | Strategic Options 1–3 (FD expansion, nonlinear/FV, FEM) |
| [`PDE_TERMS_IMPLEMENTATION_PLAN.md`](PDE_TERMS_IMPLEMENTATION_PLAN.md) | Tiered PDE term features mapped to Option 1 phases |
| [`GPU_PARITY_PLAN.md`](GPU_PARITY_PLAN.md) | Backend capability matrix and Phase 7.1 GPU priorities |
| [`GUI_UX_IMPROVEMENTS.md`](GUI_UX_IMPROVEMENTS.md) | Original GUI/UX recommendations and architecture notes |
| [`DOCKING_SYSTEM_PLAN.md`](DOCKING_SYSTEM_PLAN.md) | Splittable docking UI design and migration plan |
| [`COMPLETED_PHASES.md`](COMPLETED_PHASES.md) | Log of completed GUI Phase 1–2 work (partially stale) |

## How to use

1. Pick the next open work package from `MASTER_ROADMAP.md` (top of backlog is ordered).
2. Implement against the acceptance criteria in that package.
3. Update the status column/checkbox in `MASTER_ROADMAP.md` when done.
4. Update user-facing docs (`README.md`, `PROJECT_DOCS.md`) if behavior or CLI changes.

## Related non-planning docs (repo root)

- [`README.md`](../../README.md) — build, run, usage
- [`PROJECT_DOCS.md`](../../PROJECT_DOCS.md) — module inventory
- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) — module boundaries
