# Tests

## Regression
- `./build/pde_sim --self-test` (regression suite with analytic golden outputs)
- `./build/regression_suite` (CI-style regression runner)
- `./build/checkpoint_equivalence` (checkpoint restart equivalence)

## Smoke
- `./tests/pde_poisson.sh` (runs a small Poisson solve and checks output file)

## Nonlinear / conservation-law pipeline (Option 2)
- `./build/burgers_fd_test` — FD nonlinear derivative evaluation (Burgers)
- `./build/shock_tube_test` — 1D FV Sod shock tube smoke test
- `./build/reaction_diffusion_imex_test` — IMEX advection–diffusion time stepping
- CLI: `--discretization fd|fv`, `--stability-check` (with `--validate` or full run args)

## GUI Sanity (manual)
- Build and launch `./build/pde_gui`
- Load a simple PDE, run, verify image renders without crash

