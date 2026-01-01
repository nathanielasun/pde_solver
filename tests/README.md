# Tests

## Regression
- `./build/pde_sim --self-test` (regression suite with analytic golden outputs)
- `./build/regression_suite` (CI-style regression runner)
- `./build/checkpoint_equivalence` (checkpoint restart equivalence)

## Smoke
- `./tests/pde_poisson.sh` (runs a small Poisson solve and checks output file)

## GUI Sanity (manual)
- Build and launch `./build/pde_gui`
- Load a simple PDE, run, verify image renders without crash

