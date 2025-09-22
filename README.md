# Generalized Adaptive Systems Demo

This repository contains a demonstrative implementation of the Level‑0 primitives ⟨D, R, T, M⟩ and the Level‑1 differential operators described in *Foundational Principles for Generalized Adaptive Systems*. The goal is to provide a concrete, inspectable simulation that shows how the primitives cooperate to drive Lyapunov drift and how the Level‑1 refinements accelerate adaptation.

The repository now includes two vivid demonstrations:

1. **Scalar Lyapunov descent.** A one‑dimensional controlled process with a quadratic objective where Level‑0 and Level‑1 agents can be compared head‑to‑head using the canonical primitives.
2. **Inertial stabilization.** A damped mass‑spring plant with inertia. The Level‑0 agent only observes position, while the Level‑1 agent sees the full state, leverages relation derivatives, and annihilates overshoot to settle dramatically faster.

## Repository layout

```
README.md                # This file.
gas_demo/                # GAS primitives, agents, and simulation utilities.
    __init__.py
    objectives.py
    primitives.py
    simulation.py
    demo.py              # Legacy scalar comparison.
    inertial.py          # High-impact inertial showcase primitives.
    showcase.py          # Combined "wow" demo runner.
    visualization.py     # Textual visualization helpers.
tests/
    test_adaptivity.py   # Ensures both agents reduce the Lyapunov objective.
```

## Running the demo

```bash
python -m gas_demo.showcase
```

This command first runs the scalar Lyapunov descent demo and then launches the inertial showcase. Expect to see the Level‑1 controller slash overshoot and settle in a fraction of the time while still respecting the Lyapunov drift guarantees.

The legacy scalar comparison is still available if you prefer the minimal setup:

```bash
python -m gas_demo.demo
```

## Running the tests

```bash
python -m unittest discover -s tests
```

The tests verify that both agents decrease the Lyapunov functional and that the Level‑1 agent achieves strictly better convergence after the same number of steps.

## Extending the demo

The code is intentionally modular. You can experiment with alternative objectives, relation kernels, and memory dynamics by composing new primitives:

- Implement a new `Distinction` to extract richer features (e.g., higher-order inertial moments).
- Swap the `Relation` for a nonlinear or data‑driven kernel.
- Replace the `Transformation` with a reinforcement‑learning policy.
- Encode advanced filtering or estimator logic inside the `Memory`.

Each component remains measurable and composable, preserving the Level‑0 guarantees while enabling exploration of higher‑level operators.
