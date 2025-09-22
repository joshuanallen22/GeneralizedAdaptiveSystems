"""Regression tests for the inertial GAS showcase."""

from __future__ import annotations

from gas_demo.inertial import (
    FullInertialDistinction,
    InertialEnergyObjective,
    InertialGradientController,
    InertialLookaheadController,
    InertialMemory,
    InertialRelation,
    InertialState,
    PositionOnlyDistinction,
    energy_threshold_time,
    overshoot,
    run_inertial_agent,
)
from gas_demo.simulation import AgentComponents


def test_inertial_level1_dominates_level0() -> None:
    relation = InertialRelation(dt=0.08, damping=0.32, stiffness=1.55, noise_std=0.0)
    objective = InertialEnergyObjective(position_weight=1.0, velocity_weight=0.45)
    initial_state = InertialState(position=2.5, velocity=0.0)

    level0 = AgentComponents(
        name="Level-0",
        distinction=PositionOnlyDistinction(),
        transformation=InertialGradientController(base_step=0.6),
        memory=InertialMemory(beta=0.1),
    )
    level1 = AgentComponents(
        name="Level-1",
        distinction=FullInertialDistinction(),
        transformation=InertialLookaheadController(position_gain=1.6, velocity_gain=1.2),
        memory=InertialMemory(beta=0.1),
    )

    traj0 = run_inertial_agent(
        components=level0,
        relation=relation,
        objective=objective,
        initial_state=initial_state,
        horizon=120,
        seed=0,
    )
    traj1 = run_inertial_agent(
        components=level1,
        relation=relation,
        objective=objective,
        initial_state=initial_state,
        horizon=120,
        seed=0,
    )

    assert traj1.final_cost < 0.15 * traj0.final_cost
    assert energy_threshold_time(traj1) <= 0.5 * energy_threshold_time(traj0)
