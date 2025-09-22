"""Regression tests for the GAS demo."""

from __future__ import annotations

import unittest

from gas_demo.demo import DemoConfig, build_agents
from gas_demo.objectives import QuadraticObjective
from gas_demo.primitives import LinearRelation
from gas_demo.simulation import run_agent


class TestAdaptivity(unittest.TestCase):
    """Ensure the demo agents satisfy the Level-0 drift guarantees."""

    def test_level0_and_level1_reduce_cost(self) -> None:
        relation = LinearRelation(dt=0.15, relaxation=0.35, noise_std=0.0)
        objective = QuadraticObjective(target=0.0)
        config = DemoConfig(horizon=50, initial_state=3.0, seed=5)
        agents = build_agents(relation)
        initial_cost = objective.value(config.initial_state)
        trajectories = [
            run_agent(
                components=agent,
                relation=relation,
                objective=objective,
                initial_state=config.initial_state,
                horizon=config.horizon,
                seed=config.seed,
            )
            for agent in agents
        ]
        for traj in trajectories:
            with self.subTest(agent=traj.agent):
                self.assertLess(traj.final_cost, initial_cost)
        self.assertLess(trajectories[1].final_cost, trajectories[0].final_cost)
        self.assertGreaterEqual(trajectories[1].action_change_rate, trajectories[0].action_change_rate)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
