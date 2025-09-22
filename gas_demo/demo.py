"""Command-line demo for the GAS primitives and derivatives."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from .objectives import QuadraticObjective
from .primitives import (
    DifferentialDistinction,
    DifferentialMemory,
    DifferentialTransformation,
    GradientDescentTransformation,
    IdentityDistinction,
    LinearRelation,
    RobbinsMonroMemory,
)
from .simulation import AgentComponents, run_agent
from .visualization import describe_trajectory, format_summary_table


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the closed-loop simulation."""

    horizon: int = 60
    initial_state: float = 3.0
    seed: int = 7


def build_agents(relation: LinearRelation) -> Sequence[AgentComponents]:
    del relation  # Relation-specific tuning can be added later.
    level0 = AgentComponents(
        name="Level-0 (Robbins-Monro)",
        distinction=IdentityDistinction(),
        transformation=GradientDescentTransformation(base_step=0.7),
        memory=RobbinsMonroMemory(beta=0.15),
    )
    level1 = AgentComponents(
        name="Level-1 (Differential)",
        distinction=DifferentialDistinction(decay=0.3),
        transformation=DifferentialTransformation(base_step=0.9, damping=0.3),
        memory=DifferentialMemory(beta=0.15, momentum=0.5),
    )
    return (level0, level1)


def run_demo(config: DemoConfig) -> None:
    relation = LinearRelation(dt=0.15, relaxation=0.35, noise_std=0.01)
    objective = QuadraticObjective(target=0.0)
    agents = build_agents(relation)
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
    print("Generalized Adaptive Systems Demo")
    print("=================================")
    print(format_summary_table(trajectories))
    print()
    for traj in trajectories:
        print(describe_trajectory(traj, prefix="- "))


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=60, help="number of time steps to simulate")
    parser.add_argument("--initial-state", type=float, default=3.0, help="initial state u_0")
    parser.add_argument("--seed", type=int, default=7, help="random seed for the environment noise")
    args = parser.parse_args()
    return DemoConfig(horizon=args.horizon, initial_state=args.initial_state, seed=args.seed)


if __name__ == "__main__":
    run_demo(parse_args())
