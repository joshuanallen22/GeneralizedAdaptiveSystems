"""High-energy showcase comparing Level-0 and Level-1 GAS controllers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from .demo import build_agents as build_scalar_agents
from .inertial import (
    FullInertialDistinction,
    InertialEnergyObjective,
    InertialGradientController,
    InertialLookaheadController,
    InertialMemory,
    InertialRelation,
    InertialState,
    PositionOnlyDistinction,
    format_inertial_summary,
    run_inertial_agent,
)
from .objectives import QuadraticObjective
from .primitives import LinearRelation
from .simulation import AgentComponents, run_agent
from .visualization import describe_trajectory, format_summary_table


@dataclass(frozen=True)
class ShowcaseConfig:
    """Configuration for both scalar and inertial demos."""

    horizon: int = 80
    initial_scalar_state: float = 3.0
    initial_inertial_position: float = 2.5
    initial_inertial_velocity: float = 0.0
    seed: int = 11


def run_scalar_demo(config: ShowcaseConfig) -> Sequence[str]:
    relation = LinearRelation(dt=0.12, relaxation=0.3, noise_std=0.01)
    objective = QuadraticObjective(target=0.0)
    agents = build_scalar_agents(relation)
    trajectories = [
        run_agent(
            components=agent,
            relation=relation,
            objective=objective,
            initial_state=config.initial_scalar_state,
            horizon=config.horizon,
            seed=config.seed,
        )
        for agent in agents
    ]
    output = ["Scalar Lyapunov Descent", "-------------------------", format_summary_table(trajectories), ""]
    output.extend(describe_trajectory(traj, prefix="  ") for traj in trajectories)
    output.append("")
    return output


def build_inertial_agents() -> Sequence[AgentComponents]:
    level0 = AgentComponents(
        name="Level-0 (Position only)",
        distinction=PositionOnlyDistinction(),
        transformation=InertialGradientController(base_step=0.6),
        memory=InertialMemory(beta=0.1),
    )
    level1 = AgentComponents(
        name="Level-1 (Lookahead)",
        distinction=FullInertialDistinction(),
        transformation=InertialLookaheadController(position_gain=1.6, velocity_gain=1.2),
        memory=InertialMemory(beta=0.1),
    )
    return (level0, level1)


def run_inertial_demo(config: ShowcaseConfig) -> Sequence[str]:
    relation = InertialRelation(dt=0.08, damping=0.32, stiffness=1.55, noise_std=0.015)
    objective = InertialEnergyObjective(position_weight=1.0, velocity_weight=0.45)
    agents = build_inertial_agents()
    initial_state = InertialState(position=config.initial_inertial_position, velocity=config.initial_inertial_velocity)
    trajectories = [
        run_inertial_agent(
            components=agent,
            relation=relation,
            objective=objective,
            initial_state=initial_state,
            horizon=config.horizon,
            seed=config.seed,
        )
        for agent in agents
    ]
    header = ["Inertial Stabilization", "----------------------", format_inertial_summary(trajectories), ""]
    details = [describe_trajectory(traj, prefix="  ") for traj in trajectories]
    return [*header, *details]


def parse_args() -> ShowcaseConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=80, help="number of steps to simulate")
    parser.add_argument("--seed", type=int, default=11, help="random seed for reproducibility")
    parser.add_argument("--scalar-state", type=float, default=3.0, help="initial scalar state u_0")
    parser.add_argument(
        "--inertial-position",
        type=float,
        default=2.5,
        help="initial position for the inertial plant",
    )
    parser.add_argument(
        "--inertial-velocity",
        type=float,
        default=0.0,
        help="initial velocity for the inertial plant",
    )
    args = parser.parse_args()
    return ShowcaseConfig(
        horizon=args.horizon,
        seed=args.seed,
        initial_scalar_state=args.scalar_state,
        initial_inertial_position=args.inertial_position,
        initial_inertial_velocity=args.inertial_velocity,
    )


def main() -> None:
    config = parse_args()
    sections = [
        "Generalized Adaptive Systems Showcase",
        "====================================",
    ]
    sections.extend(run_scalar_demo(config))
    sections.append("")
    sections.extend(run_inertial_demo(config))
    print("\n".join(sections))


if __name__ == "__main__":
    main()

