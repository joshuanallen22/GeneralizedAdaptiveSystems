"""Simulation utilities for composing GAS primitives into agents.""" 

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any
import random

from .objectives import Objective
from .primitives import (
    Distinction,
    Memory,
    MemoryState,
    Relation,
    Transformation,
    simulate_step,
)


@dataclass(frozen=True)
class AgentComponents:
    """Bundle of primitives that define a GAS agent."""

    name: str
    distinction: Distinction
    transformation: Transformation
    memory: Memory


@dataclass(frozen=True)
class TrajectoryStep:
    """Recorded information for a single simulation step."""

    time: int
    state: object
    action: float
    cost: float
    memory_state: MemoryState


@dataclass(frozen=True)
class Trajectory:
    """Full rollout for an agent over a fixed horizon."""

    agent: str
    steps: List[TrajectoryStep]
    final_state: object
    final_cost: float
    average_cost: float
    action_change_rate: float

    def costs(self) -> List[float]:
        return [step.cost for step in self.steps]


def run_agent(
    *,
    components: AgentComponents,
    relation: Relation,
    objective: Objective,
    initial_state: float,
    horizon: int,
    seed: int = 0,
) -> Trajectory:
    """Simulate the closed-loop system for a given agent."""

    rng = random.Random(seed)
    state = initial_state
    memory_state = components.memory.init_state()
    steps: List[TrajectoryStep] = []
    cumulative_cost = 0.0
    for t in range(horizon):
        next_state, memory_state, action, cost, _ = simulate_step(
            state,
            distinction=components.distinction,
            relation=relation,
            transformation=components.transformation,
            memory=components.memory,
            memory_state=memory_state,
            objective=objective,
            rng=rng,
        )
        cumulative_cost += cost
        steps.append(
            TrajectoryStep(
                time=t,
                state=state,
                action=action,
                cost=cost,
                memory_state=memory_state,
            )
        )
        state = next_state
    final_cost = objective.value(state)
    average_cost = cumulative_cost / horizon if horizon else 0.0
    return Trajectory(
        agent=components.name,
        steps=steps,
        final_state=state,
        final_cost=final_cost,
        average_cost=average_cost,
        action_change_rate=memory_state.action_change_rate(),
    )
