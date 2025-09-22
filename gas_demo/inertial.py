"""High-impact inertial control showcase for GAS primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
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
from .simulation import AgentComponents, Trajectory, TrajectoryStep


# ---------------------------------------------------------------------------
# State and relation definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InertialState:
    """Second-order plant state with position and velocity."""

    position: float
    velocity: float


@dataclass(frozen=True)
class InertialObservation:
    """Observation exposing both position and velocity."""

    position: float
    velocity: float


@dataclass
class InertialRelation(Relation):
    """Discrete-time damped mass-spring system."""

    dt: float = 0.08
    damping: float = 0.35
    stiffness: float = 1.6
    control_gain: float = 1.0
    noise_std: float = 0.02

    def expected(self, state: InertialState, action: float) -> InertialState:
        accel = (
            -self.stiffness * state.position
            - self.damping * state.velocity
            + self.control_gain * action
        )
        next_velocity = state.velocity + self.dt * accel
        next_position = state.position + self.dt * next_velocity
        return InertialState(position=next_position, velocity=next_velocity)

    def sample(self, state: InertialState, action: float, rng: random.Random) -> InertialState:
        deterministic = self.expected(state, action)
        if self.noise_std <= 0:
            return deterministic
        noise = rng.gauss(0.0, self.noise_std)
        return InertialState(
            position=deterministic.position + self.dt * noise,
            velocity=deterministic.velocity + noise,
        )

    def jacobian_action(self, state: InertialState, action: float) -> float:
        del state, action
        return (self.dt ** 2) * self.control_gain


# ---------------------------------------------------------------------------
# Distinctions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionOnlyDistinction:
    """Expose only the position component (Level-0 view)."""

    def observe(self, state: InertialState, memory_state: MemoryState) -> float:
        del memory_state
        return state.position


@dataclass(frozen=True)
class FullInertialDistinction:
    """Expose both position and velocity (Level-1 view)."""

    def observe(self, state: InertialState, memory_state: MemoryState) -> InertialObservation:
        del memory_state
        return InertialObservation(position=state.position, velocity=state.velocity)


# ---------------------------------------------------------------------------
# Objective tailored to inertial control
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InertialEnergyObjective(Objective):
    """Lyapunov function combining position error and velocity energy."""

    position_weight: float = 1.0
    velocity_weight: float = 0.35

    def value(self, state) -> float:
        if isinstance(state, InertialState):
            pos = state.position
            vel = state.velocity
        else:
            pos = float(state)
            vel = 0.0
        return 0.5 * (self.position_weight * pos * pos + self.velocity_weight * vel * vel)

    def gradient(self, state) -> float:
        if isinstance(state, InertialState):
            pos = state.position
        else:
            pos = float(state)
        return self.position_weight * pos

    def velocity_penalty(self, state: InertialState) -> float:
        return self.velocity_weight * state.velocity


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@dataclass
class InertialMemory(Memory):
    """Track kinetic energy and responsiveness statistics."""

    beta: float = 0.1

    def init_state(self) -> MemoryState:
        return MemoryState()

    def update(
        self,
        memory_state: MemoryState,
        *,
        observation: object,
        action: float,
        cost: float,
        next_state: InertialState,
        objective: Objective,
    ) -> MemoryState:
        return MemoryState(
            step=memory_state.step + 1,
            avg_cost=memory_state.avg_cost + self.beta * (cost - memory_state.avg_cost),
            change_count=
            memory_state.change_count
            + (
                1
                if memory_state.prev_action is not None
                and abs(memory_state.prev_action - action) > 1e-9
                else 0
            ),
            prev_action=action,
            prev_state=next_state,
            velocity=getattr(next_state, "velocity", getattr(observation, "velocity", memory_state.velocity)),
            ema_gradient=
            (1 - self.beta) * memory_state.ema_gradient
            + self.beta * objective.gradient(next_state),
        )


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


@dataclass
class InertialGradientController(Transformation):
    """Naive gradient descent that ignores inertia."""

    base_step: float = 0.6

    def act(
        self,
        observation: object,
        memory_state: MemoryState,
        objective: Objective,
        relation: Relation,
    ) -> float:
        del relation
        del memory_state
        position = float(observation)
        gradient = objective.gradient(position)
        action = -self.base_step * gradient
        return max(min(action, 4.0), -4.0)


@dataclass
class InertialLookaheadController(Transformation):
    """Level-1 controller solving the inertial control law analytically."""

    position_gain: float = 1.6
    velocity_gain: float = 1.2
    max_action: float = 6.0

    def act(
        self,
        observation: object,
        memory_state: MemoryState,
        objective: Objective,
        relation: Relation,
    ) -> float:
        if not isinstance(observation, InertialObservation):
            raise TypeError("InertialLookaheadController expects InertialObservation")
        if not isinstance(relation, InertialRelation):
            raise TypeError("InertialLookaheadController requires InertialRelation")
        del memory_state
        pos = observation.position
        vel = observation.velocity
        target_velocity = -self.position_gain * pos - self.velocity_gain * vel
        velocity_change = (target_velocity - vel) / relation.dt
        required_accel = velocity_change + relation.stiffness * pos + relation.damping * vel
        action = required_accel / relation.control_gain
        if isinstance(objective, InertialEnergyObjective):
            grad = objective.gradient(InertialState(position=pos, velocity=vel))
            action -= 0.1 * grad
        action = max(min(action, self.max_action), -self.max_action)
        return action


# ---------------------------------------------------------------------------
# Rollout utilities dedicated to the inertial showcase
# ---------------------------------------------------------------------------


def run_inertial_agent(
    *,
    components: AgentComponents,
    relation: InertialRelation,
    objective: Objective,
    initial_state: InertialState,
    horizon: int,
    seed: int = 0,
) -> Trajectory:
    rng = random.Random(seed)
    state = initial_state
    memory_state = components.memory.init_state()
    steps: list[TrajectoryStep] = []
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
        state = next_state  # type: ignore[assignment]
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


def overshoot(traj: Trajectory) -> float:
    """Compute the maximum absolute position encountered."""

    max_dev = 0.0
    for step in traj.steps:
        state = step.state
        if isinstance(state, InertialState):
            if step.time > 0:
                max_dev = max(max_dev, abs(state.position))
    final_state = traj.final_state
    if isinstance(final_state, InertialState):
        max_dev = max(max_dev, abs(final_state.position))
    return max_dev


def settling_time(traj: Trajectory, tolerance: float = 0.05) -> int:
    """Return the first step index at which the position stays within tolerance."""

    positions: list[Tuple[int, float]] = []
    for step in traj.steps:
        state = step.state
        if isinstance(state, InertialState):
            positions.append((step.time, abs(state.position)))
    last_time = traj.steps[-1].time if traj.steps else 0
    final = traj.final_state
    if isinstance(final, InertialState):
        positions.append((last_time + 1, abs(final.position)))
    window = 5
    for idx in range(len(positions)):
        window_vals = [pos for _, pos in positions[idx : idx + window]]
        if len(window_vals) < window:
            break
        if all(val <= tolerance for val in window_vals):
            return positions[idx][0]
    return last_time


def energy_threshold_time(traj: Trajectory, fraction: float = 0.1) -> int:
    """Return the first step index where cost <= fraction of the initial cost."""

    if not traj.steps:
        return 0
    baseline = traj.steps[0].cost
    target = baseline * fraction
    for step in traj.steps:
        if step.cost <= target:
            return step.time
    return traj.steps[-1].time


def format_inertial_summary(trajectories: Iterable[Trajectory]) -> str:
    """Produce a table with overshoot and settling-time comparisons."""

    headers = ["Agent", "Final J", "Overshoot", "Settling", "10% energy"]
    rows = [headers]
    for traj in trajectories:
        rows.append(
            [
                traj.agent,
                f"{traj.final_cost:8.5f}",
                f"{overshoot(traj):7.4f}",
                f"{settling_time(traj):5d}",
                f"{energy_threshold_time(traj):5d}",
            ]
        )
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    formatted = []
    for row in rows:
        formatted.append(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    separator = "-+-".join("-" * width for width in col_widths)
    return "\n".join([formatted[0], separator, *formatted[1:]])

