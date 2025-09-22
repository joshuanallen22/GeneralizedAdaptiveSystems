"""Core primitives and derivative operators for the GAS demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Any
import math
import random

from .objectives import Objective


# ---------------------------------------------------------------------------
# Distinction primitives
# ---------------------------------------------------------------------------


class Distinction(Protocol):
    """Extract observable structure from the underlying state."""

    def observe(self, state: Any, memory_state: "MemoryState") -> object:
        """Return a measurable observation derived from the current state."""


@dataclass(frozen=True)
class IdentityDistinction:
    """Return the raw state as the only distinction."""

    def observe(self, state: Any, memory_state: "MemoryState") -> float:
        return state


@dataclass(frozen=True)
class DifferentialObservation:
    """Distinction that includes a finite difference velocity estimate."""

    state: float
    velocity: float


@dataclass(frozen=True)
class DifferentialDistinction:
    """Augment the raw state with a first-order velocity estimate."""

    decay: float = 0.2

    def observe(self, state: Any, memory_state: "MemoryState") -> DifferentialObservation:
        if memory_state.prev_state is None:
            velocity = 0.0
        else:
            try:
                raw_velocity = state - memory_state.prev_state  # type: ignore[operator]
            except TypeError:
                raw_velocity = 0.0
            velocity = (1 - self.decay) * memory_state.velocity + self.decay * float(raw_velocity)
        return DifferentialObservation(state=state, velocity=velocity)


# ---------------------------------------------------------------------------
# Relation primitives
# ---------------------------------------------------------------------------


class Relation(Protocol):
    """Map state/action pairs to distributions over next states."""

    def sample(self, state: Any, action: float, rng: random.Random) -> Any:
        """Sample the next state given the current state and action."""

    def expected(self, state: Any, action: float) -> Any:
        r"""Return :math:`\mathbb E[u_{t+1} \mid u_t = state, a_t = action]`."""

    def jacobian_action(self, state: Any, action: float) -> float:
        r"""Return :math:`\partial R / \partial a` evaluated at the given point."""


@dataclass(frozen=True)
class LinearRelation:
    """Controlled linear dynamics with optional Gaussian noise."""

    dt: float = 0.1
    relaxation: float = 0.4
    noise_std: float = 0.02

    def sample(self, state: float, action: float, rng: random.Random) -> float:
        mean = self.expected(state, action)
        if self.noise_std <= 0:
            return mean
        return rng.gauss(mean, self.noise_std)

    def expected(self, state: float, action: float) -> float:
        drift = -self.relaxation * state + action
        return state + self.dt * drift

    def jacobian_action(self, state: float, action: float) -> float:
        del state, action  # Unused; jacobian is constant for linear dynamics.
        return self.dt


# ---------------------------------------------------------------------------
# Memory primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryState:
    """Internal information state maintained by the agent."""

    step: int = 0
    avg_cost: float = 0.0
    change_count: int = 0
    prev_action: Optional[float] = None
    prev_state: Optional[object] = None
    velocity: float = 0.0
    ema_gradient: float = 0.0

    def action_change_rate(self) -> float:
        if self.step == 0:
            return 0.0
        return self.change_count / self.step


class Memory(Protocol):
    """Protocol describing how memory is initialized and updated."""

    def init_state(self) -> MemoryState:
        ...

    def update(
        self,
        memory_state: MemoryState,
        *,
        observation: object,
        action: float,
        cost: float,
        next_state: Any,
        objective: Objective,
    ) -> MemoryState:
        ...


@dataclass
class RobbinsMonroMemory:
    """Track running statistics required by the Level-0 drift analysis."""

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
        next_state: Any,
        objective: Objective,
    ) -> MemoryState:
        step = memory_state.step + 1
        avg_cost = memory_state.avg_cost + self.beta * (cost - memory_state.avg_cost)
        prev_action = memory_state.prev_action
        change_count = memory_state.change_count
        if prev_action is not None and abs(prev_action - action) > 1e-9:
            change_count += 1
        gradient = objective.gradient(next_state)
        ema_gradient = (1 - self.beta) * memory_state.ema_gradient + self.beta * gradient
        velocity = 0.0
        if isinstance(observation, DifferentialObservation):
            velocity = observation.velocity
        elif hasattr(observation, "velocity"):
            try:
                velocity = float(getattr(observation, "velocity"))
            except (TypeError, ValueError):
                velocity = memory_state.velocity
        return MemoryState(
            step=step,
            avg_cost=avg_cost,
            change_count=change_count,
            prev_action=action,
            prev_state=next_state,
            velocity=velocity,
            ema_gradient=ema_gradient,
        )


@dataclass
class DifferentialMemory(RobbinsMonroMemory):
    """Memory that emphasises velocity estimates for Level-1 operators."""

    momentum: float = 0.6

    def update(
        self,
        memory_state: MemoryState,
        *,
        observation: object,
        action: float,
        cost: float,
        next_state: float,
        objective: Objective,
    ) -> MemoryState:
        base_state = super().update(
            memory_state,
            observation=observation,
            action=action,
            cost=cost,
            next_state=next_state,
            objective=objective,
        )
        velocity = base_state.velocity
        if isinstance(observation, DifferentialObservation):
            velocity = (1 - self.momentum) * observation.velocity + self.momentum * base_state.velocity
        return MemoryState(
            step=base_state.step,
            avg_cost=base_state.avg_cost,
            change_count=base_state.change_count,
            prev_action=base_state.prev_action,
            prev_state=base_state.prev_state,
            velocity=velocity,
            ema_gradient=base_state.ema_gradient,
        )


# ---------------------------------------------------------------------------
# Transformation primitives
# ---------------------------------------------------------------------------


class Transformation(Protocol):
    """Map distinctions and memory to actions."""

    def act(
        self,
        observation: object,
        memory_state: MemoryState,
        objective: Objective,
        relation: Relation,
    ) -> float:
        ...


@dataclass
class GradientDescentTransformation:
    """Level-0 policy implementing a Robbins–Monro step schedule."""

    base_step: float = 0.6

    def act(
        self,
        observation: object,
        memory_state: MemoryState,
        objective: Objective,
        relation: Relation,
    ) -> float:
        del relation  # Not used in the Level-0 controller.
        state = float(observation)
        gradient = objective.gradient(state)
        step_scale = self.base_step / math.sqrt(1 + memory_state.step)
        return -step_scale * gradient


@dataclass
class DifferentialTransformation:
    """Level-1 controller that leverages first-order dynamics of the relation."""

    base_step: float = 0.6
    damping: float = 0.2

    def act(
        self,
        observation: object,
        memory_state: MemoryState,
        objective: Objective,
        relation: Relation,
    ) -> float:
        if not isinstance(observation, DifferentialObservation):
            raise TypeError("DifferentialTransformation expects DifferentialObservation")
        state = observation.state
        gradient = objective.gradient(state)
        step_scale = self.base_step / math.sqrt(1 + memory_state.step)
        desired_delta = -step_scale * gradient - self.damping * observation.velocity
        baseline_next = relation.expected(state, 0.0)
        derivative = relation.jacobian_action(state, 0.0)
        if abs(derivative) < 1e-8:
            return 0.0
        target_next = state + desired_delta
        action = (target_next - baseline_next) / derivative
        return action


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def simulate_step(
    state: float,
    *,
    distinction: Distinction,
    relation: Relation,
    transformation: Transformation,
    memory: Memory,
    memory_state: MemoryState,
    objective: Objective,
    rng: random.Random,
) -> tuple[float, MemoryState, float, float, object]:
    """Execute a single closed-loop step of the GAS process."""

    observation = distinction.observe(state, memory_state)
    action = transformation.act(observation, memory_state, objective, relation)
    next_state = relation.sample(state, action, rng)
    cost = objective.value(state)
    next_memory = memory.update(
        memory_state,
        observation=observation,
        action=action,
        cost=cost,
        next_state=next_state,
        objective=objective,
    )
    return next_state, next_memory, action, cost, observation
