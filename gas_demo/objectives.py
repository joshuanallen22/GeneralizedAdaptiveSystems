"""Lyapunov-style objective functions for the GAS demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


class Objective(Protocol):
    """Protocol for Lyapunov-style objectives."""

    def value(self, state: Any) -> float:
        """Return :math:`J(u)` for the current state."""

    def gradient(self, state: Any) -> float:
        """Return :math:`\nabla J(u)` for the current state."""


@dataclass(frozen=True)
class QuadraticObjective:
    r"""Quadratic Lyapunov functional :math:`J(u) = 0.5 (u - u_\star)^2`."""

    target: float = 0.0

    def value(self, state) -> float:
        deviation = float(state) - self.target
        return 0.5 * deviation * deviation

    def gradient(self, state) -> float:
        return float(state) - self.target
