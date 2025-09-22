"""Text-based visualization helpers for the GAS demo."""

from __future__ import annotations

from typing import Iterable

from .simulation import Trajectory


def format_summary_table(trajectories: Iterable[Trajectory]) -> str:
    """Return a formatted table comparing multiple trajectories."""

    headers = ["Agent", "Final J", "Avg J", "Action change rate"]
    rows = [headers]
    for traj in trajectories:
        rows.append(
            [
                traj.agent,
                f"{traj.final_cost:8.5f}",
                f"{traj.average_cost:8.5f}",
                f"{traj.action_change_rate:6.3f}",
            ]
        )
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
    separator = "-+-".join("-" * width for width in col_widths)
    return "\n".join([formatted_rows[0], separator, *formatted_rows[1:]])


def describe_trajectory(traj: Trajectory, *, prefix: str = "") -> str:
    """Return a line-by-line description of a trajectory's evolution."""

    final_state = traj.final_state
    if isinstance(final_state, float):
        state_repr = f"{final_state:.4f}"
    else:
        state_repr = repr(final_state)
    lines = [
        f"{prefix}{traj.agent}: final state = {state_repr}, final cost = {traj.final_cost:.6f}",
        f"{prefix}Action change rate: {traj.action_change_rate:.3f} (Inflection responsiveness)",
        f"{prefix}First 5 costs: {', '.join(f'{c:.5f}' for c in traj.costs()[:5])}",
    ]
    return "\n".join(lines)
