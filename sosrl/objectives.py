"""Shared objective accounting for GP and GP-aligned baselines."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ARCHITECTURE_CHANGE_WEIGHT = 0.01
DEFAULT_PEAK_BUDGET_PENALTY = 20.0


@dataclass(frozen=True)
class GPCostBreakdown:
    """Cost-side terms used by the direct-GP episode objective."""

    final_net_cost: float
    peak_net_cost: float
    budget: float
    architecture_changes: int
    architecture_change_weight: float
    peak_budget_penalty_weight: float
    net_cost_ratio: float
    peak_budget_excess_ratio: float
    architecture_change_penalty_score: float
    peak_budget_penalty_score: float
    gp_cost_score: float
    architecture_change_penalty: float
    peak_budget_penalty: float
    effective_cost: float


def gp_cost_breakdown(
    *,
    final_net_cost: float,
    peak_net_cost: float,
    budget: float,
    architecture_changes: int,
    architecture_change_weight: float = DEFAULT_ARCHITECTURE_CHANGE_WEIGHT,
    peak_budget_penalty_weight: float = DEFAULT_PEAK_BUDGET_PENALTY,
) -> GPCostBreakdown:
    """Return the GP cost terms in normalized and budget-scaled units."""
    normalized_budget = max(float(budget), 1.0)
    changes = int(architecture_changes)
    change_weight = float(architecture_change_weight)
    peak_weight = float(peak_budget_penalty_weight)
    if changes < 0:
        raise ValueError("architecture_changes cannot be negative.")
    if change_weight < 0.0 or peak_weight < 0.0:
        raise ValueError("GP cost weights cannot be negative.")

    final_cost = float(final_net_cost)
    peak_cost = float(peak_net_cost)
    net_ratio = final_cost / normalized_budget
    excess_ratio = max(peak_cost / normalized_budget - 1.0, 0.0)
    change_score = change_weight * changes
    peak_score = peak_weight * excess_ratio**2
    score = net_ratio + change_score + peak_score
    return GPCostBreakdown(
        final_net_cost=final_cost,
        peak_net_cost=peak_cost,
        budget=normalized_budget,
        architecture_changes=changes,
        architecture_change_weight=change_weight,
        peak_budget_penalty_weight=peak_weight,
        net_cost_ratio=float(net_ratio),
        peak_budget_excess_ratio=float(excess_ratio),
        architecture_change_penalty_score=float(change_score),
        peak_budget_penalty_score=float(peak_score),
        gp_cost_score=float(score),
        architecture_change_penalty=float(change_score * normalized_budget),
        peak_budget_penalty=float(peak_score * normalized_budget),
        effective_cost=float(score * normalized_budget),
    )
