"""
WW5: Irreversibility detection. Under high uncertainty, irreversible actions are forbidden.
One-way doors (large upside, irreversible, fragile assumptions) must be rejected.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .inputs import StrategyInputs


class Reversibility(Enum):
    REVERSIBLE = "reversible"
    PARTIALLY_REVERSIBLE = "partially_reversible"
    IRREVERSIBLE = "irreversible"


@dataclass
class ActionClassification:
    """Classification of an allocation action for one strategy."""
    strategy_id: str
    reversibility: Reversibility
    is_one_way_door: bool
    reason: str = ""


def classify_allocation_action(
    inp: StrategyInputs,
    proposed_weight_delta: float,
    uncertainty_score: float,
    liquidity_ratio: float,
) -> ActionClassification:
    """
    Classify: increasing exposure in illiquid strategy = partially or irreversible.
    One-way door: large upside (high edge), irreversible or partially, fragile (high uncertainty / single assumption).
    """
    # Illiquid + increasing = partially reversible or irreversible
    liquidity_ratio = liquidity_ratio if liquidity_ratio is not None and liquidity_ratio > 0 else 0.0
    if proposed_weight_delta <= 0:
        rev = Reversibility.REVERSIBLE
        one_way = False
        reason = "reducing_or_flat"
    elif liquidity_ratio >= 0.5:
        rev = Reversibility.REVERSIBLE
        one_way = False
        reason = "liquid"
    elif liquidity_ratio >= 0.2:
        rev = Reversibility.PARTIALLY_REVERSIBLE
        one_way = False
        reason = "partially_liquid"
    else:
        rev = Reversibility.IRREVERSIBLE
        one_way = uncertainty_score > 0.5 and inp.net_edge_bps > 80
        reason = "illiquid"

    if one_way:
        reason = "one_way_door: high_edge_illiquid_fragile"

    return ActionClassification(
        strategy_id=inp.strategy_id,
        reversibility=rev,
        is_one_way_door=one_way,
        reason=reason,
    )


def block_irreversible_under_uncertainty(
    classifications: List[ActionClassification],
    uncertainty_threshold: float = 0.6,
) -> List[str]:
    """Return strategy_ids that must be blocked (irreversible or one-way door under high uncertainty)."""
    return [
        c.strategy_id
        for c in classifications
        if c.reversibility == Reversibility.IRREVERSIBLE
        or c.is_one_way_door
    ]
