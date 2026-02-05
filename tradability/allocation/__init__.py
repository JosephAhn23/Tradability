"""
Level 4: Capital allocation under uncertainty.

Policy-based allocator: sizes and throttles strategies using feasibility bounds,
regime confidence, and model uncertainty. Prioritizes survivability over return maximization.
"""

from .config import load_config, AllocationConfig, StrategySpec
from .inputs import StrategyInputs, load_inputs, load_inputs_from_feasibility
from .uncertainty import uncertainty_penalty_factor, feasible_capacity_share
from .policies import apply_shutdown_rules, apply_throttles, ThrottleRecord, ShutdownRecord
from .allocator import compute_allocation, AllocationResult
from .stress import run_stress_tests, stress_2x_estimation_error, stress_correlation_one, stress_liquidity_shock
from .report import write_report
from .run import run_allocation
from .ww4_state import SurvivalState, WW4StateMachine, compute_ww4_state
from .ww4_consensus import consensus_regime, check_sensor_poisoning, ConsensusResult
from .ww4_tokens import RiskTokenBudget, compute_budget, tokens_for_state
from .ww4 import (
    WW4Context,
    WW4AllocationResult,
    compute_allocation_ww4,
    build_ww4_context,
)
from .ww5_state import CosmicSurvivalState, WW5State, compute_ww5_state
from .ww5_irreversibility import Reversibility, classify_allocation_action, ActionClassification
from .ww5_confidence_decay import decayed_confidence, optionality_score, ConfidenceDecayState
from .ww5 import (
    WW5Context,
    WW5AllocationResult,
    compute_allocation_ww5,
    build_ww5_context,
)
from .ww5_bluff import (
    BluffAudit,
    compute_bluff_risk_score,
    build_bluff_audit,
    check_confidence_monotonicity,
    check_silence_over_certainty,
    check_assumptions_disclosed,
)
from .ww5_evidence import EvidenceLedger, EvidenceEvent, EvidenceType, max_gross_from_hazard_and_bluff
from .ww5_unknown import compute_unknown_conditions, UnknownConditions

__all__ = [
    "load_config",
    "AllocationConfig",
    "StrategySpec",
    "StrategyInputs",
    "load_inputs",
    "load_inputs_from_feasibility",
    "uncertainty_penalty_factor",
    "feasible_capacity_share",
    "apply_shutdown_rules",
    "apply_throttles",
    "ThrottleRecord",
    "ShutdownRecord",
    "compute_allocation",
    "AllocationResult",
    "run_stress_tests",
    "stress_2x_estimation_error",
    "stress_correlation_one",
    "stress_liquidity_shock",
    "write_report",
    "run_allocation",
    "SurvivalState",
    "WW4StateMachine",
    "compute_ww4_state",
    "consensus_regime",
    "check_sensor_poisoning",
    "ConsensusResult",
    "RiskTokenBudget",
    "compute_budget",
    "tokens_for_state",
    "WW4Context",
    "WW4AllocationResult",
    "compute_allocation_ww4",
    "build_ww4_context",
    "CosmicSurvivalState",
    "WW5State",
    "compute_ww5_state",
    "Reversibility",
    "classify_allocation_action",
    "ActionClassification",
    "decayed_confidence",
    "optionality_score",
    "ConfidenceDecayState",
    "WW5Context",
    "WW5AllocationResult",
    "compute_allocation_ww5",
    "build_ww5_context",
    "BluffAudit",
    "compute_bluff_risk_score",
    "build_bluff_audit",
    "check_confidence_monotonicity",
    "check_silence_over_certainty",
    "check_assumptions_disclosed",
    "EvidenceLedger",
    "EvidenceEvent",
    "EvidenceType",
    "max_gross_from_hazard_and_bluff",
    "compute_unknown_conditions",
    "UnknownConditions",
]
