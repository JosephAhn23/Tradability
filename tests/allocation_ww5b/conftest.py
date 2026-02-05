import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from tradability.allocation.config import AllocationConfig, StrategySpec
from tradability.allocation.ww5 import compute_allocation_ww5, build_ww5_context


@pytest.fixture
def base_config():
    return AllocationConfig(total_capital=1_000_000, strategies=[StrategySpec(name="A"), StrategySpec(name="B")],
                            max_weight_per_strategy=0.4, min_weight_threshold=0.02, min_regime_confidence=0.2)
