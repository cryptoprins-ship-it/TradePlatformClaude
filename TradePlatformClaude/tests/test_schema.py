import pytest
from pydantic import ValidationError

from agent.llm.schema import Decision, hold


def test_valid_decision_parses():
    d = Decision.model_validate({
        "action": "OPEN_LONG", "conviction": 0.8, "size_pct": 0.5,
        "leverage": 3, "stop_loss_pct": 0.02, "take_profit_pct": 0.06,
        "reasoning": "uptrend",
    })
    assert d.action == "OPEN_LONG"
    assert d.leverage == 3


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate({
            "action": "HOLD", "conviction": 0.0, "size_pct": 0.0,
            "leverage": 1, "stop_loss_pct": 0.0, "reasoning": "x",
            "sneaky": "injected",
        })


def test_conviction_out_of_range_rejected():
    with pytest.raises(ValidationError):
        Decision.model_validate({
            "action": "HOLD", "conviction": 1.5, "size_pct": 0.0,
            "leverage": 1, "stop_loss_pct": 0.0, "reasoning": "x",
        })


def test_hold_helper_builds_valid_hold():
    d = hold("data fetch failed")
    assert d.action == "HOLD"
    assert d.size_pct == 0.0
    assert d.reasoning == "data fetch failed"
