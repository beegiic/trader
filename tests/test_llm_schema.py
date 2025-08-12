import pytest
import json
from pydantic import ValidationError

from utils.pydantic_models import (
    DecisionInput, DecisionOutput, Action, Side, 
    TimeframeSnapshot, Trend, Fees
)

def test_decision_output_validation():
    """Test DecisionOutput validates correctly and fails closed on invalid fields."""
    
    # Valid decision output
    valid_data = {
        "action": "OPEN",
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry": {"type": "market", "price": None},
        "size_pct": 0.02,
        "leverage": 10,
        "stop": {"price": 61000.0, "reason": "ATR stop"},
        "tp": [{"price": 62000.0, "type": "R"}],
        "trail": {"type": "ATR", "mult": 0.8},
        "valid_for_sec": 90,
        "confidence": 0.75,
        "notes": "High probability setup"
    }
    
    # Should validate successfully
    decision = DecisionOutput(**valid_data)
    assert decision.action == Action.OPEN
    assert decision.side == Side.LONG
    assert decision.size_pct == 0.02
    
    # Test invalid action - should fail
    invalid_action = valid_data.copy()
    invalid_action["action"] = "INVALID_ACTION"
    
    with pytest.raises(ValidationError):
        DecisionOutput(**invalid_action)
    
    # Test invalid size_pct (too large) - should fail
    invalid_size = valid_data.copy()
    invalid_size["size_pct"] = 1.5  # > 1.0 maximum
    
    with pytest.raises(ValidationError):
        DecisionOutput(**invalid_size)
    
    # Test invalid leverage (too high) - should fail
    invalid_leverage = valid_data.copy()
    invalid_leverage["leverage"] = 200  # > 125 maximum
    
    with pytest.raises(ValidationError):
        DecisionOutput(**invalid_leverage)
    
    # Test invalid confidence - should fail  
    invalid_confidence = valid_data.copy()
    invalid_confidence["confidence"] = 1.5  # > 1.0 maximum
    
    with pytest.raises(ValidationError):
        DecisionOutput(**invalid_confidence)

def test_timeframe_snapshot_validation():
    """Test TimeframeSnapshot validation."""
    
    valid_tf = {
        "trend": 1,  # UP
        "mom": 1,
        "atr_pct": 0.5,
        "rsi": 45,
        "bb_pos": 0.3,
        "sr_top_dist_bp": 150,
        "sr_bot_dist_bp": 200,
        "imb5": 0.55,
        "breakout_pending": True
    }
    
    tf = TimeframeSnapshot(**valid_tf)
    assert tf.trend == Trend.UP
    assert tf.rsi == 45
    
    # Test invalid RSI - should fail
    invalid_rsi = valid_tf.copy()
    invalid_rsi["rsi"] = 150  # > 100 maximum
    
    with pytest.raises(ValidationError):
        TimeframeSnapshot(**invalid_rsi)
    
    # Test invalid ATR percentage - should fail
    invalid_atr = valid_tf.copy() 
    invalid_atr["atr_pct"] = 10.0  # > 5.0 maximum
    
    with pytest.raises(ValidationError):
        TimeframeSnapshot(**invalid_atr)

def test_decision_input_validation():
    """Test DecisionInput validation."""
    
    tf_data = {
        "trend": 1,
        "mom": 1, 
        "atr_pct": 0.5,
        "rsi": 45,
        "bb_pos": 0.3,
        "sr_top_dist_bp": 150,
        "sr_bot_dist_bp": 200,
        "imb5": 0.55,
        "breakout_pending": True
    }
    
    valid_input = {
        "decision_id": "test123",
        "symbol": "BTCUSDT",
        "tfs": {"1m": tf_data, "5m": tf_data},
        "fees": {"taker_bps": 5, "maker_bps": 1},
        "spread_bp": 1.5,
        "volatility_regime": "medium",
        "funding_bps": -10,
        "oi_5m_chg_pct": 2.5,
        "time": "2025-08-12T10:00:00Z"
    }
    
    di = DecisionInput(**valid_input)
    assert di.symbol == "BTCUSDT"
    assert di.volatility_regime == "medium"
    
    # Test invalid volatility regime - should fail
    invalid_vol = valid_input.copy()
    invalid_vol["volatility_regime"] = "extreme"  # Not in allowed values
    
    with pytest.raises(ValidationError):
        DecisionInput(**invalid_vol)

def test_json_schema_fuzzing():
    """Fuzz test with various invalid JSON structures."""
    
    # Missing required fields
    with pytest.raises(ValidationError):
        DecisionOutput(action="OPEN")  # Missing symbol
    
    # Wrong types
    with pytest.raises(ValidationError):
        DecisionOutput(
            action="OPEN",
            symbol="BTCUSDT", 
            size_pct="invalid_string"  # Should be float
        )
    
    # Out of range values
    with pytest.raises(ValidationError):
        DecisionOutput(
            action="OPEN",
            symbol="BTCUSDT",
            size_pct=-0.1  # Negative not allowed
        )

if __name__ == "__main__":
    test_decision_output_validation()
    test_timeframe_snapshot_validation() 
    test_decision_input_validation()
    test_json_schema_fuzzing()
    print("All schema validation tests passed!")