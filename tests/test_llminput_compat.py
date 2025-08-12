from datetime import datetime, timezone
from utils.pydantic_models import LLMInput

def test_llminput_ts_coercion():
    li = LLMInput(symbol="ETHUSDT", last_trade_ts=datetime(2025,8,12,14,45, tzinfo=timezone.utc))
    j = li.model_dump()
    assert isinstance(j["last_trade_ts"], str)
    assert j["last_trade_ts"].startswith("2025-08-12T14:45")

def test_llminput_extra_fields():
    """Test that LLMInput allows extra fields without failing."""
    li = LLMInput(
        symbol="BTCUSDT",
        unknown_field="should_be_allowed",
        another_extra=123
    )
    j = li.model_dump()
    assert j["symbol"] == "BTCUSDT"
    assert "unknown_field" in j
    assert j["unknown_field"] == "should_be_allowed"

def test_llminput_datetime_string_passthrough():
    """Test that string timestamps pass through unchanged."""
    ts_str = "2025-08-12T14:45:00"
    li = LLMInput(symbol="BTCUSDT", last_trade_ts=ts_str)
    j = li.model_dump()
    assert j["last_trade_ts"] == ts_str

if __name__ == "__main__":
    test_llminput_ts_coercion()
    test_llminput_extra_fields()
    test_llminput_datetime_string_passthrough()
    print("All LLMInput compatibility tests passed!")