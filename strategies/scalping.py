from dataclasses import dataclass
from typing import List, Optional
from utils.pydantic_models import Side

@dataclass
class Candidate:
    symbol: str
    side: Side
    entry_type: str  # "market" or "limit"
    entry_price: Optional[float]  # None for market
    stop_price: float
    tp_R: List[float]            # e.g., [0.5, 1.0]
    trail_type: str              # "ATR"
    trail_mult: float
    ttl_sec: int
    rationale: str

def build_scalp_candidates(state: dict, last_price: float) -> List[Candidate]:
    """
    Build scalping candidates using deterministic mean-reversion rules.
    
    Logic:
    - Use on low/medium volatility for mean-reversion
    - Use on high volatility only if 1m+5m momentum both align for continuation
    - LONG scalp when 1m bb_pos <= 0.1, rsi<30, 5m not DOWN trend, sr_bot_dist_bp small (<300bp)
    - SHORT scalp when 1m bb_pos >= 0.9, rsi>70, 5m not UP trend, sr_top_dist_bp small (<300bp)
    """
    sym = state["symbol"]
    t1 = state["tfs"]["1m"]
    t5 = state["tfs"].get("5m", None)
    vol_regime = state.get("volatility_regime", "medium")
    
    out = []
    
    # Skip scalping in high volatility unless momentum aligns
    if vol_regime == "high":
        # Only scalp in high vol if 1m and 5m momentum align
        if t5 and (t1["mom"] * t5["mom"] <= 0):  # momentum doesn't align
            return out
    
    # LONG mean-reversion conditions
    if (t1["bb_pos"] <= 0.1 and 
        t1["rsi"] < 30 and 
        (t5 is None or t5["trend"] >= 0) and  # 5m not DOWN trend
        t1["sr_bot_dist_bp"] <= 300):  # close to support
        
        # Stop loss: min of recent swing low or entry - 0.6*ATR_1m
        atr_stop_distance = t1["atr_pct"] / 100 * 0.6 * last_price
        swing_low_stop = last_price * (1 - 0.006)  # approximate recent swing low
        stop = max(swing_low_stop, last_price - atr_stop_distance)
        
        out.append(Candidate(
            symbol=sym,
            side=Side.LONG,
            entry_type="market",
            entry_price=None,
            stop_price=stop,
            tp_R=[0.5, 1.0],  # TP1 at +0.5R, TP2 at +1.0R
            trail_type="ATR",
            trail_mult=0.8,
            ttl_sec=90,
            rationale=f"1m BB low ({t1['bb_pos']:.2f}) + RSI<30 ({t1['rsi']})"
        ))
    
    # SHORT mean-reversion conditions
    if (t1["bb_pos"] >= 0.9 and 
        t1["rsi"] > 70 and 
        (t5 is None or t5["trend"] <= 0) and  # 5m not UP trend
        t1["sr_top_dist_bp"] <= 300):  # close to resistance
        
        # Stop loss: max of recent swing high or entry + 0.6*ATR_1m
        atr_stop_distance = t1["atr_pct"] / 100 * 0.6 * last_price
        swing_high_stop = last_price * (1 + 0.006)  # approximate recent swing high
        stop = min(swing_high_stop, last_price + atr_stop_distance)
        
        out.append(Candidate(
            symbol=sym,
            side=Side.SHORT,
            entry_type="market",
            entry_price=None,
            stop_price=stop,
            tp_R=[0.5, 1.0],  # TP1 at +0.5R, TP2 at +1.0R
            trail_type="ATR",
            trail_mult=0.8,
            ttl_sec=90,
            rationale=f"1m BB high ({t1['bb_pos']:.2f}) + RSI>70 ({t1['rsi']})"
        ))
    
    return out