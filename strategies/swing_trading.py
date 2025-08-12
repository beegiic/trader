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
    tp_R: List[float]            # e.g., [1.0, 2.0]
    trail_type: str              # "ATR"
    trail_mult: float
    ttl_sec: int
    rationale: str

def build_swing_candidates(state: dict, last_price: float) -> List[Candidate]:
    """
    Build swing trading candidates using deterministic breakout/continuation logic.
    
    Logic:
    - Use only when volatility_regime == "high" and 1h trend agrees with 5m/15m
    - LONG breakout if 5m breakout_pending true, 1h trend >= 1, and 1m mom>=0
    - Entry = market or limit slightly above last_price
    - SL = last swing low or 1.2*ATR_5m, TPs 1R/2R, trail ATR 0.8 after +1R
    - SHORT symmetric
    """
    sym = state["symbol"]
    t1 = state["tfs"]["1m"]
    t5 = state["tfs"].get("5m", None)
    t15 = state["tfs"].get("15m", None)
    t1h = state["tfs"].get("1h", None)
    vol_regime = state.get("volatility_regime", "medium")
    
    out = []
    
    # Only trade breakouts in high volatility
    if vol_regime != "high":
        return out
    
    # Need at least 5m and 1h data
    if not t5 or not t1h:
        return out
    
    # LONG breakout conditions
    if (t5["breakout_pending"] and 
        t1h["trend"] >= 1 and  # 1h trend UP or STRONG_UP
        t1["mom"] >= 0):  # 1m momentum positive or neutral
        
        # Entry slightly above current price for breakout confirmation
        entry_price = last_price * 1.0001  # +1 bp above
        
        # Stop loss: last 5m swing low or 1.2*ATR_5m below entry
        atr_stop_distance = t5["atr_pct"] / 100 * 1.2 * entry_price
        swing_low_stop = last_price * (1 - 0.008)  # approximate 5m swing low
        stop = max(swing_low_stop, entry_price - atr_stop_distance)
        
        out.append(Candidate(
            symbol=sym,
            side=Side.LONG,
            entry_type="limit",
            entry_price=entry_price,
            stop_price=stop,
            tp_R=[1.0, 2.0],  # TP1 at +1.0R, TP2 at +2.0R
            trail_type="ATR",
            trail_mult=0.8,
            ttl_sec=120,
            rationale=f"5m breakout pending, 1h trend {t1h['trend']}, 1m mom {t1['mom']}"
        ))
    
    # SHORT breakout conditions (symmetric)
    if (t5["breakout_pending"] and 
        t1h["trend"] <= -1 and  # 1h trend DOWN or STRONG_DOWN
        t1["mom"] <= 0):  # 1m momentum negative or neutral
        
        # Entry slightly below current price for breakout confirmation
        entry_price = last_price * 0.9999  # -1 bp below
        
        # Stop loss: last 5m swing high or 1.2*ATR_5m above entry
        atr_stop_distance = t5["atr_pct"] / 100 * 1.2 * entry_price
        swing_high_stop = last_price * (1 + 0.008)  # approximate 5m swing high
        stop = min(swing_high_stop, entry_price + atr_stop_distance)
        
        out.append(Candidate(
            symbol=sym,
            side=Side.SHORT,
            entry_type="limit",
            entry_price=entry_price,
            stop_price=stop,
            tp_R=[1.0, 2.0],  # TP1 at +1.0R, TP2 at +2.0R
            trail_type="ATR",
            trail_mult=0.8,
            ttl_sec=120,
            rationale=f"5m breakout pending, 1h trend {t1h['trend']}, 1m mom {t1['mom']}"
        ))
    
    return out