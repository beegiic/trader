from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np

# Basic technical analysis functions
def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    alpha = 2 / (period + 1)
    ema_values = np.zeros_like(prices)
    ema_values[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i-1]
    
    return ema_values

def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))
    
    # Initial values
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # Calculate smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values

def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range"""
    high_low = highs - lows
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    
    # Pad arrays to same length
    high_close = np.concatenate([[high_low[0]], high_close])
    low_close = np.concatenate([[high_low[0]], low_close])
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate ATR using EMA
    atr_values = np.zeros_like(true_range)
    atr_values[0] = true_range[0]
    
    for i in range(1, len(true_range)):
        atr_values[i] = ((period - 1) * atr_values[i-1] + true_range[i]) / period
    
    return atr_values

def bollinger(closes: np.ndarray, period: int = 20, multiplier: float = 2.0) -> tuple:
    """Calculate Bollinger Bands"""
    sma = np.convolve(closes, np.ones(period), 'valid') / period
    
    # Pad SMA to match original length
    sma_padded = np.concatenate([np.full(period-1, sma[0]), sma])
    
    # Calculate standard deviation
    std_dev = np.zeros_like(closes)
    for i in range(period-1, len(closes)):
        std_dev[i] = np.std(closes[i-period+1:i+1])
    
    # Fill early values
    for i in range(period-1):
        std_dev[i] = std_dev[period-1]
    
    upper_band = sma_padded + (multiplier * std_dev)
    lower_band = sma_padded - (multiplier * std_dev)
    
    return sma_padded, upper_band, lower_band

TF = Literal["1m","5m","15m","1h"]

@dataclass
class TFOut:
    trend: int
    mom: int
    atr_pct: float
    rsi: int
    bb_pos: float
    sr_top_dist_bp: int
    sr_bot_dist_bp: int
    imb5: float
    breakout_pending: bool

class FeatureEngine:
    def __init__(self, tick_size: float):
        self.tick = tick_size

    def _trend(self, closes, fast=20, slow=100):
        e1, e2 = ema(closes, fast), ema(closes, slow)
        if e1[-1] > e2[-1] * 1.003 and e1[-1] > e1[-2]: return 2
        if e1[-1] > e2[-1]: return 1
        if e1[-1] < e2[-1] * 0.997 and e1[-1] < e1[-2]: return -2
        if e1[-1] < e2[-1]: return -1
        return 0

    def _mom(self, closes, look=10):
        diff = closes[-1] - closes[-look]
        return 1 if diff > 0 else (-1 if diff < 0 else 0)

    def _bb_pos(self, closes, period=20, mult=2.0):
        mid, upper, lower = bollinger(closes, period, mult)
        c = closes[-1]
        return float(np.clip((c - lower[-1])/max(upper[-1] - lower[-1], 1e-9), 0, 1))

    def _atr_pct(self, highs, lows, closes, period=14):
        a = atr(highs, lows, closes, period)
        return float(np.clip(100 * a[-1] / closes[-1], 0, 5))

    def _sr_dists_bp(self, closes):
        # naive SR via recent swing highs/lows
        window = closes[-300:] if len(closes) >= 300 else closes
        peak = np.max(window)
        trough = np.min(window)
        last = closes[-1]
        top_bp = int(10000 * max(peak - last, 0) / last)
        bot_bp = int(10000 * max(last - trough, 0) / last)
        return top_bp, bot_bp

    def _imbalance(self, bids_vol5, asks_vol5):
        tot = bids_vol5 + asks_vol5
        return float(0.5 if tot == 0 else np.clip(bids_vol5 / tot, 0, 1))

    def build_tf(self, highs, lows, closes, bids_vol5, asks_vol5) -> TFOut:
        return TFOut(
            trend=self._trend(closes),
            mom=self._mom(closes),
            atr_pct=self._atr_pct(highs, lows, closes),
            rsi=int(np.clip(rsi(closes, 14)[-1], 0, 100)),
            bb_pos=self._bb_pos(closes),
            sr_top_dist_bp=self._sr_dists_bp(closes)[0],
            sr_bot_dist_bp=self._sr_dists_bp(closes)[1],
            imb5=self._imbalance(bids_vol5, asks_vol5),
            breakout_pending=bool(closes[-1] > np.max(closes[-50:-1])*0.999) if len(closes) >= 50 else False
        )

    def build_state(self, symbol: str, tf_data: Dict[TF, dict], fees_bps:int, spread_bp:float, vol_regime:str, funding_bps:int=0, oi_5m_chg_pct:float=0.0) -> dict:
        # tf_data: { "1m": {highs,lows,closes,bids_vol5,asks_vol5}, ... }
        tfs = {}
        for tf, d in tf_data.items():
            tfs[tf] = self.build_tf(d["highs"], d["lows"], d["closes"], d["bids_vol5"], d["asks_vol5"]).__dict__
        return {
            "symbol": symbol,
            "tfs": tfs,
            "fees": {"taker_bps": fees_bps, "maker_bps": max(1, fees_bps//5)},
            "spread_bp": float(spread_bp),
            "volatility_regime": vol_regime,
            "funding_bps": int(funding_bps),
            "oi_5m_chg_pct": float(oi_5m_chg_pct)
        }