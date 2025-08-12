"""
LLM-Powered Technical Analysis Engine
Uses GPT-4 to analyze market data and generate trading signals
"""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import os
import time
import hashlib
import pandas as pd
# import pandas_ta as ta  # Temporarily disabled due to numpy compatibility

from utils.logging_config import LoggerMixin
from core.binance_client import BinanceClient
from core.llm_coordinator import LLMCoordinator
from utils.pydantic_models import (
    LLMInput as LLMInputModel,
    TechnicalIndicators as LLMTechIndicators,
    OHLCVData as LLMOHLCV,
    LLMOutput as LLMOutModel,
    DecisionAction as LLMDecisionAction,
)


class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values"""
    # Trend indicators
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    macd: float
    macd_signal: float
    macd_histogram: float
    
    # Momentum indicators
    rsi_14: float
    stoch_k: float
    stoch_d: float
    
    # Volatility indicators
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    atr: float
    
    # Volume indicators
    volume_sma: float
    volume_ratio: float
    
    # Price action
    current_price: float
    support_level: float
    resistance_level: float
    
    # Market structure
    higher_highs: bool
    higher_lows: bool
    lower_highs: bool
    lower_lows: bool


@dataclass
class TradingSignal:
    """Complete trading signal with analysis"""
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    
    # Analysis details
    reasoning: str
    technical_summary: str
    key_levels: Dict[str, float]
    timeframe: str
    
    # Metadata
    generated_at: datetime
    valid_until: datetime


class TechnicalAnalyzer(LoggerMixin):
    """
    LLM-powered technical analysis engine that combines traditional indicators
    with AI-driven market analysis to generate high-quality trading signals.
    """
    
    def __init__(self, binance_client: BinanceClient, llm_coordinator: LLMCoordinator, 
                 risk_manager=None):
        super().__init__()
        self.binance_client = binance_client
        self.llm_coordinator = llm_coordinator
        self.risk_manager = risk_manager
        
        # Analysis configuration
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.main_timeframe = '15m'
        self.lookback_periods = 100
        
        self.logger.info("Technical analyzer initialized")
    
    def _get_trading_objectives(self) -> Dict[str, Any]:
        """Get current trading objectives and style from risk manager/settings"""
        try:
            if not self.risk_manager:
                return self._get_default_objectives()
            
            # Get current risk profile and trading mode
            risk_summary = self.risk_manager.get_risk_summary()
            trading_mode = risk_summary.get('trading_mode', 'conservative')
            
            # Load challenge configuration
            challenge_config = {}
            try:
                import yaml
                with open('/home/trader/trader/config/risk_profiles/challenge.yaml', 'r') as f:
                    challenge_config = yaml.safe_load(f)
            except Exception:
                pass
            
            # Determine objectives based on trading mode
            if trading_mode.lower() == 'challenge' or 'challenge' in trading_mode.lower():
                return {
                    'objective': 'CHALLENGE_MODE_GROWTH',
                    'goal': 'Grow $100 to $100,000 as fast as possible',
                    'risk_tolerance': 'EXTREME',
                    'position_size': '10% of portfolio per trade',
                    'leverage': 'Up to 20x leverage',
                    'strategy_focus': [
                        'Look for high-probability breakouts and momentum plays',
                        'Target 3:1+ risk/reward ratios when possible',
                        'Accept higher risk for exponential growth potential',
                        'Focus on volatile, trending markets',
                        'Quick entries/exits to capitalize on momentum',
                        'Avoid sideways/choppy markets unless clear breakout'
                    ],
                    'risk_warnings': [
                        'Challenge mode uses extreme risk (10% position, 20x leverage)',
                        'Account can be lost quickly with bad trades',
                        'Only recommend very high confidence setups (70%+)',
                        'Prioritize capital preservation while seeking growth'
                    ],
                    'preferred_patterns': [
                        'Strong trend continuations with volume',
                        'Breakouts from consolidation patterns', 
                        'Momentum reversals with clear confirmation',
                        'Multi-timeframe aligned setups',
                        'High volume breakouts of key levels'
                    ],
                    'avoid_patterns': [
                        'Low volume, choppy price action',
                        'Counter-trend trades without strong confirmation',
                        'Range-bound markets without clear direction',
                        'Conflicting multi-timeframe signals'
                    ]
                }
            else:
                return self._get_conservative_objectives()
                
        except Exception as e:
            self.logger.error("Error getting trading objectives", error=str(e))
            return self._get_default_objectives()
    
    def _get_default_objectives(self) -> Dict[str, Any]:
        """Default conservative trading objectives"""
        return {
            'objective': 'CAPITAL_GROWTH',
            'goal': 'Steady capital growth with risk management',
            'risk_tolerance': 'MODERATE',
            'position_size': '2% of portfolio per trade',
            'leverage': 'Up to 3x leverage',
            'strategy_focus': [
                'High-probability setups with good risk/reward',
                'Strong trend following strategies',
                'Clear technical patterns with confirmation'
            ]
        }
    
    def _get_conservative_objectives(self) -> Dict[str, Any]:
        """Conservative trading objectives"""
        return {
            'objective': 'CAPITAL_PRESERVATION',
            'goal': 'Steady growth with minimal risk',
            'risk_tolerance': 'LOW',
            'position_size': '1% of portfolio per trade',
            'leverage': 'Up to 2x leverage',
            'strategy_focus': [
                'Very high-probability setups only',
                'Strong confirmation required',
                'Capital preservation priority'
            ]
        }
    
    async def analyze_symbol(self, symbol: str, timeframe: str = None) -> Optional[TradingSignal]:
        """
        Perform comprehensive technical analysis on a symbol
        Returns trading signal if opportunity is found
        """
        try:
            timeframe = timeframe or self.main_timeframe
            self.logger.info("Starting comprehensive analysis", 
                           symbol=symbol, 
                           timeframe=timeframe,
                           action="analysis_start")
            
            # 1. Get market data
            market_data = await self._get_market_data(symbol, timeframe)
            if market_data is None or len(market_data) < 50:
                self.logger.warning("Insufficient market data for analysis", 
                                  symbol=symbol, 
                                  data_points=len(market_data) if market_data is not None else 0,
                                  minimum_required=50)
                return None
            
            self.logger.debug("Market data retrieved successfully", 
                           symbol=symbol,
                           data_points=len(market_data),
                           latest_price=market_data['close'].iloc[-1] if not market_data.empty else None)
            
            # 2. Calculate technical indicators
            indicators = self._calculate_indicators(market_data)
            
            # 3. Perform multi-timeframe analysis
            mtf_context = await self._multi_timeframe_analysis(symbol)
            
            # 4. LLM prefilter + throttling before LLM analysis
            try:
                mid_price = float(indicators.current_price)
                atr_5m = float(indicators.atr)
                vol_bps = (atr_5m / max(mid_price, 1e-9)) * 1e4 if mid_price else 0.0
                high_min = int(os.getenv("VOL_BPS_HIGH_MIN", "25"))
                med_min = int(os.getenv("VOL_BPS_MED_MIN", "12"))
                if vol_bps >= high_min:
                    band, min_interval = "high", 60
                elif vol_bps >= med_min:
                    band, min_interval = "medium", 300
                else:
                    band, min_interval = "low", 1800

                # Proximity to key levels
                bb_u = float(getattr(indicators, 'bb_upper', np.nan))
                bb_l = float(getattr(indicators, 'bb_lower', np.nan))
                sma20 = float(getattr(indicators, 'sma_20', np.nan))
                sma50 = float(getattr(indicators, 'sma_50', np.nan))
                distances = []
                for lvl in [bb_u, bb_l, sma20, sma50]:
                    if np.isfinite(lvl):
                        distances.append(abs(mid_price - lvl))
                proximity_ok = (min(distances) <= 0.5 * atr_5m) if distances else False

                # Quick triggers (approximate)
                rsi = float(getattr(indicators, 'rsi_14', np.nan)) if hasattr(indicators, 'rsi_14') else float('nan')
                trigger = (np.isfinite(rsi) and (rsi <= 30 or rsi >= 70)) or proximity_ok or (vol_bps >= high_min)

                # Deduplicate by compact feature summary
                summary = {
                    "mid": round(mid_price, 6) if mid_price else 0,
                    "atr": round(atr_5m, 6) if atr_5m else 0,
                    "rsi": round(rsi, 2) if np.isfinite(rsi) else None,
                    "sma20": round(sma20, 6) if np.isfinite(sma20) else None,
                    "sma50": round(sma50, 6) if np.isfinite(sma50) else None,
                    "bb_u": round(bb_u, 6) if np.isfinite(bb_u) else None,
                    "bb_l": round(bb_l, 6) if np.isfinite(bb_l) else None,
                    "band": band,
                    "prox": proximity_ok,
                    "trig": bool(trigger),
                }
                sha = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16]
                now = time.time()
                drift_bps_thr = float(os.getenv("DRIFT_BPS", "8"))
                cache = getattr(self, "_llm_cache", {})
                last = cache.get(symbol, {})
                last_mid = last.get("last_mid")
                drift_bps = abs((mid_price - last_mid) / mid_price) * 1e4 if last_mid else 0.0
                if sha == last.get("last_sha") and (now - last.get("last_ts", 0)) < min_interval and drift_bps < drift_bps_thr:
                    self.logger.debug("LLM debounce", symbol=symbol, band=band, vol_bps=vol_bps)
                    return None
                # Update cache just before LLM call
                cache[symbol] = {"last_sha": sha, "last_ts": now, "band": band, "last_mid": mid_price}
                self._llm_cache = cache

                # Prefilter guard: if no trigger and low vol, skip LLM
                if not trigger and band == "low":
                    self.logger.debug("LLM skip (prefilter)", symbol=symbol, reason="low_vol_no_trigger")
                    return None
            except Exception as e:
                self.logger.warning("LLM prefilter error; proceeding without throttle", symbol=symbol, error=str(e))

            # 5. LLM-powered analysis
            signal = await self._llm_analysis(symbol, market_data, indicators, mtf_context, timeframe)
            
            if signal and signal.confidence >= 0.6:  # Minimum confidence threshold
                self.logger.info("Signal generated", 
                               symbol=symbol, 
                               direction=signal.direction.value,
                               strength=signal.strength.value,
                               confidence=signal.confidence)
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error("Error analyzing symbol", symbol=symbol, error=str(e))
            return None
    
    async def _get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get historical market data for analysis"""
        try:
            # Get klines data
            klines = await self.binance_client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=self.lookback_periods
            )
            
            if not klines:
                return None
            
            # Convert OHLCVData objects to DataFrame
            data = []
            for kline in klines:
                data.append({
                    'timestamp': kline.timestamp,
                    'open': kline.open,
                    'high': kline.high,
                    'low': kline.low,
                    'close': kline.close,
                    'volume': kline.volume,
                    'quote_volume': kline.volume * kline.close,  # Approximate quote volume
                })
            
            df = pd.DataFrame(data)
            
            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error("Error getting market data", symbol=symbol, error=str(e))
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate comprehensive technical indicators"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Simple stochastic (approximation)
            df['stoch_k'] = ((df['close'] - df['low'].rolling(14).min()) / 
                            (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR (Average True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = abs(df['high'] - df['close'].shift())
            df['low_close'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Support/Resistance levels (simplified)
            recent_data = df.tail(20)
            support_level = recent_data['low'].min()
            resistance_level = recent_data['high'].max()
            
            # Market structure analysis
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            current_high = df['high'].iloc[-1]
            prev_high = df['high'].iloc[-10:-1].max()
            current_low = df['low'].iloc[-1]
            prev_low = df['low'].iloc[-10:-1].min()
            
            higher_highs = current_high > prev_high
            higher_lows = current_low > prev_low
            lower_highs = current_high < prev_high
            lower_lows = current_low < prev_low
            
            # Get latest values
            latest = df.iloc[-1]
            
            return TechnicalIndicators(
                sma_20=latest['sma_20'],
                sma_50=latest['sma_50'],
                ema_12=latest['ema_12'],
                ema_26=latest['ema_26'],
                macd=latest['macd'],
                macd_signal=latest['macd_signal'],
                macd_histogram=latest['macd_histogram'],
                rsi_14=latest['rsi'],
                stoch_k=latest['stoch_k'],
                stoch_d=latest['stoch_d'],
                bb_upper=latest['bb_upper'],
                bb_middle=latest['bb_middle'],
                bb_lower=latest['bb_lower'],
                bb_width=latest['bb_width'],
                atr=latest['atr'],
                volume_sma=latest['volume_sma'],
                volume_ratio=latest['volume_ratio'],
                current_price=latest['close'],
                support_level=support_level,
                resistance_level=resistance_level,
                higher_highs=higher_highs,
                higher_lows=higher_lows,
                lower_highs=lower_highs,
                lower_lows=lower_lows
            )
            
        except Exception as e:
            self.logger.error("Error calculating indicators", error=str(e))
            raise
    
    async def _multi_timeframe_analysis(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive multi-timeframe analysis for GPT-5"""
        try:
            timeframes = ['5m', '15m', '1h', '4h']
            context = {}
            
            for tf in timeframes:
                try:
                    # Get comprehensive data for each timeframe
                    data = await self._get_market_data(symbol, tf)
                    if data is None or len(data) < 50:
                        continue
                    
                    # Calculate all indicators for this timeframe
                    indicators = self._calculate_indicators(data)
                    latest = data.iloc[-1]
                    
                    # Trend analysis
                    sma_20 = data['sma_20'].iloc[-1] if 'sma_20' in data else None
                    sma_50 = data['sma_50'].iloc[-1] if 'sma_50' in data else None
                    ema_12 = data['ema_12'].iloc[-1] if 'ema_12' in data else None
                    ema_26 = data['ema_26'].iloc[-1] if 'ema_26' in data else None
                    
                    # Determine trend strength
                    current_price = latest['close']
                    if pd.notna(sma_20) and pd.notna(sma_50):
                        if current_price > sma_20 > sma_50:
                            trend = "strong_bullish"
                            trend_strength = (current_price - sma_50) / sma_50
                        elif current_price > sma_20 and sma_20 < sma_50:
                            trend = "weak_bullish"
                            trend_strength = (current_price - sma_20) / sma_20
                        elif current_price < sma_20 < sma_50:
                            trend = "strong_bearish"
                            trend_strength = (sma_50 - current_price) / sma_50
                        elif current_price < sma_20 and sma_20 > sma_50:
                            trend = "weak_bearish"
                            trend_strength = (sma_20 - current_price) / sma_20
                        else:
                            trend = "neutral"
                            trend_strength = 0
                    else:
                        trend = "neutral"
                        trend_strength = 0
                    
                    # Volume analysis for this timeframe
                    volume_sma = data['volume_sma'].iloc[-1] if 'volume_sma' in data else None
                    volume_ratio = data['volume_ratio'].iloc[-1] if 'volume_ratio' in data else 1
                    
                    # Momentum indicators
                    rsi = data['rsi'].iloc[-1] if 'rsi' in data else None
                    macd = data['macd'].iloc[-1] if 'macd' in data else None
                    macd_signal = data['macd_signal'].iloc[-1] if 'macd_signal' in data else None
                    
                    # Support/Resistance for this timeframe
                    recent_data = data.tail(20)
                    support = recent_data['low'].min()
                    resistance = recent_data['high'].max()
                    
                    # Price action patterns
                    price_patterns = self._identify_price_patterns(data)
                    
                    context[tf] = {
                        'timeframe': tf,
                        'current_price': float(current_price),
                        'trend': trend,
                        'trend_strength': float(trend_strength),
                        'technical_indicators': {
                            'sma_20': float(sma_20) if pd.notna(sma_20) else None,
                            'sma_50': float(sma_50) if pd.notna(sma_50) else None,
                            'ema_12': float(ema_12) if pd.notna(ema_12) else None,
                            'ema_26': float(ema_26) if pd.notna(ema_26) else None,
                            'rsi': float(rsi) if pd.notna(rsi) else None,
                            'macd': float(macd) if pd.notna(macd) else None,
                            'macd_signal': float(macd_signal) if pd.notna(macd_signal) else None
                        },
                        'volume_analysis': {
                            'current_volume': float(latest['volume']),
                            'volume_ratio': float(volume_ratio),
                            'volume_trend': 'high' if volume_ratio > 1.2 else 'normal' if volume_ratio > 0.8 else 'low'
                        },
                        'key_levels': {
                            'support': float(support),
                            'resistance': float(resistance),
                            'range_percent': float((resistance - support) / current_price * 100)
                        },
                        'market_structure': price_patterns['trend_pattern'],
                        'recent_candles': [
                            {
                                'timestamp': idx.strftime('%m-%d %H:%M'),
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': float(row['volume'])
                            }
                            for idx, row in data.tail(10).iterrows()
                        ]
                    }
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing {tf} timeframe for {symbol}", error=str(e))
                    continue
            
            return context
            
        except Exception as e:
            self.logger.error("Error in multi-timeframe analysis", symbol=symbol, error=str(e))
            return {}
    
    async def _llm_analysis(self, symbol: str, market_data: pd.DataFrame, 
                           indicators: TechnicalIndicators, mtf_context: Dict[str, Any],
                           timeframe: str) -> Optional[TradingSignal]:
        """Use LLM to analyze market data and generate trading signal"""
        try:
            # Prepare comprehensive market data for GPT-5
            # Get full 100 candles with all technical indicators
            full_data = market_data.tail(100).copy()
            
            # Prepare complete OHLCV + indicators dataset
            comprehensive_data = []
            for idx, row in full_data.iterrows():
                candle_data = {
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'quote_volume': float(row.get('quote_volume', 0)),
                    
                    # Technical indicators for each candle
                    'sma_20': float(row.get('sma_20', 0)) if pd.notna(row.get('sma_20')) else None,
                    'sma_50': float(row.get('sma_50', 0)) if pd.notna(row.get('sma_50')) else None,
                    'ema_12': float(row.get('ema_12', 0)) if pd.notna(row.get('ema_12')) else None,
                    'ema_26': float(row.get('ema_26', 0)) if pd.notna(row.get('ema_26')) else None,
                    'rsi': float(row.get('rsi', 0)) if pd.notna(row.get('rsi')) else None,
                    'macd': float(row.get('macd', 0)) if pd.notna(row.get('macd')) else None,
                    'macd_signal': float(row.get('macd_signal', 0)) if pd.notna(row.get('macd_signal')) else None,
                    'macd_histogram': float(row.get('macd_histogram', 0)) if pd.notna(row.get('macd_histogram')) else None,
                    'bb_upper': float(row.get('bb_upper', 0)) if pd.notna(row.get('bb_upper')) else None,
                    'bb_middle': float(row.get('bb_middle', 0)) if pd.notna(row.get('bb_middle')) else None,
                    'bb_lower': float(row.get('bb_lower', 0)) if pd.notna(row.get('bb_lower')) else None,
                    'stoch_k': float(row.get('stoch_k', 0)) if pd.notna(row.get('stoch_k')) else None,
                    'stoch_d': float(row.get('stoch_d', 0)) if pd.notna(row.get('stoch_d')) else None,
                    'atr': float(row.get('atr', 0)) if pd.notna(row.get('atr')) else None,
                    'volume_sma': float(row.get('volume_sma', 0)) if pd.notna(row.get('volume_sma')) else None,
                    'volume_ratio': float(row.get('volume_ratio', 0)) if pd.notna(row.get('volume_ratio')) else None,
                    'true_range': float(row.get('true_range', 0)) if pd.notna(row.get('true_range')) else None
                }
                comprehensive_data.append(candle_data)
            
            # Volume analysis summary
            volume_analysis = self._analyze_volume_patterns(full_data)
            
            # Price action patterns
            price_patterns = self._identify_price_patterns(full_data)
            
            # Get trading objectives based on current settings
            trading_objectives = self._get_trading_objectives()

            # NEW: Route through LLMCoordinator with full context; legacy prompt below is bypassed
            try:
                # Bid/ask/spread
                mid_price = float(indicators.current_price) if indicators and indicators.current_price else None
                bid = None
                ask = None
                spread_bps = None
                try:
                    ob = await self.binance_client.get_order_book(symbol=symbol, limit=5)
                    best_bid = float(ob['bids'][0][0]) if ob and ob.get('bids') else None
                    best_ask = float(ob['asks'][0][0]) if ob and ob.get('asks') else None
                    if best_bid and best_ask:
                        bid, ask = best_bid, best_ask
                        mid_price = (bid + ask) / 2.0
                        spread_bps = (ask - bid) / max(mid_price, 1e-9) * 1e4
                except Exception:
                    pass

                if mid_price is None:
                    last_close = float(market_data['close'].iloc[-1]) if not market_data.empty else 0.0
                    mid_price = last_close
                if spread_bps is None:
                    spread_bps = float(os.getenv("DEFAULT_SPREAD_BPS", "5"))
                if bid is None or ask is None:
                    half_spread = mid_price * (spread_bps / 1e4) / 2.0
                    bid = mid_price - half_spread
                    ask = mid_price + half_spread

                # OHLCV tails
                try:
                    ohlcv_1m = await self.binance_client.get_klines(symbol=symbol, interval='1m', limit=12)
                except Exception:
                    ohlcv_1m = []
                try:
                    ohlcv_5m = await self.binance_client.get_klines(symbol=symbol, interval='5m', limit=12)
                except Exception:
                    ohlcv_5m = []
                try:
                    ohlcv_1h = await self.binance_client.get_klines(symbol=symbol, interval='1h', limit=6)
                except Exception:
                    ohlcv_1h = []

                # Indicators for LLM
                try:
                    ema20 = float(market_data['close'].ewm(span=20).mean().iloc[-1])
                except Exception:
                    ema20 = None
                try:
                    ema200 = float(market_data['close'].ewm(span=200).mean().iloc[-1])
                except Exception:
                    ema200 = None
                rsi_14 = float(getattr(indicators, 'rsi_14', np.nan)) if hasattr(indicators, 'rsi_14') else None
                if rsi_14 is not None and not np.isfinite(rsi_14):
                    rsi_14 = None
                macd_val = float(getattr(indicators, 'macd', np.nan)) if hasattr(indicators, 'macd') else None
                if macd_val is not None and not np.isfinite(macd_val):
                    macd_val = None
                macd_signal = float(getattr(indicators, 'macd_signal', np.nan)) if hasattr(indicators, 'macd_signal') else None
                if macd_signal is not None and not np.isfinite(macd_signal):
                    macd_signal = None
                bb_upper = float(getattr(indicators, 'bb_upper', np.nan)) if hasattr(indicators, 'bb_upper') else None
                if bb_upper is not None and not np.isfinite(bb_upper):
                    bb_upper = None
                bb_lower = float(getattr(indicators, 'bb_lower', np.nan)) if hasattr(indicators, 'bb_lower') else None
                if bb_lower is not None and not np.isfinite(bb_lower):
                    bb_lower = None
                bb_middle = float(getattr(indicators, 'bb_middle', np.nan)) if hasattr(indicators, 'bb_middle') else None
                if bb_middle is not None and not np.isfinite(bb_middle):
                    bb_middle = None
                atr_14 = float(getattr(indicators, 'atr', np.nan)) if hasattr(indicators, 'atr') else None
                if atr_14 is not None and not np.isfinite(atr_14):
                    atr_14 = None

                llm_ind = LLMTechIndicators(
                    rsi_14=rsi_14,
                    macd=macd_val,
                    macd_signal=macd_signal,
                    bb_upper=bb_upper,
                    bb_lower=bb_lower,
                    bb_middle=bb_middle,
                    ema_20=ema20,
                    ema_200=ema200,
                    atr_14=atr_14,
                )

                # Market/account context
                realized_vol_24h = None
                try:
                    funding_rate = await self.binance_client.get_funding_rate(symbol)
                except Exception:
                    funding_rate = None
                fees_bps = float(os.getenv("DEFAULT_FEES_BPS", "8"))
                slippage_bps = float(os.getenv("DEFAULT_SLIPPAGE_BPS", "5"))
                portfolio_balance = float(os.getenv("PORTFOLIO_BALANCE_DEFAULT", "10000"))
                try:
                    daily_pnl = float(self.risk_manager.get_risk_summary().get('daily_pnl', 0.0)) if self.risk_manager else 0.0
                except Exception:
                    daily_pnl = 0.0

                try:
                    last_trade_ts = market_data.index[-1].to_pydatetime()
                except Exception:
                    last_trade_ts = datetime.now(timezone.utc)

                llm_input = LLMInputModel(
                    symbol=symbol,
                    bid=float(bid),
                    ask=float(ask),
                    spread_bps=float(spread_bps),
                    last_trade_ts=last_trade_ts,
                    indicators=llm_ind,
                    realized_vol_24h=realized_vol_24h,
                    funding_rate=funding_rate,
                    fees_bps=fees_bps,
                    slippage_bps=slippage_bps,
                    portfolio_balance=portfolio_balance,
                    portfolio_heat=0.0,
                    daily_pnl=daily_pnl,
                    sentiment_score=None,
                    sentiment_strength=None,
                    ohlcv_1m=ohlcv_1m,
                    ohlcv_5m=ohlcv_5m,
                    ohlcv_1h=ohlcv_1h,
                )

                # Pre-LLM preview log
                preview = (
                    f"SYMBOL={symbol} rsi={llm_ind.rsi_14 if llm_ind.rsi_14 is not None else 'NA'} "
                    f"ema20={llm_ind.ema_20 if llm_ind.ema_20 is not None else 'NA'} "
                    f"ema200={llm_ind.ema_200 if llm_ind.ema_200 is not None else 'NA'} "
                    f"atr14={llm_ind.atr_14 if llm_ind.atr_14 is not None else 'NA'} "
                    f"spread_bps={spread_bps:.2f} ohlcv_1m={len(ohlcv_1m)} 5m={len(ohlcv_5m)} 1h={len(ohlcv_1h)}"
                )
                try:
                    sha_obj = {
                        "symbol": symbol,
                        "rsi": llm_ind.rsi_14,
                        "ema20": llm_ind.ema_20,
                        "ema200": llm_ind.ema_200,
                        "atr14": llm_ind.atr_14,
                        "spread_bps": spread_bps,
                        "c1": len(ohlcv_1m),
                        "c5": len(ohlcv_5m),
                        "c60": len(ohlcv_1h),
                    }
                    payload_sha16 = hashlib.sha256(json.dumps(sha_obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
                except Exception:
                    payload_sha16 = ""
                self.logger.info(
                    "Dispatching LLM (coordinator)",
                    symbol=symbol,
                    timeframe=timeframe,
                    payload_preview=preview,
                    payload_sha16=payload_sha16,
                )

                decision, human_summary = await self.llm_coordinator.generate_decision(llm_input)

                # Per-symbol cooldown: bump last_ts after any successful LLM call
                try:
                    cache = getattr(self, "_llm_cache", {})
                    last = cache.get(symbol, {})
                    last["last_ts"] = time.time()
                    last["last_mid"] = mid_price
                    cache[symbol] = last
                    self._llm_cache = cache
                except Exception:
                    pass

                # Non-actionable decision: HOLD, missing prices, or failed EV gate
                if (
                    decision is None
                    or getattr(decision, "action", None) is None
                    or getattr(decision, "action", None).value == "HOLD"
                    or decision.entry_price is None
                    or decision.stop_loss_price is None
                    or decision.take_profit_price is None
                ):
                    # Log + bump cooldown so we don't hammer the LLM again immediately
                    self.logger.debug(
                        "LLM decision not actionable",
                        symbol=symbol,
                        action=(getattr(decision, "action", None).value if getattr(decision, "action", None) else "NONE"),
                    )
                    try:
                        now_ts = time.time()
                        cache = getattr(self, "_llm_cache", {})
                        _cache = cache.get(symbol, {})
                        _cache["last_ts"] = now_ts
                        cache[symbol] = _cache
                        self._llm_cache = cache
                    except Exception:
                        pass
                    return None

                conf = float(getattr(decision, 'confidence', 0.0))
                if conf >= 0.85:
                    strength = SignalStrength.VERY_STRONG
                elif conf >= 0.70:
                    strength = SignalStrength.STRONG
                elif conf >= 0.50:
                    strength = SignalStrength.MODERATE
                else:
                    strength = SignalStrength.WEAK

                rr = 0.0
                try:
                    ep = decision.entry_price
                    sl = decision.stop_loss_price
                    tp = decision.take_profit_price
                    if ep and sl and tp:
                        if decision.action == LLMDecisionAction.LONG:
                            risk = abs(ep - sl)
                            reward = abs(tp - ep)
                        else:
                            risk = abs(sl - ep)
                            reward = abs(ep - tp)
                        rr = (reward / max(risk, 1e-9)) if risk > 0 else 0.0
                except Exception:
                    rr = 0.0

                direction = SignalDirection.LONG if decision.action == LLMDecisionAction.LONG else SignalDirection.SHORT

                signal = TradingSignal(
                    symbol=symbol,
                    direction=direction,
                    strength=strength,
                    confidence=conf,
                    entry_price=float(decision.entry_price) if decision.entry_price is not None else float(mid_price),
                    stop_loss=float(decision.stop_loss_price) if decision.stop_loss_price is not None else 0.0,
                    take_profit=float(decision.take_profit_price) if decision.take_profit_price is not None else 0.0,
                    risk_reward_ratio=float(rr),
                    reasoning=str(getattr(decision, 'reasoning', '')),
                    technical_summary=f"Coordinator decision: {decision.action.value} size={decision.position_size_pct:.2%}",
                    key_levels={
                        'support': float(indicators.support_level) if indicators.support_level is not None else 0.0,
                        'resistance': float(indicators.resistance_level) if indicators.resistance_level is not None else 0.0,
                    },
                    timeframe=timeframe,
                    generated_at=datetime.now(timezone.utc),
                    valid_until=datetime.now(timezone.utc) + timedelta(hours=1)
                )

                self.logger.info(
                    "Signal generated via coordinator",
                    symbol=symbol,
                    action=decision.action.value,
                    confidence=conf,
                    rr=rr,
                )
                return signal
            except Exception as _e:
                # Do not fall back to legacy prompt; update cooldown and stop
                self.logger.warning("Coordinator path failed; no legacy fallback", error=str(_e))
                try:
                    now_ts = time.time()
                    cache = getattr(self, "_llm_cache", {})
                    _cache = cache.get(symbol, {})
                    _cache["last_ts"] = now_ts
                    cache[symbol] = _cache
                    self._llm_cache = cache
                except Exception:
                    pass
                return None
            
        except Exception as e:
            self.logger.error("Error in LLM analysis", symbol=symbol, error=str(e))
            return None
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns and trends"""
        try:
            recent_volume = df['volume'].tail(20)
            avg_volume = df['volume'].rolling(50).mean().iloc[-1]
            volume_trend = df['volume'].rolling(10).mean()
            
            # Volume momentum
            volume_increasing = volume_trend.iloc[-1] > volume_trend.iloc[-5]
            volume_spike = recent_volume.iloc[-1] > (avg_volume * 1.5)
            
            # Volume price correlation
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            correlation = price_change.tail(20).corr(volume_change.tail(20))
            
            # On Balance Volume approximation
            obv_changes = []
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv_changes.append(df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv_changes.append(-df['volume'].iloc[i])
                else:
                    obv_changes.append(0)
            
            obv_trend = 'bullish' if sum(obv_changes[-10:]) > 0 else 'bearish'
            
            return {
                'current_volume': float(recent_volume.iloc[-1]),
                'average_volume_50': float(avg_volume),
                'volume_ratio': float(recent_volume.iloc[-1] / avg_volume),
                'volume_increasing': bool(volume_increasing),
                'volume_spike': bool(volume_spike),
                'volume_price_correlation': float(correlation) if pd.notna(correlation) else 0.0,
                'obv_trend': obv_trend,
                'high_volume_candles': int(sum(recent_volume > avg_volume * 1.2))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_price_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key price action patterns"""
        try:
            recent_data = df.tail(50)
            
            # Trend analysis
            highs = recent_data['high'].rolling(5).max()
            lows = recent_data['low'].rolling(5).min()
            
            # Higher highs/lows pattern
            recent_highs = highs.tail(10)
            recent_lows = lows.tail(10)
            
            higher_highs = (recent_highs.iloc[-1] > recent_highs.iloc[-5] > recent_highs.iloc[-10])
            higher_lows = (recent_lows.iloc[-1] > recent_lows.iloc[-5] > recent_lows.iloc[-10])
            lower_highs = (recent_highs.iloc[-1] < recent_highs.iloc[-5] < recent_highs.iloc[-10])
            lower_lows = (recent_lows.iloc[-1] < recent_lows.iloc[-5] < recent_lows.iloc[-10])
            
            # Support and resistance levels
            support_levels = []
            resistance_levels = []
            
            # Find pivot points
            for i in range(5, len(recent_data)-5):
                # Resistance (local high)
                if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-5:i].max() and
                    recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1:i+6].max()):
                    resistance_levels.append(recent_data['high'].iloc[i])
                
                # Support (local low)  
                if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-5:i].min() and
                    recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1:i+6].min()):
                    support_levels.append(recent_data['low'].iloc[i])
            
            # Candlestick patterns
            last_candle = recent_data.iloc[-1]
            prev_candle = recent_data.iloc[-2]
            
            # Doji pattern
            body_size = abs(last_candle['close'] - last_candle['open'])
            candle_range = last_candle['high'] - last_candle['low']
            is_doji = body_size < (candle_range * 0.1) if candle_range > 0 else False
            
            # Engulfing pattern
            bullish_engulfing = (last_candle['close'] > last_candle['open'] and 
                               prev_candle['close'] < prev_candle['open'] and
                               last_candle['open'] < prev_candle['close'] and
                               last_candle['close'] > prev_candle['open'])
            
            bearish_engulfing = (last_candle['close'] < last_candle['open'] and 
                               prev_candle['close'] > prev_candle['open'] and
                               last_candle['open'] > prev_candle['close'] and
                               last_candle['close'] < prev_candle['open'])
            
            return {
                'trend_pattern': {
                    'higher_highs': bool(higher_highs),
                    'higher_lows': bool(higher_lows),
                    'lower_highs': bool(lower_highs),
                    'lower_lows': bool(lower_lows)
                },
                'support_levels': support_levels[-3:] if support_levels else [],
                'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
                'candlestick_patterns': {
                    'doji': bool(is_doji),
                    'bullish_engulfing': bool(bullish_engulfing),
                    'bearish_engulfing': bool(bearish_engulfing)
                },
                'price_position': {
                    'near_high': bool((last_candle['close'] / recent_data['high'].max()) > 0.98),
                    'near_low': bool((last_candle['close'] / recent_data['low'].min()) < 1.02),
                    'middle_range': bool(0.4 < (last_candle['close'] / recent_data['high'].max()) < 0.6)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def batch_analyze(self, symbols: List[str], timeframe: str = None) -> List[TradingSignal]:
        """Analyze multiple symbols concurrently"""
        tasks = [self.analyze_symbol(symbol, timeframe) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        signals = []
        for result in results:
            if isinstance(result, TradingSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                self.logger.error("Batch analysis error", error=str(result))
        
        return signals
