from enum import Enum
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, conlist, conint, confloat, model_validator, field_validator
from datetime import datetime, timezone
import os
import yaml

# New deterministic trading models
class Trend(Enum):
    DOWN = -1
    FLAT = 0
    UP = 1
    STRONG_UP = 2
    STRONG_DOWN = -2

class Action(str, Enum):
    OPEN = "OPEN"
    HOLD = "HOLD"
    FLAT = "FLAT"
    CANCEL = "CANCEL"

class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class EntryType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class TimeframeSnapshot(BaseModel):
    trend: Trend
    mom: conint(ge=-2, le=2)                 # momentum sign/strength
    atr_pct: confloat(ge=0, le=5)            # ATR/price * 100
    rsi: conint(ge=0, le=100)
    bb_pos: confloat(ge=0, le=1)             # 0=lower band, 1=upper band
    sr_top_dist_bp: conint(ge=0, le=20000)   # basis points distance to overhead SR
    sr_bot_dist_bp: conint(ge=0, le=20000)   # bps to support
    imb5: confloat(ge=0, le=1)               # simple orderbook imbalance 0..1
    breakout_pending: bool = False

class Fees(BaseModel):
    taker_bps: conint(ge=0, le=100) = 5
    maker_bps: conint(ge=0, le=100) = 1

class DecisionInput(BaseModel):
    decision_id: str
    symbol: str                               # e.g., "BTCUSDT"
    tfs: dict                                 # {"1m": TimeframeSnapshot, "5m": ..., "1h": ...}
    fees: Fees
    spread_bp: confloat(ge=0, le=100)
    volatility_regime: Literal["low","medium","high"]
    funding_bps: Optional[conint(ge=-200, le=200)] = 0
    oi_5m_chg_pct: Optional[confloat(ge=-50, le=50)] = 0
    time: str

class StopSpec(BaseModel):
    price: Optional[confloat(gt=0)]
    reason: Optional[str] = None

class TPLevel(BaseModel):
    price: confloat(gt=0)
    type: Literal["R","ABS"] = "R"           # R multiple or absolute price

class TrailSpec(BaseModel):
    type: Literal["ATR","SWING","NONE"] = "ATR"
    mult: Optional[confloat(gt=0, le=5)] = 0.8

class DecisionOutput(BaseModel):
    action: Action
    symbol: str
    side: Optional[Side] = None
    entry: dict = Field(default_factory=lambda: {"type":"market","price":None})
    size_pct: confloat(gt=0, le=1) = 0.25     # of allocatable capital for this trade
    leverage: confloat(gt=1, le=125) = 10
    stop: Optional[StopSpec] = None
    tp: conlist(TPLevel, min_length=0, max_length=4) = []
    trail: TrailSpec = TrailSpec()
    valid_for_sec: conint(gt=10, le=600) = 90
    confidence: confloat(ge=0, le=1) = 0.5
    notes: Optional[str] = None

# Legacy compatibility models
class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HALT = "halt"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    OCO = "OCO"

class TradeState(str, Enum):
    SCANNING = "SCANNING"
    PROPOSAL = "PROPOSAL"
    AWAIT_APPROVAL = "AWAIT_APPROVAL"
    EXECUTION = "EXECUTION"
    MONITOR = "MONITOR"
    EXIT = "EXIT"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"

class DecisionAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    EXIT = "EXIT"

# Configuration Models
class BinanceConfig(BaseModel):
    api_key: str
    secret_key: str
    use_testnet: bool = True
    futures: bool = False
    base_url: Optional[str] = None

class MatrixConfig(BaseModel):
    homeserver: str
    access_token: str
    room_id: str
    admin_users: List[str] = Field(default_factory=list)

class RiskConfig(BaseModel):
    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.1)
    max_daily_loss: float = Field(default=0.05, ge=0.01, le=0.5)
    heat_cap: float = Field(default=0.1, ge=0.02, le=1.0)
    leverage_cap: int = Field(default=3, ge=1, le=20)
    require_double_confirm: bool = False

class MarketConfig(BaseModel):
    top_symbols_count: int = Field(default=5, ge=1, le=20)
    min_24h_volume: float = Field(default=1000000)
    max_spread_bps: float = Field(default=20)
    scan_interval_minutes: int = Field(default=30, ge=5, le=120)

class AppConfig(BaseModel):
    binance: BinanceConfig
    matrix: MatrixConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    
    @classmethod
    def load_from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        return cls(
            binance=BinanceConfig(
                api_key=os.getenv("BINANCE_API_KEY", ""),
                secret_key=os.getenv("BINANCE_SECRET_KEY", ""),
                use_testnet=os.getenv("BINANCE_USE_TESTNET", "true").lower() == "true",
                futures=os.getenv("BINANCE_FUTURES", "false").lower() == "true"
            ),
            matrix=MatrixConfig(
                homeserver=os.getenv("MATRIX_HOMESERVER", ""),
                access_token=os.getenv("MATRIX_ACCESS_TOKEN", ""),
                room_id=os.getenv("MATRIX_ROOM_ID", ""),
                admin_users=os.getenv("MATRIX_ADMIN_USERS", "").split(",") if os.getenv("MATRIX_ADMIN_USERS") else []
            ),
            risk=RiskConfig(
                risk_per_trade=float(os.getenv("DEFAULT_RISK_PER_TRADE", "0.01")),
                max_daily_loss=float(os.getenv("DEFAULT_MAX_DAILY_LOSS", "0.05")),
                heat_cap=float(os.getenv("DEFAULT_HEAT_CAP", "0.10")),
                leverage_cap=int(os.getenv("DEFAULT_LEVERAGE_CAP", "3"))
            )
        )

# Trading Models
class OHLCVData(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None

class TechnicalIndicators(BaseModel):
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_middle: Optional[float] = None
    ema_20: Optional[float] = None
    ema_200: Optional[float] = None
    atr_14: Optional[float] = None

class LLMInput(BaseModel):
    """
    Legacy/compat input envelope used by some analysis paths.
    Only fields used by llm_coordinator should be present; others are allowed via extras.
    """
    symbol: str
    timeframe: Optional[str] = None
    state: Dict[str, Any] = {}
    last_trade_ts: Optional[str] = None  # coerced to ISO string
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_bps: Optional[float] = None
    indicators: Optional[TechnicalIndicators] = None
    realized_vol_24h: Optional[float] = None
    funding_rate: Optional[float] = None
    fees_bps: Optional[float] = None
    slippage_bps: Optional[float] = None
    portfolio_balance: Optional[float] = None
    portfolio_heat: Optional[float] = None
    daily_pnl: Optional[float] = None
    sentiment_score: Optional[float] = None
    sentiment_strength: Optional[float] = None
    ohlcv_1m: Optional[List[OHLCVData]] = None
    ohlcv_5m: Optional[List[OHLCVData]] = None
    ohlcv_1h: Optional[List[OHLCVData]] = None
    # allow extra keys without failing
    model_config = {"extra": "allow"}

    @field_validator("last_trade_ts", mode="before")
    @classmethod
    def _coerce_ts(cls, v):
        if isinstance(v, datetime):
            return v.replace(microsecond=0).isoformat()
        return v

class LLMOutput(BaseModel):
    """Legacy LLM output format for backward compatibility."""
    action: DecisionAction
    confidence: float
    position_size_pct: float
    entry_type: str
    entry_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str

# Database Models
class TradeRecord(BaseModel):
    id: Optional[int] = None
    decision_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: TradeState
    created_at: datetime
    updated_at: datetime
    pnl_realized: Optional[float] = None
    fees_paid: Optional[float] = None

# Utility functions
def load_yaml_config(file_path: str, env_overrides: bool = True) -> Dict[str, Any]:
    """Load YAML configuration with optional environment variable overrides."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if env_overrides:
        # Simple env override for nested configs
        for key, value in os.environ.items():
            if key.startswith(('BINANCE_', 'MATRIX_', 'RISK_', 'MARKET_')):
                parts = key.lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    subkey = '_'.join(parts[1:])
                    if section in config and isinstance(config[section], dict):
                        config[section][subkey] = value
    
    return config

# --- Compatibility for Matrix UI (hot-fix) ---
class HumanProposal(BaseModel):
    """
    UI-facing proposal for Matrix. Mirrors DecisionOutput fields needed for display/approval.
    Safe to remove once matrix_client uses DecisionOutput directly.
    """
    decision_id: str
    symbol: str
    side: Side
    entry: dict = Field(default_factory=lambda: {"type":"market","price":None})  # {"type":"market|limit","price": Optional[float]}
    stop: StopSpec
    tp: List[TPLevel] = []
    trail: TrailSpec = TrailSpec()
    size_pct: confloat(gt=0, le=1) = 0.25
    leverage: confloat(gt=1, le=125) = 10
    valid_for_sec: conint(gt=10, le=600) = 90
    confidence: confloat(ge=0, le=1) = 0.5
    notes: Optional[str] = None