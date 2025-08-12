import os
import yaml
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict, validator, model_validator


class TradingMode(str, Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    PAPER = "paper"
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
    model_config = ConfigDict(str_strip_whitespace=True)
    
    api_key: str
    secret_key: str
    use_testnet: bool = True
    futures: bool = False
    base_url: Optional[str] = None


class MatrixConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
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


class BriefingConfig(BaseModel):
    enabled: bool = True
    time_local: str = "07:00"
    timezone: str = "Europe/Ljubljana"
    symbols: List[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    include_futures_calendar: bool = True


class TavilyConfig(BaseModel):
    enabled: bool = False
    api_key: Optional[str] = None
    freshness_days: int = Field(default=2, ge=1, le=7)
    per_symbol_queries: int = Field(default=3, ge=1, le=10)
    max_requests_per_day: int = Field(default=60, ge=10, le=1000)
    query_templates: List[str] = Field(default_factory=lambda: [
        "latest news {symbol} price drivers",
        "{symbol} funding, liquidation clusters, derivatives sentiment",
        "crypto market risk today macro events"
    ])


class AppConfig(BaseModel):
    binance: BinanceConfig
    matrix: MatrixConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    briefing: BriefingConfig = Field(default_factory=BriefingConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    
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
            ),
            briefing=BriefingConfig(
                time_local=os.getenv("BRIEF_TIME_LOCAL", "07:00"),
                timezone=os.getenv("BRIEF_TIMEZONE", "Europe/Ljubljana")
            ),
            tavily=TavilyConfig(
                enabled=os.getenv("TAVILY_ENABLE", "false").lower() == "true",
                api_key=os.getenv("TAVILY_API_KEY")
            )
        )


# Trading Models
class OHLCVData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
        return v


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
    """Sanitized numeric input for LLM - no raw text or URLs."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    symbol: str
    bid: float
    ask: float
    spread_bps: float
    last_trade_ts: datetime
    
    # OHLCV tails for different timeframes
    ohlcv_1m: List[OHLCVData] = Field(max_length=60)
    ohlcv_5m: List[OHLCVData] = Field(max_length=288)
    ohlcv_1h: List[OHLCVData] = Field(max_length=168)
    
    # Technical indicators
    indicators: TechnicalIndicators
    
    # Market context
    realized_vol_24h: Optional[float] = None
    funding_rate: Optional[float] = None
    fees_bps: float
    slippage_bps: float
    
    # Account context
    portfolio_balance: float
    portfolio_heat: float  # Current risk exposure as percentage
    daily_pnl: float
    
    # Sentiment (numeric summary only)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    sentiment_strength: Optional[float] = Field(None, ge=0.0, le=1.0)


class LLMOutput(BaseModel):
    """Strict action schema for LLM decisions."""
    action: DecisionAction
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Position sizing (as percentage of portfolio)
    position_size_pct: float = Field(ge=0.0, le=0.1)  # Max 10% position
    
    # Entry parameters
    entry_type: OrderType = OrderType.LIMIT
    entry_price: Optional[float] = None  # None for market orders
    
    # Risk management
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    # Reasoning (brief)
    reasoning: str = Field(max_length=500)
    
    @model_validator(mode='after')
    def validate_prices(self):
        if self.action in [DecisionAction.LONG, DecisionAction.SHORT]:
            if self.entry_type == OrderType.LIMIT and self.entry_price is None:
                raise ValueError("Entry price required for limit orders")
            if self.stop_loss_price is None:
                raise ValueError("Stop loss is required for new positions")
        return self


class HumanProposal(BaseModel):
    """Human-readable proposal with LLM decision."""
    decision_id: str
    symbol: str
    timestamp: datetime
    summary_text: str  # Human-readable explanation
    decision: LLMOutput
    requires_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


# News and Sentiment Models
class NewsHeadline(BaseModel):
    title: str
    source: str
    age_hours: float
    url_hash: str  # Hash of URL for deduplication
    relevance_score: Optional[float] = None


class SentimentAnalysis(BaseModel):
    symbol: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_strength: float = Field(ge=0.0, le=1.0)
    top_headlines: List[NewsHeadline] = Field(max_length=10)
    risk_keywords: List[str] = Field(default_factory=list)
    generated_at: datetime


# Daily Brief Models
class BriefSection(BaseModel):
    title: str
    content: str
    priority: int = 0  # Higher number = higher priority


class BriefDocument(BaseModel):
    date: datetime
    mode: TradingMode
    symbols: List[str]
    sections: Dict[str, List[BriefSection]]  # symbol -> sections
    generated_at: datetime
    
    def to_matrix_message(self) -> str:
        """Convert to Matrix-formatted message."""
        lines = [
            f"📊 **DAILY BRIEF** {self.date.strftime('%Y-%m-%d')} — Mode: {self.mode.value.title()}",
            f"**Symbols:** {', '.join(self.symbols)}",
            ""
        ]
        
        for symbol in self.symbols:
            if symbol in self.sections:
                lines.append(f"**{symbol}**")
                for section in sorted(self.sections[symbol], key=lambda x: x.priority, reverse=True):
                    lines.append(f"• **{section.title}**: {section.content}")
                lines.append("")
        
        lines.append("💬 Commands: `/approve <id>` for proposals, `/brief` to refresh, `/mode aggressive`, `/halt`")
        return "\n".join(lines)


# Database Models (for SQLAlchemy integration)
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


class RiskEvent(BaseModel):
    id: Optional[int] = None
    event_type: str
    symbol: Optional[str] = None
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    triggered_at: datetime
    resolved_at: Optional[datetime] = None


# Utility functions
def load_yaml_config(file_path: str, env_overrides: bool = True) -> Dict[str, Any]:
    """Load YAML configuration with optional environment variable overrides."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if env_overrides:
        # Simple env override for nested configs
        # Format: SECTION_SUBSECTION_KEY=value
        for key, value in os.environ.items():
            if key.startswith(('BINANCE_', 'MATRIX_', 'RISK_', 'MARKET_', 'BRIEF_', 'TAVILY_')):
                parts = key.lower().split('_')
                if len(parts) >= 2:
                    section = parts[0]
                    subkey = '_'.join(parts[1:])
                    if section in config and isinstance(config[section], dict):
                        config[section][subkey] = value
    
    return config


def create_default_config_file(file_path: str):
    """Create a default configuration YAML file."""
    default_config = {
        'risk': {
            'risk_per_trade': 0.01,
            'max_daily_loss': 0.05,
            'heat_cap': 0.1,
            'leverage_cap': 3,
            'require_double_confirm': False
        },
        'market': {
            'top_symbols_count': 5,
            'min_24h_volume': 1000000,
            'max_spread_bps': 20,
            'scan_interval_minutes': 30
        },
        'briefing': {
            'enabled': True,
            'time_local': '07:00',
            'timezone': 'Europe/Ljubljana',
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'include_futures_calendar': True
        },
        'tavily': {
            'enabled': False,
            'freshness_days': 2,
            'per_symbol_queries': 3,
            'max_requests_per_day': 60,
            'query_templates': [
                'latest news {symbol} price drivers',
                '{symbol} funding, liquidation clusters, derivatives sentiment',
                'crypto market risk today macro events'
            ]
        }
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)