# Advanced AI Trading Bot

A sophisticated cryptocurrency trading bot that combines AI decision-making with comprehensive risk management, real-time market analysis, and Matrix chat integration for human oversight.

## 🏗️ Architecture Overview

The trading bot follows a modular architecture with clear separation of concerns:

```
trader/
├── main.py                    # Main orchestrator and entry point
├── .env.template             # Environment configuration template
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── market_settings.yaml # Market and trading settings
│   └── risk_profiles/       # Risk profile configurations
├── core/                    # Core trading infrastructure
│   ├── binance_client.py    # Binance REST API client
│   ├── binance_ws.py        # Binance WebSocket client
│   ├── order_constraints.py # Order validation and constraints
│   ├── matrix_client.py     # Matrix chat integration
│   ├── llm_coordinator.py   # AI/LLM decision coordination
│   ├── execution.py         # Trade execution engine
│   ├── trade_manager.py     # Trade lifecycle management
│   └── state_machine.py     # Trading state management
├── analyzers/               # Market analysis modules
│   ├── market_scanner.py    # Symbol selection and filtering
│   ├── features.py          # Feature extraction
│   ├── technical_analysis.py # Technical indicators
│   └── sentiment_analysis.py # News and sentiment analysis
├── strategies/              # Trading strategies and risk management
│   ├── risk_manager.py      # Comprehensive risk management
│   ├── scalping.py         # Scalping strategy implementation
│   └── swing_trading.py    # Swing trading strategy
├── utils/                   # Utility modules
│   ├── database.py          # Database models and management
│   ├── logging_config.py    # Structured logging setup
│   ├── pydantic_models.py   # Data models and validation
│   ├── formatters.py        # Message and data formatting
│   └── notifications.py     # Alert and notification system
├── backtester/             # Backtesting framework
│   ├── engine.py           # Backtesting engine
│   └── paper_trader.py     # Paper trading implementation
└── tests/                  # Test suite
    ├── test_risk_manager.py
    ├── test_order_constraints.py
    └── test_llm_schema.py
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd /home/trader/trader

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt

# Copy and configure environment file
cp .env.template .env
nano .env  # Edit with your API keys and settings
```

### 2. Configuration

Edit `.env` with your credentials:

```env
# Binance API (get from binance.com)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_USE_TESTNET=true  # Start with testnet!
BINANCE_FUTURES=false     # Spot trading recommended

# Matrix Chat (for bot communication)
MATRIX_HOMESERVER=https://chat.niflaire.com
MATRIX_ACCESS_TOKEN=your_matrix_access_token
MATRIX_ROOM_ID=!your_room_id:chat.niflaire.com
MATRIX_ADMIN_USERS=@yourusername:chat.niflaire.com

# AI API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: News Analysis
TAVILY_API_KEY=your_tavily_api_key
TAVILY_ENABLE=false
```

### 3. Initial Run

```bash
# Run the bot
python main.py
```

The bot will:
- Initialize all components
- Connect to Binance (testnet by default)
- Join your Matrix room
- Send a startup message
- Begin monitoring markets

## 🎛️ Matrix Chat Commands

Once running, control the bot via Matrix chat:

### Basic Commands
- `/help` - Show all available commands
- `/portfolio` - Display current portfolio status
- `/recent` - Show recent trades and performance

### Trading Control
- `/mode <aggressive|conservative|paper|halt>` - Set trading mode
- `/auto on size<=<usd> confidence>=<x>` - Enable auto-approval
- `/auto off` - Disable auto-approval
- `/approve <decision_id>` - Approve a trade proposal
- `/reject <decision_id>` - Reject a trade proposal

### Risk Management
- `/set risk_per_trade <pct>` - Set risk per trade (e.g., 0.02 = 2%)
- `/set max_daily_loss <pct>` - Set daily loss limit
- `/set heat_cap <pct>` - Set portfolio heat cap
- `/flatten` - Emergency: close all positions
- `/halt` - Stop all trading immediately

### Daily Briefings
- `/brief` - Generate immediate market brief
- `/brief schedule <HH:MM>` - Set daily brief time
- `/brief symbols add <SYMBOL>` - Add symbol to brief coverage
- `/brief symbols list` - Show covered symbols

### Profile Management
- `/profile load challenge` - Load high-risk challenge profile
- `/profile load conservative` - Load low-risk conservative profile

## 🧠 How It Works

### 1. Market Scanning
The bot continuously scans markets every 30 minutes:
- Identifies top symbols by volume and liquidity
- Filters by spread and trading conditions
- Maintains an allowlist of tradeable pairs

### 2. Data Collection
For each symbol:
- **Real-time prices**: WebSocket feeds for bid/ask, trades
- **OHLCV data**: Rolling windows (1m, 5m, 1h candles)
- **Technical indicators**: RSI, MACD, Bollinger Bands, EMAs, ATR
- **Market metrics**: Volatility, funding rates, fees
- **News sentiment**: Optional Tavily integration for news analysis

### 3. AI Decision Making
The LLM coordinator:
- Aggregates all market data into structured input
- Sends sanitized, numeric-only data to AI models
- Uses self-consistency (multiple samples) or ensemble (multiple models)
- Validates responses and parses structured decisions
- Generates human-readable summaries

### 4. Risk Management
Multi-layered risk system:
- **Per-trade limits**: Max 1-5% risk per position
- **Portfolio heat**: Total exposure cap (5-25%)
- **Daily loss limits**: Stop trading if daily loss exceeds threshold
- **Leverage limits**: Maximum leverage multiplier
- **Symbol allowlists**: Only trade vetted, liquid pairs

### 5. Human Oversight
- All trades require approval unless auto-approved
- Rich Matrix notifications with reasoning
- Emergency controls (halt, flatten)
- Real-time portfolio monitoring

### 6. Execution
- Binance order constraints validation
- Tick size and lot size rounding
- Order type selection (market/limit)
- OCO orders for stop loss and take profit
- Fill tracking and P&L calculation

## 📊 Trading Modes

### Conservative Mode (Default)
- **Risk per trade**: 0.5%
- **Daily loss limit**: 2%
- **Portfolio heat**: 5%
- **Leverage**: 2x max
- **Auto-approval**: $50 max, 90%+ confidence
- **Focus**: Capital preservation, high-quality setups

### Aggressive Mode
- **Risk per trade**: 2-5%
- **Daily loss limit**: 10-15%
- **Portfolio heat**: 15-25%
- **Leverage**: 5-10x max
- **Auto-approval**: $200+ max, 70%+ confidence
- **Focus**: Higher returns, more frequent trading

### Challenge Mode
- **Risk per trade**: 5%
- **Daily loss limit**: 15%
- **Portfolio heat**: 25%
- **Leverage**: 10x max
- **Double confirmation**: Required
- **Focus**: High-risk trading for experienced users

### Paper Trading Mode
- All features enabled but no real trades
- Perfect for testing strategies and settings
- Full simulation with realistic fills and slippage

## 🔐 Security & Safety

### API Security
- API keys stored in environment variables only
- Testnet enabled by default
- No withdrawal permissions required
- Read-only portfolio access where possible

### Risk Controls
- Multiple layers of position sizing validation
- Hard stops on daily losses
- Emergency flatten functionality
- Administrative access controls via Matrix

### Data Protection
- Local SQLite database by default
- Structured logging without sensitive data
- Optional encryption for database files

## 📈 Performance Monitoring

### Real-time Metrics
- Portfolio balance and P&L
- Daily/weekly/monthly performance
- Win rate and average R/R
- Maximum drawdown tracking
- Heat and leverage monitoring

### Logging
- Structured JSON logs with trace IDs
- Rotating file logs
- Configurable log levels
- Error tracking and alerting

### Database Storage
- Trade history and outcomes
- Decision logs with reasoning
- Risk events and violations
- Performance analytics
- News and sentiment data

## 🧪 Testing & Development

### Testnet Trading
Always start with testnet:
```env
BINANCE_USE_TESTNET=true
```

### Paper Trading
Enable paper mode for risk-free testing:
```bash
# In Matrix chat
/mode paper
```

### Backtesting
```bash
# Run backtests on historical data
python -m backtester.engine --symbol BTCUSDT --days 30
```

### Unit Tests
```bash
# Run test suite
python -m pytest tests/
```

## 🔧 Configuration Files

### Market Settings (`config/market_settings.yaml`)
```yaml
risk:
  risk_per_trade: 0.01      # 1% per trade
  max_daily_loss: 0.05      # 5% daily limit
  heat_cap: 0.1             # 10% portfolio heat
  leverage_cap: 3           # 3x max leverage

market:
  top_symbols_count: 5      # Number of symbols to trade
  scan_interval_minutes: 30 # Market scan frequency
  
briefing:
  time_local: "07:00"       # Daily brief time
  timezone: "Europe/Ljubljana"
  symbols: ["BTCUSDT", "ETHUSDT"]
```

### Risk Profiles
- `config/risk_profiles/conservative.yaml` - Low-risk settings
- `config/risk_profiles/challenge.yaml` - High-risk settings

## 🚨 Troubleshooting

### Common Issues

**1. Bot won't start**
```bash
# Check logs
tail -f logs/trading_bot.log

# Verify configuration
python -c "from utils.pydantic_models import AppConfig; print('Config OK')"
```

**2. Matrix connection failed**
- Verify homeserver URL and access token
- Check room ID format (starts with !)
- Ensure bot account has room permissions

**3. Binance API errors**
- Check API key permissions
- Verify testnet vs mainnet settings
- Review rate limits and restrictions

**4. No trading decisions**
- Check LLM API keys and credits
- Verify market scanner is finding symbols
- Review risk manager logs for blocks

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py
```

### Health Checks
The bot performs automatic health checks:
- Binance API connectivity
- WebSocket connection status
- Matrix client status
- Database connectivity
- Component initialization

## 📚 Advanced Features

### Custom Indicators
Extend `analyzers/technical_analysis.py`:
```python
def custom_indicator(ohlcv_data):
    # Your indicator logic
    return indicator_value
```

### Strategy Development
Create new strategies in `strategies/`:
```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    async def generate_signal(self, market_data):
        # Your strategy logic
        return signal
```

### Notification Channels
Extend `utils/notifications.py` for:
- Discord webhooks
- Telegram bots
- Email alerts
- Custom API integrations

## 📞 Support & Community

### Getting Help
1. Check this README thoroughly
2. Review log files for error details
3. Test with paper trading first
4. Start with conservative settings

### Best Practices
- **Start small**: Use testnet and conservative settings
- **Monitor closely**: Watch bot behavior for first few days
- **Risk management**: Never risk more than you can afford to lose
- **Regular updates**: Keep dependencies and strategies updated
- **Backup data**: Regular database backups for trade history

### Disclaimer
This trading bot is provided for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and never trade with funds you cannot afford to lose.

## 🔄 Updates & Maintenance

### Regular Tasks
- Monitor log files for errors
- Update API keys before expiration
- Review and adjust risk parameters
- Backup trade data regularly
- Update dependencies monthly

### Version Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart bot
python main.py
```

---

**Happy Trading! 🚀**

Remember: Start with paper trading, use testnet, and always prioritize risk management over profits.