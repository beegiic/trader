#!/usr/bin/env python3
"""
Advanced AI Trading Bot
Main entry point for the trading bot with async task orchestration.
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

# Import core modules
from utils.logging_config import init_logging, get_logger
from utils.database import init_database, get_db_manager
from utils.pydantic_models import AppConfig, TradingMode
from core.binance_client import BinanceClient
from core.binance_ws import BinanceWebSocket
from core.order_constraints import OrderConstraints
from core.matrix_client import MatrixClient
from core.llm_coordinator import LLMCoordinator
from strategies.risk_manager import RiskManager
from strategies.trading_engine import TradingEngine
from analyzers.market_scanner import MarketScanner, ScanConfig
from analyzers.technical_analysis import TechnicalAnalyzer


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self):
        self.logger = get_logger("TradingBot")
        self.config: Optional[AppConfig] = None
        self.running = False
        
        # Core components
        self.binance_client: Optional[BinanceClient] = None
        self.binance_ws: Optional[BinanceWebSocket] = None
        self.matrix_client: Optional[MatrixClient] = None
        self.llm_coordinator: Optional[LLMCoordinator] = None
        self.risk_manager: Optional[RiskManager] = None
        self.order_constraints: Optional[OrderConstraints] = None
        
        # Trading engine components
        self.market_scanner: Optional[MarketScanner] = None
        self.technical_analyzer: Optional[TechnicalAnalyzer] = None
        self.trading_engine: Optional[TradingEngine] = None
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # State
        self.start_time = datetime.now(timezone.utc)
        self.last_heartbeat = datetime.now(timezone.utc)
        
        self.logger.info("Trading bot initialized")
    
    async def initialize(self):
        """Initialize all bot components."""
        try:
            # Load configuration
            self.config = AppConfig.load_from_env()
            
            # Initialize database
            init_database()
            
            # Initialize order constraints
            self.order_constraints = OrderConstraints()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                risk_config=self.config.risk,
                trading_mode=TradingMode.CONSERVATIVE  # Default mode
            )
            
            # Initialize Binance client
            self.binance_client = BinanceClient(
                api_key=self.config.binance.api_key,
                secret_key=self.config.binance.secret_key,
                testnet=self.config.binance.use_testnet,
                futures=self.config.binance.futures,
                order_constraints=self.order_constraints,
                risk_manager=self.risk_manager
            )
            
            # Initialize Binance WebSocket
            self.binance_ws = BinanceWebSocket(
                testnet=self.config.binance.use_testnet,
                futures=self.config.binance.futures
            )
            
            # Initialize LLM coordinator
            self.llm_coordinator = LLMCoordinator(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                grok_api_key=os.getenv("GROK_API_KEY")
            )
            
            # Initialize Matrix client
            self.matrix_client = MatrixClient(
                homeserver=self.config.matrix.homeserver,
                access_token=self.config.matrix.access_token,
                room_id=self.config.matrix.room_id,
                admin_users=self.config.matrix.admin_users
            )
            
            # Initialize trading engine components
            await self._initialize_trading_engine()
            
            # Register Matrix command handlers
            await self._setup_matrix_handlers()
            
            # Initialize Binance client
            await self.binance_client.initialize()
            
            # Connect to services
            await self.binance_ws.connect()
            await self.matrix_client.connect()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize bot", error=str(e))
            raise
    
    async def _initialize_trading_engine(self):
        """Initialize the trading engine components"""
        try:
            # Initialize market scanner
            scan_config = ScanConfig(
                idle_scan_interval=300,    # 5 minutes when idle
                active_scan_interval=20,   # 20 seconds when trades active
                min_24h_volume=1000000,   # $1M minimum volume
                max_symbols=10            # Track top 10 symbols
            )
            self.market_scanner = MarketScanner(self.binance_client, scan_config)
            
            # Initialize technical analyzer
            self.technical_analyzer = TechnicalAnalyzer(
                self.binance_client, 
                self.llm_coordinator,
                self.risk_manager
            )
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(
                binance_client=self.binance_client,
                risk_manager=self.risk_manager,
                matrix_client=self.matrix_client,
                scanner=self.market_scanner,
                analyzer=self.technical_analyzer
            )
            
            self.logger.info("Trading engine components initialized")
            
        except Exception as e:
            self.logger.error("Failed to initialize trading engine", error=str(e))
            raise
    
    async def _setup_matrix_handlers(self):
        """Set up Matrix command handlers."""
        
        async def handle_portfolio_command(command):
            """Handle /portfolio command."""
            try:
                # Get portfolio status
                account_info = await self.binance_client.get_account()
                risk_summary = self.risk_manager.get_risk_summary()
                
                # Format portfolio message
                balance = float(account_info.get('totalWalletBalance', 0)) if self.config.binance.futures else sum(
                    float(asset['free']) + float(asset['locked']) 
                    for asset in account_info['balances'] 
                    if asset['asset'] == 'USDT'
                )
                
                message = f"""
📊 **PORTFOLIO STATUS**

💰 **Balance:** ${balance:.2f}
📈 **Daily P&L:** ${risk_summary['daily_pnl']:.2f}
🔥 **Heat:** {risk_summary.get('current_heat', 0):.1%}
🎯 **Mode:** {risk_summary['trading_mode'].title()}

⚙️ **Risk Limits:**
• Per Trade: {risk_summary['effective_limits']['risk_per_trade']:.1%}
• Daily Loss: {risk_summary['effective_limits']['max_daily_loss']:.1%}
• Heat Cap: {risk_summary['effective_limits']['heat_cap']:.1%}

📊 **Today:** {risk_summary['daily_trades_count']} trades
                """
                
                await self.matrix_client.send_message(message)
                
            except Exception as e:
                self.logger.error("Error handling portfolio command", error=str(e))
                await self.matrix_client.send_message(f"❌ Error retrieving portfolio: {str(e)}")
        
        async def handle_brief_command(command):
            """Handle /brief command."""
            try:
                await self.matrix_client.send_message("📰 Generating market brief...")
                
                # Get market data for briefing
                brief_data = await self._generate_market_brief()
                await self.matrix_client.send_message(brief_data)
                
            except Exception as e:
                self.logger.error("Error generating brief", error=str(e))
                await self.matrix_client.send_message(f"❌ Error generating brief: {str(e)}")
        
        async def handle_flatten_command(command):
            """Handle /flatten command."""
            try:
                self.risk_manager.flatten_all_positions()
                await self.matrix_client.send_message("🚨 **FLATTEN ALL** command executed. Check positions manually.")
            except Exception as e:
                await self.matrix_client.send_message(f"❌ Error executing flatten: {str(e)}")
        
        async def handle_approve_command(command):
            """Handle /approve command."""
            if not self.trading_engine:
                await self.matrix_client.send_message("❌ Trading engine not initialized")
                return
            
            if not command.args:
                await self.matrix_client.send_message("❌ Usage: /approve <proposal_id>")
                return
            
            proposal_id = command.args[0]
            success = await self.trading_engine.approve_proposal(proposal_id, command.sender)
        
        async def handle_reject_command(command):
            """Handle /reject command."""
            if not self.trading_engine:
                await self.matrix_client.send_message("❌ Trading engine not initialized")
                return
            
            if not command.args:
                await self.matrix_client.send_message("❌ Usage: /reject <proposal_id>")
                return
            
            proposal_id = command.args[0]
            success = await self.trading_engine.reject_proposal(proposal_id, command.sender)
        
        async def handle_status_command(command):
            """Handle /status command."""
            try:
                if not self.trading_engine:
                    await self.matrix_client.send_message("❌ Trading engine not initialized")
                    return
                
                status = self.trading_engine.get_status()
                scanner_status = self.market_scanner.get_scanner_status()
                
                message = f"""
📊 **TRADING ENGINE STATUS**

🔄 **Engine:** {'Running' if status['running'] else 'Stopped'}
📈 **Active Trades:** {status['active_trades']}/{status['max_concurrent_trades']}
⏳ **Pending Proposals:** {status['pending_proposals']}

🔍 **Scanner:** {scanner_status['mode'].title()} Mode
📊 **Symbols:** {scanner_status['candidates_count']} candidates
⏱️ **Next Scan:** {scanner_status.get('next_scan_seconds', 0)}s
🎯 **Active Symbols:** {', '.join(status['active_symbols']) or 'None'}

💰 **Account Balance:** ${self.risk_manager.get_risk_summary().get('balance', 0):.2f}
🔥 **Heat:** {self.risk_manager.get_risk_summary().get('current_heat', 0):.1%}
                """
                
                await self.matrix_client.send_message(message)
                
            except Exception as e:
                await self.matrix_client.send_message(f"❌ Error getting status: {str(e)}")
        
        # Register handlers
        self.matrix_client.register_command_handler('portfolio', handle_portfolio_command)
        self.matrix_client.register_command_handler('brief', handle_brief_command)
        self.matrix_client.register_command_handler('flatten', handle_flatten_command)
        self.matrix_client.register_command_handler('approve', handle_approve_command)
        self.matrix_client.register_command_handler('reject', handle_reject_command)
        self.matrix_client.register_command_handler('status', handle_status_command)
    
    async def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Bot is already running")
            return
        
        self.running = True
        self.logger.info("Starting trading bot...")
        
        try:
            # Start background tasks
            await self._start_background_tasks()
            
            # Start trading engine
            if self.trading_engine:
                await self.trading_engine.start()
                self.logger.info("Trading engine started")
            
            # Send startup message
            await self.matrix_client.send_message(
                f"🚀 **Trading Bot Started**\n"
                f"📅 **Time:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                f"🔧 **Mode:** {self.risk_manager.trading_mode.value.title()}\n"
                f"🌐 **Testnet:** {'Yes' if self.config.binance.use_testnet else 'No'}\n"
                f"📊 **Futures:** {'Yes' if self.config.binance.futures else 'No'}\n"
                f"🤖 **Trading Engine:** {'Active' if self.trading_engine and self.trading_engine.is_running else 'Inactive'}"
            )
            
            self.logger.info("Trading bot started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start bot", error=str(e))
            await self.stop()
            raise
    
    async def _start_background_tasks(self):
        """Start all background tasks."""
        
        # Heartbeat task
        self.tasks.append(asyncio.create_task(self._heartbeat_task()))
        
        # WebSocket connection monitor
        self.tasks.append(asyncio.create_task(self._monitor_websocket()))
        
        # Health check task
        self.tasks.append(asyncio.create_task(self._health_check_task()))
        
        # Database cleanup task
        self.tasks.append(asyncio.create_task(self._cleanup_task()))
        
        self.logger.info("Background tasks started", task_count=len(self.tasks))
    
    async def _heartbeat_task(self):
        """Periodic heartbeat task."""
        while self.running:
            try:
                self.last_heartbeat = datetime.now(timezone.utc)
                
                # Log periodic status
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                self.logger.debug("Heartbeat", uptime_seconds=int(uptime))
                
                await asyncio.sleep(60)  # Every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Heartbeat task error", error=str(e))
                await asyncio.sleep(30)
    
    async def _monitor_websocket(self):
        """Monitor WebSocket connection health."""
        while self.running:
            try:
                if self.binance_ws and not self.binance_ws.is_connected:
                    self.logger.warning("WebSocket disconnected, attempting reconnection...")
                    try:
                        await self.binance_ws.connect()
                        await self.matrix_client.send_alert(
                            "websocket", 
                            "WebSocket reconnected successfully"
                        )
                    except Exception as e:
                        await self.matrix_client.send_alert(
                            "websocket", 
                            f"WebSocket reconnection failed: {str(e)}",
                            urgent=True
                        )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("WebSocket monitor error", error=str(e))
                await asyncio.sleep(60)
    
    async def _health_check_task(self):
        """Periodic health checks."""
        while self.running:
            try:
                # Check Binance API health
                binance_healthy = await self.binance_client.health_check()
                
                # Check WebSocket health
                ws_health = self.binance_ws.health_check()
                
                # Check Matrix connection
                matrix_healthy = self.matrix_client.is_connected
                
                # Check database
                db_healthy = get_db_manager().health_check()
                
                if not all([binance_healthy, ws_health['connected'], matrix_healthy, db_healthy]):
                    unhealthy_services = []
                    if not binance_healthy:
                        unhealthy_services.append("Binance API")
                    if not ws_health['connected']:
                        unhealthy_services.append("WebSocket")
                    if not matrix_healthy:
                        unhealthy_services.append("Matrix")
                    if not db_healthy:
                        unhealthy_services.append("Database")
                    
                    self.logger.warning("Health check failed", unhealthy_services=unhealthy_services)
                    
                    await self.matrix_client.send_alert(
                        "health_check",
                        f"Unhealthy services: {', '.join(unhealthy_services)}",
                        urgent=True
                    )
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check error", error=str(e))
                await asyncio.sleep(300)
    
    async def _cleanup_task(self):
        """Periodic database cleanup."""
        while self.running:
            try:
                # Run cleanup every 6 hours
                await asyncio.sleep(6 * 3600)
                
                from utils.database import cleanup_old_data, get_db_session
                
                with get_db_session() as session:
                    cleanup_old_data(session, days_to_keep=30)
                
                self.logger.info("Database cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cleanup task error", error=str(e))
    
    async def stop(self):
        """Stop the trading bot gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        try:
            # Send shutdown message
            if self.matrix_client and self.matrix_client.is_connected:
                await self.matrix_client.send_message("🛑 **Trading Bot Shutting Down**")
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
                self.logger.info("Trading engine stopped")
            
            # Cancel background tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close connections
            if self.binance_ws:
                await self.binance_ws.disconnect()
            
            if self.matrix_client:
                await self.matrix_client.disconnect()
            
            self.logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        return {
            'running': self.running,
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': int(uptime),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'tasks_count': len(self.tasks),
            'components': {
                'binance_client': self.binance_client is not None,
                'binance_ws': self.binance_ws.is_connected if self.binance_ws else False,
                'matrix_client': self.matrix_client.is_connected if self.matrix_client else False,
                'llm_coordinator': self.llm_coordinator is not None,
                'risk_manager': self.risk_manager is not None
            }
        }

    async def _generate_market_brief(self) -> str:
        """Generate market brief with current data."""
        try:
            # Get symbols from config
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
            brief_parts = ["📊 **MARKET BRIEF**\n"]
            
            # Get 24hr ticker data for each symbol
            for symbol in symbols:
                try:
                    self.logger.info(f"Fetching ticker for {symbol}")
                    ticker = await self.binance_client.get_24hr_ticker(symbol)
                    self.logger.info(f"Ticker result for {symbol}: {ticker is not None}")
                    
                    if ticker:
                        self.logger.info(f"Ticker data: {ticker}")
                        price = float(ticker['lastPrice'])
                        change = float(ticker['priceChangePercent'])
                        volume = float(ticker['volume'])
                        
                        # Format change with emoji
                        if change > 0:
                            change_str = f"📈 +{change:.2f}%"
                        else:
                            change_str = f"📉 {change:.2f}%"
                        
                        # Format volume in millions/billions
                        if volume >= 1e9:
                            volume_str = f"{volume/1e9:.1f}B"
                        elif volume >= 1e6:
                            volume_str = f"{volume/1e6:.1f}M"
                        else:
                            volume_str = f"{volume:.0f}"
                        
                        brief_parts.append(
                            f"• **{symbol.replace('USDT', '')}**: ${price:.4f} {change_str} (Vol: {volume_str})"
                        )
                    else:
                        self.logger.warning(f"No ticker data returned for {symbol}")
                        brief_parts.append(f"• **{symbol.replace('USDT', '')}**: Data unavailable")
                        
                except Exception as e:
                    self.logger.error(f"Exception getting ticker for {symbol}", error=str(e), exc_info=True)
                    brief_parts.append(f"• **{symbol.replace('USDT', '')}**: Data unavailable")
            
            # Add account summary
            try:
                self.logger.info("Fetching account info")
                account_info = await self.binance_client.get_account()
                self.logger.info(f"Account result: {account_info is not None}")
                
                if account_info:
                    self.logger.info(f"Account keys: {list(account_info.keys())}")
                    if self.config.binance.futures:
                        balance = float(account_info.get('totalWalletBalance', 0))
                        self.logger.info(f"Futures balance: {balance}")
                    else:
                        usdt_balance = next(
                            (float(asset['free']) + float(asset['locked']) 
                             for asset in account_info['balances'] 
                             if asset['asset'] == 'USDT'), 0
                        )
                        balance = usdt_balance
                        self.logger.info(f"Spot balance: {balance}")
                    
                    brief_parts.append(f"\n💰 **Account Balance**: ${balance:.2f}")
                    
                    # Add risk summary
                    risk_summary = self.risk_manager.get_risk_summary()
                    brief_parts.append(f"🔥 **Heat**: {risk_summary.get('current_heat', 0):.1%}")
                    brief_parts.append(f"📊 **Daily P&L**: ${risk_summary['daily_pnl']:.2f}")
                else:
                    self.logger.warning("Account info is None")
                    brief_parts.append("\n💰 **Account**: Data unavailable")
                
            except Exception as e:
                self.logger.error("Failed to get account info for brief", error=str(e), exc_info=True)
                brief_parts.append("\n💰 **Account**: Data unavailable")
            
            # Add timestamp
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            brief_parts.append(f"\n⏰ **Updated**: {now.strftime('%H:%M:%S')} UTC")
            
            return "\n".join(brief_parts)
            
        except Exception as e:
            self.logger.error("Error generating market brief", error=str(e))
            return f"❌ **Error generating brief**: {str(e)}"


# Global bot instance
bot: Optional[TradingBot] = None


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        if bot and bot.running:
            # Create new event loop for shutdown if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(bot.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    global bot
    
    # Load environment variables
    load_dotenv()
    
    # Initialize logging
    init_logging()
    logger = get_logger("main")
    
    logger.info("Starting AI Trading Bot...")
    
    try:
        # Set up signal handlers
        setup_signal_handlers()
        
        # Create and initialize bot
        bot = TradingBot()
        await bot.initialize()
        
        # Start bot
        await bot.start()
        
        # Keep running
        while bot.running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error("Fatal error in main", error=str(e))
    finally:
        if bot:
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())