"""
Trading Strategy Engine
Prepares trade proposals and manages trade execution workflow
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

from utils.logging_config import LoggerMixin
from utils.database import get_db_session, get_db_manager
from analyzers.market_scanner import MarketScanner, MarketCandidate
from analyzers.technical_analysis import TechnicalAnalyzer, TradingSignal, SignalDirection, SignalStrength
from strategies.risk_manager import RiskManager
from core.binance_client import BinanceClient
from core.matrix_client import MatrixClient


class TradeStatus(Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradeProposal:
    """Trade proposal awaiting approval"""
    proposal_id: str
    symbol: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    
    # Entry details
    entry_price: float
    entry_type: OrderType
    position_size_usdt: float
    position_size_base: float
    leverage: int
    
    # Risk management
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    risk_amount: float
    
    # Analysis
    signal: TradingSignal
    reasoning: str
    technical_summary: str
    
    # Metadata
    created_at: datetime
    expires_at: datetime
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    status: TradeStatus = TradeStatus.PENDING_APPROVAL


@dataclass
class ActiveTrade:
    """Active trade being managed"""
    trade_id: str
    proposal_id: str
    symbol: str
    direction: SignalDirection
    entry_price: float
    position_size: float
    leverage: int
    stop_loss: float
    take_profit: float
    
    # Orders (with defaults)
    entry_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
    
    # Risk management (with defaults)
    trailing_stop: bool = False
    trailing_stop_distance: float = 0.0
    
    # Status (with defaults)
    status: TradeStatus = TradeStatus.ACTIVE
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # P&L tracking (with defaults)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class TradingEngine(LoggerMixin):
    """
    Main trading strategy engine that:
    1. Coordinates market scanning and analysis
    2. Generates trade proposals
    3. Manages approval workflow
    4. Executes approved trades
    5. Manages active positions
    """
    
    def __init__(self, binance_client: BinanceClient, risk_manager: RiskManager, 
                 matrix_client: MatrixClient, scanner: MarketScanner, analyzer: TechnicalAnalyzer):
        super().__init__()
        
        self.binance_client = binance_client
        self.risk_manager = risk_manager
        self.matrix_client = matrix_client
        self.scanner = scanner
        self.analyzer = analyzer
        
        # State tracking
        self.pending_proposals: Dict[str, TradeProposal] = {}
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.processed_signals: Set[str] = set()  # Prevent duplicate signals
        
        # Engine control
        self.is_running = False
        self.engine_task: Optional[asyncio.Task] = None
        self.trade_monitor_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_concurrent_trades = 5
        self.proposal_expiry_hours = 2
        self.signal_cooldown_minutes = 15  # Prevent spam
        
        self.logger.info("Trading engine initialized")
    
    async def start(self):
        """Start the trading engine"""
        if self.is_running:
            self.logger.warning("Trading engine already running")
            return
        
        self.is_running = True
        
        # Start engine tasks
        self.engine_task = asyncio.create_task(self._strategy_loop())
        self.trade_monitor_task = asyncio.create_task(self._trade_monitor_loop())
        
        # Start market scanner
        await self.scanner.start()
        
        self.logger.info("Trading engine started")
    
    async def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        
        # Stop tasks
        if self.engine_task:
            self.engine_task.cancel()
        if self.trade_monitor_task:
            self.trade_monitor_task.cancel()
        
        # Stop scanner
        await self.scanner.stop()
        
        self.logger.info("Trading engine stopped")
    
    async def _strategy_loop(self):
        """Main strategy loop - generates trade proposals"""
        while self.is_running:
            try:
                await self._generate_trade_proposals()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in strategy loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _trade_monitor_loop(self):
        """Monitor active trades and manage positions"""
        while self.is_running:
            try:
                await self._monitor_active_trades()
                await self._cleanup_expired_proposals()
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in trade monitor loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _generate_trade_proposals(self):
        """Generate new trade proposals from market analysis"""
        try:
            self.logger.info("Checking for new trade opportunities", 
                           active_trades=len(self.active_trades),
                           max_trades=self.max_concurrent_trades,
                           pending_proposals=len(self.pending_proposals),
                           action="proposal_generation_start")
            
            # Check if we can create more trades
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.logger.debug("Max concurrent trades reached, skipping proposal generation",
                                max_trades=self.max_concurrent_trades)
                return
            
            if len(self.pending_proposals) >= 3:  # Max pending proposals
                self.logger.debug("Max pending proposals reached, skipping generation",
                                max_pending=3)
                return
            
            # Get market candidates
            candidates = self.scanner.get_candidates()
            if not candidates:
                self.logger.debug("No market candidates available from scanner")
                return
            
            self.logger.info("Market candidates available for analysis", 
                           candidate_count=len(candidates),
                           top_candidates=list(candidates.keys())[:5])
            
            # Analyze top candidates
            top_symbols = list(candidates.keys())[:5]  # Top 5 by volume
            
            for symbol in top_symbols:
                self.logger.debug("Evaluating candidate symbol", symbol=symbol)
                
                # Skip if we already have a trade or proposal for this symbol
                if (any(t.symbol == symbol for t in self.active_trades.values()) or
                    any(p.symbol == symbol for p in self.pending_proposals.values())):
                    self.logger.debug("Symbol already has active trade or pending proposal, skipping", 
                                    symbol=symbol)
                    continue
                
                # Analyze symbol
                self.logger.info("Sending symbol for technical analysis", 
                               symbol=symbol,
                               action="requesting_analysis")
                signal = await self.analyzer.analyze_symbol(symbol)
                
                if signal:
                    self.logger.info(
                        "Analysis completed for symbol",
                        symbol=symbol,
                        confidence=signal.confidence,
                        direction=signal.direction.value,
                    )
                else:
                    self.logger.info("No trading signal generated for symbol", 
                                   symbol=symbol,
                                   reason="analysis_returned_none")
                
                if signal:
                    # EV-based gating using RiskManager
                    ok, metrics = self.risk_manager.passes_ev_gate(
                        signal,
                        getattr(signal, 'entry_price', None),
                        getattr(signal, 'stop_loss', None),
                        getattr(signal, 'take_profit', None),
                    )
                    if not ok:
                        self.logger.info("EV gate rejected", symbol=symbol, **metrics)
                        continue
                    
                    # Create signal key to prevent duplicates
                    signal_key = f"{symbol}_{signal.direction.value}_{int(signal.generated_at.timestamp())}"
                    
                    if signal_key in self.processed_signals:
                        continue
                    
                    # Create trade proposal
                    proposal = await self._create_trade_proposal(signal)
                    
                    if proposal:
                        self.pending_proposals[proposal.proposal_id] = proposal
                        self.processed_signals.add(signal_key)
                        
                        # Send proposal to Matrix for approval
                        await self._send_proposal_to_matrix(proposal)
                        
                        self.logger.info("Trade proposal created", 
                                       proposal_id=proposal.proposal_id,
                                       symbol=proposal.symbol,
                                       direction=proposal.direction.value,
                                       confidence=proposal.confidence)
        
        except Exception as e:
            self.logger.error("Error generating trade proposals", error=str(e))
    
    async def _create_trade_proposal(self, signal: TradingSignal) -> Optional[TradeProposal]:
        """Create a trade proposal from a trading signal"""
        try:
            # Get account info for position sizing
            account_info = await self.binance_client.get_account()
            if not account_info:
                return None
            
            balance = float(account_info.get('totalWalletBalance', 0))
            
            # Calculate position size based on risk management
            risk_per_trade = self.risk_manager.get_effective_risk_per_trade()  # This will be 0.10 (10%)
            
            # For challenge mode: use 10% of portfolio as position size
            position_size_usdt = balance * risk_per_trade
            
            # Calculate actual risk amount based on stop distance
            stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            risk_amount = position_size_usdt * stop_distance
            
            # Calculate base currency amount
            position_size_base = position_size_usdt / signal.entry_price
            
            # Determine leverage (based on signal strength and risk settings)
            leverage = self._calculate_leverage(signal.strength, signal.confidence)
            
            # Create proposal
            proposal = TradeProposal(
                proposal_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                direction=signal.direction,
                strength=signal.strength,
                confidence=signal.confidence,
                entry_price=signal.entry_price,
                entry_type=OrderType.MARKET,  # For now, use market orders
                position_size_usdt=position_size_usdt,
                position_size_base=position_size_base,
                leverage=leverage,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_reward_ratio=signal.risk_reward_ratio,
                risk_amount=risk_amount,
                signal=signal,
                reasoning=signal.reasoning,
                technical_summary=signal.technical_summary,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.proposal_expiry_hours)
            )
            
            return proposal
            
        except Exception as e:
            self.logger.error("Error creating trade proposal", symbol=signal.symbol, error=str(e))
            return None
    
    def _calculate_leverage(self, strength: SignalStrength, confidence: float) -> int:
        """Calculate appropriate leverage based on signal quality"""
        # For challenge mode - use higher base leverage
        base_leverage = {
            SignalStrength.WEAK: 8,
            SignalStrength.MODERATE: 12,
            SignalStrength.STRONG: 16,
            SignalStrength.VERY_STRONG: 20
        }
        
        leverage = base_leverage.get(strength, 8)
        
        # Adjust based on confidence
        if confidence >= 0.9:
            leverage = 20  # Max leverage for highest confidence
        elif confidence >= 0.8:
            leverage = min(leverage + 4, 20)
        elif confidence < 0.7:
            leverage = max(leverage - 4, 5)  # Minimum 5x
        
        # Cap at risk manager limits
        max_leverage = self.risk_manager.get_effective_leverage_cap()
        return min(leverage, max_leverage)
    
    async def _send_proposal_to_matrix(self, proposal: TradeProposal):
        """Send trade proposal to Matrix for approval"""
        try:
            # Format proposal message
            direction_emoji = "🟢" if proposal.direction == SignalDirection.LONG else "🔴"
            strength_emoji = {
                SignalStrength.WEAK: "⚪",
                SignalStrength.MODERATE: "🟡", 
                SignalStrength.STRONG: "🟠",
                SignalStrength.VERY_STRONG: "🔴"
            }
            
            message = f"""
{direction_emoji} **TRADE PROPOSAL** {strength_emoji[proposal.strength]}

**{proposal.symbol}** | {proposal.direction.value.upper()} | {proposal.leverage}x
💪 **Strength:** {proposal.strength.value.title()} ({proposal.confidence:.1%} confidence)

📊 **Entry:** ${proposal.entry_price:.4f}
🛑 **Stop Loss:** ${proposal.stop_loss:.4f}
🎯 **Take Profit:** ${proposal.take_profit:.4f}
⚖️ **R:R Ratio:** 1:{proposal.risk_reward_ratio:.1f}

💰 **Position:** ${proposal.position_size_usdt:.2f} ({proposal.position_size_base:.4f} {proposal.symbol.replace('USDT', '')})
🎲 **Risk:** ${proposal.risk_amount:.2f}

📈 **Analysis:** {proposal.technical_summary}

🔍 **Reasoning:** {proposal.reasoning[:200]}...

⏰ **Expires:** {proposal.expires_at.strftime('%H:%M UTC')}
🆔 **ID:** `{proposal.proposal_id[:8]}`

**Commands:**
• `/approve {proposal.proposal_id[:8]}` - Execute trade
• `/reject {proposal.proposal_id[:8]}` - Reject proposal
• `/details {proposal.proposal_id[:8]}` - Full analysis
            """
            
            await self.matrix_client.send_message(message)
            
        except Exception as e:
            self.logger.error("Error sending proposal to Matrix", proposal_id=proposal.proposal_id, error=str(e))
    
    async def approve_proposal(self, proposal_id: str, user_id: str = None) -> bool:
        """Approve a trade proposal and execute it"""
        try:
            # Find proposal
            proposal = None
            for pid, p in self.pending_proposals.items():
                if pid.startswith(proposal_id) or pid == proposal_id:
                    proposal = p
                    break
            
            if not proposal:
                await self.matrix_client.send_message(f"❌ Proposal {proposal_id[:8]} not found")
                return False
            
            if proposal.status != TradeStatus.PENDING_APPROVAL:
                await self.matrix_client.send_message(f"❌ Proposal {proposal_id[:8]} already processed")
                return False
            
            if datetime.now(timezone.utc) > proposal.expires_at:
                await self.matrix_client.send_message(f"❌ Proposal {proposal_id[:8]} expired")
                return False
            
            # Update proposal status
            proposal.status = TradeStatus.APPROVED
            proposal.approved_at = datetime.now(timezone.utc)
            
            # Execute trade
            success = await self._execute_trade(proposal)
            
            if success:
                await self.matrix_client.send_message(f"✅ Trade {proposal_id[:8]} executed successfully!")
                # Remove from pending
                del self.pending_proposals[proposal.proposal_id]
            else:
                await self.matrix_client.send_message(f"❌ Failed to execute trade {proposal_id[:8]}")
                proposal.status = TradeStatus.CANCELLED
            
            return success
            
        except Exception as e:
            self.logger.error("Error approving proposal", proposal_id=proposal_id, error=str(e))
            await self.matrix_client.send_message(f"❌ Error approving trade {proposal_id[:8]}: {str(e)}")
            return False
    
    async def reject_proposal(self, proposal_id: str, user_id: str = None) -> bool:
        """Reject a trade proposal"""
        try:
            # Find and reject proposal
            proposal = None
            for pid, p in self.pending_proposals.items():
                if pid.startswith(proposal_id) or pid == proposal_id:
                    proposal = p
                    break
            
            if not proposal:
                await self.matrix_client.send_message(f"❌ Proposal {proposal_id[:8]} not found")
                return False
            
            proposal.status = TradeStatus.REJECTED
            proposal.rejected_at = datetime.now(timezone.utc)
            
            await self.matrix_client.send_message(f"❌ Trade {proposal_id[:8]} rejected")
            
            # Remove from pending
            del self.pending_proposals[proposal.proposal_id]
            return True
            
        except Exception as e:
            self.logger.error("Error rejecting proposal", proposal_id=proposal_id, error=str(e))
            return False
    
    async def _execute_trade(self, proposal: TradeProposal) -> bool:
        """Execute an approved trade proposal"""
        try:
            self.logger.info("Executing trade", proposal_id=proposal.proposal_id, symbol=proposal.symbol)
            
            # Set leverage
            await self.binance_client.set_leverage(proposal.symbol, proposal.leverage)
            
            # Place entry order
            side = "BUY" if proposal.direction == SignalDirection.LONG else "SELL"
            
            order_result = await self.binance_client.place_order(
                symbol=proposal.symbol,
                side=side,
                order_type="MARKET",
                quantity=proposal.position_size_base
            )
            
            if not order_result:
                return False
            
            # Create active trade
            trade = ActiveTrade(
                trade_id=str(uuid.uuid4()),
                proposal_id=proposal.proposal_id,
                symbol=proposal.symbol,
                direction=proposal.direction,
                entry_price=float(order_result['avgPrice']) if 'avgPrice' in order_result else proposal.entry_price,
                position_size=proposal.position_size_base,
                leverage=proposal.leverage,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
                entry_order_id=order_result['orderId'],
                opened_at=datetime.now(timezone.utc)
            )
            
            # Place stop loss and take profit orders
            await self._place_risk_management_orders(trade)
            
            # Add to active trades
            self.active_trades[trade.trade_id] = trade
            
            # Update scanner to fast mode
            self.scanner.add_active_symbol(proposal.symbol)
            
            self.logger.info("Trade executed successfully", 
                           trade_id=trade.trade_id, 
                           symbol=proposal.symbol,
                           entry_price=trade.entry_price)
            
            return True
            
        except Exception as e:
            self.logger.error("Error executing trade", proposal_id=proposal.proposal_id, error=str(e))
            return False
    
    async def _place_risk_management_orders(self, trade: ActiveTrade):
        """Place stop loss and take profit orders"""
        try:
            # Determine order sides (opposite of entry for closing)
            if trade.direction == SignalDirection.LONG:
                sl_side = "SELL"
                tp_side = "SELL"
            else:
                sl_side = "BUY"
                tp_side = "BUY"
            
            # Place stop loss
            if trade.stop_loss:
                sl_order = await self.binance_client.place_order(
                    symbol=trade.symbol,
                    side=sl_side,
                    order_type="STOP_MARKET",
                    quantity=trade.position_size,
                    stop_price=trade.stop_loss
                )
                if sl_order:
                    trade.stop_loss_order_id = sl_order['orderId']
            
            # Place take profit
            if trade.take_profit:
                tp_order = await self.binance_client.place_order(
                    symbol=trade.symbol,
                    side=tp_side,
                    order_type="LIMIT",
                    quantity=trade.position_size,
                    price=trade.take_profit
                )
                if tp_order:
                    trade.take_profit_order_id = tp_order['orderId']
            
        except Exception as e:
            self.logger.error("Error placing risk management orders", trade_id=trade.trade_id, error=str(e))
    
    async def _monitor_active_trades(self):
        """Monitor and manage active trades"""
        for trade_id, trade in list(self.active_trades.items()):
            try:
                # Check if trade is still active
                position = await self.binance_client.get_position(trade.symbol)
                
                if not position or float(position.get('positionAmt', 0)) == 0:
                    # Position closed
                    await self._handle_trade_closure(trade)
                    continue
                
                # Update unrealized P&L
                trade.unrealized_pnl = float(position.get('unrealizedProfit', 0))
                
                # Check for trailing stop updates
                if trade.trailing_stop:
                    await self._update_trailing_stop(trade)
                
            except Exception as e:
                self.logger.error("Error monitoring trade", trade_id=trade_id, error=str(e))
    
    async def _handle_trade_closure(self, trade: ActiveTrade):
        """Handle a closed trade"""
        try:
            trade.status = TradeStatus.CLOSED
            trade.closed_at = datetime.now(timezone.utc)
            
            # Get final P&L
            # This would typically come from the position or order history
            
            # Remove from active trades
            del self.active_trades[trade.trade_id]
            
            # Update scanner mode
            self.scanner.remove_active_symbol(trade.symbol)
            
            # Send closure notification
            pnl_emoji = "💰" if trade.unrealized_pnl >= 0 else "💸"
            await self.matrix_client.send_message(
                f"{pnl_emoji} **TRADE CLOSED**\n"
                f"**{trade.symbol}** {trade.direction.value.upper()}\n"
                f"P&L: ${trade.unrealized_pnl:.2f}\n"
                f"Duration: {(trade.closed_at - trade.opened_at).total_seconds() / 3600:.1f} hours"
            )
            
            self.logger.info("Trade closed", 
                           trade_id=trade.trade_id,
                           symbol=trade.symbol,
                           pnl=trade.unrealized_pnl)
            
        except Exception as e:
            self.logger.error("Error handling trade closure", trade_id=trade.trade_id, error=str(e))
    
    async def _cleanup_expired_proposals(self):
        """Clean up expired trade proposals"""
        now = datetime.now(timezone.utc)
        expired = []
        
        for proposal_id, proposal in self.pending_proposals.items():
            if now > proposal.expires_at:
                expired.append(proposal_id)
        
        for proposal_id in expired:
            proposal = self.pending_proposals.pop(proposal_id)
            self.logger.info("Proposal expired", proposal_id=proposal_id, symbol=proposal.symbol)
    
    def get_status(self) -> Dict:
        """Get trading engine status"""
        return {
            'running': self.is_running,
            'active_trades': len(self.active_trades),
            'pending_proposals': len(self.pending_proposals),
            'max_concurrent_trades': self.max_concurrent_trades,
            'scanner_mode': self.scanner.current_mode.value,
            'active_symbols': list(self.scanner.active_symbols)
        }
