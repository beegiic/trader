from datetime import datetime, timezone, timedelta
import os
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from utils.logging_config import LoggerMixin
from utils.pydantic_models import OrderSide, OrderType, RiskConfig, TradingMode


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskCheckResult:
    """Result of a risk management check."""
    approved: bool
    reason: Optional[str] = None
    adjusted_quantity: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.LOW
    metadata: Dict[str, Any] = None


@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics."""
    total_balance: float
    unrealized_pnl: float
    daily_pnl: float
    current_heat: float  # % of portfolio at risk
    max_daily_loss_pct: float
    positions_count: int
    leverage_used: float


class RiskManager(LoggerMixin):
    """
    Comprehensive risk management system with hard gates and position sizing.
    """
    
    def __init__(self, risk_config: RiskConfig, trading_mode: TradingMode = TradingMode.CONSERVATIVE):
        super().__init__()
        
        self.risk_config = risk_config
        self.trading_mode = trading_mode
        
        # EV gating thresholds (env defaults; overridden by profile YAML if present)
        self.MIN_RR = float(os.getenv("MIN_RR", "1.5"))
        self.MIN_EV_BPS = float(os.getenv("MIN_EV_BPS", "8"))
        self.BASE_MIN_CONF = float(os.getenv("BASE_MIN_CONF", "0.60"))
        self.CONF_FLOOR = float(os.getenv("CONF_FLOOR", "0.35"))
        self.RR_CONF_SLOPE = float(os.getenv("RR_CONF_SLOPE", "0.10"))
        self._load_ev_profile_overrides()
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.risk_events: List[Dict[str, Any]] = []
        
        # Emergency states
        self.is_halted = False
        self.halt_reason: Optional[str] = None
        self.max_daily_loss_hit = False
        
        # Symbol allowlist (updated by market scanner)
        self.allowed_symbols: List[str] = []
        
        # Risk limits by mode
        self.mode_multipliers = {
            TradingMode.CONSERVATIVE: {
                'risk_per_trade': 0.8,
                'heat_cap': 0.8,
                'max_daily_loss': 0.8
            },
            TradingMode.AGGRESSIVE: {
                'risk_per_trade': 1.2,
                'heat_cap': 1.2,
                'max_daily_loss': 1.0
            },
            TradingMode.PAPER: {
                'risk_per_trade': 1.0,
                'heat_cap': 1.0,
                'max_daily_loss': 1.0
            },
            TradingMode.HALT: {
                'risk_per_trade': 0.0,
                'heat_cap': 0.0,
                'max_daily_loss': 0.0
            }
        }
        
        self.logger.info("Risk manager initialized", 
                        mode=trading_mode.value,
                        risk_per_trade=risk_config.risk_per_trade,
                        max_daily_loss=risk_config.max_daily_loss,
                        min_rr=self.MIN_RR,
                        min_ev_bps=self.MIN_EV_BPS,
                        base_min_conf=self.BASE_MIN_CONF)

    def _load_ev_profile_overrides(self):
        """Optionally override EV gating thresholds from profile YAMLs."""
        try:
            profile_file = None
            if self.trading_mode == TradingMode.CONSERVATIVE:
                profile_file = "/home/trader/trader/config/risk_profiles/conservative.yaml"
            elif self.trading_mode == TradingMode.AGGRESSIVE:
                # Map aggressive to challenge profile overrides
                profile_file = "/home/trader/trader/config/risk_profiles/challenge.yaml"
            if not profile_file or not os.path.exists(profile_file):
                return
            with open(profile_file, "r") as f:
                data = yaml.safe_load(f) or {}
            # Keys at root if present
            self.MIN_RR = float(data.get("min_rr", self.MIN_RR))
            self.MIN_EV_BPS = float(data.get("min_ev_bps", self.MIN_EV_BPS))
            self.BASE_MIN_CONF = float(data.get("base_min_conf", self.BASE_MIN_CONF))
            self.CONF_FLOOR = float(data.get("conf_floor", self.CONF_FLOOR))
            self.RR_CONF_SLOPE = float(data.get("rr_conf_slope", self.RR_CONF_SLOPE))
            self.logger.info("EV profile overrides loaded",
                            profile=os.path.basename(profile_file),
                            min_rr=self.MIN_RR,
                            min_ev_bps=self.MIN_EV_BPS,
                            base_min_conf=self.BASE_MIN_CONF)
        except Exception as e:
            self.logger.warning("Failed to load EV profile overrides", error=str(e))

    def passes_ev_gate(self, decision, entry_price: float, sl_price: float, tp_price: float) -> tuple[bool, dict]:
        """Evaluate Expected Value gate based on confidence, RR, and EV in bps."""
        try:
            if entry_price is None or sl_price is None or tp_price is None:
                return False, {"reason": "missing_sl_tp"}
            risk_bps = abs(entry_price - sl_price) / max(entry_price, 1e-9) * 1e4
            reward_bps = abs(tp_price - entry_price) / max(entry_price, 1e-9) * 1e4
            rr = (reward_bps / max(risk_bps, 1e-9)) if risk_bps > 0 else 0.0
            p = float(getattr(decision, "confidence", 0.0))
            ev_bps = p * reward_bps - (1 - p) * risk_bps
            # dynamic confidence floor: higher RR -> lower required confidence
            eff_min_conf = max(self.BASE_MIN_CONF - self.RR_CONF_SLOPE * max(0.0, math.log10(max(rr, 1.0))), self.CONF_FLOOR)
            ok = (rr >= self.MIN_RR) and (ev_bps >= self.MIN_EV_BPS) and (p >= eff_min_conf)
            return ok, {"rr": rr, "ev_bps": ev_bps, "eff_min_conf": eff_min_conf, "risk_bps": risk_bps, "reward_bps": reward_bps}
        except Exception as e:
            self.logger.error("EV gate evaluation failed", error=str(e))
            return False, {"reason": "ev_gate_error", "error": str(e)}
    
    async def validate_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float,
        portfolio_balance: float = None,
        current_positions: List[Dict] = None
    ) -> RiskCheckResult:
        """
        Comprehensive order validation through all risk gates.
        """
        
        # Check if halted
        if self.is_halted or self.trading_mode == TradingMode.HALT:
            return RiskCheckResult(
                approved=False,
                reason=f"Trading halted: {self.halt_reason or 'Manual halt'}",
                risk_level=RiskLevel.CRITICAL
            )
        
        # Check symbol allowlist
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            return RiskCheckResult(
                approved=False,
                reason=f"Symbol {symbol} not in allowlist",
                risk_level=RiskLevel.HIGH
            )
        
        # Check daily P&L limits
        daily_check = await self._check_daily_limits(portfolio_balance or 10000)
        if not daily_check.approved:
            return daily_check
        
        # Calculate position risk
        notional_value = quantity * price
        position_risk = await self._calculate_position_risk(
            notional_value, portfolio_balance or 10000, current_positions or []
        )
        
        # Check per-trade risk limit
        trade_check = await self._check_trade_risk_limit(position_risk)
        if not trade_check.approved:
            return trade_check
        
        # Check portfolio heat
        heat_check = await self._check_portfolio_heat(
            position_risk, portfolio_balance or 10000, current_positions or []
        )
        if not heat_check.approved:
            return heat_check
        
        # Check leverage limits
        leverage_check = await self._check_leverage_limit(
            notional_value, portfolio_balance or 10000
        )
        if not leverage_check.approved:
            return leverage_check
        
        # Additional mode-specific checks
        mode_check = await self._check_mode_specific_limits(symbol, notional_value)
        if not mode_check.approved:
            return mode_check
        
        # All checks passed
        self.logger.debug("Order passed all risk checks", 
                         symbol=symbol,
                         notional=notional_value,
                         risk_level=RiskLevel.LOW.value)
        
        return RiskCheckResult(
            approved=True,
            risk_level=RiskLevel.LOW,
            metadata={
                'position_risk': position_risk,
                'notional_value': notional_value
            }
        )
    
    async def _check_daily_limits(self, portfolio_balance: float) -> RiskCheckResult:
        """Check daily P&L and trade count limits."""
        
        # Reset daily counters if new day
        current_date = datetime.now(timezone.utc).date()
        if current_date != self.last_reset_date:
            await self._reset_daily_counters()
        
        # Check if max daily loss already hit
        if self.max_daily_loss_hit:
            return RiskCheckResult(
                approved=False,
                reason="Maximum daily loss limit already reached",
                risk_level=RiskLevel.CRITICAL
            )
        
        # Check daily P&L
        effective_max_loss = self.risk_config.max_daily_loss * self.mode_multipliers[self.trading_mode]['max_daily_loss']
        max_loss_amount = portfolio_balance * effective_max_loss
        
        if self.daily_pnl < -max_loss_amount:
            self.max_daily_loss_hit = True
            self.is_halted = True
            self.halt_reason = f"Daily loss limit reached: ${abs(self.daily_pnl):.2f} > ${max_loss_amount:.2f}"
            
            await self._emit_risk_event(
                "MAX_DAILY_LOSS",
                RiskLevel.CRITICAL,
                self.halt_reason
            )
            
            return RiskCheckResult(
                approved=False,
                reason=self.halt_reason,
                risk_level=RiskLevel.CRITICAL
            )
        
        return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
    
    async def _calculate_position_risk(
        self, 
        notional_value: float, 
        portfolio_balance: float,
        current_positions: List[Dict]
    ) -> float:
        """Calculate risk percentage for this position."""
        return notional_value / portfolio_balance
    
    async def _check_trade_risk_limit(self, position_risk: float) -> RiskCheckResult:
        """Check if trade exceeds per-trade risk limit."""
        
        effective_risk_limit = (
            self.risk_config.risk_per_trade * 
            self.mode_multipliers[self.trading_mode]['risk_per_trade']
        )
        
        if position_risk > effective_risk_limit:
            # Calculate adjusted quantity
            risk_ratio = effective_risk_limit / position_risk
            
            return RiskCheckResult(
                approved=True,  # Approved with adjustment
                adjusted_quantity=risk_ratio,
                reason=f"Position resized to meet risk limit: {effective_risk_limit:.1%}",
                risk_level=RiskLevel.MEDIUM,
                metadata={'original_risk': position_risk, 'risk_limit': effective_risk_limit}
            )
        
        return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
    
    async def _check_portfolio_heat(
        self, 
        position_risk: float, 
        portfolio_balance: float,
        current_positions: List[Dict]
    ) -> RiskCheckResult:
        """Check portfolio heat (total risk exposure)."""
        
        # Calculate current heat from existing positions
        current_heat = sum(pos.get('risk_pct', 0) for pos in current_positions)
        
        # Add new position risk
        total_heat = current_heat + position_risk
        
        effective_heat_cap = (
            self.risk_config.heat_cap * 
            self.mode_multipliers[self.trading_mode]['heat_cap']
        )
        
        if total_heat > effective_heat_cap:
            # Check if we can reduce position size
            available_heat = effective_heat_cap - current_heat
            
            if available_heat <= 0:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Portfolio heat limit reached: {current_heat:.1%} >= {effective_heat_cap:.1%}",
                    risk_level=RiskLevel.HIGH
                )
            
            # Adjust position size to fit available heat
            adjustment_ratio = available_heat / position_risk
            
            return RiskCheckResult(
                approved=True,
                adjusted_quantity=adjustment_ratio,
                reason=f"Position resized for heat limit: {effective_heat_cap:.1%}",
                risk_level=RiskLevel.MEDIUM,
                metadata={
                    'current_heat': current_heat,
                    'heat_limit': effective_heat_cap,
                    'available_heat': available_heat
                }
            )
        
        return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
    
    async def _check_leverage_limit(
        self, 
        notional_value: float, 
        portfolio_balance: float
    ) -> RiskCheckResult:
        """Check leverage limits."""
        
        implied_leverage = notional_value / portfolio_balance
        
        if implied_leverage > self.risk_config.leverage_cap:
            return RiskCheckResult(
                approved=False,
                reason=f"Leverage too high: {implied_leverage:.1f}x > {self.risk_config.leverage_cap}x",
                risk_level=RiskLevel.HIGH
            )
        
        return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
    
    async def _check_mode_specific_limits(self, symbol: str, notional_value: float) -> RiskCheckResult:
        """Apply mode-specific additional checks."""
        
        # Paper mode - allow all (just logging)
        if self.trading_mode == TradingMode.PAPER:
            return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
        
        # Conservative mode - extra restrictions
        if self.trading_mode == TradingMode.CONSERVATIVE:
            # Limit position size in conservative mode
            if notional_value > 1000:  # $1000 max per position
                return RiskCheckResult(
                    approved=False,
                    reason="Position size too large for conservative mode (max $1000)",
                    risk_level=RiskLevel.MEDIUM
                )
        
        return RiskCheckResult(approved=True, risk_level=RiskLevel.LOW)
    
    async def _emit_risk_event(self, event_type: str, level: RiskLevel, description: str):
        """Emit a risk event for logging/alerting."""
        
        event = {
            'timestamp': datetime.now(timezone.utc),
            'event_type': event_type,
            'level': level.value,
            'description': description,
            'trading_mode': self.trading_mode.value,
            'daily_pnl': self.daily_pnl
        }
        
        self.risk_events.append(event)
        
        self.logger.warning("Risk event emitted",
                           event_type=event_type,
                           level=level.value,
                           description=description)
        
        # Trigger alerts for high/critical events
        if level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # This would integrate with notification system
            pass
    
    async def _reset_daily_counters(self):
        """Reset daily tracking counters."""
        
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.max_daily_loss_hit = False
        
        # Auto-resume if halted due to daily limits
        if self.is_halted and "daily loss" in (self.halt_reason or "").lower():
            self.is_halted = False
            self.halt_reason = None
        
        self.logger.info("Daily risk counters reset")
    
    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking."""
        self.daily_pnl += pnl_change
        self.logger.debug("Daily P&L updated", daily_pnl=self.daily_pnl, change=pnl_change)
    
    def update_trading_mode(self, mode: TradingMode):
        """Update trading mode."""
        old_mode = self.trading_mode
        self.trading_mode = mode
        
        self.logger.info("Trading mode updated", old_mode=old_mode.value, new_mode=mode.value)
        
        # Auto-halt if switching to HALT mode
        if mode == TradingMode.HALT:
            self.is_halted = True
            self.halt_reason = "Manual halt via mode change"
        
        # Resume if switching away from HALT
        elif old_mode == TradingMode.HALT and mode != TradingMode.HALT:
            self.is_halted = False
            self.halt_reason = None
    
    def update_risk_config(self, config: RiskConfig):
        """Update risk configuration."""
        self.risk_config = config
        self.logger.info("Risk configuration updated", 
                        risk_per_trade=config.risk_per_trade,
                        max_daily_loss=config.max_daily_loss)
    
    def update_symbol_allowlist(self, symbols: List[str]):
        """Update allowed symbols list."""
        self.allowed_symbols = symbols
        self.logger.info("Symbol allowlist updated", count=len(symbols), symbols=symbols[:5])
    
    def manual_halt(self, reason: str = "Manual halt"):
        """Manually halt all trading."""
        self.is_halted = True
        self.halt_reason = reason
        self.trading_mode = TradingMode.HALT
        
        self.logger.warning("Manual halt activated", reason=reason)
    
    def resume_trading(self, mode: TradingMode = TradingMode.CONSERVATIVE):
        """Resume trading after halt."""
        self.is_halted = False
        self.halt_reason = None
        self.trading_mode = mode
        
        self.logger.info("Trading resumed", mode=mode.value)
    
    def flatten_all_positions(self) -> bool:
        """Signal to flatten all positions (emergency)."""
        self.logger.critical("FLATTEN ALL POSITIONS REQUESTED")
        
        # This would integrate with position manager
        # For now, just log and halt
        self.manual_halt("Emergency flatten requested")
        return True
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        return {
            'trading_mode': self.trading_mode.value,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'daily_pnl': self.daily_pnl,
            'daily_trades_count': self.daily_trades_count,
            'max_daily_loss_hit': self.max_daily_loss_hit,
            'allowed_symbols_count': len(self.allowed_symbols),
            'recent_risk_events': len([e for e in self.risk_events if 
                                     (datetime.now(timezone.utc) - e['timestamp']).days < 1]),
            'risk_config': {
                'risk_per_trade': self.risk_config.risk_per_trade,
                'max_daily_loss': self.risk_config.max_daily_loss,
                'heat_cap': self.risk_config.heat_cap,
                'leverage_cap': self.risk_config.leverage_cap
            },
            'effective_limits': {
                'risk_per_trade': self.risk_config.risk_per_trade * 
                                self.mode_multipliers[self.trading_mode]['risk_per_trade'],
                'heat_cap': self.risk_config.heat_cap * 
                           self.mode_multipliers[self.trading_mode]['heat_cap'],
                'max_daily_loss': self.risk_config.max_daily_loss * 
                                self.mode_multipliers[self.trading_mode]['max_daily_loss']
            }
        }
    
    def get_recent_risk_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent risk events."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [event for event in self.risk_events if event['timestamp'] >= cutoff_time]
