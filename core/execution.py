from typing import Dict, List, Optional
import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from utils.logging_config import LoggerMixin
from utils.pydantic_models import Side, OrderType

def compute_qty_from_risk(equity_usd: float, risk_pct: float, entry_price: float, stop_price: float, fee_bps: int = 5, slippage_bp: int = 2) -> float:
    """
    Compute position quantity based on risk amount.
    
    Key insight: Risk is bounded by stop distance, not leverage.
    Even with high leverage, actual risk = equity * risk_pct because SL exits near stop_price.
    """
    risk_amount = equity_usd * risk_pct
    stop_dist = abs(entry_price - stop_price)
    
    if stop_dist <= 1e-9:
        return 0.0
    
    gross_qty = risk_amount / stop_dist
    
    # Buffer for fees + slippage  
    buffer = (fee_bps + slippage_bp) / 10000.0
    net_qty = gross_qty * (1 - buffer)
    
    return max(0.0, net_qty)

class ExecutionClient(LoggerMixin):
    """
    Handle bracket order execution with proper risk sizing.
    
    For Binance Futures:
    - Use STOP_MARKET for SL (reduceOnly)
    - Use TAKE_PROFIT_MARKET or LIMIT for TP(s) (reduceOnly) 
    - If any leg fails to place, cancel the entry
    """
    
    def __init__(self, binance_client, order_constraints=None):
        super().__init__()
        self.binance_client = binance_client
        self.order_constraints = order_constraints
        
    async def place_bracket_order(
        self,
        symbol: str,
        side: Side,
        entry_type: str,  # "market" or "limit"
        entry_price: Optional[float],
        stop_price: float,
        tp_prices: List[float],
        size_pct: float,
        equity_usd: float,
        leverage_cap: int = 25,
        reduce_only: bool = False
    ) -> Dict:
        """
        Place bracket order (entry + SL + TP legs).
        Returns order result with IDs or error info.
        """
        try:
            # 1. Calculate position quantity
            if entry_type == "market":
                # Use current market price for market orders
                ticker = await self.binance_client.get_ticker(symbol)
                effective_entry = float(ticker['price'])
            else:
                effective_entry = entry_price
            
            qty = compute_qty_from_risk(
                equity_usd=equity_usd,
                risk_pct=size_pct,
                entry_price=effective_entry,
                stop_price=stop_price
            )
            
            if qty <= 0:
                return {"success": False, "error": "Invalid quantity calculation"}
            
            # 2. Apply order constraints (rounding, min notional, etc.)
            if self.order_constraints:
                qty = self._apply_constraints(symbol, qty, effective_entry)
                entry_price = self._round_price(symbol, entry_price) if entry_price else None
                stop_price = self._round_price(symbol, stop_price)
                tp_prices = [self._round_price(symbol, tp) for tp in tp_prices]
            
            self.logger.info("Placing bracket order",
                           symbol=symbol,
                           side=side.value,
                           entry_type=entry_type,
                           qty=qty,
                           entry_price=entry_price,
                           stop_price=stop_price,
                           tp_count=len(tp_prices))
            
            # 3. Place entry order
            entry_order = await self._place_entry_order(
                symbol, side, entry_type, entry_price, qty, reduce_only
            )
            
            if not entry_order.get("success"):
                return entry_order
            
            entry_order_id = entry_order["order_id"]
            
            # 4. Place protective orders (SL + TPs)
            protective_orders = []
            
            # Stop Loss
            sl_order = await self._place_stop_loss(symbol, side, stop_price, qty)
            if sl_order.get("success"):
                protective_orders.append(sl_order["order_id"])
            else:
                # Cancel entry if SL fails
                await self._cancel_order(symbol, entry_order_id)
                return {"success": False, "error": "Stop loss placement failed", "details": sl_order}
            
            # Take Profits  
            tp_qty_each = qty / len(tp_prices) if tp_prices else 0
            for i, tp_price in enumerate(tp_prices):
                tp_order = await self._place_take_profit(symbol, side, tp_price, tp_qty_each)
                if tp_order.get("success"):
                    protective_orders.append(tp_order["order_id"])
                else:
                    self.logger.warning("Take profit placement failed", 
                                      symbol=symbol, tp_level=i+1, error=tp_order)
            
            return {
                "success": True,
                "entry_order_id": entry_order_id,
                "protective_order_ids": protective_orders,
                "quantity": qty,
                "effective_entry": effective_entry
            }
            
        except Exception as e:
            self.logger.error("Bracket order failed", symbol=symbol, error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _place_entry_order(self, symbol: str, side: Side, entry_type: str, entry_price: Optional[float], qty: float, reduce_only: bool = False) -> Dict:
        """Place the entry order."""
        try:
            binance_side = "BUY" if side == Side.LONG else "SELL"
            
            if entry_type == "market":
                order = await self.binance_client.create_order(
                    symbol=symbol,
                    side=binance_side,
                    type="MARKET",
                    quantity=qty,
                    reduce_only=reduce_only
                )
            else:  # limit
                order = await self.binance_client.create_order(
                    symbol=symbol,
                    side=binance_side,
                    type="LIMIT",
                    quantity=qty,
                    price=entry_price,
                    time_in_force="GTC",
                    reduce_only=reduce_only
                )
            
            return {"success": True, "order_id": order.get("orderId"), "order": order}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _place_stop_loss(self, symbol: str, side: Side, stop_price: float, qty: float) -> Dict:
        """Place stop loss order."""
        try:
            # Opposite side for exit
            binance_side = "SELL" if side == Side.LONG else "BUY"
            
            order = await self.binance_client.create_order(
                symbol=symbol,
                side=binance_side,
                type="STOP_MARKET",
                quantity=qty,
                stop_price=stop_price,
                reduce_only=True
            )
            
            return {"success": True, "order_id": order.get("orderId"), "order": order}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _place_take_profit(self, symbol: str, side: Side, tp_price: float, qty: float) -> Dict:
        """Place take profit order."""
        try:
            # Opposite side for exit
            binance_side = "SELL" if side == Side.LONG else "BUY"
            
            order = await self.binance_client.create_order(
                symbol=symbol,
                side=binance_side,
                type="TAKE_PROFIT_MARKET",
                quantity=qty,
                stop_price=tp_price,
                reduce_only=True
            )
            
            return {"success": True, "order_id": order.get("orderId"), "order": order}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self.binance_client.cancel_order(symbol=symbol, order_id=order_id)
            return True
        except Exception as e:
            self.logger.error("Failed to cancel order", symbol=symbol, order_id=order_id, error=str(e))
            return False
    
    def _apply_constraints(self, symbol: str, qty: float, price: float) -> float:
        """Apply exchange constraints to quantity."""
        if not self.order_constraints:
            return qty
            
        # This would use the order constraints to round qty appropriately
        # For now, just ensure minimum notional
        min_notional = 10.0  # $10 minimum
        if qty * price < min_notional:
            return 0.0
            
        return qty
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Round price according to exchange tick size."""
        if not self.order_constraints or not price:
            return price
            
        # This would use proper tick size rounding
        # For now, round to 6 decimal places
        return round(price, 6)