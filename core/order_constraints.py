from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Optional, Tuple, Any
import math

from utils.logging_config import LoggerMixin
from utils.pydantic_models import OrderSide, OrderType


class OrderConstraints(LoggerMixin):
    """
    Handles Binance order constraints including tick size, step size, 
    and minimum notional validations with automatic rounding.
    """
    
    def __init__(self):
        super().__init__()
        
        # Cache for symbol filters
        self._symbol_filters: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Order constraints manager initialized")
    
    def update_symbol_info(self, symbol: str, symbol_info: Dict[str, Any]):
        """Update cached symbol information."""
        filters = {}
        
        # Extract relevant filters
        for filter_info in symbol_info.get('filters', []):
            filter_type = filter_info['filterType']
            
            if filter_type == 'PRICE_FILTER':
                filters['tick_size'] = float(filter_info['tickSize'])
                filters['min_price'] = float(filter_info['minPrice'])
                filters['max_price'] = float(filter_info['maxPrice'])
                
            elif filter_type == 'LOT_SIZE':
                filters['step_size'] = float(filter_info['stepSize'])
                filters['min_qty'] = float(filter_info['minQty'])
                filters['max_qty'] = float(filter_info['maxQty'])
                
            elif filter_type == 'MIN_NOTIONAL':
                filters['min_notional'] = float(filter_info['minNotional'])
                
            elif filter_type == 'NOTIONAL':
                filters['min_notional'] = float(filter_info.get('minNotional', 0))
                filters['max_notional'] = float(filter_info.get('maxNotional', float('inf')))
                
            elif filter_type == 'MARKET_LOT_SIZE':
                filters['market_step_size'] = float(filter_info['stepSize'])
                filters['market_min_qty'] = float(filter_info['minQty'])
                filters['market_max_qty'] = float(filter_info['maxQty'])
        
        # Store base asset and quote asset info
        filters['base_asset'] = symbol_info.get('baseAsset')
        filters['quote_asset'] = symbol_info.get('quoteAsset')
        filters['base_precision'] = int(symbol_info.get('baseAssetPrecision', 8))
        filters['quote_precision'] = int(symbol_info.get('quoteAssetPrecision', 8))
        
        self._symbol_filters[symbol] = filters
        
        self.logger.debug("Symbol filters updated", 
                         symbol=symbol, 
                         tick_size=filters.get('tick_size'),
                         step_size=filters.get('step_size'),
                         min_notional=filters.get('min_notional'))
    
    def get_symbol_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached symbol filters."""
        return self._symbol_filters.get(symbol)
    
    def round_price(self, symbol: str, price: float, round_up: bool = False) -> float:
        """Round price to valid tick size."""
        filters = self.get_symbol_filters(symbol)
        if not filters or 'tick_size' not in filters:
            return price
        
        tick_size = filters['tick_size']
        
        if tick_size <= 0:
            return price
        
        # Use Decimal for precise calculations
        price_decimal = Decimal(str(price))
        tick_decimal = Decimal(str(tick_size))
        
        # Round to nearest tick
        if round_up:
            rounded = (price_decimal / tick_decimal).quantize(Decimal('1'), rounding=ROUND_UP) * tick_decimal
        else:
            rounded = (price_decimal / tick_decimal).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_decimal
        
        return float(rounded)
    
    def round_quantity(self, symbol: str, quantity: float, order_type: OrderType = OrderType.LIMIT, round_up: bool = False) -> float:
        """Round quantity to valid step size."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return quantity
        
        # Use market step size for market orders if available
        if order_type == OrderType.MARKET and 'market_step_size' in filters:
            step_size = filters['market_step_size']
        else:
            step_size = filters.get('step_size', 0)
        
        if step_size <= 0:
            return quantity
        
        # Use Decimal for precise calculations
        qty_decimal = Decimal(str(quantity))
        step_decimal = Decimal(str(step_size))
        
        # Round to nearest step
        if round_up:
            rounded = (qty_decimal / step_decimal).quantize(Decimal('1'), rounding=ROUND_UP) * step_decimal
        else:
            rounded = (qty_decimal / step_decimal).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_decimal
        
        return float(rounded)
    
    def validate_price(self, symbol: str, price: float) -> Tuple[bool, Optional[str]]:
        """Validate if price meets constraints."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return True, None
        
        min_price = filters.get('min_price', 0)
        max_price = filters.get('max_price', float('inf'))
        tick_size = filters.get('tick_size', 0)
        
        if price < min_price:
            return False, f"Price {price} below minimum {min_price}"
        
        if price > max_price:
            return False, f"Price {price} above maximum {max_price}"
        
        # Check tick size alignment
        if tick_size > 0:
            remainder = price % tick_size
            if abs(remainder) > 1e-8 and abs(remainder - tick_size) > 1e-8:
                return False, f"Price {price} not aligned to tick size {tick_size}"
        
        return True, None
    
    def validate_quantity(self, symbol: str, quantity: float, order_type: OrderType = OrderType.LIMIT) -> Tuple[bool, Optional[str]]:
        """Validate if quantity meets constraints."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return True, None
        
        # Use appropriate constraints based on order type
        if order_type == OrderType.MARKET:
            min_qty = filters.get('market_min_qty', filters.get('min_qty', 0))
            max_qty = filters.get('market_max_qty', filters.get('max_qty', float('inf')))
            step_size = filters.get('market_step_size', filters.get('step_size', 0))
        else:
            min_qty = filters.get('min_qty', 0)
            max_qty = filters.get('max_qty', float('inf'))
            step_size = filters.get('step_size', 0)
        
        if quantity < min_qty:
            return False, f"Quantity {quantity} below minimum {min_qty}"
        
        if quantity > max_qty:
            return False, f"Quantity {quantity} above maximum {max_qty}"
        
        # Check step size alignment
        if step_size > 0:
            remainder = quantity % step_size
            if abs(remainder) > 1e-8 and abs(remainder - step_size) > 1e-8:
                return False, f"Quantity {quantity} not aligned to step size {step_size}"
        
        return True, None
    
    def validate_notional(self, symbol: str, quantity: float, price: float) -> Tuple[bool, Optional[str]]:
        """Validate if notional value meets minimum requirements."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return True, None
        
        notional = quantity * price
        min_notional = filters.get('min_notional', 0)
        max_notional = filters.get('max_notional', float('inf'))
        
        if notional < min_notional:
            return False, f"Notional {notional:.6f} below minimum {min_notional}"
        
        if notional > max_notional:
            return False, f"Notional {notional:.6f} above maximum {max_notional}"
        
        return True, None
    
    def validate_and_adjust(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        symbol_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate and adjust order parameters to meet all constraints.
        Returns: (is_valid, error_message, adjusted_parameters)
        """
        
        # Update symbol filters if provided
        if symbol_info:
            self.update_symbol_info(symbol, symbol_info)
        
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return False, f"No filters available for symbol {symbol}", {}
        
        adjusted_params = {}
        
        # Validate and adjust quantity
        rounded_quantity = self.round_quantity(symbol, quantity, order_type, round_up=False)
        
        # Ensure we meet minimum quantity after rounding
        min_qty = filters.get('market_min_qty' if order_type == OrderType.MARKET else 'min_qty', 0)
        if rounded_quantity < min_qty:
            # Try rounding up
            rounded_quantity = self.round_quantity(symbol, quantity, order_type, round_up=True)
            if rounded_quantity < min_qty:
                return False, f"Cannot meet minimum quantity {min_qty} with provided quantity {quantity}", {}
        
        is_valid, error = self.validate_quantity(symbol, rounded_quantity, order_type)
        if not is_valid:
            return False, error, {}
        
        adjusted_params['quantity'] = rounded_quantity
        
        # Validate and adjust price if provided
        if price is not None:
            rounded_price = self.round_price(symbol, price, round_up=(side == OrderSide.BUY))
            
            is_valid, error = self.validate_price(symbol, rounded_price)
            if not is_valid:
                return False, error, {}
            
            adjusted_params['price'] = rounded_price
            
            # Check notional value
            is_valid, error = self.validate_notional(symbol, rounded_quantity, rounded_price)
            if not is_valid:
                # Try to adjust quantity to meet min notional
                min_notional = filters.get('min_notional', 0)
                if min_notional > 0:
                    required_quantity = min_notional / rounded_price
                    adjusted_quantity = self.round_quantity(symbol, required_quantity, order_type, round_up=True)
                    
                    # Check if adjusted quantity is valid
                    is_valid, error = self.validate_quantity(symbol, adjusted_quantity, order_type)
                    if is_valid:
                        adjusted_params['quantity'] = adjusted_quantity
                        is_valid, error = self.validate_notional(symbol, adjusted_quantity, rounded_price)
                        if is_valid:
                            self.logger.info("Adjusted quantity to meet min notional",
                                           symbol=symbol,
                                           original_quantity=quantity,
                                           adjusted_quantity=adjusted_quantity,
                                           min_notional=min_notional)
                        else:
                            return False, error, {}
                    else:
                        return False, f"Cannot adjust quantity to meet min notional: {error}", {}
                else:
                    return False, error, {}
        
        # Validate and adjust stop price if provided
        if stop_price is not None:
            rounded_stop_price = self.round_price(symbol, stop_price, round_up=(side == OrderSide.SELL))
            
            is_valid, error = self.validate_price(symbol, rounded_stop_price)
            if not is_valid:
                return False, f"Stop price validation failed: {error}", {}
            
            adjusted_params['stop_price'] = rounded_stop_price
        
        self.logger.debug("Order validation successful",
                         symbol=symbol,
                         original_quantity=quantity,
                         adjusted_quantity=adjusted_params['quantity'],
                         original_price=price,
                         adjusted_price=adjusted_params.get('price'))
        
        return True, None, adjusted_params
    
    def get_min_order_size(self, symbol: str, price: float) -> Optional[float]:
        """Get minimum order size in base asset for a given price."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return None
        
        min_qty = filters.get('min_qty', 0)
        min_notional = filters.get('min_notional', 0)
        
        if min_notional > 0 and price > 0:
            min_qty_for_notional = min_notional / price
            min_qty = max(min_qty, min_qty_for_notional)
        
        # Round up to meet step size
        if min_qty > 0:
            return self.round_quantity(symbol, min_qty, round_up=True)
        
        return None
    
    def get_max_order_size(self, symbol: str) -> Optional[float]:
        """Get maximum order size in base asset."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return None
        
        return filters.get('max_qty')
    
    def calculate_commission_adjusted_quantity(
        self, 
        symbol: str, 
        target_notional: float, 
        price: float, 
        commission_rate: float = 0.001
    ) -> Optional[float]:
        """Calculate quantity accounting for trading fees."""
        # Adjust target notional for commission
        effective_notional = target_notional / (1 + commission_rate)
        base_quantity = effective_notional / price
        
        # Round and validate
        adjusted_quantity = self.round_quantity(symbol, base_quantity, round_up=False)
        
        # Ensure we still meet minimum requirements
        is_valid, _ = self.validate_notional(symbol, adjusted_quantity, price)
        if not is_valid:
            # Try rounding up
            adjusted_quantity = self.round_quantity(symbol, base_quantity, round_up=True)
        
        return adjusted_quantity
    
    def get_precision_info(self, symbol: str) -> Dict[str, int]:
        """Get decimal precision information for a symbol."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return {}
        
        info = {}
        
        # Calculate price precision from tick size
        tick_size = filters.get('tick_size')
        if tick_size:
            price_precision = len(str(tick_size).rstrip('0').split('.')[-1]) if '.' in str(tick_size) else 0
            info['price_precision'] = price_precision
        
        # Calculate quantity precision from step size
        step_size = filters.get('step_size')
        if step_size:
            qty_precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
            info['quantity_precision'] = qty_precision
        
        # Asset precisions
        info['base_precision'] = filters.get('base_precision', 8)
        info['quote_precision'] = filters.get('quote_precision', 8)
        
        return info