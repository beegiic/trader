import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logging_config import LoggerMixin
from utils.pydantic_models import OrderSide, OrderType, OHLCVData
from core.order_constraints import OrderConstraints
from strategies.risk_manager import RiskManager


class BinanceClient(LoggerMixin):
    """
    Binance REST API client with order constraints and risk management integration.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        futures: bool = False,
        order_constraints: OrderConstraints = None,
        risk_manager: RiskManager = None
    ):
        super().__init__()
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.futures = futures
        self.order_constraints = order_constraints
        self.risk_manager = risk_manager
        
        # Initialize Binance client
        self.client = Client(
            api_key=api_key,
            api_secret=secret_key,
            testnet=testnet
        )
        
        # Cache for exchange info and symbol data
        self._exchange_info: Optional[Dict] = None
        self._symbol_info: Dict[str, Dict] = {}
        
        self.logger.info(
            "Binance client initialized",
            testnet=testnet,
            futures=futures,
            has_constraints=order_constraints is not None,
            has_risk_manager=risk_manager is not None
        )
    
    async def initialize(self):
        """Initialize client by fetching exchange info."""
        try:
            await self.get_exchange_info()
            self.logger.info("Binance client initialization completed")
        except Exception as e:
            self.logger.error("Failed to initialize Binance client", error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(BinanceAPIException)
    )
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange info with caching and retry logic."""
        if self._exchange_info is None:
            try:
                if self.futures:
                    info = self.client.futures_exchange_info()
                else:
                    info = self.client.get_exchange_info()
                
                self._exchange_info = info
                
                # Cache symbol info for quick lookups
                for symbol_data in info['symbols']:
                    symbol = symbol_data['symbol']
                    self._symbol_info[symbol] = symbol_data
                
                self.logger.info(
                    "Exchange info fetched",
                    symbol_count=len(self._symbol_info),
                    futures=self.futures
                )
            except BinanceAPIException as e:
                self.logger.error("Binance API error fetching exchange info", error=str(e))
                raise
        
        return self._exchange_info
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached symbol info."""
        return self._symbol_info.get(symbol)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[OHLCVData]:
        """Get kline/candlestick data."""
        try:
            if self.futures:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_time,
                    endTime=end_time
                )
            else:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start_time,
                    endTime=end_time
                )
            
            # Convert to OHLCVData objects
            ohlcv_data = []
            for kline in klines:
                ohlcv_data.append(OHLCVData(
                    timestamp=int(kline[0]),  # Will be converted by pydantic
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                ))
            
            return ohlcv_data
            
        except BinanceAPIException as e:
            self.logger.error("Error fetching klines", symbol=symbol, error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        try:
            loop = asyncio.get_event_loop()
            if self.futures:
                result = await loop.run_in_executor(
                    None, lambda: self.client.futures_account()
                )
            else:
                result = await loop.run_in_executor(
                    None, lambda: self.client.get_account()
                )
            return result
        except BinanceAPIException as e:
            self.logger.error("Error fetching account info", error=str(e))
            return None
        except Exception as e:
            self.logger.error("Unexpected error fetching account", error=str(e))
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions (futures only)."""
        if not self.futures:
            raise ValueError("Positions only available for futures trading")
        
        try:
            positions = self.client.futures_position_information()
            # Filter out positions with zero quantity
            return [pos for pos in positions if float(pos['positionAmt']) != 0]
        except BinanceAPIException as e:
            self.logger.error("Error fetching positions", error=str(e))
            raise
    
    async def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate order through constraints and risk manager."""
        
        # Get symbol info if not cached
        if symbol not in self._symbol_info:
            await self.get_exchange_info()
        
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return False, f"Symbol {symbol} not found", {}
        
        # Apply order constraints
        if self.order_constraints:
            is_valid, error_msg, adjusted_params = self.order_constraints.validate_and_adjust(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                symbol_info=symbol_info
            )
            
            if not is_valid:
                self.logger.warning("Order failed constraints", symbol=symbol, error=error_msg)
                return False, error_msg, {}
            
            # Update parameters with adjusted values
            quantity = adjusted_params.get('quantity', quantity)
            price = adjusted_params.get('price', price)
            stop_price = adjusted_params.get('stop_price', stop_price)
        else:
            adjusted_params = {
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price
            }
        
        # Apply risk management
        if self.risk_manager:
            risk_result = await self.risk_manager.validate_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price or (await self.get_ticker_price(symbol))
            )
            
            if not risk_result.approved:
                self.logger.warning("Order failed risk check", symbol=symbol, reason=risk_result.reason)
                return False, risk_result.reason, {}
            
            # Apply risk adjustments
            if risk_result.adjusted_quantity:
                adjusted_params['quantity'] = risk_result.adjusted_quantity
        
        return True, None, adjusted_params
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """Place an order with validation."""
        
        # Validate order
        is_valid, error_msg, adjusted_params = await self._validate_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        if not is_valid:
            raise ValueError(f"Order validation failed: {error_msg}")
        
        # Use adjusted parameters
        final_quantity = adjusted_params['quantity']
        final_price = adjusted_params['price']
        final_stop_price = adjusted_params['stop_price']
        
        # Prepare order parameters
        order_params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': final_quantity,
            'timeInForce': time_in_force
        }
        
        if client_order_id:
            order_params['newClientOrderId'] = client_order_id
        
        if final_price is not None:
            order_params['price'] = final_price
        
        if final_stop_price is not None:
            order_params['stopPrice'] = final_stop_price
        
        # Place the order
        try:
            if self.futures:
                result = self.client.futures_create_order(**order_params)
            else:
                result = self.client.create_order(**order_params)
            
            self.logger.info(
                "Order placed successfully",
                symbol=symbol,
                side=side.value,
                type=order_type.value,
                quantity=final_quantity,
                order_id=result.get('orderId'),
                client_order_id=result.get('clientOrderId')
            )
            
            return result
            
        except (BinanceAPIException, BinanceOrderException) as e:
            self.logger.error(
                "Failed to place order",
                symbol=symbol,
                side=side.value,
                error=str(e)
            )
            raise
    
    async def place_test_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Place a test order (validation only)."""
        
        # Still validate the order
        is_valid, error_msg, adjusted_params = await self._validate_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        if not is_valid:
            return {
                'success': False,
                'error': error_msg
            }
        
        # Prepare test order parameters
        order_params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': adjusted_params['quantity'],
            'timeInForce': 'GTC'
        }
        
        if adjusted_params['price'] is not None:
            order_params['price'] = adjusted_params['price']
        
        if adjusted_params['stop_price'] is not None:
            order_params['stopPrice'] = adjusted_params['stop_price']
        
        try:
            if self.futures:
                result = self.client.futures_create_test_order(**order_params)
            else:
                result = self.client.create_test_order(**order_params)
            
            return {
                'success': True,
                'result': result,
                'adjusted_params': adjusted_params
            }
            
        except (BinanceAPIException, BinanceOrderException) as e:
            self.logger.error("Test order failed", symbol=symbol, error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_ticker_price(self, symbol: str) -> float:
        """Get current ticker price."""
        try:
            if self.futures:
                ticker = self.client.futures_ticker_price(symbol=symbol)
            else:
                ticker = self.client.get_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error("Error fetching ticker price", symbol=symbol, error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book depth."""
        try:
            if self.futures:
                return self.client.futures_order_book(symbol=symbol, limit=limit)
            else:
                return self.client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            self.logger.error("Error fetching order book", symbol=symbol, error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            if self.futures:
                return self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            else:
                return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error("Error canceling order", symbol=symbol, order_id=order_id, error=str(e))
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    async def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status."""
        try:
            if self.futures:
                return self.client.futures_get_order(symbol=symbol, orderId=order_id)
            else:
                return self.client.get_order(symbol=symbol, orderId=order_id)
        except BinanceAPIException as e:
            self.logger.error("Error fetching order status", symbol=symbol, order_id=order_id, error=str(e))
            raise
    
    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker statistics."""
        try:
            loop = asyncio.get_event_loop()
            if self.futures:
                result = await loop.run_in_executor(
                    None, lambda: self.client.futures_ticker(symbol=symbol)
                )
            else:
                result = await loop.run_in_executor(
                    None, lambda: self.client.get_ticker(symbol=symbol)
                )
            return result
        except BinanceAPIException as e:
            self.logger.error("Error fetching 24hr ticker", symbol=symbol, error=str(e))
            return None
        except Exception as e:
            self.logger.error("Unexpected error fetching ticker", symbol=symbol, error=str(e))
            return None
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate (futures only)."""
        if not self.futures:
            return None
        
        try:
            funding_info = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding_info:
                return float(funding_info[0]['fundingRate'])
            return None
        except BinanceAPIException as e:
            self.logger.error("Error fetching funding rate", symbol=symbol, error=str(e))
            return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for futures trading."""
        if not self.futures:
            return True  # No leverage needed for spot
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            )
            self.logger.info("Leverage set", symbol=symbol, leverage=leverage)
            return True
        except BinanceAPIException as e:
            self.logger.error("Error setting leverage", symbol=symbol, leverage=leverage, error=str(e))
            return False

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get specific position information (futures only)."""
        if not self.futures:
            return None
            
        try:
            positions = await self.get_positions()
            for position in positions:
                if position.get('symbol') == symbol:
                    return position
            return None
        except Exception as e:
            self.logger.error("Error getting position", symbol=symbol, error=str(e))
            return None

    async def health_check(self) -> bool:
        """Check if the client is healthy and can connect to Binance."""
        try:
            if self.futures:
                self.client.futures_ping()
            else:
                self.client.ping()
            return True
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def set_order_constraints(self, constraints: OrderConstraints):
        """Set order constraints."""
        self.order_constraints = constraints
        self.logger.info("Order constraints updated")
    
    def set_risk_manager(self, risk_manager: RiskManager):
        """Set risk manager."""
        self.risk_manager = risk_manager
        self.logger.info("Risk manager updated")