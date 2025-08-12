import asyncio
import json
import websockets
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from utils.logging_config import LoggerMixin
from utils.pydantic_models import OHLCVData


@dataclass
class BookTickerData:
    """Real-time best bid/ask data."""
    symbol: str
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    timestamp: datetime


@dataclass
class TradeData:
    """Individual trade data."""
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool


class BinanceWebSocket(LoggerMixin):
    """
    Binance WebSocket client for real-time market data.
    Maintains in-memory top-of-book and rolling OHLCV data.
    """
    
    def __init__(self, testnet: bool = True, futures: bool = False):
        super().__init__()
        
        self.testnet = testnet
        self.futures = futures
        
        # WebSocket URLs
        if futures:
            self.base_url = "wss://fstream.binance.com/ws/" if not testnet else "wss://fstream.binancefuture.com/ws/"
        else:
            self.base_url = "wss://stream.binance.com:9443/ws/" if not testnet else "wss://testnet.binance.vision/ws/"
        
        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Subscriptions
        self.subscriptions: List[str] = []
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Market data storage
        self.book_tickers: Dict[str, BookTickerData] = {}
        self.recent_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # OHLCV data storage (rolling windows)
        self.ohlcv_1m: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 1 hour
        self.ohlcv_5m: Dict[str, deque] = defaultdict(lambda: deque(maxlen=288))  # 24 hours
        self.ohlcv_1h: Dict[str, deque] = defaultdict(lambda: deque(maxlen=168))  # 1 week
        
        # Current incomplete candles
        self.current_candles: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
        
        self.logger.info("Binance WebSocket client initialized", testnet=testnet, futures=futures)
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            # Connect to a basic stream first (BTCUSDT book ticker for testing)
            stream_url = f"{self.base_url}btcusdt@bookTicker"
            
            self.websocket = await websockets.connect(
                stream_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            
            self.logger.info("WebSocket connected", url=stream_url)
            
            # Start message handling
            asyncio.create_task(self._handle_messages())
            
        except Exception as e:
            self.logger.error("Failed to connect to WebSocket", error=str(e))
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.logger.info("WebSocket disconnected")
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    self.logger.warning("Failed to decode message", error=str(e))
                except Exception as e:
                    self.logger.error("Error processing message", error=str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
            await self._handle_reconnection()
        except Exception as e:
            self.logger.error("Error in message handler", error=str(e))
            self.is_connected = False
            await self._handle_reconnection()
    
    async def _handle_reconnection(self):
        """Handle WebSocket reconnection logic."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff, max 60s
        
        self.logger.info("Attempting to reconnect", attempt=self.reconnect_attempts, delay=delay)
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
            # Re-subscribe to streams
            if self.subscriptions:
                await self._resubscribe()
        except Exception as e:
            self.logger.error("Reconnection failed", error=str(e))
            await self._handle_reconnection()
    
    async def _resubscribe(self):
        """Re-subscribe to all streams after reconnection."""
        if self.subscriptions:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": self.subscriptions,
                "id": int(datetime.now().timestamp())
            }
            
            await self.websocket.send(json.dumps(subscribe_msg))
            self.logger.info("Re-subscribed to streams", count=len(self.subscriptions))
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        # Handle subscription confirmations
        if 'result' in data and 'id' in data:
            self.logger.debug("Received subscription confirmation", id=data['id'])
            return
        
        # Handle stream data
        if 'stream' in data and 'data' in data:
            stream = data['stream']
            stream_data = data['data']
            
            # Parse stream name
            if '@bookTicker' in stream:
                await self._handle_book_ticker(stream_data)
            elif '@trade' in stream:
                await self._handle_trade(stream_data)
            elif '@kline' in stream:
                await self._handle_kline(stream_data)
            elif '@depth' in stream:
                await self._handle_depth(stream_data)
            
            # Execute callbacks
            await self._execute_callbacks(stream, stream_data)
    
    async def _handle_book_ticker(self, data: Dict[str, Any]):
        """Handle bookTicker stream data."""
        symbol = data['s']
        
        book_ticker = BookTickerData(
            symbol=symbol,
            bid_price=float(data['b']),
            bid_qty=float(data['B']),
            ask_price=float(data['a']),
            ask_qty=float(data['A']),
            timestamp=datetime.now(timezone.utc)
        )
        
        self.book_tickers[symbol] = book_ticker
        self.logger.debug("Updated book ticker", symbol=symbol)
    
    async def _handle_trade(self, data: Dict[str, Any]):
        """Handle trade stream data."""
        symbol = data['s']
        
        trade = TradeData(
            symbol=symbol,
            price=float(data['p']),
            quantity=float(data['q']),
            timestamp=datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc),
            is_buyer_maker=data['m']
        )
        
        self.recent_trades[symbol].append(trade)
        self.logger.debug("Recorded trade", symbol=symbol, price=trade.price)
    
    async def _handle_kline(self, data: Dict[str, Any]):
        """Handle kline/candlestick stream data."""
        kline_data = data['k']
        symbol = kline_data['s']
        interval = kline_data['i']
        is_closed = kline_data['x']  # Whether this kline is closed
        
        ohlcv = OHLCVData(
            timestamp=int(kline_data['t']),
            open=float(kline_data['o']),
            high=float(kline_data['h']),
            low=float(kline_data['l']),
            close=float(kline_data['c']),
            volume=float(kline_data['v'])
        )
        
        # Store current incomplete candle
        self.current_candles[symbol][interval] = ohlcv
        
        # If candle is closed, add to historical data
        if is_closed:
            if interval == '1m':
                self.ohlcv_1m[symbol].append(ohlcv)
            elif interval == '5m':
                self.ohlcv_5m[symbol].append(ohlcv)
            elif interval == '1h':
                self.ohlcv_1h[symbol].append(ohlcv)
            
            self.logger.debug("Recorded closed candle", symbol=symbol, interval=interval)
    
    async def _handle_depth(self, data: Dict[str, Any]):
        """Handle depth/order book stream data."""
        symbol = data['s']
        # For now, just log depth updates
        # Full order book management would require more complex state tracking
        self.logger.debug("Received depth update", symbol=symbol)
    
    async def _execute_callbacks(self, stream: str, data: Dict[str, Any]):
        """Execute registered callbacks for a stream."""
        if stream in self.callbacks:
            for callback in self.callbacks[stream]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error("Error in callback", stream=stream, error=str(e))
    
    async def subscribe_book_ticker(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to bookTicker stream for a symbol."""
        stream = f"{symbol.lower()}@bookTicker"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_trade(self, symbol: str, callback: Optional[Callable] = None):
        """Subscribe to trade stream for a symbol."""
        stream = f"{symbol.lower()}@trade"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_kline(self, symbol: str, interval: str, callback: Optional[Callable] = None):
        """Subscribe to kline stream for a symbol and interval."""
        stream = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe_stream(stream, callback)
    
    async def subscribe_depth(self, symbol: str, levels: int = 5, update_speed: str = '100ms', callback: Optional[Callable] = None):
        """Subscribe to depth stream for a symbol."""
        stream = f"{symbol.lower()}@depth{levels}@{update_speed}"
        await self._subscribe_stream(stream, callback)
    
    async def _subscribe_stream(self, stream: str, callback: Optional[Callable] = None):
        """Subscribe to a specific stream."""
        if stream not in self.subscriptions:
            self.subscriptions.append(stream)
            
            if callback:
                self.callbacks[stream].append(callback)
            
            if self.is_connected:
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": [stream],
                    "id": int(datetime.now().timestamp())
                }
                
                await self.websocket.send(json.dumps(subscribe_msg))
                self.logger.info("Subscribed to stream", stream=stream)
            else:
                self.logger.warning("Not connected, stream queued", stream=stream)
    
    async def unsubscribe_stream(self, stream: str):
        """Unsubscribe from a specific stream."""
        if stream in self.subscriptions:
            self.subscriptions.remove(stream)
            
            if stream in self.callbacks:
                del self.callbacks[stream]
            
            if self.is_connected:
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": [stream],
                    "id": int(datetime.now().timestamp())
                }
                
                await self.websocket.send(json.dumps(unsubscribe_msg))
                self.logger.info("Unsubscribed from stream", stream=stream)
    
    def get_book_ticker(self, symbol: str) -> Optional[BookTickerData]:
        """Get latest book ticker data for a symbol."""
        return self.book_tickers.get(symbol)
    
    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[TradeData]:
        """Get recent trades for a symbol."""
        trades = list(self.recent_trades[symbol])
        return trades[-limit:] if limit else trades
    
    def get_ohlcv(self, symbol: str, interval: str, limit: int = None) -> List[OHLCVData]:
        """Get OHLCV data for a symbol and interval."""
        if interval == '1m':
            data = list(self.ohlcv_1m[symbol])
        elif interval == '5m':
            data = list(self.ohlcv_5m[symbol])
        elif interval == '1h':
            data = list(self.ohlcv_1h[symbol])
        else:
            return []
        
        return data[-limit:] if limit else data
    
    def get_current_candle(self, symbol: str, interval: str) -> Optional[OHLCVData]:
        """Get current (incomplete) candle for a symbol and interval."""
        return self.current_candles.get(symbol, {}).get(interval)
    
    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Calculate current spread in basis points."""
        ticker = self.get_book_ticker(symbol)
        if ticker and ticker.bid_price > 0:
            spread = ticker.ask_price - ticker.bid_price
            return (spread / ticker.bid_price) * 10000
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Get WebSocket health status."""
        return {
            'connected': self.is_connected,
            'subscriptions_count': len(self.subscriptions),
            'symbols_with_book_data': len(self.book_tickers),
            'reconnect_attempts': self.reconnect_attempts,
            'websocket_open': self.websocket is not None and not self.websocket.closed if self.websocket else False
        }
    
    async def start_multi_stream(self, symbols: List[str]):
        """Start multiple streams for common data types."""
        tasks = []
        
        for symbol in symbols:
            # Subscribe to essential streams for each symbol
            tasks.extend([
                self.subscribe_book_ticker(symbol),
                self.subscribe_trade(symbol),
                self.subscribe_kline(symbol, '1m'),
                self.subscribe_kline(symbol, '5m'),
                self.subscribe_kline(symbol, '1h')
            ])
        
        await asyncio.gather(*tasks)
        self.logger.info("Started multi-stream subscriptions", symbols=symbols)
    
    def register_callback(self, stream: str, callback: Callable):
        """Register a callback for a specific stream."""
        self.callbacks[stream].append(callback)
        self.logger.debug("Callback registered", stream=stream)
    
    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market summary for a symbol."""
        ticker = self.get_book_ticker(symbol)
        recent_trades = self.get_recent_trades(symbol, 10)
        ohlcv_1m = self.get_ohlcv(symbol, '1m', 20)
        current_candle = self.get_current_candle(symbol, '1m')
        
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc),
            'book_ticker': ticker,
            'spread_bps': self.get_spread_bps(symbol),
            'recent_trades_count': len(recent_trades),
            'ohlcv_1m_count': len(ohlcv_1m),
            'has_current_candle': current_candle is not None
        }
        
        if recent_trades:
            summary['last_trade_price'] = recent_trades[-1].price
            summary['last_trade_time'] = recent_trades[-1].timestamp
        
        if ohlcv_1m:
            summary['latest_close'] = ohlcv_1m[-1].close
            summary['latest_volume'] = ohlcv_1m[-1].volume
        
        return summary