"""
Market Scanner - Monitors markets and selects trading candidates
Configurable scanning intervals: slow when idle, fast when trades are active
"""
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

from utils.logging_config import LoggerMixin
from core.binance_client import BinanceClient


class ScanMode(Enum):
    IDLE = "idle"           # No active trades - slow scanning
    ACTIVE = "active"       # Active trades - fast scanning
    MANUAL = "manual"       # Manual override


@dataclass
class MarketCandidate:
    """Market symbol that meets scanning criteria"""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    volume_rank: int
    volatility: float
    spread_bps: float
    last_updated: datetime
    

@dataclass
class ScanConfig:
    """Configuration for market scanning"""
    # Scanning intervals - more frequent for more opportunities
    idle_scan_interval: int = 180      # 3 minutes when idle (was 15min)
    active_scan_interval: int = 15     # 15 seconds when trades active
    manual_scan_interval: int = 30     # 30 seconds manual override
    
    # Market selection criteria
    min_24h_volume: float = 500000     # Lower volume threshold for more opportunities
    max_symbols: int = 15              # Track more symbols
    min_price: float = 0.01           # Minimum price
    max_price: float = 100000         # Maximum price
    max_spread_bps: int = 999999      # Effectively no spread limit
    
    # Volatility filters
    min_volatility: float = 0.02      # Minimum 2% volatility
    max_volatility: float = 999.0     # Effectively no maximum volatility limit


class MarketScanner(LoggerMixin):
    """
    Market scanner that monitors cryptocurrency futures markets
    and identifies trading candidates based on volume, volatility, and spread criteria.
    """
    
    def __init__(self, binance_client: BinanceClient, config: ScanConfig = None):
        super().__init__()
        self.binance_client = binance_client
        self.config = config or ScanConfig()
        
        # State management
        self.current_mode = ScanMode.IDLE
        self.active_symbols: Set[str] = set()
        self.candidates: Dict[str, MarketCandidate] = {}
        self.last_scan: Optional[datetime] = None
        
        # Scanning control
        self.scanner_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        self.logger.info("Market scanner initialized", 
                        idle_interval=self.config.idle_scan_interval,
                        active_interval=self.config.active_scan_interval)

    def get_universe(self):
        """Get the list of symbols to scan."""
        # prefer explicit allowlist from config
        allow = (getattr(self.config, "symbols_allowlist", None) or 
                getattr(self.config, "symbols", None) or [])
        if allow:
            return list(dict.fromkeys(allow))  # dedupe preserving order
        # fallback defaults - more symbols for more opportunities
        return ["BTCUSDT", "ETHUSDT"]
    
    async def start(self):
        """Start the market scanner"""
        if self.is_running:
            self.logger.warning("Scanner already running")
            return
        
        self.is_running = True
        self.scanner_task = asyncio.create_task(self._scanning_loop())
        self.logger.info("Market scanner started")
    
    async def stop(self):
        """Stop the market scanner"""
        self.is_running = False
        if self.scanner_task:
            self.scanner_task.cancel()
            try:
                await self.scanner_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Market scanner stopped")
    
    def set_mode(self, mode: ScanMode):
        """Change scanning mode"""
        if self.current_mode != mode:
            self.logger.info("Scanning mode changed", old_mode=self.current_mode.value, new_mode=mode.value)
            self.current_mode = mode
    
    def add_active_symbol(self, symbol: str):
        """Add symbol to active trading list (triggers fast scanning)"""
        self.active_symbols.add(symbol)
        if len(self.active_symbols) > 0:
            self.set_mode(ScanMode.ACTIVE)
        self.logger.info("Added active symbol", symbol=symbol, active_count=len(self.active_symbols))
    
    def remove_active_symbol(self, symbol: str):
        """Remove symbol from active trading list"""
        self.active_symbols.discard(symbol)
        if len(self.active_symbols) == 0:
            self.set_mode(ScanMode.IDLE)
        self.logger.info("Removed active symbol", symbol=symbol, active_count=len(self.active_symbols))
    
    def get_scan_interval(self) -> int:
        """Get current scanning interval based on mode"""
        if self.current_mode == ScanMode.ACTIVE:
            return self.config.active_scan_interval
        elif self.current_mode == ScanMode.MANUAL:
            return self.config.manual_scan_interval
        else:
            return self.config.idle_scan_interval
    
    async def _scanning_loop(self):
        """Main scanning loop"""
        while self.is_running:
            try:
                scan_start = datetime.now(timezone.utc)
                
                # Perform market scan
                await self._perform_scan()
                
                # Calculate next scan time
                scan_interval = self.get_scan_interval()
                scan_duration = (datetime.now(timezone.utc) - scan_start).total_seconds()
                
                self.logger.info("Market scan completed", 
                                duration_seconds=round(scan_duration, 2),
                                next_scan_seconds=scan_interval,
                                mode=self.current_mode.value,
                                candidates_found=len(self.candidates),
                                active_symbols=len(self.active_symbols),
                                action="scan_completed")
                
                # Wait until next scan (accounting for scan duration)
                sleep_time = max(1, scan_interval - scan_duration)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in scanning loop", error=str(e))
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _perform_scan(self):
        """Perform a single market scan"""
        try:
            self.logger.info("Starting market scan", 
                           mode=self.current_mode.value,
                           interval_seconds=self.get_scan_interval(),
                           action="scan_start")
            
            # Get symbols from universe (config allowlist or defaults)
            universe_symbols = self.get_universe()
            
            self.logger.info("Scanning futures pairs", 
                           symbols=universe_symbols,
                           symbol_count=len(universe_symbols))
            
            # Get 24hr tickers for universe symbols only
            candidates = await self._analyze_symbols(universe_symbols)
            
            # Rank and select best candidates
            self.candidates = self._select_candidates(candidates)
            self.last_scan = datetime.now(timezone.utc)
            
            self.logger.info("Market scan complete", 
                           total_analyzed=len(candidates),
                           selected_candidates=len(self.candidates))
            
        except Exception as e:
            self.logger.error("Error performing market scan", error=str(e))
    
    async def _analyze_symbols(self, symbols: List[str]) -> Dict[str, MarketCandidate]:
        """Analyze list of symbols and create candidates"""
        candidates = {}
        
        for symbol in symbols:
            try:
                # Get ticker data
                ticker = await self.binance_client.get_24hr_ticker(symbol)
                if not ticker:
                    continue
                
                # Extract metrics
                price = float(ticker['lastPrice'])
                volume_24h = float(ticker['quoteVolume'])
                price_change_24h = float(ticker['priceChangePercent']) / 100
                
                # Apply basic filters
                if (price < self.config.min_price or 
                    price > self.config.max_price or
                    volume_24h < self.config.min_24h_volume):
                    continue
                
                # Calculate volatility (approximation using price change)
                volatility = abs(price_change_24h)
                
                if (volatility < self.config.min_volatility or 
                    volatility > self.config.max_volatility):
                    continue
                
                # Get order book for spread calculation
                spread_bps = await self._calculate_spread(symbol, price)
                if spread_bps > self.config.max_spread_bps:
                    continue
                
                # Create candidate
                candidate = MarketCandidate(
                    symbol=symbol,
                    price=price,
                    volume_24h=volume_24h,
                    price_change_24h=price_change_24h,
                    volume_rank=0,  # Will be set in ranking
                    volatility=volatility,
                    spread_bps=spread_bps,
                    last_updated=datetime.now(timezone.utc)
                )
                
                candidates[symbol] = candidate
                
            except Exception as e:
                self.logger.debug("Error analyzing symbol", symbol=symbol, error=str(e))
        
        return candidates
    
    async def _calculate_spread(self, symbol: str, price: float) -> float:
        """Calculate bid-ask spread in basis points"""
        try:
            # Get order book
            depth = await self.binance_client.get_order_book(symbol, limit=5)
            if not depth or not depth.get('bids') or not depth.get('asks'):
                return 999  # High spread to filter out
            
            best_bid = float(depth['bids'][0][0])
            best_ask = float(depth['asks'][0][0])
            
            spread = (best_ask - best_bid) / price
            return spread * 10000  # Convert to basis points
            
        except Exception:
            return 999  # High spread to filter out
    
    def _select_candidates(self, candidates: Dict[str, MarketCandidate]) -> Dict[str, MarketCandidate]:
        """Select and rank the best candidates"""
        if not candidates:
            return {}
        
        # Sort by volume (primary) and volatility (secondary)
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: (c.volume_24h, c.volatility),
            reverse=True
        )
        
        # Take top N candidates
        selected = sorted_candidates[:self.config.max_symbols]
        
        # Update volume ranks
        result = {}
        for rank, candidate in enumerate(selected, 1):
            candidate.volume_rank = rank
            result[candidate.symbol] = candidate
        
        return result
    
    def get_candidates(self) -> Dict[str, MarketCandidate]:
        """Get current market candidates"""
        return self.candidates.copy()
    
    def get_candidate(self, symbol: str) -> Optional[MarketCandidate]:
        """Get specific market candidate"""
        return self.candidates.get(symbol)
    
    def get_scanner_status(self) -> Dict:
        """Get scanner status information"""
        return {
            'running': self.is_running,
            'mode': self.current_mode.value,
            'active_symbols': list(self.active_symbols),
            'candidates_count': len(self.candidates),
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'next_scan_seconds': self.get_scan_interval(),
            'config': {
                'idle_interval': self.config.idle_scan_interval,
                'active_interval': self.config.active_scan_interval,
                'max_symbols': self.config.max_symbols,
                'min_volume': self.config.min_24h_volume
            }
        }