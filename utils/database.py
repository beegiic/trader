import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import event
from sqlalchemy.engine import Engine
import uuid

from utils.logging_config import get_logger

logger = get_logger("database")

Base = declarative_base()


# Enable foreign keys for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if 'sqlite' in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Trade(Base):
    """Trade execution records."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    decision_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY/SELL
    order_type = Column(String(20), nullable=False)
    
    # Execution details
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Status tracking
    status = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # P&L tracking
    pnl_realized = Column(Float, default=0.0)
    fees_paid = Column(Float, default=0.0)
    
    # Exchange details
    binance_order_id = Column(String(50), nullable=True)
    binance_client_order_id = Column(String(50), nullable=True)
    
    # Relationships
    orders = relationship("Order", back_populates="trade")
    
    __table_args__ = (
        Index('ix_trades_symbol_status', 'symbol', 'status'),
        Index('ix_trades_created_at', 'created_at'),
    )


class Order(Base):
    """Individual order records."""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False)
    
    # Order details
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    order_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), nullable=False)  # NEW, FILLED, CANCELLED, etc.
    filled_quantity = Column(Float, default=0.0)
    avg_price = Column(Float, nullable=True)
    
    # Exchange details
    binance_order_id = Column(String(50), nullable=True, unique=True)
    binance_client_order_id = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    filled_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    trade = relationship("Trade", back_populates="orders")


class Position(Base):
    """Current positions."""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True)
    side = Column(String(10), nullable=False)  # LONG/SHORT
    
    # Position details
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Related trade
    trade_id = Column(Integer, ForeignKey('trades.id'), nullable=False)


class Decision(Base):
    """LLM decision records."""
    __tablename__ = "decisions"
    
    id = Column(Integer, primary_key=True)
    decision_id = Column(String(50), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Decision details
    action = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    position_size_pct = Column(Float, nullable=False)
    
    # LLM input/output
    llm_input = Column(JSON, nullable=False)
    llm_output = Column(JSON, nullable=False)
    
    # Human interaction
    requires_approval = Column(Boolean, default=True)
    approved = Column(Boolean, nullable=True)
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Human-readable summary
    summary_text = Column(Text, nullable=True)


class RiskEvent(Base):
    """Risk management events."""
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    
    # Event details
    description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    event_metadata = Column(JSON, nullable=True)
    
    # Status
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)
    
    # Timestamps
    triggered_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        Index('ix_risk_events_severity_resolved', 'severity', 'resolved'),
    )


class FeatureSnapshot(Base):
    """Market feature snapshots for analysis."""
    __tablename__ = "feature_snapshots"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Market data
    bid = Column(Float, nullable=False)
    ask = Column(Float, nullable=False)
    spread_bps = Column(Float, nullable=False)
    
    # Technical indicators
    indicators = Column(JSON, nullable=False)
    
    # Market metrics
    realized_vol_24h = Column(Float, nullable=True)
    funding_rate = Column(Float, nullable=True)
    fees_bps = Column(Float, nullable=False)
    slippage_bps = Column(Float, nullable=False)
    
    # Portfolio context
    portfolio_balance = Column(Float, nullable=False)
    portfolio_heat = Column(Float, nullable=False)
    daily_pnl = Column(Float, nullable=False)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    
    __table_args__ = (
        Index('ix_feature_snapshots_symbol_timestamp', 'symbol', 'timestamp'),
    )


class Performance(Base):
    """Daily performance metrics."""
    __tablename__ = "performance"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime(timezone=True), nullable=False, unique=True)
    
    # P&L metrics
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    
    # Trading metrics
    trades_count = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    portfolio_heat_max = Column(Float, default=0.0)
    
    # Fees
    fees_paid = Column(Float, default=0.0)
    
    # Portfolio value
    portfolio_value = Column(Float, nullable=False)
    
    __table_args__ = (
        Index('ix_performance_date', 'date'),
    )


class NewsItem(Base):
    """News items from Tavily."""
    __tablename__ = "news_items"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=True, index=True)
    query = Column(String(200), nullable=False)
    
    # News content
    title = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)
    url_hash = Column(String(64), nullable=False, unique=True)
    content_snippet = Column(Text, nullable=True)
    
    # Metadata
    published_at = Column(DateTime(timezone=True), nullable=True)
    relevance_score = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    
    # Timestamps
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    __table_args__ = (
        Index('ix_news_items_symbol_published', 'symbol', 'published_at'),
        Index('ix_news_items_fetched_at', 'fetched_at'),
    )


class DailyBrief(Base):
    """Daily brief documents."""
    __tablename__ = "daily_briefs"
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime(timezone=True), nullable=False, unique=True)
    mode = Column(String(20), nullable=False)
    
    # Content
    symbols = Column(JSON, nullable=False)  # List of symbols covered
    sections = Column(JSON, nullable=False)  # Structured content
    matrix_message = Column(Text, nullable=False)
    
    # Metadata
    generated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    sent_to_matrix = Column(Boolean, default=False)
    sent_at = Column(DateTime(timezone=True), nullable=True)


# Database Manager
class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///bot.db")
        self.engine = create_engine(
            self.database_url,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true",
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info("Database manager initialized", url=self.database_url)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False


# Session context manager
from contextlib import contextmanager

@contextmanager
def get_db_session(db_manager: DatabaseManager = None):
    """Context manager for database sessions."""
    if db_manager is None:
        db_manager = DatabaseManager()
    
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session error", error=str(e))
        raise
    finally:
        session.close()


# Utility functions
def save_trade(session: Session, trade_data: Dict[str, Any]) -> Trade:
    """Save a trade record."""
    trade = Trade(**trade_data)
    session.add(trade)
    session.flush()  # Get the ID
    logger.info("Trade saved", trade_id=trade.id, decision_id=trade.decision_id)
    return trade


def save_decision(session: Session, decision_data: Dict[str, Any]) -> Decision:
    """Save a decision record."""
    decision = Decision(**decision_data)
    session.add(decision)
    session.flush()
    logger.info("Decision saved", decision_id=decision.decision_id)
    return decision


def save_risk_event(session: Session, event_data: Dict[str, Any]) -> RiskEvent:
    """Save a risk event."""
    event = RiskEvent(**event_data)
    session.add(event)
    session.flush()
    logger.info("Risk event saved", event_type=event.event_type, severity=event.severity)
    return event


def get_active_positions(session: Session) -> List[Position]:
    """Get all active positions."""
    return session.query(Position).all()


def get_daily_pnl(session: Session, date: datetime = None) -> float:
    """Calculate daily P&L."""
    if date is None:
        date = datetime.now(timezone.utc).date()
    
    # Get realized P&L from completed trades
    realized = session.query(Trade).filter(
        Trade.created_at >= date,
        Trade.status == 'COMPLETED'
    ).with_entities(Trade.pnl_realized).scalar() or 0.0
    
    # Get unrealized P&L from open positions
    unrealized = sum(p.unrealized_pnl for p in get_active_positions(session))
    
    return realized + unrealized


def cleanup_old_data(session: Session, days_to_keep: int = 30):
    """Clean up old data to keep database size manageable."""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
    
    # Clean old feature snapshots
    deleted_features = session.query(FeatureSnapshot).filter(
        FeatureSnapshot.timestamp < cutoff_date
    ).delete()
    
    # Clean old news items
    deleted_news = session.query(NewsItem).filter(
        NewsItem.fetched_at < cutoff_date
    ).delete()
    
    logger.info("Old data cleaned", 
                features_deleted=deleted_features,
                news_deleted=deleted_news,
                cutoff_date=cutoff_date)


# Initialize global database manager
db_manager: Optional[DatabaseManager] = None

def init_database(database_url: str = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global db_manager
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager

def get_db_manager() -> DatabaseManager:
    """Get the global database manager."""
    if db_manager is None:
        return init_database()
    return db_manager