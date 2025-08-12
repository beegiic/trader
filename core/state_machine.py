from enum import Enum
from typing import Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

class TradeState(str, Enum):
    """Trading states for the state machine."""
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED" 
    REJECTED = "REJECTED"
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    MANAGED = "MANAGED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"

class StateTransition:
    """Define valid state transitions to prevent zombie states."""
    
    # Valid transitions map: current_state -> set of allowed next states
    VALID_TRANSITIONS: Dict[TradeState, Set[TradeState]] = {
        TradeState.IDLE: {TradeState.CANDIDATE},
        
        TradeState.CANDIDATE: {
            TradeState.APPROVED, 
            TradeState.REJECTED,
            TradeState.EXPIRED,
            TradeState.CANCELLED
        },
        
        TradeState.APPROVED: {
            TradeState.PENDING_ENTRY,
            TradeState.CANCELLED,
            TradeState.ERROR
        },
        
        TradeState.REJECTED: {TradeState.IDLE},
        
        TradeState.PENDING_ENTRY: {
            TradeState.OPEN,
            TradeState.CANCELLED,
            TradeState.ERROR,
            TradeState.EXPIRED
        },
        
        TradeState.OPEN: {
            TradeState.MANAGED,
            TradeState.CLOSED,
            TradeState.ERROR
        },
        
        TradeState.MANAGED: {
            TradeState.CLOSED,
            TradeState.ERROR
        },
        
        TradeState.CLOSED: {TradeState.IDLE},
        TradeState.CANCELLED: {TradeState.IDLE},
        TradeState.EXPIRED: {TradeState.IDLE},
        TradeState.ERROR: {TradeState.IDLE, TradeState.CANCELLED}
    }
    
    @classmethod
    def is_valid_transition(cls, from_state: TradeState, to_state: TradeState) -> bool:
        """Check if state transition is valid."""
        return to_state in cls.VALID_TRANSITIONS.get(from_state, set())
    
    @classmethod
    def get_valid_next_states(cls, current_state: TradeState) -> Set[TradeState]:
        """Get all valid next states from current state."""
        return cls.VALID_TRANSITIONS.get(current_state, set())

@dataclass
class StateTransitionEvent:
    """Event representing a state transition."""
    from_state: TradeState
    to_state: TradeState
    timestamp: datetime
    reason: Optional[str] = None
    metadata: Optional[Dict] = None

class TradeStateMachine:
    """
    State machine for managing trade lifecycle.
    Prevents zombie states and ensures graceful error handling.
    """
    
    def __init__(self, initial_state: TradeState = TradeState.IDLE):
        self.current_state = initial_state
        self.transition_history = []
        self.created_at = datetime.now(timezone.utc)
        
    def transition_to(self, new_state: TradeState, reason: str = None, metadata: Dict = None) -> bool:
        """
        Attempt to transition to new state.
        Returns True if successful, False if invalid transition.
        """
        if not StateTransition.is_valid_transition(self.current_state, new_state):
            return False
        
        # Record transition
        event = StateTransitionEvent(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            metadata=metadata
        )
        
        self.transition_history.append(event)
        self.current_state = new_state
        
        return True
    
    def force_transition_to(self, new_state: TradeState, reason: str = "FORCED") -> bool:
        """
        Force transition to new state (for error recovery).
        Use with caution - bypasses validation.
        """
        event = StateTransitionEvent(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            metadata={"forced": True}
        )
        
        self.transition_history.append(event)
        self.current_state = new_state
        
        return True
    
    def can_transition_to(self, state: TradeState) -> bool:
        """Check if transition to state is valid."""
        return StateTransition.is_valid_transition(self.current_state, state)
    
    def get_valid_next_states(self) -> Set[TradeState]:
        """Get valid states that can be transitioned to."""
        return StateTransition.get_valid_next_states(self.current_state)
    
    def is_terminal_state(self) -> bool:
        """Check if current state is terminal (needs reset to continue)."""
        terminal_states = {TradeState.CLOSED, TradeState.CANCELLED, TradeState.EXPIRED, TradeState.ERROR}
        return self.current_state in terminal_states
    
    def is_active_state(self) -> bool:
        """Check if trade is in an active state (not terminal or rejected)."""
        active_states = {
            TradeState.CANDIDATE,
            TradeState.APPROVED,
            TradeState.PENDING_ENTRY,
            TradeState.OPEN,
            TradeState.MANAGED
        }
        return self.current_state in active_states
    
    def reset(self) -> bool:
        """Reset to IDLE state (only allowed from terminal states)."""
        if self.is_terminal_state():
            return self.transition_to(TradeState.IDLE, "RESET")
        return False
    
    def get_last_transition(self) -> Optional[StateTransitionEvent]:
        """Get the most recent state transition."""
        return self.transition_history[-1] if self.transition_history else None
    
    def get_state_duration(self) -> float:
        """Get how long trade has been in current state (seconds)."""
        if not self.transition_history:
            return (datetime.now(timezone.utc) - self.created_at).total_seconds()
        
        last_transition = self.transition_history[-1]
        return (datetime.now(timezone.utc) - last_transition.timestamp).total_seconds()
    
    def to_dict(self) -> Dict:
        """Export state machine to dictionary."""
        return {
            "current_state": self.current_state.value,
            "created_at": self.created_at.isoformat(),
            "transition_count": len(self.transition_history),
            "state_duration_sec": self.get_state_duration(),
            "is_terminal": self.is_terminal_state(),
            "is_active": self.is_active_state(),
            "valid_next_states": [s.value for s in self.get_valid_next_states()]
        }