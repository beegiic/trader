import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

from utils.pydantic_models import DecisionInput, DecisionOutput, Action, Side, Fees
from utils.logging_config import LoggerMixin
from core.llm_coordinator import LLMCoordinator
from strategies.scalping import build_scalp_candidates, Candidate as ScalpCandidate
from strategies.swing_trading import build_swing_candidates, Candidate as SwingCandidate

class CandidateState(str, Enum):
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

@dataclass
class ManagedCandidate:
    """Trade candidate with state management."""
    decision_id: str
    symbol: str
    candidate: Any  # ScalpCandidate or SwingCandidate
    state: CandidateState = CandidateState.CANDIDATE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    llm_decision: Optional[DecisionOutput] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    entry_order_id: Optional[str] = None
    position_qty: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if candidate has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def time_remaining_sec(self) -> int:
        """Get remaining seconds before expiry."""
        if not self.expires_at:
            return 0
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))

class TradeManager(LoggerMixin):
    """
    Manages the complete candidate pipeline:
    1. Collect candidates from strategies
    2. Build DecisionInput 
    3. Ask LLM for approval/tuning
    4. Apply risk sizing
    5. Pass to execution upon approval
    """
    
    def __init__(
        self,
        llm_coordinator: LLMCoordinator,
        risk_manager = None,
        matrix_client = None,
        execution_client = None
    ):
        super().__init__()
        self.llm_coordinator = llm_coordinator
        self.risk_manager = risk_manager
        self.matrix_client = matrix_client
        self.execution_client = execution_client
        
        # Active candidates being managed
        self.candidates: Dict[str, ManagedCandidate] = {}
        
        # Auto-approval settings (can be toggled via Matrix)
        self.auto_approve = False
        self.auto_approve_max_confidence = 0.7
        self.auto_approve_max_notional = 1000.0  # USD
        
        self.logger.info("Trade Manager initialized")
    
    async def process_symbol(self, symbol: str, state: dict, last_price: float) -> List[str]:
        """
        Process a symbol through the complete pipeline.
        Returns list of decision_ids created.
        """
        try:
            decision_ids = []
            
            # 1. Get candidates from deterministic strategies
            scalp_candidates = build_scalp_candidates(state, last_price)
            swing_candidates = build_swing_candidates(state, last_price)
            
            all_candidates = scalp_candidates + swing_candidates
            
            if not all_candidates:
                self.logger.debug("No candidates generated", symbol=symbol)
                return decision_ids
            
            self.logger.info(f"Generated {len(all_candidates)} candidates", 
                           symbol=symbol, 
                           scalp=len(scalp_candidates),
                           swing=len(swing_candidates))
            
            # 2. Process each candidate through the pipeline
            for candidate in all_candidates:
                decision_id = await self._process_candidate(candidate, state, last_price)
                if decision_id:
                    decision_ids.append(decision_id)
            
            return decision_ids
            
        except Exception as e:
            self.logger.error("Error processing symbol", symbol=symbol, error=str(e))
            return []
    
    async def _process_candidate(self, candidate: Any, state: dict, last_price: float) -> Optional[str]:
        """Process a single candidate through the pipeline."""
        try:
            # Generate unique decision ID
            decision_id = str(uuid.uuid4())[:8]
            
            # 3a. Pre-check risk constraints  
            if self.risk_manager:
                risk_check = await self._pre_check_risk(candidate)
                if not risk_check["allowed"]:
                    self.logger.info("Candidate blocked by risk check",
                                   decision_id=decision_id,
                                   symbol=candidate.symbol,
                                   reason=risk_check["reason"])
                    return None
            
            # 3b. Build DecisionInput from state + candidate
            decision_input = self._build_decision_input(decision_id, candidate, state)
            
            # 3c. Call LLM for approval/tuning
            llm_decision = await self.llm_coordinator.propose(decision_input)
            
            if not llm_decision or llm_decision.action not in [Action.OPEN]:
                self.logger.debug("LLM rejected candidate", 
                                decision_id=decision_id,
                                action=llm_decision.action.value if llm_decision else "None")
                return None
            
            # 4. Create managed candidate with TTL
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=llm_decision.valid_for_sec)
            
            managed = ManagedCandidate(
                decision_id=decision_id,
                symbol=candidate.symbol,
                candidate=candidate,
                state=CandidateState.CANDIDATE,
                expires_at=expires_at,
                llm_decision=llm_decision
            )
            
            self.candidates[decision_id] = managed
            
            # 5. Check for auto-approval or send to Matrix
            if self._should_auto_approve(llm_decision):
                await self._approve_candidate(decision_id, "AUTO")
            else:
                await self._request_manual_approval(managed)
            
            return decision_id
            
        except Exception as e:
            self.logger.error("Error processing candidate", error=str(e))
            return None
    
    def _build_decision_input(self, decision_id: str, candidate: Any, state: dict) -> DecisionInput:
        """Build DecisionInput from state and candidate info."""
        return DecisionInput(
            decision_id=decision_id,
            symbol=candidate.symbol,
            tfs=state.get("tfs", {}),
            fees=Fees(
                taker_bps=state.get("fees", {}).get("taker_bps", 5),
                maker_bps=state.get("fees", {}).get("maker_bps", 1)
            ),
            spread_bp=state.get("spread_bp", 2.0),
            volatility_regime=state.get("volatility_regime", "medium"),
            funding_bps=state.get("funding_bps", 0),
            oi_5m_chg_pct=state.get("oi_5m_chg_pct", 0.0),
            time=datetime.now(timezone.utc).isoformat()
        )
    
    async def _pre_check_risk(self, candidate: Any) -> Dict[str, Any]:
        """Pre-trade risk checks."""
        if not self.risk_manager:
            return {"allowed": True, "reason": None}
        
        # Implement risk checks via risk_manager
        # - symbol allowlist
        # - max open positions  
        # - max leverage_cap
        # - max portfolio heat
        # - daily loss halt
        
        try:
            # This would call actual risk manager methods
            return {"allowed": True, "reason": None}
        except Exception as e:
            return {"allowed": False, "reason": f"Risk check error: {e}"}
    
    def _should_auto_approve(self, llm_decision: DecisionOutput) -> bool:
        """Check if decision meets auto-approval criteria."""
        if not self.auto_approve:
            return False
        
        if llm_decision.confidence < self.auto_approve_max_confidence:
            return False
            
        # Estimate notional value (rough approximation)
        notional = llm_decision.size_pct * 10000  # Assume $10k account
        if notional > self.auto_approve_max_notional:
            return False
            
        return True
    
    async def _request_manual_approval(self, managed: ManagedCandidate):
        """Send approval request to Matrix."""
        if not self.matrix_client:
            self.logger.warning("No Matrix client for manual approval", 
                              decision_id=managed.decision_id)
            return
        
        try:
            # Build Matrix message with approval buttons
            candidate = managed.candidate
            llm = managed.llm_decision
            
            message = (
                f"🎯 **TRADE PROPOSAL** `{managed.decision_id}`\n"
                f"**Symbol:** {candidate.symbol} {llm.side.value if llm.side else 'N/A'}\n"
                f"**Entry:** {candidate.entry_type} @ {candidate.entry_price or 'market'}\n"
                f"**Stop:** {candidate.stop_price:.6f}\n"
                f"**TPs:** {candidate.tp_R}\n"
                f"**Size:** {llm.size_pct:.1%} ({llm.size_pct * 10000:.0f} USD)\n"
                f"**R/R:** {len(candidate.tp_R)}:1\n"
                f"**Confidence:** {llm.confidence:.1%}\n"
                f"**TTL:** {managed.time_remaining_sec()}s\n"
                f"**Rationale:** {candidate.rationale}\n\n"
                f"React with ✅ to approve or ❌ to reject"
            )
            
            await self.matrix_client.send_message(message)
            
        except Exception as e:
            self.logger.error("Error sending Matrix approval request", 
                            decision_id=managed.decision_id, error=str(e))
    
    async def approve_candidate(self, decision_id: str, approved_by: str = "MANUAL") -> bool:
        """Approve a candidate for execution."""
        return await self._approve_candidate(decision_id, approved_by)
    
    async def _approve_candidate(self, decision_id: str, approved_by: str) -> bool:
        """Internal approval processing."""
        try:
            if decision_id not in self.candidates:
                self.logger.warning("Decision ID not found", decision_id=decision_id)
                return False
            
            managed = self.candidates[decision_id]
            
            if managed.is_expired():
                managed.state = CandidateState.EXPIRED
                self.logger.info("Candidate expired before approval", decision_id=decision_id)
                return False
            
            # Update state
            managed.state = CandidateState.APPROVED
            managed.approved_by = approved_by
            managed.approved_at = datetime.now(timezone.utc)
            
            # Pass to execution
            if self.execution_client:
                success = await self._execute_candidate(managed)
                if success:
                    managed.state = CandidateState.PENDING_ENTRY
                else:
                    managed.state = CandidateState.REJECTED
                    return False
            
            self.logger.info("Candidate approved and executed", 
                           decision_id=decision_id, 
                           approved_by=approved_by)
            return True
            
        except Exception as e:
            self.logger.error("Error approving candidate", decision_id=decision_id, error=str(e))
            return False
    
    async def reject_candidate(self, decision_id: str, rejected_by: str = "MANUAL") -> bool:
        """Reject a candidate."""
        try:
            if decision_id not in self.candidates:
                return False
                
            managed = self.candidates[decision_id]
            managed.state = CandidateState.REJECTED
            
            self.logger.info("Candidate rejected", decision_id=decision_id, rejected_by=rejected_by)
            return True
            
        except Exception as e:
            self.logger.error("Error rejecting candidate", decision_id=decision_id, error=str(e))
            return False
    
    async def _execute_candidate(self, managed: ManagedCandidate) -> bool:
        """Execute approved candidate via execution client."""
        try:
            # This would integrate with execution.py
            # For now, just log the intent
            
            candidate = managed.candidate  
            llm = managed.llm_decision
            
            self.logger.info("Executing trade", 
                           decision_id=managed.decision_id,
                           symbol=candidate.symbol,
                           side=llm.side.value if llm.side else None,
                           entry_type=candidate.entry_type,
                           size_pct=llm.size_pct)
            
            # TODO: Call execution_client.place_bracket_order()
            return True
            
        except Exception as e:
            self.logger.error("Execution failed", decision_id=managed.decision_id, error=str(e))
            return False
    
    async def cleanup_expired(self):
        """Clean up expired candidates."""
        expired_ids = []
        
        for decision_id, managed in self.candidates.items():
            if managed.is_expired() and managed.state == CandidateState.CANDIDATE:
                managed.state = CandidateState.EXPIRED
                expired_ids.append(decision_id)
        
        if expired_ids:
            self.logger.info(f"Expired {len(expired_ids)} candidates", expired=expired_ids)
    
    def get_active_candidates(self) -> Dict[str, ManagedCandidate]:
        """Get all active candidates."""
        return {k: v for k, v in self.candidates.items() 
                if v.state not in [CandidateState.CLOSED, CandidateState.CANCELLED, CandidateState.EXPIRED]}
    
    def set_auto_approval(self, enabled: bool, max_confidence: float = 0.7, max_notional: float = 1000.0):
        """Configure auto-approval settings."""
        self.auto_approve = enabled
        self.auto_approve_max_confidence = max_confidence
        self.auto_approve_max_notional = max_notional
        
        self.logger.info("Auto-approval updated", 
                        enabled=enabled, 
                        max_confidence=max_confidence,
                        max_notional=max_notional)