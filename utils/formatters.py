import json
from datetime import datetime
from typing import Dict, Any, Optional
from utils.pydantic_models import DecisionInput, DecisionOutput

def format_decision_input(di: DecisionInput, trace_id: str = None) -> Dict[str, Any]:
    """Format DecisionInput for structured logging."""
    return {
        "decision_id": di.decision_id,
        "symbol": di.symbol,
        "trace_id": trace_id,
        "volatility_regime": di.volatility_regime,
        "spread_bp": di.spread_bp,
        "funding_bps": di.funding_bps,
        "timeframes": list(di.tfs.keys()),
        "timestamp": di.time
    }

def format_decision_output(do: DecisionOutput, trace_id: str = None) -> Dict[str, Any]:
    """Format DecisionOutput for structured logging."""
    return {
        "decision_id": getattr(do, "decision_id", None),
        "action": do.action.value,
        "symbol": do.symbol,
        "side": do.side.value if do.side else None,
        "trace_id": trace_id,
        "confidence": do.confidence,
        "size_pct": do.size_pct,
        "leverage": do.leverage,
        "valid_for_sec": do.valid_for_sec,
        "entry_type": do.entry.get("type") if do.entry else None,
        "stop_price": do.stop.price if do.stop else None,
        "tp_count": len(do.tp) if do.tp else 0,
        "notes": do.notes
    }

def format_execution_result(result: Dict[str, Any], trace_id: str = None) -> Dict[str, Any]:
    """Format execution result for structured logging."""
    return {
        "trace_id": trace_id,
        "success": result.get("success", False),
        "entry_order_id": result.get("entry_order_id"),
        "protective_orders": len(result.get("protective_order_ids", [])),
        "quantity": result.get("quantity"),
        "effective_entry": result.get("effective_entry"),
        "error": result.get("error")
    }

def format_risk_check(check_result: Dict[str, Any], symbol: str, trace_id: str = None) -> Dict[str, Any]:
    """Format risk check result for structured logging.""" 
    return {
        "trace_id": trace_id,
        "symbol": symbol,
        "allowed": check_result.get("allowed", False),
        "reason": check_result.get("reason"),
        "timestamp": datetime.utcnow().isoformat()
    }