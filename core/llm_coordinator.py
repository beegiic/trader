import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import hashlib

import openai
import anthropic

# Helpers for Responses API handling and safe logging/dumping
def _safe_usage(resp) -> int:
    try:
        u = getattr(resp, "usage", None)
        if not u:
            return 0
        return getattr(u, "total_tokens", None) or (
            (getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0))
        )
    except Exception:
        return 0

def _safe_dump(resp) -> Optional[Dict[str, Any]]:
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if isinstance(resp, dict):
            return resp
        return json.loads(str(resp)) if isinstance(str(resp), str) else None
    except Exception:
        return None

def _extract_responses_text(resp: Any) -> str:
    # Prefer resp.output_text if present (SDK helper). Fall back to concatenating message text blocks.
    try:
        t = getattr(resp, "output_text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    try:
        output = getattr(resp, "output", None)
        if output:
            parts: list[str] = []
            for item in output:
                typ = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                if typ == "message":
                    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    if content and isinstance(content, list):
                        for block in content:
                            # Prefer blocks labeled as output_text; otherwise any text field
                            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                            txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                            if btype in {"output_text", "summary_text"} and isinstance(txt, str) and txt:
                                parts.append(txt)
                            elif isinstance(txt, str) and txt:
                                parts.append(txt)
            if parts:
                return " ".join(parts)
    except Exception:
        pass
    return ""

import re
def _first_json_block(text: str) -> Optional[str]:
    try:
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
        return m.group(0) if m else None
    except Exception:
        start = text.find("{"); end = text.rfind("}")
        return text[start:end+1] if start != -1 and end != -1 and end > start else None

from utils.logging_config import LoggerMixin
from utils.pydantic_models import LLMInput, LLMOutput, DecisionAction


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    raw_response: Optional[Dict] = None


class LLMCoordinator(LoggerMixin):
    """
    Coordinates LLM requests with self-consistency and multi-LLM ensemble support.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        grok_api_key: Optional[str] = None,
        use_ensemble: bool = False,
        consistency_samples: int = 3
    ):
        super().__init__()
        
        self.use_ensemble = use_ensemble
        # Default to env-driven sample count to avoid tripling requests by default
        try:
            self.consistency_samples = int(os.getenv("LLM_SAMPLES", "1"))
        except Exception:
            self.consistency_samples = max(1, consistency_samples)
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self.grok_client = None
        
        # Configurable OpenAI model and Responses API knobs
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-5")
        try:
            self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "256"))
        except Exception:
            self.max_output_tokens = 256
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "minimal")
        
        # Initialize OpenAI client (do not log secrets)
        if openai_api_key:
            try:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
                self.logger.info(
                    "OpenAI client initialized",
                    openai_key_present=True,
                    openai_key_length=len(openai_api_key) if openai_api_key else 0,
                )
            except Exception as e:
                self.logger.error("Failed to initialize OpenAI client", error=str(e))
        
        if anthropic_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        
        # Grok would be initialized similarly when available
        if grok_api_key:
            # Placeholder for Grok API client
            self.grok_client = None  # Not implemented yet
        
        # Available models
        self.available_models = []
        if self.openai_client:
            self.available_models.append(self.openai_model)
        if self.anthropic_client:
            self.available_models.append("claude-3-sonnet-20240229")
        if self.grok_client:
            self.available_models.append("grok-1")
        
        self.logger.info(
            "LLM Coordinator initialized",
            models=self.available_models,
            use_ensemble=use_ensemble,
            consistency_samples=consistency_samples,
        )

    def _openai_kwargs_for(self, model: str, messages, *, timeout: int = 30, json_only: bool = True) -> dict:
        """Centralize OpenAI Chat Completions kwargs for non-GPT-5 models."""
        kwargs = {"model": model, "messages": messages, "timeout": timeout}
        # Only non-gpt-5 uses Chat Completions here
        kwargs["max_tokens"] = 200
        kwargs["temperature"] = 0.1
        if json_only:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs
    
    async def generate_decision(self, llm_input: LLMInput) -> tuple[LLMOutput, str]:
        """
        Generate trading decision from LLM input.
        Returns: (LLMOutput, human_summary)
        """
        
        if self.use_ensemble and len(self.available_models) > 1:
            return await self._ensemble_decision(llm_input)
        else:
            return await self._self_consistency_decision(llm_input)
    
    async def _self_consistency_decision(self, llm_input: LLMInput) -> tuple[LLMOutput, str]:
        """Generate decision using self-consistency with one model."""
        
        if not self.available_models:
            raise ValueError("No LLM models available")
        
        # Use configured OpenAI model when available
        model = self.openai_model if self.openai_client else self.available_models[0]
        prompt = self._build_trading_prompt(llm_input)
        
        # Generate samples; coalesce identical prompts unless explicitly >1
        tasks = []
        if self.consistency_samples > 1:
            # Coalesce to a single request per identical payload
            tasks = [self._call_llm(model, prompt)]
        else:
            tasks = [self._call_llm(model, prompt)]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        valid_responses = []
        for resp in responses:
            if isinstance(resp, LLMResponse):
                try:
                    decision = self._parse_llm_response(resp.content)
                    if decision:
                        valid_responses.append((decision, resp))
                except Exception as e:
                    self.logger.warning("Failed to parse LLM response", error=str(e))
        
        if not valid_responses:
            raise ValueError("No valid LLM responses received")
        
        # Apply self-consistency logic
        final_decision, summary = self._apply_self_consistency(valid_responses, llm_input.symbol)
        
        return final_decision, summary
    
    async def _ensemble_decision(self, llm_input: LLMInput) -> tuple[LLMOutput, str]:
        """Generate decision using multi-LLM ensemble."""
        
        prompt = self._build_trading_prompt(llm_input)
        
        # Get responses from all available models
        tasks = []
        for model in self.available_models:
            tasks.append(self._call_llm(model, prompt))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Parse responses
        valid_decisions = []
        for i, resp in enumerate(responses):
            if isinstance(resp, LLMResponse):
                try:
                    decision = self._parse_llm_response(resp.content)
                    if decision:
                        valid_decisions.append((decision, self.available_models[i], resp))
                except Exception as e:
                    self.logger.warning("Failed to parse ensemble response", 
                                      model=self.available_models[i], error=str(e))
        
        if not valid_decisions:
            raise ValueError("No valid ensemble responses received")
        
        # Apply ensemble voting
        final_decision, summary = self._apply_ensemble_voting(valid_decisions, llm_input.symbol)
        
        return final_decision, summary
    
    def _build_trading_prompt(self, llm_input: LLMInput) -> str:
        """Build trading prompt from LLM input."""
        
        # Convert OHLCV data to string representation (further trimmed)
        ohlcv_1m_str = self._ohlcv_to_string(llm_input.ohlcv_1m[-6:], "1m")
        ohlcv_5m_str = self._ohlcv_to_string(llm_input.ohlcv_5m[-6:], "5m")
        ohlcv_1h_str = self._ohlcv_to_string(llm_input.ohlcv_1h[-4:], "1h")
        
        prompt = f"""
You are a professional cryptocurrency trading algorithm. Analyze the following market data and provide a trading decision.

SYMBOL: {llm_input.symbol}
CURRENT BID: ${llm_input.bid:.6f}
CURRENT ASK: ${llm_input.ask:.6f}
SPREAD: {llm_input.spread_bps:.1f} bps
LAST TRADE: {llm_input.last_trade_ts}

TECHNICAL INDICATORS:
- RSI(14): {llm_input.indicators.rsi_14 or 'N/A'}
- MACD: {llm_input.indicators.macd or 'N/A'}
- BB Upper: ${llm_input.indicators.bb_upper or 'N/A'}
- BB Lower: ${llm_input.indicators.bb_lower or 'N/A'}
- EMA(20): ${llm_input.indicators.ema_20 or 'N/A'}
- EMA(200): ${llm_input.indicators.ema_200 or 'N/A'}
- ATR(14): {llm_input.indicators.atr_14 or 'N/A'}

MARKET DATA:
24h Realized Vol: {llm_input.realized_vol_24h or 'N/A'}
Funding Rate: {llm_input.funding_rate or 'N/A'}
Trading Fees: {llm_input.fees_bps} bps
Est. Slippage: {llm_input.slippage_bps} bps

PORTFOLIO CONTEXT:
Balance: ${llm_input.portfolio_balance:.2f}
Current Heat: {llm_input.portfolio_heat:.1%}
Daily P&L: ${llm_input.daily_pnl:.2f}

SENTIMENT (if available):
Score: {llm_input.sentiment_score or 'N/A'} (-1 to +1)
Strength: {llm_input.sentiment_strength or 'N/A'} (0 to 1)

RECENT PRICE ACTION:
1m Candles (last 6): {ohlcv_1m_str}
5m Candles (last 6): {ohlcv_5m_str}
1h Candles (last 4): {ohlcv_1h_str}

INSTRUCTIONS:
Analyze this data and respond with a JSON object containing your trading decision. Consider:
1. Technical setup quality and risk/reward
2. Market conditions and volatility
3. Sentiment alignment with technical picture
4. Position sizing relative to portfolio
5. Current market structure and trend

Respond ONLY with valid JSON in this exact format:
{{
    "action": "LONG" | "SHORT" | "HOLD" | "EXIT",
    "confidence": 0.75,
    "position_size_pct": 0.02,
    "entry_type": "LIMIT" | "MARKET",
    "entry_price": 61500.50,
    "stop_loss_price": 61000.00,
    "take_profit_price": 62500.00,
    "reasoning": "Brief explanation of your decision logic (max 500 chars)"
}}

Be conservative with position sizing. Only trade when you see a clear, high-probability setup.
        """
        
        return prompt.strip()
    
    def _ohlcv_to_string(self, ohlcv_data: List, interval: str) -> str:
        """Convert OHLCV data to readable string."""
        if not ohlcv_data:
            return "No data"
        
        data_points = []
        for candle in ohlcv_data:
            data_points.append(f"O:{candle.open:.2f} H:{candle.high:.2f} L:{candle.low:.2f} C:{candle.close:.2f}")
        
        return f"[{interval}] " + " | ".join(data_points[-10:])  # Last 10 for brevity
    
    async def _call_llm(self, model: str, prompt: str) -> LLMResponse:
        """Call specific LLM API."""
        # Temporary debug logging to confirm dispatch
        try:
            payload_preview = (prompt[:160] + "...") if len(prompt) > 160 else prompt
            payload_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            api = "responses" if model.startswith("gpt-5") else (
                "chat.completions" if model.startswith("gpt-") else "other"
            )
            self.logger.info(
                "Dispatching LLM request",
                model=model,
                api=api,
                timestamp=datetime.now(timezone.utc).isoformat(),
                payload_preview=payload_preview,
                payload_sha16=payload_hash,
            )
        except Exception:
            # Never let logging break the call path
            pass

        try:
            if model.startswith("gpt-") and self.openai_client:
                return await self._call_openai(model, prompt)
            elif model.startswith("claude-") and self.anthropic_client:
                return await self._call_anthropic(model, prompt)
            elif model.startswith("grok-") and self.grok_client:
                return await self._call_grok(model, prompt)
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            self.logger.error("LLM API call failed", model=model, error=str(e))
            raise
    
    async def _call_openai(self, model: str, prompt: str) -> LLMResponse:
        """Call OpenAI API; GPT-5 via Responses API, others via Chat Completions."""
        try:
            if model.startswith("gpt-5"):
                # RESPONSES API path (no response_format; enforce strict JSON via instruction)
                sys = (
                    "SYSTEM: Return ONLY a single JSON object matching this schema. No prose, no steps.\n"
                    "Keep \"reasoning\" ≤ 160 chars.\n\n"
                    "Schema:\n{\n  \"action\": \"LONG\"|\"SHORT\"|\"HOLD\"|\"EXIT\",\n  \"confidence\": 0.0-1.0,\n  \"position_size_pct\": 0.0-1.0,\n  \"entry_type\": \"LIMIT\"|\"MARKET\",\n  \"entry_price\": number|null,\n  \"stop_loss_price\": number|null,\n  \"take_profit_price\": number|null,\n  \"reasoning\": string\n}"
                )
                full_prompt = f"{sys}\n\n{prompt}"

                # Correct structured input shape: message with input_text content
                kwargs = dict(
                    model=model,
                    input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": full_prompt}]}],
                    reasoning={"effort": self.reasoning_effort},
                    text={"verbosity": "low"},
                    max_output_tokens=self.max_output_tokens,
                    timeout=30,
                )

                try:
                    resp = await self.openai_client.responses.create(**kwargs)
                except Exception as e:
                    emsg = str(e)
                    if (("invalid_value" in emsg) or ("invalid_type" in emsg)) and ("input[0]" in emsg or "input" in emsg):
                        # Final fallback to bare string
                        self.logger.info("LLM_RETRY", api="responses", retry_shape="string")
                        kwargs["input"] = full_prompt
                        resp = await self.openai_client.responses.create(**kwargs)
                    else:
                        raise

                content_text = _extract_responses_text(resp)
                finish_reason = getattr(resp, "finish_reason", None)
                used = _safe_usage(resp)

                # Single retry on empty/length or missing JSON block
                has_json = bool(_first_json_block(content_text)) if content_text else False
                if ((not content_text) or (not has_json and (finish_reason in (None, "length")))):
                    new_tokens = min(self.max_output_tokens + 128, 512)
                    self.logger.info("LLM_RETRY", model=model, api="responses", new_max_output_tokens=new_tokens)
                    kwargs["max_output_tokens"] = new_tokens
                    resp = await self.openai_client.responses.create(**kwargs)
                    content_text = _extract_responses_text(resp)
                    finish_reason = getattr(resp, "finish_reason", None)
                    used = _safe_usage(resp)

                content = content_text or ""
                if content and content.strip() and not content.strip().startswith("{"):
                    content = _first_json_block(content) or content

                if content and content.strip() and not content.strip().startswith("{"):
                    content = _first_json_block(content) or content

                self.logger.info(
                    "OpenAI API response received",
                    model=model, api="responses",
                    content_length=len(content) if content else 0,
                    has_content=bool(content), finish_reason=finish_reason,
                    tokens_used=used,
                )
                return LLMResponse(
                    content=content or "",
                    model=model,
                    timestamp=datetime.now(timezone.utc),
                    tokens_used=used,
                    raw_response=_safe_dump(resp),
                )

            # EXISTING CHAT COMPLETIONS PATH (non-gpt-5)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY a single JSON object. "
                        "Do NOT include any steps, thoughts, analysis, or explanations. "
                        "Keep \"reasoning\" ≤ 200 characters."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            kwargs = self._openai_kwargs_for(model, messages, timeout=30, json_only=True)

            try:
                response = await self.openai_client.chat.completions.create(**kwargs)
            except Exception as e:
                emsg = str(e)
                if "unsupported_parameter" in emsg and "response_format" in emsg:
                    kwargs.pop("response_format", None)
                    response = await self.openai_client.chat.completions.create(**kwargs)
                else:
                    raise

            content = response.choices[0].message.content if response.choices else ""
            finish_reason = None
            try:
                finish_reason = response.choices[0].finish_reason if response.choices else None
            except Exception:
                finish_reason = None

            self.logger.info(
                "OpenAI API response received",
                model=model,
                api="chat.completions",
                created=getattr(response, "created", None),
                response_id=getattr(response, "id", None),
                content_length=len(content) if content else 0,
                has_content=bool(content),
                tokens_used=(response.usage.total_tokens if getattr(response, "usage", None) else 0),
                finish_reason=finish_reason,
            )

            # Auto-retry once if cut off or empty content
            if (not content) or (finish_reason == "length"):
                self.logger.info(
                    "LLM_RETRY",
                    api="chat.completions",
                    reason="length_cutoff_or_empty",
                )
                try:
                    response = await self.openai_client.chat.completions.create(**kwargs)
                except Exception as e:
                    emsg = str(e)
                    if "unsupported_parameter" in emsg and "response_format" in emsg:
                        kwargs.pop("response_format", None)
                        response = await self.openai_client.chat.completions.create(**kwargs)
                    else:
                        raise
                content = response.choices[0].message.content if response.choices else ""

            return LLMResponse(
                content=content or "",
                model=model,
                timestamp=datetime.now(timezone.utc),
                tokens_used=response.usage.total_tokens if getattr(response, "usage", None) else 0,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            status_code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            self.logger.error(
                "OpenAI API call error",
                model=model,
                api=("responses" if model.startswith("gpt-5") else "chat.completions"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                error_type=type(e).__name__,
                error_message=str(e),
                http_status=status_code,
            )
            raise
    
    async def _call_anthropic(self, model: str, prompt: str) -> LLMResponse:
        """Call Anthropic API."""
        
        response = await self.anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=model,
            timestamp=datetime.now(timezone.utc),
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            raw_response=response.model_dump()
        )
    
    async def _call_grok(self, model: str, prompt: str) -> LLMResponse:
        """Call Grok API (placeholder)."""
        # This would be implemented when Grok API is available
        raise NotImplementedError("Grok API not yet implemented")
    
    def _parse_llm_response(self, content: str) -> Optional[LLMOutput]:
        """Parse LLM response content into LLMOutput."""
        
        try:
            # Extract JSON from response
            content = content.strip()
            
            # Find JSON block
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start == -1 or end == 0:
                self.logger.warning("No JSON found in LLM response")
                return None
            
            json_str = content[start:end]
            data = json.loads(json_str)
            
            # Validate and create LLMOutput
            return LLMOutput(
                action=DecisionAction(data['action']),
                confidence=float(data['confidence']),
                position_size_pct=float(data['position_size_pct']),
                entry_type=data['entry_type'],
                entry_price=float(data['entry_price']) if data.get('entry_price') else None,
                stop_loss_price=float(data['stop_loss_price']) if data.get('stop_loss_price') else None,
                take_profit_price=float(data['take_profit_price']) if data.get('take_profit_price') else None,
                reasoning=data.get('reasoning', '')
            )
            
        except Exception as e:
            self.logger.error("Failed to parse LLM response", error=str(e), content=content[:200])
            return None
    
    def _apply_self_consistency(self, responses: List[tuple], symbol: str) -> tuple[LLMOutput, str]:
        """Apply self-consistency logic to multiple responses."""
        
        decisions = [resp[0] for resp in responses]
        
        # Count actions
        action_counts = {}
        for decision in decisions:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Find consensus action
        consensus_action = max(action_counts.items(), key=lambda x: x[1])
        
        # If no clear majority, default to HOLD
        if consensus_action[1] < len(decisions) / 2:
            majority_decision = LLMOutput(
                action=DecisionAction.HOLD,
                confidence=0.3,
                position_size_pct=0.0,
                entry_type="MARKET",
                reasoning="No consensus from self-consistency check"
            )
        else:
            # Use consensus action, average other parameters
            consensus_decisions = [d for d in decisions if d.action == consensus_action[0]]
            
            avg_confidence = sum(d.confidence for d in consensus_decisions) / len(consensus_decisions)
            avg_position_size = sum(d.position_size_pct for d in consensus_decisions) / len(consensus_decisions)
            
            # Use first decision as template, adjust averages
            majority_decision = consensus_decisions[0]
            majority_decision.confidence = avg_confidence
            majority_decision.position_size_pct = avg_position_size
        
        summary = f"Self-consistency: {consensus_action[1]}/{len(decisions)} votes for {consensus_action[0].value}"
        
        self.logger.info("Self-consistency applied", 
                        symbol=symbol,
                        consensus=consensus_action,
                        final_action=majority_decision.action.value)
        
        return majority_decision, summary
    
    def _apply_ensemble_voting(self, responses: List[tuple], symbol: str) -> tuple[LLMOutput, str]:
        """Apply ensemble voting to multi-model responses."""
        
        # Simple majority voting for now
        decisions = [resp[0] for resp in responses]
        models = [resp[1] for resp in responses]
        
        action_counts = {}
        for decision in decisions:
            action = decision.action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        consensus_action = max(action_counts.items(), key=lambda x: x[1])
        
        # Get decisions for consensus action
        consensus_decisions = [decisions[i] for i, d in enumerate(decisions) 
                             if d.action == consensus_action[0]]
        
        if consensus_decisions:
            # Average the consensus decisions
            avg_confidence = sum(d.confidence for d in consensus_decisions) / len(consensus_decisions)
            avg_position_size = sum(d.position_size_pct for d in consensus_decisions) / len(consensus_decisions)
            
            final_decision = consensus_decisions[0]
            final_decision.confidence = avg_confidence
            final_decision.position_size_pct = avg_position_size
        else:
            final_decision = LLMOutput(
                action=DecisionAction.HOLD,
                confidence=0.3,
                position_size_pct=0.0,
                entry_type="MARKET",
                reasoning="No ensemble consensus"
            )
        
        participating_models = ", ".join(models)
        summary = f"Ensemble: {consensus_action[1]}/{len(decisions)} models chose {consensus_action[0].value} ({participating_models})"
        
        self.logger.info("Ensemble voting applied",
                        symbol=symbol,
                        consensus=consensus_action,
                        models=participating_models)
        
        return final_decision, summary
