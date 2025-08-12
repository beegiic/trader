import json
import time
import uuid
import os
import logging
from typing import Optional
import openai
import anthropic
from datetime import datetime, timezone

from utils.pydantic_models import DecisionInput, DecisionOutput, Action, Side, EntryType
from utils.logging_config import LoggerMixin

class LLMCoordinator(LoggerMixin):
    """
    Provider-agnostic LLM coordinator for trading decisions.
    Receives DecisionInput and returns DecisionOutput.
    No raw candles or images - only numeric state vectors.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 512,
        self_consistency: int = 1,
        **kwargs
    ):
        """
        Backward-compatible initializer.
        - provider is optional; we infer it from aliases or available API keys/env.
        - Accepts legacy aliases: type, vendor, engine, backend.
        - Accepts both provider-specific keys and generic api_key; also reads env vars.
        - Ignores unknown kwargs instead of crashing.
        """
        super().__init__()
        self.log = logging.getLogger(__name__)

        # Accept legacy aliases for provider
        alias_provider = None
        for k in ("type", "vendor", "engine", "backend", "llm_provider"):
            if k in kwargs and isinstance(kwargs[k], str):
                alias_provider = kwargs.pop(k)
                break

        # Infer provider if not explicitly set
        prov = (provider or alias_provider or "").strip().lower()
        if not prov:
            # Heuristics: prefer anthropic if given; else openai; else generic
            if anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
                prov = "anthropic"
            elif openai_api_key or os.getenv("OPENAI_API_KEY"):
                prov = "openai"
            elif api_key or os.getenv("LLM_API_KEY"):
                # default to openai when only a generic key is present
                prov = "openai"
            else:
                # final fallback
                prov = "openai"

        self.provider = prov
        self.model = model
        self.max_tokens = int(max_tokens)
        self.n = max(1, int(self_consistency))

        # Resolve effective API key by provider
        eff_key = None
        if self.provider in ("openai", "gpt", "oai", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-5"):
            eff_key = openai_api_key or api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        elif self.provider in ("anthropic", "claude"):
            eff_key = anthropic_api_key or api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        else:
            # Unknown provider: try generic, then common envs
            eff_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        self.api_key = eff_key
        if not self.api_key:
            self.log.warning("LLMCoordinator: No API key resolved for provider=%s; API calls will fail if attempted.", self.provider)

        # Soft-log any unexpected kwargs to help debug config drift
        unexpected = set(kwargs.keys())
        if unexpected:
            self.log.debug("LLMCoordinator: Ignoring unexpected kwargs: %s", ", ".join(sorted(unexpected)))
        
        # Initialize clients
        if self.provider in ("openai", "gpt", "gpt-4", "gpt-4o", "gpt-4.1", "gpt-5", "oai"):
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
            self.model = self.model or os.getenv("OPENAI_MODEL", "gpt-4")
        elif self.provider in ("anthropic", "claude"):
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.model = self.model or "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        self.logger.info(f"LLM Coordinator initialized with {self.provider}, model: {self.model}")

    def _normalize_input_payload(self, payload):
        """
        Accepts either a DecisionInput instance, LLMInput instance, or plain dict.
        Returns a tuple: (kind, dict_json_string)
        kind in {"decision","llm"} used only for logging/routing.
        """
        from utils.pydantic_models import DecisionInput, LLMInput
        if isinstance(payload, DecisionInput):
            return "decision", payload.model_dump_json()
        if isinstance(payload, LLMInput):
            return "llm", payload.model_dump_json()
        if isinstance(payload, dict):
            # try decision first
            try:
                di = DecisionInput(**payload)
                return "decision", di.model_dump_json()
            except Exception:
                # try llm compat
                try:
                    li = LLMInput(**payload)
                    return "llm", li.model_dump_json()
                except Exception as e:
                    self.log.warning("LLMCoordinator: input normalization failed: %s", e)
                    return "invalid", None
        self.log.warning("LLMCoordinator: unsupported input type: %s", type(payload))
        return "invalid", None

    def build_prompt(self, di: DecisionInput) -> str:
        """Build concise, numeric-only prompt for trading decision."""
        return self.build_prompt_json(di.model_dump_json(), kind="decision")

    def build_prompt_json(self, payload_json: str, kind: str = "decision") -> str:
        return (
            "You are a trading strategist. Approve/tune parameters or cancel.\n"
            "Return ONLY valid JSON matching this EXACT schema:\n"
            "{\n"
            '  "action": "OPEN|CANCEL|FLAT",\n'
            '  "symbol": "REQUIRED_SYMBOL_FROM_INPUT",\n'
            '  "side": "LONG|SHORT",\n'
            '  "entry": {"type": "market", "price": null},\n'
            '  "size_pct": 0.25,\n'
            '  "leverage": 10,\n'
            '  "stop": {"price": 123.45},\n'
            '  "tp": [{"price": 125.0, "type": "R"}],\n'
            '  "trail": {"type": "ATR", "mult": 0.8},\n'
            '  "valid_for_sec": 90,\n'
            '  "confidence": 0.5,\n'
            '  "notes": "reasoning"\n'
            "}\n\n"
            f"CRITICAL: Always include the exact symbol from input. Input: {payload_json}\n"
            "Rules: If setup aligns across TFs -> action=OPEN else CANCEL. Stop required for OPEN.\n"
        )

    async def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API."""
        try:
            if self.model.startswith("gpt-5"):
                # Use Responses API for GPT-5
                response = await self.client.responses.create(
                    model=self.model,
                    input=[{
                        "type": "message",
                        "role": "user", 
                        "content": [{"type": "input_text", "text": prompt}]
                    }],
                    max_output_tokens=self.max_tokens,
                    timeout=30
                )
                # Extract text from responses format
                content = getattr(response, "output_text", "") or ""
                if not content:
                    try:
                        parts = []
                        for item in getattr(response, "output", []) or []:
                            typ = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                            if typ == "message":
                                msg_content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                                if msg_content and isinstance(msg_content, list):
                                    for block in msg_content:
                                        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                                        txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                                        if isinstance(txt, str) and txt:
                                            parts.append(txt)
                        content = " ".join(parts)
                    except Exception:
                        pass
                return content
            else:
                # Use Chat Completions for other models
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON matching the DecisionOutput schema."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    timeout=30
                )
                return response.choices[0].message.content if response.choices else None
        except Exception as e:
            self.logger.error("OpenAI API call failed", error=str(e))
            return None

    async def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text if response.content else None
        except Exception as e:
            self.logger.error("Anthropic API call failed", error=str(e))
            return None

    async def _provider_call(self, prompt: str) -> Optional[str]:
        """Call the configured provider."""
        if self.provider == "openai":
            return await self._call_openai(prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            return None

    async def _call(self, prompt: str) -> Optional[DecisionOutput]:
        """Make LLM call and parse response."""
        raw = await self._provider_call(prompt)
        if not raw:
            return None
            
        try:
            # Handle JSON wrapped in tags
            if "<json>" in raw:
                raw = raw.split("<json>")[1].split("</json>")[0]
            elif "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "{" in raw and "}" in raw:
                # Extract JSON from text
                start = raw.find("{")
                end = raw.rfind("}") + 1
                raw = raw[start:end]
            
            data = json.loads(raw)
            return DecisionOutput(**data)
        except Exception as e:
            self.logger.warning("Failed to parse LLM response", error=str(e), raw_preview=raw[:200])
            return None

    async def propose(self, di) -> Optional[DecisionOutput]:
        """
        Generate trading decision proposal from DecisionInput or LLMInput.
        Returns None if LLM fails or produces invalid output.
        """
        try:
            # Normalize input payload
            kind, payload_json = self._normalize_input_payload(di)
            if payload_json is None:
                return None  # drop safely
            
            # Extract symbol and decision_id for logging (works for both input types)
            symbol = getattr(di, 'symbol', 'UNKNOWN')
            decision_id = getattr(di, 'decision_id', getattr(di, 'symbol', 'UNKNOWN'))
            
            self.logger.info(
                "LLM decision request",
                decision_id=decision_id,
                symbol=symbol,
                input_kind=kind
            )
            
            # Try multiple times for self-consistency
            for attempt in range(self.n):
                prompt = self.build_prompt_json(payload_json, kind=kind)
                out = await self._call(prompt)
                
                if out:
                    # Validate output matches input symbol
                    if out.symbol != symbol:
                        self.logger.warning("Symbol mismatch in LLM output", 
                                          input_symbol=symbol, 
                                          output_symbol=out.symbol)
                        continue
                    
                    self.logger.info(
                        "LLM decision generated", 
                        decision_id=decision_id,
                        action=out.action.value,
                        side=out.side.value if out.side else None,
                        confidence=out.confidence,
                        size_pct=out.size_pct
                    )
                    return out
                
                self.logger.debug("LLM attempt failed, retrying", attempt=attempt+1)
            
            self.logger.warning("All LLM attempts failed", decision_id=decision_id)
            return None
            
        except Exception as e:
            self.logger.error("Error in LLM propose", error=str(e), decision_id=getattr(di, 'decision_id', 'UNKNOWN'))
            return None

    # Legacy methods for backward compatibility
    async def generate_decision(self, llm_input) -> tuple:
        """Legacy method - converts old format to new and back."""
        # This is a compatibility shim - in practice should be removed
        # when old code is fully migrated
        decision_id = str(uuid.uuid4())[:8]
        
        # Create mock DecisionInput from legacy LLMInput
        di = DecisionInput(
            decision_id=decision_id,
            symbol=llm_input.symbol,
            tfs={
                "1m": {
                    "trend": 0, "mom": 0, "atr_pct": 1.0, "rsi": 50,
                    "bb_pos": 0.5, "sr_top_dist_bp": 100, "sr_bot_dist_bp": 100,
                    "imb5": 0.5, "breakout_pending": False
                }
            },
            fees={"taker_bps": 5, "maker_bps": 1},
            spread_bp=llm_input.spread_bps,
            volatility_regime="medium",
            time=datetime.now(timezone.utc).isoformat()
        )
        
        output = await self.propose(di)
        if not output:
            return None, "LLM failed to generate decision"
            
        # Convert back to legacy format
        from utils.pydantic_models import LLMOutput, DecisionAction
        
        legacy_output = LLMOutput(
            action=DecisionAction.HOLD,  # Default fallback
            confidence=output.confidence,
            position_size_pct=output.size_pct,
            entry_type=output.entry.get("type", "market"),
            entry_price=output.entry.get("price"),
            reasoning=output.notes or "No reasoning provided"
        )
        
        return legacy_output, f"Decision: {output.action.value}"