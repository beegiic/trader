#!/usr/bin/env python3
"""
Simple smoketest for the LLM call path used by the bot.

What it does:
- Loads .env to get OPENAI_API_KEY (no hard-coded keys)
- Uses the same AsyncOpenAI client and model name as the bot (gpt-5)
- Sends a minimal JSON-only prompt to verify reachability/auth/model
- Prints the response content or the exact error encountered
"""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
import openai


def _openai_kwargs_for(model: str, messages, *, timeout: int = 30, json_only: bool = True) -> dict:
    # Chat Completions kwargs for non-gpt-5 models
    kwargs = {"model": model, "messages": messages, "timeout": timeout}
    kwargs["max_tokens"] = 200
    kwargs["temperature"] = 0.1
    if json_only:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


async def run_smoketest():
    load_dotenv()  # Ensure .env is loaded when running outside systemd

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"[smoketest] OPENAI_API_KEY present: {bool(api_key)}; length: {len(api_key) if api_key else 0}")

    if not api_key:
        print("[smoketest] ERROR: OPENAI_API_KEY is not set in environment.")
        return 2

    # Initialize client the same way as the bot
    client = openai.AsyncOpenAI(api_key=api_key)

    # Use the same model name as the bot (configurable)
    model = os.getenv("OPENAI_MODEL", "gpt-5")

    # Minimal, strictly-JSON prompt
    prompt = (
        "Respond ONLY with valid JSON in this exact format: {\"ok\": true, \"ts\": \"YYYY-MM-DDTHH:MM:SSZ\"}"
    )

    print(
        f"[smoketest] Sending request to model={model} at {datetime.now(timezone.utc).isoformat()}"
    )

    try:
        if model.startswith("gpt-5"):
            prompt = (
                "Return ONLY one JSON object with keys: action, confidence, position_size_pct, entry_type, "
                "entry_price, stop_loss_price, take_profit_price, reasoning (<=160 chars). Now reply with: "
                "{\"action\":\"HOLD\",\"confidence\":0.5,\"position_size_pct\":0.0,\"entry_type\":\"MARKET\",\"entry_price\":0,\"stop_loss_price\":0,\"take_profit_price\":0,\"reasoning\":\"smoke\"}"
            )
            kwargs = dict(
                model=model,
                input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                reasoning={"effort": os.getenv("OPENAI_REASONING_EFFORT","minimal")},
                text={"verbosity": "low"},
                max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS","384")),
                timeout=30,
            )
            try:
                resp = await client.responses.create(**kwargs)
            except Exception as e:
                emsg = str(e)
                if (("invalid_value" in emsg) or ("invalid_type" in emsg)) and ("input[0]" in emsg or "input" in emsg):
                    # Final fallback to bare string
                    kwargs["input"] = prompt
                    resp = await client.responses.create(**kwargs)
                else:
                    raise

            out = getattr(resp, "output_text", "") or ""
            if not out:
                try:
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        typ = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                        if typ == "message":
                            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                            if content and isinstance(content, list):
                                for block in content:
                                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                                    txt = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                                    if btype in {"output_text", "summary_text"} and isinstance(txt, str) and txt:
                                        parts.append(txt)
                                    elif isinstance(txt, str) and txt:
                                        parts.append(txt)
                    out = " ".join(parts)
                except Exception:
                    out = ""
            print(f"OK (responses): len={len(out)} preview={out[:200]!r}")
            return 0 if out else 1
        else:
            messages = [
                {"role": "system", "content": "Return ONLY a single JSON object with keys action, confidence."},
                {"role": "user", "content": '{"symbol":"BTCUSDT","bid":61000,"ask":61001}'},
            ]
            kwargs = _openai_kwargs_for(model, messages, timeout=30, json_only=True)
            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as e:
                emsg = str(e)
                if "unsupported_parameter" in emsg and "response_format" in emsg:
                    kwargs.pop("response_format", None)
                    resp = await client.chat.completions.create(**kwargs)
                else:
                    raise

            content = resp.choices[0].message.content if resp.choices else ""
            finish_reason = None
            try:
                finish_reason = resp.choices[0].finish_reason if resp.choices else None
            except Exception:
                pass
            usage = getattr(resp, "usage", None)
            print(
                f"[smoketest] Result: status=200 id={getattr(resp,'id',None)} finish_reason={finish_reason} has_content={bool(content)} tokens={getattr(usage,'total_tokens',None)}"
            )
            if not content:
                return 1
            print(content)
            return 0

    except Exception as e:
        status_code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
        print(
            f"[smoketest] ERROR calling OpenAI: type={type(e).__name__} status={status_code} message={str(e)}"
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_smoketest())
    raise SystemExit(exit_code)
