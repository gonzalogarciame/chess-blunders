"""
Shared LLM plumbing -- one place for the Groq/Anthropic client calls so report.py (the
coaching narrative) and explain.py (per-blunder "why best / why blunder") don't each carry
their own copy.

Groq is the default (free tier, no credit card -- console.groq.com); ANTHROPIC_API_KEY is
used only if GROQ_API_KEY isn't set. If neither is set, `call_llm` returns None and the
caller falls back to whatever it does without an LLM.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Loads GROQ_API_KEY / ANTHROPIC_API_KEY from a .env file in the project root if present, so
# the key only has to be entered once (not re-set as a shell env var every session).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_MODEL = "claude-sonnet-5"


def _call_groq(prompt: str, api_key: str, max_tokens: int = 1024,
               *, model: str | None = None, json_mode: bool = False) -> str:
    import groq

    client = groq.Groq(api_key=api_key)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, api_key: str, max_tokens: int = 1024,
                    *, model: str | None = None, json_mode: bool = False) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def llm_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def call_llm(prompt: str, max_tokens: int = 1024,
             *, groq_model: str | None = None, json_mode: bool = False) -> str | None:
    """Groq first (this project's default), then Anthropic, then None. Raises nothing on a
    missing key -- returns None so the caller can fall back. `groq_model` / `json_mode` only
    take effect on the Groq path (Anthropic has no JSON response mode; steer it via the
    prompt instead)."""
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if groq_key:
        return _call_groq(prompt, groq_key, max_tokens, model=groq_model, json_mode=json_mode)
    if anthropic_key:
        return _call_anthropic(prompt, anthropic_key, max_tokens)
    return None
