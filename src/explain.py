"""
Set 8 (part 3) -- a short, plain-English "why" for each trainer puzzle.

motifs.py already tags every leak-category blunder with the engine's best replies, the line
that refutes the move played, and a few heuristic labels. This module turns that into two
sentences per puzzle:

- **why_best**   -- the idea behind the engine's move.
- **why_blunder** -- exactly what goes wrong after the move the player actually played.

Two ways it's produced, in order of preference:

1. **LLM** (Groq's free `openai/gpt-oss-20b` by default, Anthropic if that key is set
   instead) -- same plumbing as report.py's coaching narrative (`src/llm.py`). The engine
   lines and tags are handed to the model; it writes the prose. Groq's free tier caps
   tokens *per day*, so a first run may run out and finish the remainder on heuristics --
   re-run the next day and it keeps the LLM notes it has and upgrades the rest.
2. **Heuristic fallback** -- if no API key is set, or a given call fails / doesn't return
   valid JSON, the sentence is templated from motifs.py's tags instead. Blunt but always
   available, and unit-tested.

Output is a sidecar file, `data/processed/blunder_explanations_<user>.json`
(`{ "<game_id>-<ply>": {"why_best": ..., "why_blunder": ..., "source": "llm"|"heuristic"} }`),
which trainer.py merges into each puzzle. motifs.py's own output is left untouched, so this
step is independently re-runnable.

Resumable like motifs.py: work is checkpointed per batch, so a killed run picks up where it
left off. The LLM callable is injected (same pattern as motifs.py's Analyzer) so
tests/test_explain.py runs against a deterministic fake with no network.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import call_llm, llm_available

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

USERNAME = "gonzalopelotas"
OUT_PATH = PROCESSED_DIR / f"blunder_explanations_{USERNAME}.json"

SLEEP_BETWEEN_CALLS = 3.0   # seconds -- pace calls under Groq's per-minute rate limit
GROQ_MODEL = "openai/gpt-oss-20b"  # free tier; 120b's daily token cap is small and 20b is
                                   # much faster. Both write fine for this summarise-the-lines task.
EXPLAIN_MAX_TOKENS = 1200   # gpt-oss "thinks" before answering -- leave room or it returns ""
RETRY_WAITS = (6, 18)       # backoff on a per-minute rate-limit / transient error
SAVE_EVERY = 10             # flush the sidecar this often, so a killed run resumes cleanly
CIRCUIT_BREAK_AFTER = 3     # consecutive LLM misses -> assume the daily quota is gone,
                            # finish the rest on heuristics fast (re-run later to upgrade)

LLMFn = Callable[[str], "str | None"]


# --------------------------------------------------------------------------------------------
# heuristic fallback (pure -- unit-tested)
# --------------------------------------------------------------------------------------------

def _acceptable_sans(rec: dict) -> str:
    sans = [m["san"] for m in rec.get("acceptable_moves", []) if m.get("san")]
    return " or ".join(sans) if sans else (rec.get("best_line_san", "").split() or ["the engine move"])[0]


def _refutation_reply(rec: dict) -> str | None:
    parts = (rec.get("refutation_line_san") or "").split()
    return parts[1] if len(parts) >= 2 else None


def build_heuristic_explanation(rec: dict) -> dict:
    """Two sentences templated from motifs.py's tags. Deliberately blunt -- this is the
    no-API-key path, not the headline feature."""
    played = rec["played"]["san"]
    best = _acceptable_sans(rec)
    top = (rec.get("top_lines") or [{}])[0]
    ev, main_line = top.get("eval", "?"), top.get("san", rec.get("best_line_san", ""))
    material = rec.get("material_label", "positional")
    refutation = rec.get("refutation_type", "quiet")
    ref_line = rec.get("refutation_line_san") or played
    reply = _refutation_reply(rec)
    wp = round(rec.get("wp_loss", 0))
    gone = f"~{wp} win% gone"

    lead = "You were winning here. " if rec.get("advantage_state") == "threw away a winning position" else ""
    why_best = f"{lead}The engine's move is {best} ({ev}). The line runs {main_line}." if main_line \
        else f"{lead}The engine's move is {best} ({ev})."

    # material_label is a past-tense verb phrase ("hung a piece", "dropped a pawn", ...) --
    # phrase the sentences so it reads as "you {material_label}".
    if refutation == "allowed_mate":
        why_blunder = f"{played} allows forced mate: {ref_line}."
    elif refutation == "back_rank" and reply:
        why_blunder = f"{played} drops the back rank to {reply} — {ref_line} ({gone})."
    elif refutation == "fork" and reply:
        tail = f" You {material}." if material != "positional" else ""
        why_blunder = f"{played} runs into {reply}, a fork.{tail} Line: {ref_line} ({gone})."
    elif refutation == "capture" and reply:
        if material != "positional":
            why_blunder = f"After {played}, {reply} just takes — you {material} for nothing ({gone})."
        else:
            why_blunder = f"After {played}, {reply} wins the initiative — {ref_line} ({gone})."
    elif refutation == "check" and reply:
        worse = f"you {material}" if material != "positional" else "you're clearly worse"
        why_blunder = f"{played} allows {reply}, and after {ref_line} {worse} ({gone})."
    else:
        if material != "positional":
            why_blunder = f"{played} loses material — after {ref_line} you {material} ({gone})."
        else:
            why_blunder = f"{played} throws the position away with no clear tactic; {ref_line} leaves you worse ({gone})."

    return {"why_best": why_best, "why_blunder": why_blunder}


# --------------------------------------------------------------------------------------------
# LLM path
# --------------------------------------------------------------------------------------------

def build_llm_prompt(rec: dict) -> str:
    side = rec.get("side_to_move", "white")
    played = rec["played"]["san"]
    tops = "\n".join(
        f"  {i + 1}. {l.get('san', '')}   ({l.get('eval', '?')})"
        for i, l in enumerate(rec.get("top_lines", [])[:3])
    )
    return f"""Chess coach note on one position from a club player's own game. Keep it short and concrete.

Position (FEN, {side} to move): {rec['fen_before']}
The player played: {played}
Engine's line refuting that move: {rec.get('refutation_line_san') or '(none found)'}
Engine's top choices instead:
{tops}
Tags (rough): material result "{rec.get('material_label')}"; refutation type "{rec.get('refutation_type')}"; the player {rec.get('advantage_state')}.

Output a single JSON object and nothing else:
{{"why_best": "1-2 plain sentences: the idea behind the engine's move -- the pieces and squares it works with, not just notation", "why_blunder": "1-2 plain sentences: what {played} allows -- name the refuting move and what it wins or forces"}}"""


# Non-breaking hyphen -> hyphen, ellipsis char -> "..."; str.split() below folds the
# assorted no-break spaces the model emits into ordinary spaces.
# Non-breaking hyphen -> hyphen, ellipsis char -> "..."; str.split() below folds
# the assorted no-break spaces the model emits into ordinary spaces.
_TIDY = {"‑": "-", "…": "..."}


def _tidy(s: str) -> str:
    for a, b in _TIDY.items():
        s = s.replace(a, b)
    return " ".join(s.split()).strip()


def _parse_llm_json(text: str | None) -> dict | None:
    if not text:
        return None
    s = text.strip()
    if "{" in s and "}" in s:
        s = s[s.index("{"): s.rindex("}") + 1]
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    wb, wbl = obj.get("why_best"), obj.get("why_blunder")
    if isinstance(wb, str) and isinstance(wbl, str) and wb.strip() and wbl.strip():
        return {"why_best": _tidy(wb), "why_blunder": _tidy(wbl)}
    return None


class QuotaExhausted(Exception):
    """Groq's per-day token cap is spent -- no point calling the LLM again this run."""


# Message fragments that mean "try again", not "your code is wrong". gpt-oss under Groq's
# JSON mode intermittently emits JSON the server rejects ("failed to validate/generate
# JSON") -- resampling usually fixes it, so it's treated as transient too.
_TRANSIENT = ("overloaded", "timeout", "timed out", "503", "502", "connection",
              "temporarily", "validate json", "generate json",
              "per minute", "tpm", "rpm", "requests per")
_DAILY_CAP = ("per day", "tpd", "tokens per day")


def default_llm_fn(prompt: str) -> str | None:
    """The real LLM call used by `run()`: `call_llm` on gpt-oss in JSON mode, with backoff
    on Groq's per-minute rate limits and its occasional JSON-validation hiccups. Returns None
    (-> heuristic for this one puzzle) if it still can't get a usable answer; raises
    QuotaExhausted if the *daily* token cap is hit (run() then finishes on heuristics).
    Injected, so the tests never touch it."""
    for wait in (0, *RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            out = call_llm(prompt, EXPLAIN_MAX_TOKENS, groq_model=GROQ_MODEL, json_mode=True)
        except Exception as exc:  # noqa: BLE001 -- classify by message, re-raise real bugs
            msg = str(exc).lower()
            if any(t in msg for t in _DAILY_CAP):
                raise QuotaExhausted(str(exc)[:200]) from exc
            if "429" in msg or "rate limit" in msg or any(t in msg for t in _TRANSIENT):
                continue
            raise
        if out and out.strip():
            return out
    return None


def explain_one(rec: dict, llm_fn: LLMFn | None) -> dict:
    heuristic = build_heuristic_explanation(rec)
    if llm_fn is None:
        return {**heuristic, "source": "heuristic"}
    try:
        parsed = _parse_llm_json(llm_fn(build_llm_prompt(rec)))
    except QuotaExhausted:
        raise
    except Exception as exc:  # network / rate-limit / SDK error -- fall back for this one
        print(f"  ! LLM call failed ({exc.__class__.__name__}); using the heuristic", flush=True)
        parsed = None
    if parsed is None:
        return {**heuristic, "source": "heuristic"}
    return {**parsed, "source": "llm"}


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------

def puzzle_id(rec: dict) -> str:
    return f"{rec['game_id']}-{rec['ply']}"


def load_motifs() -> list[dict]:
    path = PROCESSED_DIR / f"blunder_motifs_{USERNAME}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `python src/motifs.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def explain_records(records: list[dict], llm_fn: LLMFn | None, *, sleep_s: float = 0.0) -> dict[str, dict]:
    """id -> {why_best, why_blunder, source}, one entry per record. Pure and in-memory --
    the tests call this directly; `run()` adds resumption, pacing and the circuit breaker."""
    out: dict[str, dict] = {}
    for i, rec in enumerate(records):
        out[puzzle_id(rec)] = explain_one(rec, llm_fn)
        if llm_fn is not None and sleep_s and i < len(records) - 1:
            time.sleep(sleep_s)
    return out


def _load_existing() -> dict[str, dict]:
    if not OUT_PATH.exists():
        return {}
    try:
        data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _save(explanations: dict[str, dict]) -> None:
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    os.replace(tmp, OUT_PATH)


def run(llm_fn: "LLMFn | None | str" = "auto") -> dict[str, dict]:
    records = load_motifs()
    if llm_fn == "auto":
        if not llm_available():
            print("Neither GROQ_API_KEY nor ANTHROPIC_API_KEY is set -- explanations will be the "
                  "templated heuristic ones. Get a free key at console.groq.com, set "
                  "GROQ_API_KEY, and rerun for LLM-written explanations.\n")
        llm_fn = default_llm_fn if llm_available() else None

    # The sidecar itself is the resume point: keep every entry that already has an LLM note,
    # re-attempt everything else (heuristic entries from an earlier quota-limited run, too).
    have = _load_existing()
    out = dict(have)
    todo = [r for r in records if have.get(puzzle_id(r), {}).get("source") != "llm"]
    kept = len(records) - len(todo)
    mode = "LLM + heuristic fallback" if llm_fn is not None else "heuristic only"
    print(f"{USERNAME}: {len(records)} puzzles -- {kept} already have an LLM note, "
          f"{len(todo)} to do ({mode})")

    llm_live = llm_fn is not None
    misses = 0
    for i, rec in enumerate(todo):
        if llm_live:
            try:
                res = explain_one(rec, llm_fn)
            except QuotaExhausted as exc:
                llm_live = False
                res = {**build_heuristic_explanation(rec), "source": "heuristic"}
                print(f"  Groq's daily token cap is spent ({exc}) -- finishing the rest on "
                      f"heuristics. Re-run tomorrow to upgrade them.", flush=True)
            else:
                misses = misses + 1 if res["source"] != "llm" else 0
                if misses >= CIRCUIT_BREAK_AFTER:
                    llm_live = False
                    print(f"  LLM missed {misses}x in a row -- finishing the rest on "
                          f"heuristics. Re-run later to upgrade them.", flush=True)
        else:
            res = {**build_heuristic_explanation(rec), "source": "heuristic"}
        out[puzzle_id(rec)] = res

        if (i + 1) % SAVE_EVERY == 0 or i == len(todo) - 1:
            _save(out)
            n = sum(1 for e in out.values() if e["source"] == "llm")
            print(f"  {i + 1}/{len(todo)}  ({n} LLM, {len(out) - n} heuristic)", flush=True)
        if llm_live and i < len(todo) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    _save(out)
    n_llm = sum(1 for e in out.values() if e["source"] == "llm")
    print(f"\nwrote {OUT_PATH} ({len(out)} puzzles: {n_llm} LLM, {len(out) - n_llm} heuristic)")
    if n_llm < len(out) and llm_available():
        print(f"  {len(out) - n_llm} are still heuristic -- re-run `python src/explain.py` "
              f"later (the free-tier quota resets daily) to turn them into LLM notes.")
    return out


def main() -> None:
    run()


if __name__ == "__main__":
    main()
