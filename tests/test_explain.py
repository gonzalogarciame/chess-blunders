import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import explain  # noqa: E402
from explain import (  # noqa: E402
    _parse_llm_json,
    build_heuristic_explanation,
    build_llm_prompt,
    default_llm_fn,
    explain_one,
    explain_records,
    puzzle_id,
)


def _rec(**kw):
    base = dict(
        game_id="g1", ply=21, side_to_move="white", fen_before="8/8/8/8/8/8/8/8 w - - 0 1",
        played={"san": "Re1", "from": "e1", "to": "e1", "uci": "e1e1", "promotion": None},
        acceptable_moves=[{"san": "Nc3"}],
        best_line_san="Nc3 Bb7 Re1",
        top_lines=[{"san": "Nc3 Bb7", "eval": "+1.2"}, {"san": "d4", "eval": "-0.3"}],
        material_label="hung a piece",
        refutation_type="capture",
        refutation_line_san="Re1 Nxe1 Kxe1",
        advantage_state="lost from an equal position",
        wp_loss=24.0,
    )
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------------
# heuristic templating (pure)
# --------------------------------------------------------------------------------------------

def test_heuristic_names_the_engine_move_and_eval():
    e = build_heuristic_explanation(_rec())
    assert "Nc3" in e["why_best"] and "+1.2" in e["why_best"]


@pytest.mark.parametrize("refutation,material,needle", [
    ("capture", "hung a piece", "just takes"),
    ("fork", "hung a piece", "a fork"),
    ("allowed_mate", "positional", "forced mate"),
    ("back_rank", "positional", "back rank"),
    ("quiet", "positional", "throws the position away"),
    ("quiet", "hung a rook", "loses material"),
])
def test_heuristic_why_blunder_branches(refutation, material, needle):
    e = build_heuristic_explanation(_rec(refutation_type=refutation, material_label=material))
    assert needle in e["why_blunder"]
    assert "24 win%" in e["why_blunder"] or "forced mate" in e["why_blunder"]


def test_heuristic_flags_a_thrown_win():
    e = build_heuristic_explanation(_rec(advantage_state="threw away a winning position"))
    assert e["why_best"].startswith("You were winning here.")


# --------------------------------------------------------------------------------------------
# LLM JSON parsing
# --------------------------------------------------------------------------------------------

def test_parse_llm_json_plain():
    got = _parse_llm_json('{"why_best": "a", "why_blunder": "b"}')
    assert got == {"why_best": "a", "why_blunder": "b"}


def test_parse_llm_json_with_prose_and_fence():
    raw = 'Sure!\n```json\n{"why_best": "a ", "why_blunder": " b"}\n```\n'
    assert _parse_llm_json(raw) == {"why_best": "a", "why_blunder": "b"}


@pytest.mark.parametrize("raw", [None, "", "not json", '{"why_best": "only one key"}',
                                 '{"why_best": "", "why_blunder": "b"}'])
def test_parse_llm_json_rejects_bad_input(raw):
    assert _parse_llm_json(raw) is None


# --------------------------------------------------------------------------------------------
# explain_one / explain_records wiring (injected fake LLM)
# --------------------------------------------------------------------------------------------

def test_explain_one_uses_heuristic_when_no_llm():
    out = explain_one(_rec(), None)
    assert out["source"] == "heuristic"
    assert out["why_best"] and out["why_blunder"]


def test_explain_one_uses_llm_when_it_returns_valid_json():
    fake = lambda prompt: '{"why_best": "engine idea", "why_blunder": "your move hangs it"}'
    out = explain_one(_rec(), fake)
    assert out == {"why_best": "engine idea", "why_blunder": "your move hangs it", "source": "llm"}


def test_explain_one_falls_back_when_llm_returns_junk():
    assert explain_one(_rec(), lambda p: "no json here")["source"] == "heuristic"


def test_explain_one_falls_back_when_llm_raises():
    def boom(prompt):
        raise RuntimeError("rate limited")
    assert explain_one(_rec(), boom)["source"] == "heuristic"


def test_explain_records_keys_by_puzzle_id():
    recs = [_rec(game_id="g1", ply=21), _rec(game_id="g2", ply=8)]
    out = explain_records(recs, None)
    assert set(out) == {"g1-21", "g2-8"}
    assert puzzle_id(recs[0]) == "g1-21"


def test_build_llm_prompt_mentions_the_played_move_and_asks_for_json():
    prompt = build_llm_prompt(_rec())
    assert "Re1" in prompt and "why_best" in prompt and "why_blunder" in prompt
    assert "JSON" in prompt


def test_default_llm_fn_retries_transient_errors_then_succeeds(monkeypatch):
    calls = []

    def fake_call_llm(prompt, max_tokens, *, groq_model=None, json_mode=False):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("Error code: 429 - rate limit reached for model")
        return '{"why_best": "a", "why_blunder": "b"}'

    monkeypatch.setattr(explain, "call_llm", fake_call_llm)
    monkeypatch.setattr(explain.time, "sleep", lambda _s: None)
    assert default_llm_fn("p") == '{"why_best": "a", "why_blunder": "b"}'
    assert len(calls) == 3


def test_default_llm_fn_reraises_non_transient_errors(monkeypatch):
    def fake_call_llm(prompt, max_tokens, **kw):
        raise ValueError("KeyError: 'fen_before' is missing from the record")

    monkeypatch.setattr(explain, "call_llm", fake_call_llm)
    monkeypatch.setattr(explain.time, "sleep", lambda _s: None)
    with pytest.raises(ValueError):
        default_llm_fn("p")


def test_default_llm_fn_gives_up_to_heuristic_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(explain, "call_llm", lambda *a, **k: "")  # always empty
    monkeypatch.setattr(explain.time, "sleep", lambda _s: None)
    assert default_llm_fn("p") is None
