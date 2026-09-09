import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trainer import (  # noqa: E402
    PIECE_FILES,
    build_page,
    build_puzzle,
    chess_js,
    fmt_clock,
    group_summary,
    lead_in,
    piece_defs,
)

RUY_LOPEZ = (
    '[Event "t"]\n[Site "?"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 *"
)


@pytest.mark.parametrize("secs,out", [(0, "0:00"), (45.3, "0:45"), (60, "1:00"), (125, "2:05")])
def test_fmt_clock(secs, out):
    assert fmt_clock(secs) == out


def test_piece_defs_has_all_twelve_symbols():
    defs = piece_defs()
    for sid in PIECE_FILES:
        assert f'<symbol id="pc-{sid}"' in defs
    assert defs.count("<symbol") == 12


def test_lead_in_ends_on_the_puzzle_position():
    # ply 12 = Black's 6th move (...b5); the position before it is after 11 plies.
    steps = lead_in(RUY_LOPEZ, ply=12, n=4)
    assert len(steps) == 4
    assert steps[-1]["san"] == "Re1"          # the move just before the puzzle ply
    assert steps[-1]["fen"].split()[1] == "b"  # Black to move in the puzzle position


def _rec(**kw):
    base = dict(
        game_id=123, ply=21, utc_date="2025-03-04", opening="Ruy Lopez",
        opening_family="Ruy Lopez", mover_color="white", fen_before=RUY_LOPEZ,
        move_uci="f1e1", move_san="Re1", wp_before=61.0, wp_loss=24.0, cp_before=120.0,
        clock_before=95.4, increment=2, base_time=180, leak_tags=["ply_bucket=21-40"],
        motif_group="Hung a piece", phase="middlegame", advantage_state="lost from an equal position",
        material_label="hung a piece", refutation_type="capture",
        acceptable_moves=[{"uci": "b1c3", "san": "Nc3", "from": "b1", "to": "c3", "promotion": None}],
        played={"uci": "f1e1", "san": "Re1", "from": "f1", "to": "e1", "promotion": None},
        best_line_san="Nc3 Bb7", best_line_fens=["fen0", "fen1", "fen2"],
        top_lines=[{"san": "Nc3 Bb7", "eval": "+1.2"}],
        refutation_line_san="Re1 Nxe1 Kxe1", refutation_line_fens=["fen0", "fen1", "fen2", "fen3"],
    )
    base.update(kw)
    return base


def test_build_puzzle_shape_and_game_url():
    p = build_puzzle(_rec(), None)
    assert p["id"] == "123-21"
    assert p["orientation"] == "white"
    assert p["meta"]["clock"] == "1:35"
    assert p["meta"]["gameUrl"] == "https://www.chess.com/game/live/123"
    assert p["leadIn"] == []  # no movetext supplied
    assert p["whyBest"] == "" and p["whyBlunder"] == ""  # no explanations supplied
    assert p["refutationSan"] == "Re1 Nxe1 Kxe1"


def test_build_puzzle_merges_explanations_by_id():
    ex = {"123-21": {"why_best": "Nc3 keeps the extra piece.",
                     "why_blunder": "Re1 drops it to Nxe1.", "source": "llm"}}
    p = build_puzzle(_rec(), None, ex)
    assert p["whyBest"] == "Nc3 keeps the extra piece."
    assert p["whyBlunder"] == "Re1 drops it to Nxe1."
    assert build_puzzle(_rec(ply=99), None, ex)["whyBest"] == ""  # id miss -> empty


def test_group_summary_orders_by_the_fixed_priority():
    puzzles = [build_puzzle(_rec(motif_group="Dropped a pawn"), None),
               build_puzzle(_rec(motif_group="Allowed forced mate"), None),
               build_puzzle(_rec(motif_group="Dropped a pawn"), None)]
    summary = group_summary(puzzles)
    assert summary[0]["name"] == "Allowed forced mate"  # higher priority than "Dropped a pawn"
    assert {s["name"]: s["count"] for s in summary} == {"Allowed forced mate": 1, "Dropped a pawn": 2}


def test_build_page_inlines_data_and_pieces_without_html_wrapper():
    page = build_page([build_puzzle(_rec(), None)], [{"name": "Hung a piece", "count": 1}])
    assert page.startswith("<title>Blunder Trainer</title>")
    assert "<!doctype" not in page.lower() and "<html" not in page.lower()
    assert "window.__TRAINER__ = " in page
    assert '<symbol id="pc-wk"' in page
    assert "/*__DATA__*/" not in page and "<!--__PIECES__-->" not in page


def test_build_page_inlines_chess_js_with_no_cdn():
    page = build_page([build_puzzle(_rec(), None)], [{"name": "Hung a piece", "count": 1}])
    assert "<!--__CHESSJS__-->" not in page
    assert "cdnjs.cloudflare.com" not in page and "<script src=" not in page
    assert "var Chess=function" in page  # the vendored library, inline


def test_chess_js_is_a_script_tag():
    assert chess_js().startswith("<script>") and chess_js().rstrip().endswith("</script>")
