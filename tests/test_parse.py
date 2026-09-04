import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parse import cp_to_winpct_white, parse_clock, parse_eval, parse_game  # noqa: E402

# Ruy Lopez opening, 10 plies, with distinct hand-picked eval/clock values per ply so an
# off-by-one shift is impossible to miss. All clocks are Base 5+0 (increment 0).
HAND_CHECKED_MOVETEXT = (
    "1. e4 { [%eval 0.10] [%clk 0:05:00] } "
    "1... e5 { [%eval 0.20] [%clk 0:05:00] } "
    "2. Nf3 { [%eval 0.30] [%clk 0:04:58] } "
    "2... Nc6 { [%eval 0.40] [%clk 0:04:59] } "
    "3. Bb5 { [%eval 0.50] [%clk 0:04:56] } "
    "3... a6 { [%eval 0.60] [%clk 0:04:57] } "
    "4. Ba4 { [%eval 0.70] [%clk 0:04:54] } "
    "4... Nf6 { [%eval 0.80] [%clk 0:04:55] } "
    "5. O-O { [%eval 0.90] [%clk 0:04:52] } "
    "5... Be7 { [%eval 1.00] [%clk 0:04:53] } "
    "1-0"
)

ROW = {
    "game_id": "handcheck1",
    "white": "alice",
    "black": "bob",
    "white_elo": 1500,
    "black_elo": 1500,
    "time_control": "300+0",
    "result": "1-0",
    "termination": "Normal",
    "movetext": HAND_CHECKED_MOVETEXT,
}


def test_parse_eval_centipawns_and_clamp():
    assert parse_eval("0.10") == 10.0
    assert parse_eval("-0.43") == -43.0
    assert parse_eval("20.0") == 1000.0  # clamped
    assert parse_eval("-20.0") == -1000.0  # clamped


def test_parse_eval_mate():
    assert parse_eval("#5") == 1000.0  # mate for the side to move from White's POV, clamped
    assert parse_eval("#-3") == -1000.0


def test_parse_clock():
    assert parse_clock("0:05:00") == 300
    assert parse_clock("1:02:03") == 3723


def test_win_pct_formula_independent_reimplementation():
    # Reimplemented from the spec directly (not calling the module twice) to catch typos
    # in the module's own formula.
    k = 0.00368208
    for cp in (0.0, 250.0, -400.0, 1000.0, -1000.0):
        expected = 50 + 50 * (2 / (1 + math.exp(-k * cp)) - 1)
        assert cp_to_winpct_white(cp) == pytest.approx(expected)


def test_win_pct_midpoint_is_fifty():
    assert cp_to_winpct_white(0.0) == pytest.approx(50.0)


def test_offbyone_hand_checked_game():
    """
    Ply 9 (White's 5th move, O-O): eval_before must come from ply 8, eval_after from ply 9;
    the mover's (White's) clock before the move is White's own previous clock, on ply 7 --
    NOT ply 8, which is Black's clock.
    Ply 10 (Black's 5th move, Be7): mirrors the same check for Black.
    Plies 1-8 are dropped by the ply < 9 exclusion, so exactly 2 rows should survive.
    """
    rows = parse_game(ROW)
    assert len(rows) == 2

    ply9, ply10 = rows
    assert ply9["ply"] == 9
    assert ply10["ply"] == 10

    # --- ply 9: White plays O-O ---
    assert ply9["mover"] == "alice"
    assert ply9["mover_color"] == "white"
    assert ply9["cp_before"] == 80.0   # eval on ply 8 (0.80)
    assert ply9["cp_after"] == 90.0    # eval on ply 9 (0.90)
    assert ply9["clock_before"] == 294  # ply 7 clock (0:04:54), White's own previous move
    assert ply9["clock_after"] == 292   # ply 9 clock (0:04:52)
    assert ply9["opp_clock_before"] == 295  # ply 8 clock (0:04:55), Black's last update
    assert ply9["move_san"] == "O-O"

    # --- ply 10: Black plays Be7 ---
    assert ply10["mover"] == "bob"
    assert ply10["mover_color"] == "black"
    assert ply10["cp_before"] == 90.0   # eval on ply 9 (0.90)
    assert ply10["cp_after"] == 100.0   # eval on ply 10 (1.00)
    assert ply10["clock_before"] == 295  # ply 8 clock (0:04:55), Black's own previous move
    assert ply10["clock_after"] == 293   # ply 10 clock (0:04:53)
    assert ply10["opp_clock_before"] == 292  # ply 9 clock (0:04:52), White's last update
    assert ply10["move_san"] == "Be7"

    # eval improved for White on both moves here, so neither mover lost win probability
    assert ply9["wp_loss"] < 0
    assert ply10["wp_loss"] > 0  # eval moved further in White's favor -> bad for Black
    assert not ply9["blunder"]
    assert not ply10["blunder"]


def test_final_move_of_time_forfeit_game_is_dropped():
    row = dict(ROW)
    row["termination"] = "Time forfeit"
    rows = parse_game(row)
    # ply 10 was the last ply in the movetext and the game ended on time, so it must be
    # dropped even though its eval/clock happen to be present in this synthetic example.
    plies = [r["ply"] for r in rows]
    assert 10 not in plies
    assert plies == [9]
