import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parse import cp_to_winpct_white, parse_clock, parse_game, parse_time_control  # noqa: E402


class FakeEvaluator:
    """Deterministic stand-in for the real Stockfish-backed evaluator -- returns cp values from
    a fixed list, one per call, in the order extract_plies() calls it (starting position first,
    then after every ply). Lets the off-by-one alignment logic be tested without depending on
    the real engine binary."""

    def __init__(self, cps: list[float]):
        self.cps = list(cps)
        self.calls = 0

    def __call__(self, board):
        cp = self.cps[self.calls]
        self.calls += 1
        return cp, "e2e4"  # placeholder PV, not exercised by these tests


# Ruy Lopez opening, 10 plies, clocks only (no [%eval] -- chess.com never provides it; eval
# comes from the injected FakeEvaluator instead). All clocks are Base 5+0 (increment 0).
RUY_LOPEZ_10_PLY = (
    "1. e4 { [%clk 0:05:00] } "
    "1... e5 { [%clk 0:05:00] } "
    "2. Nf3 { [%clk 0:04:58] } "
    "2... Nc6 { [%clk 0:04:59] } "
    "3. Bb5 { [%clk 0:04:56] } "
    "3... a6 { [%clk 0:04:57] } "
    "4. Ba4 { [%clk 0:04:54] } "
    "4... Nf6 { [%clk 0:04:55] } "
    "5. O-O { [%clk 0:04:52] } "
    "5... Be7 { [%clk 0:04:53] } "
    "1-0"
)

# cp after each of the 10 plies is 10, 20, ..., 100 (from White's POV) -- distinct, hand-picked
# values so an off-by-one shift is impossible to miss. Index 0 is the (unused by these
# assertions) starting-position eval.
FAKE_CPS_10_PLY = [0.0] + [10.0 * n for n in range(1, 11)]

ROW_WHITE = {
    "game_id": "handcheck_white", "white": "Gonzalopelotas", "black": "bob",
    "white_elo": 1500, "black_elo": 1500, "time_control": "300", "result": "1-0",
    "termination": "Normal", "utc_date": "2023-01-01", "opening": "Ruy Lopez",
    "movetext": RUY_LOPEZ_10_PLY,
}

ROW_BLACK = {
    "game_id": "handcheck_black", "white": "alice", "black": "Gonzalopelotas",
    "white_elo": 1500, "black_elo": 1500, "time_control": "300", "result": "1-0",
    "termination": "Normal", "utc_date": "2023-01-01", "opening": "Ruy Lopez",
    "movetext": RUY_LOPEZ_10_PLY,
}


def test_parse_clock_handles_fractional_seconds():
    # chess.com carries tenths of a second, unlike Lichess's whole-second format.
    assert parse_clock("0:05:00") == 300.0
    assert parse_clock("0:05:00.4") == pytest.approx(300.4)
    assert parse_clock("1:02:03") == 3723.0


def test_parse_time_control_handles_missing_plus():
    # chess.com omits "+0" for zero increment (e.g. "600"), unlike Lichess's "600+0".
    assert parse_time_control("600") == (600, 0)
    assert parse_time_control("180+2") == (180, 2)


def test_win_pct_formula_independent_reimplementation():
    # Reimplemented from the spec directly (not calling the module twice) to catch typos
    # in the module's own formula.
    k = 0.00368208
    for cp in (0.0, 250.0, -400.0, 1000.0, -1000.0):
        expected = 50 + 50 * (2 / (1 + math.exp(-k * cp)) - 1)
        assert cp_to_winpct_white(cp) == pytest.approx(expected)


def test_win_pct_midpoint_is_fifty():
    assert cp_to_winpct_white(0.0) == pytest.approx(50.0)


def test_offbyone_hand_checked_game_white_mover():
    """
    Ply 9 (White's 5th move, O-O), Gonzalopelotas playing White: eval_before must come from
    ply 8, eval_after from ply 9; the mover's (White's) clock before the move is White's own
    previous clock, on ply 7 -- NOT ply 8, which is Black's clock. Ply 10 (Black/bob's move)
    must be dropped entirely now that only the tracked player's own moves are emitted. Plies
    1-8 are dropped by the ply < 9 exclusion, so exactly 1 row should survive.
    """
    rows = parse_game(ROW_WHITE, FakeEvaluator(FAKE_CPS_10_PLY))
    assert len(rows) == 1

    ply9 = rows[0]
    assert ply9["ply"] == 9
    assert ply9["mover"] == "Gonzalopelotas"
    assert ply9["mover_color"] == "white"
    assert ply9["cp_before"] == 80.0   # eval after ply 8
    assert ply9["cp_after"] == 90.0    # eval after ply 9
    assert ply9["clock_before"] == 294  # ply 7 clock (0:04:54), White's own previous move
    assert ply9["clock_after"] == 292   # ply 9 clock (0:04:52)
    assert ply9["opp_clock_before"] == 295  # ply 8 clock (0:04:55), Black's last update
    assert ply9["move_san"] == "O-O"
    assert ply9["wp_loss"] < 0  # eval improved for White here, so White didn't lose win%
    assert not ply9["blunder"]


def test_offbyone_hand_checked_game_black_mover():
    """Mirrors the White-mover check for ply 10 (Black's 5th move, Be7), Gonzalopelotas
    playing Black. Only ply 10 should survive (ply 9 is White/alice's move, now dropped)."""
    rows = parse_game(ROW_BLACK, FakeEvaluator(FAKE_CPS_10_PLY))
    assert len(rows) == 1

    ply10 = rows[0]
    assert ply10["ply"] == 10
    assert ply10["mover"] == "Gonzalopelotas"
    assert ply10["mover_color"] == "black"
    assert ply10["cp_before"] == 90.0   # eval after ply 9
    assert ply10["cp_after"] == 100.0   # eval after ply 10
    assert ply10["clock_before"] == 295  # ply 8 clock (0:04:55), Black's own previous move
    assert ply10["clock_after"] == 293   # ply 10 clock (0:04:53)
    assert ply10["opp_clock_before"] == 292  # ply 9 clock (0:04:52), White's last update
    assert ply10["move_san"] == "Be7"
    assert ply10["wp_loss"] > 0  # eval moved further in White's favor -> bad for Black
    assert not ply10["blunder"]


def test_final_move_of_time_forfeit_game_is_dropped():
    """Extends the same opening by one White move (ply 11) so the tracked mover (White) has
    two qualifying plies (9 and 11). The game ends on ply 11 by time forfeit, so ply 11 must be
    dropped even though its clock happens to be present -- ply 9 must still survive."""
    movetext_11_ply = RUY_LOPEZ_10_PLY.replace("1-0", "6. Re1 { [%clk 0:04:50] } 1-0")
    row = dict(ROW_WHITE, movetext=movetext_11_ply, termination="Time forfeit")
    fake_cps = FAKE_CPS_10_PLY + [110.0]

    rows = parse_game(row, FakeEvaluator(fake_cps))
    plies = [r["ply"] for r in rows]
    assert 11 not in plies
    assert plies == [9]


def test_mover_filter_drops_games_with_no_tracked_player():
    row = dict(ROW_WHITE, white="alice", black="bob")
    rows = parse_game(row, FakeEvaluator(FAKE_CPS_10_PLY))
    assert rows == []
