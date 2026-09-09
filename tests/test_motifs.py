import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motifs import (  # noqa: E402
    LineInfo,
    _advantage_state,
    _material_label,
    analyse_one,
    classify_blunder,
    select_leak_blunders,
)
import pandas as pd  # noqa: E402


def line(uci_moves: list[str], score_cp: float, mate: int | None = None) -> LineInfo:
    return LineInfo(score_cp=score_cp, mate=mate, pv=[chess.Move.from_uci(u) for u in uci_moves])


# --------------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("swing,label", [
    (-9, "hung a queen"), (-5, "hung a rook"), (-3, "hung a piece"),
    (-2, "lost the exchange"), (-1, "dropped a pawn"), (0, "positional"),
])
def test_material_label_thresholds(swing, label):
    assert _material_label(swing) == label


@pytest.mark.parametrize("wp,state", [
    (80.0, "threw away a winning position"),
    (50.0, "lost from an equal position"),
    (20.0, "compounded a worse position"),
])
def test_advantage_state_thresholds(wp, state):
    assert _advantage_state(wp) == state


# --------------------------------------------------------------------------------------------
# classify_blunder
# --------------------------------------------------------------------------------------------

def test_hung_queen_is_tagged_from_the_material_swing():
    # White Qe2, Ke1; Black Ke8, pawn f7. White plays Qe6??, f7xe6 wins the queen.
    fen = "4k3/5p2/8/8/8/8/4Q3/4K3 w - - 0 1"
    tags = classify_blunder(
        fen_before=fen,
        played_uci="e2e6",
        wp_before=70.0,
        ply=25,
        lines_before=[line(["e2e5"], 60.0), line(["e2d3"], 40.0)],
        lines_after=[line(["f7e6"], -850.0)],
    )
    assert tags["material_label"] == "hung a queen"
    assert tags["refutation_type"] == "capture"
    assert tags["motif_key"] == "hung a queen"
    assert tags["motif_group"] == "Hung the queen"
    assert tags["advantage_state"] == "threw away a winning position"
    assert tags["phase"] == "middlegame"
    # the "punishment" line: the move played, then the engine's refutation of it
    assert tags["refutation_line_san"].split()[0] == "Qe6+"
    assert "fxe6" in tags["refutation_line_san"]
    assert len(tags["refutation_line_fens"]) >= 2
    assert tags["refutation_line_fens"][0] == fen  # starts from the pre-move position


def test_allowed_mate_wins_over_material_tag():
    # Black Ra8/Kh8; White Kg1, pawns f2 g2 h2. White plays Kh1??, ...Ra1#.
    fen = "r6k/8/8/8/8/8/5PPP/6K1 w - - 0 1"
    tags = classify_blunder(
        fen_before=fen,
        played_uci="g1h1",
        wp_before=48.0,
        ply=55,
        lines_before=[line(["g1f1"], 0.0)],
        lines_after=[line(["a8a1"], -30000.0, mate=-1)],
    )
    assert tags["refutation_type"] == "allowed_mate"
    assert tags["motif_key"] == "allowed_mate"
    assert tags["advantage_state"] == "lost from an equal position"


def test_back_rank_refutation_detected_for_black_victim():
    # White Ra1/Kh1; Black Kg8, pawns f7 g7 h7 (Black to move). Black plays Kh8??, Ra8+ back-rank.
    fen = "6k1/5ppp/8/8/8/8/8/R6K b - - 0 1"
    tags = classify_blunder(
        fen_before=fen,
        played_uci="g8h8",
        wp_before=45.0,
        ply=61,
        lines_before=[line(["f7f6"], 0.0)],
        lines_after=[line(["a1a8"], 900.0)],
    )
    assert tags["refutation_type"] == "back_rank"
    assert tags["motif_key"] == "back_rank"
    assert tags["phase"] == "endgame"


def test_two_engine_moves_within_margin_are_both_acceptable():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tags = classify_blunder(
        fen_before=fen,
        played_uci="a2a3",  # not among the acceptable set
        wp_before=50.0,
        ply=9,
        lines_before=[line(["e2e4"], 30.0), line(["d2d4"], 5.0), line(["g1f3"], -40.0)],
        lines_after=[line(["e7e5"], -20.0)],
    )
    sans = {m["san"] for m in tags["acceptable_moves"]}
    assert sans == {"e4", "d4"}  # g1f3 is 70cp below best -> excluded
    assert "a3" not in sans


# --------------------------------------------------------------------------------------------
# selection + wiring
# --------------------------------------------------------------------------------------------

def test_select_leak_blunders_keeps_only_rows_in_a_leak_slice():
    df = pd.DataFrame({
        "blunder": [True, True, False, True],
        "mover_color": ["black", "white", "black", "white"],
        "ply_bucket": ["21-40", "9-20", "21-40", "9-20"],
    })
    leaks = pd.DataFrame({"dimension": ["mover_color", "ply_bucket"], "value": ["black", "21-40"]})
    out = select_leak_blunders(df, leaks)
    assert len(out) == 1  # only row 0 is a blunder AND in a leak slice
    assert out.iloc[0]["leak_tags"] == ["mover_color=black", "ply_bucket=21-40"]


def test_analyse_one_wires_through_a_fake_analyzer():
    fen = "4k3/5p2/8/8/8/8/4Q3/4K3 w - - 0 1"
    row = pd.Series({
        "game_id": "g1", "ply": 25, "utc_date": "2025-01-01", "opening": "Test",
        "opening_family": "Test Opening", "mover_color": "white", "fen_before": fen,
        "move_uci": "e2e6", "move_san": "Qe6", "wp_before": 70.0, "wp_loss": 40.0,
        "cp_before": 300.0, "clock_before": 45.0, "increment": 2, "base_time": 180,
        "leak_tags": ["ply_bucket=21-40"],
    })

    def fake(board, multipv, depth):
        if board.turn == chess.WHITE:
            return [line(["e2e5"], 60.0), line(["e2d3"], 40.0)]
        return [line(["f7e6"], -850.0)]

    rec = analyse_one(row, fake)
    assert rec["motif_group"] == "Hung the queen"
    assert rec["played"]["san"] == "Qe6+"
    assert rec["side_to_move"] == "white"
    assert rec["refutation_line_san"].startswith("Qe6+ fxe6")
