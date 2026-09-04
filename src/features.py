"""
Set 3 -- build a leakage-free feature matrix from data/processed/moves_{month}.parquet.

Every feature must be computable strictly before the move is played: board features come
from python-chess applied to the pre-move FEN, eval/clock features only ever look at
_before values or backward-looking history, and no feature may depend on eval_after,
wp_after, wp_loss, the game result, or any whole-game aggregate (those are only known once
the game -- or at least the move -- is over). See README for the full leakage-rule list.
"""

import multiprocessing as mp
from pathlib import Path

import chess
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MONTHS = ["2024-03", "2024-09"]

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# Anything that describes the position/outcome after the move, or a whole-game aggregate,
# is banned from the feature list -- these can only be known once the future has happened.
BANNED_COLUMNS = {
    "eval_after", "cp_after", "wp_after", "wp_loss", "result", "termination",
    "total_accuracy", "total_blunders",
}

FEATURE_COLUMNS = [
    # board
    "legal_move_count", "is_check", "captures_available", "checks_available",
    "material_balance", "material_total", "mover_piece_count", "opp_piece_count",
    # eval
    "wp_before", "abs_cp_before", "wp_volatility_3", "wp_swing_last",
    # clock
    "clock_before", "clock_frac", "log_clock", "opp_clock_before", "clock_diff",
    "time_spent_prev", "increment", "base_time",
    # player / game
    "mover_elo", "opp_elo", "elo_diff", "is_white", "ply",
]


def board_features(fen: str, is_white: bool) -> dict:
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    captures = sum(1 for m in legal_moves if board.is_capture(m))
    checks = sum(1 for m in legal_moves if board.gives_check(m))

    pieces = board.piece_map().values()
    white_material = sum(PIECE_VALUES[p.piece_type] for p in pieces if p.color == chess.WHITE)
    black_material = sum(PIECE_VALUES[p.piece_type] for p in pieces if p.color == chess.BLACK)
    white_count = sum(1 for p in pieces if p.color == chess.WHITE)
    black_count = sum(1 for p in pieces if p.color == chess.BLACK)

    return {
        "legal_move_count": len(legal_moves),
        "is_check": board.is_check(),
        "captures_available": captures,
        "checks_available": checks,
        "material_balance": (white_material - black_material) if is_white else (black_material - white_material),
        "material_total": white_material + black_material,
        "mover_piece_count": white_count if is_white else black_count,
        "opp_piece_count": black_count if is_white else white_count,
    }


def board_features_batch(rows: list[dict]) -> list[dict]:
    return [board_features(r["fen_before"], r["mover_color"] == "white") for r in rows]


def add_board_features(df: pd.DataFrame) -> pd.DataFrame:
    records = df[["fen_before", "mover_color"]].to_dict("records")
    chunk = 5000
    batches = [records[i:i + chunk] for i in range(0, len(records), chunk)]
    with mp.Pool() as pool:
        results = pool.map(board_features_batch, batches)
    board_df = pd.DataFrame([r for batch in results for r in batch], index=df.index)
    return pd.concat([df, board_df], axis=1)


def add_eval_features(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_cp_before"] = df["cp_before"].abs()

    by_game = df.sort_values(["game_id", "ply"]).groupby("game_id")["wp_before"]
    df["wp_volatility_3"] = by_game.transform(lambda s: s.shift(1).rolling(3, min_periods=1).std())

    by_mover = df.sort_values(["game_id", "mover_color", "ply"]).groupby(["game_id", "mover_color"])["wp_before"]
    df["wp_swing_last"] = by_mover.transform(lambda s: s.diff())
    return df


def add_clock_features(df: pd.DataFrame) -> pd.DataFrame:
    df["clock_frac"] = df["clock_before"] / df["base_time"]
    df["log_clock"] = np.log1p(df["clock_before"])
    df["clock_diff"] = df["clock_before"] - df["opp_clock_before"]

    # time_spent_prev = clk[t-4] - clk[t-2] + increment. clk[t-2] is this row's own
    # clock_before; clk[t-4] is the mover's *previous* move's clock_before.
    prev_clock_before = df.sort_values(["game_id", "mover_color", "ply"]) \
        .groupby(["game_id", "mover_color"])["clock_before"].shift(1)
    df["time_spent_prev"] = prev_clock_before - df["clock_before"] + df["increment"]
    return df


def add_player_features(df: pd.DataFrame) -> pd.DataFrame:
    df["elo_diff"] = df["mover_elo"] - df["opp_elo"]
    df["is_white"] = df["mover_color"] == "white"
    return df


def build_features(month: str) -> pd.DataFrame:
    moves_path = PROCESSED_DIR / f"moves_{month}.parquet"
    df = pd.read_parquet(moves_path)
    print(f"\n{month}: {len(df):,} move rows loaded")

    df = add_board_features(df)
    df = add_eval_features(df)
    df = add_clock_features(df)
    df = add_player_features(df)

    assert BANNED_COLUMNS.isdisjoint(FEATURE_COLUMNS), (
        f"leakage: banned columns found in feature list: {BANNED_COLUMNS.intersection(FEATURE_COLUMNS)}"
    )
    print(f"leakage assertion passed: none of {sorted(BANNED_COLUMNS)} are in FEATURE_COLUMNS")

    print(f"feature matrix shape: {df[FEATURE_COLUMNS].shape}")
    null_counts = df[FEATURE_COLUMNS].isnull().sum()
    print("null counts per feature column:")
    print(null_counts[null_counts > 0] if null_counts.any() else "  none")

    out_path = PROCESSED_DIR / f"features_{month}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"{month}: wrote {out_path}")
    return df


def main() -> None:
    for month in MONTHS:
        build_features(month)


if __name__ == "__main__":
    main()
