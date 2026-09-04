"""
Set 2 -- turn each game (data/raw/games_{month}.parquet) into one row per move, with
eval-before/after, clock-before/after, and a blunder label.

Annotation semantics: the comment after the move played at ply t describes the position
AFTER that move. So for the move at ply t: eval_after = eval on ply t; eval_before = eval
on ply t-1 (the position it was played from). Clocks only change on their own side's move,
so the mover's clock before ply t is their own last clock update, on ply t-2; the opponent's
clock before ply t is their last update, on ply t-1.
"""

import io
import multiprocessing as mp
import re
from pathlib import Path

import chess.pgn
import duckdb
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MONTHS = ["2024-03", "2024-09"]

MIN_PLY = 9
CP_CLAMP = 1000
MATE_CP = 10000
WP_K = 0.00368208
THRESHOLD = 20.0
MAX_ROWS = 8_000_000

EVAL_RE = re.compile(r"\[%eval\s+([^\]]+)\]")
CLK_RE = re.compile(r"\[%clk\s+([^\]]+)\]")


def parse_eval(raw: str) -> float:
    """Centipawns from White's point of view, clamped to +/- CP_CLAMP. Mate #N -> +/-MATE_CP first."""
    raw = raw.strip()
    if raw.startswith("#"):
        cp = MATE_CP if int(raw[1:]) > 0 else -MATE_CP
    else:
        cp = float(raw) * 100.0
    return max(-CP_CLAMP, min(CP_CLAMP, cp))


def parse_clock(raw: str) -> int:
    h, m, s = (int(p) for p in raw.strip().split(":"))
    return h * 3600 + m * 60 + s


def cp_to_winpct_white(cp: float) -> float:
    return 50 + 50 * (2 / (1 + np.exp(-WP_K * cp)) - 1)


def parse_time_control(tc: str) -> tuple[int, int]:
    base, inc = tc.split("+")
    return int(base), int(inc)


def extract_plies(movetext: str) -> tuple[list, list, list, list]:
    """Replay a game, returning per-ply (fen_before, san, uci, cp, clock) aligned lists,
    one entry per ply, in playing order."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    board = game.board()

    fens_before, sans, ucis, cps, clocks = [], [], [], [], []
    for node in game.mainline():
        move = node.move
        fens_before.append(board.fen())
        sans.append(board.san(move))
        ucis.append(move.uci())
        board.push(move)

        comment = node.comment or ""
        eval_match = EVAL_RE.search(comment)
        clk_match = CLK_RE.search(comment)
        cps.append(parse_eval(eval_match.group(1)) if eval_match else None)
        clocks.append(parse_clock(clk_match.group(1)) if clk_match else None)

    return fens_before, sans, ucis, cps, clocks


def parse_game(row: dict) -> list[dict]:
    fens_before, sans, ucis, cps, clocks = extract_plies(row["movetext"])
    n_plies = len(cps)
    base_time, increment = parse_time_control(row["time_control"])
    last_move_is_time_forfeit = row["termination"] == "Time forfeit"

    out_rows = []
    for i in range(n_plies):
        ply = i + 1
        if ply < MIN_PLY:
            continue
        if last_move_is_time_forfeit and ply == n_plies:
            continue

        cp_after = cps[i]
        cp_before = cps[i - 1] if i - 1 >= 0 else None
        if cp_before is None or cp_after is None:
            continue

        clock_after = clocks[i]
        clock_before = clocks[i - 2] if i - 2 >= 0 else None
        opp_clock_before = clocks[i - 1] if i - 1 >= 0 else None
        if clock_before is None or clock_after is None:
            continue

        is_white = ply % 2 == 1
        wp_white_before = cp_to_winpct_white(cp_before)
        wp_white_after = cp_to_winpct_white(cp_after)
        wp_before = wp_white_before if is_white else 100 - wp_white_before
        wp_after = wp_white_after if is_white else 100 - wp_white_after
        wp_loss = wp_before - wp_after

        out_rows.append({
            "game_id": row["game_id"],
            "ply": ply,
            "mover": row["white"] if is_white else row["black"],
            "mover_elo": row["white_elo"] if is_white else row["black_elo"],
            "opp_elo": row["black_elo"] if is_white else row["white_elo"],
            "mover_color": "white" if is_white else "black",
            "time_control": row["time_control"],
            "base_time": base_time,
            "increment": increment,
            "cp_before": cp_before,
            "cp_after": cp_after,
            "wp_before": wp_before,
            "wp_after": wp_after,
            "wp_loss": wp_loss,
            "blunder": wp_loss >= THRESHOLD,
            "clock_before": clock_before,
            "clock_after": clock_after,
            "opp_clock_before": opp_clock_before,
            "move_san": sans[i],
            "move_uci": ucis[i],
            "fen_before": fens_before[i],
        })
    return out_rows


def print_verification(df: pd.DataFrame, month: str) -> None:
    print(f"\n=== {month} verification ===")
    print(f"total move rows: {len(df):,}")
    print(f"overall blunder rate: {df['blunder'].mean() * 100:.2f}%")
    print(f"mean wp_loss: {df['wp_loss'].mean():.2f}")

    elo_bins = list(range(800, 2601, 200))
    elo_band = pd.cut(df["mover_elo"], bins=elo_bins, right=False)
    print("\nblunder rate by mover Elo band (%):")
    print((df.groupby(elo_band, observed=True)["blunder"].mean() * 100).round(2))

    print("\nblunder rate by increment (%):")
    print((df.groupby("increment", observed=True)["blunder"].mean() * 100).round(2))


def process_month(month: str) -> None:
    games_path = RAW_DIR / f"games_{month}.parquet"
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{games_path}')").fetchdf()
    print(f"\n{month}: {len(df):,} games loaded")

    rows = df.to_dict("records")
    with mp.Pool() as pool:
        per_game = pool.map(parse_game, rows, chunksize=200)

    total = sum(len(g) for g in per_game)
    print(f"{month}: {total:,} move rows before subsampling")

    if total > MAX_ROWS:
        rng = np.random.default_rng(42)
        frac = MAX_ROWS / total
        n_games = len(per_game)
        keep = rng.choice(n_games, size=int(n_games * frac), replace=False)
        move_rows = [r for i in keep for r in per_game[i]]
        print(f"{month}: subsampled {len(keep):,}/{n_games:,} games -> {len(move_rows):,} move rows")
    else:
        move_rows = [r for g in per_game for r in g]

    moves_df = pd.DataFrame(move_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"moves_{month}.parquet"
    moves_df.to_parquet(out_path, index=False)
    print(f"{month}: wrote {out_path}")

    print_verification(moves_df, month)


def main() -> None:
    for month in MONTHS:
        process_month(month)


if __name__ == "__main__":
    main()
