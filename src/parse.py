"""
Set 2 -- turn each game (data/raw/games_gonzalopelotas.parquet) into one row per move *of this
player's own moves only* (not the opponent's -- see README), with eval-before/after,
clock-before/after, and a blunder label.

chess.com never exposes engine [%eval] via its public API (verified empirically -- see
ingest.py), so unlike the original Lichess-based design, eval isn't parsed out of the PGN here.
Instead, every position along each game's mainline is evaluated locally with a persistent
Stockfish process per worker (one engine launch per pool worker, reused across every game/
position it handles -- launching Stockfish per-position would dominate runtime). The evaluator
is injected as a plain callable so tests can use a deterministic fake instead of depending on
the real engine binary.

Annotation semantics carried over unchanged from the original design: the comment after the
move played at ply t describes the position AFTER that move, so for the row at ply t:
eval_after = eval on ply t, eval_before = eval on ply t-1. Clocks only change on their own
side's move, so the mover's clock before ply t is their own last clock update, on ply t-2; the
opponent's clock before ply t is their last update, on ply t-1.
"""

import atexit
import io
import multiprocessing as mp
import re
import shutil
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = PROCESSED_DIR / "_parse_checkpoints"

USERNAME = "gonzalopelotas"
GAMES_PATH = RAW_DIR / "games_gonzalopelotas.parquet"

MIN_PLY = 9
CP_CLAMP = 1000
MATE_CP = 10000
WP_K = 0.00368208
THRESHOLD = 20.0

ENGINE_DEPTH = 14
BATCH_SIZE = 50  # games per checkpoint -- a crash/interrupt only costs one batch, not the run

CLK_RE = re.compile(r"\[%clk\s+([^\]]+)\]")

_ENGINE: chess.engine.SimpleEngine | None = None
_LIMIT = chess.engine.Limit(depth=ENGINE_DEPTH)


def find_stockfish() -> str:
    path = shutil.which("stockfish")
    if path:
        return path
    winget_glob = list(Path.home().glob(
        "AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_*/stockfish/"
        "stockfish-windows-*.exe"
    ))
    if winget_glob:
        return str(winget_glob[0])
    raise FileNotFoundError(
        "stockfish executable not found on PATH or in the usual winget install location -- "
        "install it with: winget install --id Stockfish.Stockfish -e"
    )


def parse_clock(raw: str) -> float:
    """chess.com clocks carry tenths of a second (e.g. '0:03:01.4'), unlike Lichess's
    whole-second format -- kept as float rather than truncated, for free extra precision."""
    h, m, s = raw.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_time_control(tc: str) -> tuple[int, int]:
    """chess.com omits '+0' for zero increment (e.g. '600', not '600+0'), unlike Lichess."""
    if "+" in tc:
        base, inc = tc.split("+")
        return int(base), int(inc)
    return int(tc), 0


def cp_to_winpct_white(cp: float) -> float:
    return 50 + 50 * (2 / (1 + np.exp(-WP_K * cp)) - 1)


def score_to_cp(score: chess.engine.PovScore) -> float:
    """Signed centipawns from White's point of view, clamped to +/- CP_CLAMP. Mate scores are
    substituted with +/- MATE_CP (then clamped), matching how the original text-parsed '#N'
    annotations were handled."""
    cp = score.white().score(mate_score=MATE_CP)
    return max(-CP_CLAMP, min(CP_CLAMP, float(cp)))


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board,
                       limit: chess.engine.Limit) -> tuple[float, str | None]:
    """One engine call, returning (clamped cp from White's POV, best move UCI or None if the
    position has no legal moves)."""
    info = engine.analyse(board, limit)
    cp = score_to_cp(info["score"])
    pv = info.get("pv")
    best_uci = pv[0].uci() if pv else None
    return cp, best_uci


def extract_plies(movetext: str, evaluator) -> tuple[list, list, list, list, list, list]:
    """Replay a game, returning per-position/per-ply aligned lists. `cps`/`pvs` have one entry
    per position along the mainline INCLUDING the starting position (length n_plies + 1); the
    others have one entry per ply (length n_plies). `evaluator` is `board -> (cp, best_uci)`,
    injected so tests don't need a real engine."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    board = game.board()

    fens_before, sans, ucis, clocks = [], [], [], []
    cps, pvs = [], []

    cp0, pv0 = evaluator(board)
    cps.append(cp0)
    pvs.append(pv0)

    for node in game.mainline():
        move = node.move
        fens_before.append(board.fen())
        sans.append(board.san(move))
        ucis.append(move.uci())
        board.push(move)

        comment = node.comment or ""
        clk_match = CLK_RE.search(comment)
        clocks.append(parse_clock(clk_match.group(1)) if clk_match else None)

        cp, pv = evaluator(board)
        cps.append(cp)
        pvs.append(pv)

    return fens_before, sans, ucis, clocks, cps, pvs


def parse_game(row: dict, evaluator) -> list[dict]:
    fens_before, sans, ucis, clocks, cps, pvs = extract_plies(row["movetext"], evaluator)
    n_plies = len(ucis)
    base_time, increment = parse_time_control(row["time_control"])
    last_move_is_time_forfeit = row["termination"] == "Time forfeit"

    out_rows = []
    for i in range(n_plies):
        ply = i + 1
        if ply < MIN_PLY:
            continue
        if last_move_is_time_forfeit and ply == n_plies:
            continue

        is_white = ply % 2 == 1
        mover = row["white"] if is_white else row["black"]
        if mover.lower() != USERNAME.lower():
            continue

        cp_before, cp_after = cps[i], cps[i + 1]
        if cp_before is None or cp_after is None:
            continue

        clock_after = clocks[i]
        clock_before = clocks[i - 2] if i - 2 >= 0 else None
        opp_clock_before = clocks[i - 1] if i - 1 >= 0 else None
        if clock_before is None or clock_after is None:
            continue

        wp_white_before = cp_to_winpct_white(cp_before)
        wp_white_after = cp_to_winpct_white(cp_after)
        wp_before = wp_white_before if is_white else 100 - wp_white_before
        wp_after = wp_white_after if is_white else 100 - wp_white_after
        wp_loss = wp_before - wp_after

        out_rows.append({
            "game_id": row["game_id"],
            "ply": ply,
            "mover": mover,
            "mover_elo": row["white_elo"] if is_white else row["black_elo"],
            "opp_elo": row["black_elo"] if is_white else row["white_elo"],
            "mover_color": "white" if is_white else "black",
            "time_control": row["time_control"],
            "base_time": base_time,
            "increment": increment,
            "utc_date": row["utc_date"],
            "opening": row["opening"],
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
            # engine's suggested move from the pre-move position (what it would have played
            # instead) and from the post-move position (the opponent's best reply to what was
            # actually played) -- report.py uses the latter to coarsely tag blunders that hang
            # material outright vs. subtler ones. Free from the same analyse() calls as cp.
            "engine_pv_before": pvs[i],
            "engine_pv_after": pvs[i + 1],
        })
    return out_rows


def _init_worker(stockfish_path: str) -> None:
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    atexit.register(_ENGINE.quit)


def _worker_evaluator(board: chess.Board) -> tuple[float, str | None]:
    return evaluate_position(_ENGINE, board, _LIMIT)


def _parse_game_task(row: dict) -> list[dict]:
    return parse_game(row, _worker_evaluator)


def print_verification(df: pd.DataFrame) -> None:
    print(f"\n=== verification ===")
    print(f"total move rows: {len(df):,}")
    print(f"overall blunder rate: {df['blunder'].mean() * 100:.2f}%")
    print(f"mean wp_loss: {df['wp_loss'].mean():.2f}")

    elo_bins = list(range(400, 2001, 100))
    elo_band = pd.cut(df["mover_elo"], bins=elo_bins, right=False)
    print("\nblunder rate by own-Elo band at time of game (%) -- reflects rating drift over "
          "~5 years, not population variation:")
    print((df.groupby(elo_band, observed=True)["blunder"].mean() * 100).round(2))

    print("\nblunder rate by increment (%):")
    print((df.groupby("increment", observed=True)["blunder"].mean() * 100).round(2))

    print("\nblunder rate by mover_color (%):")
    print((df.groupby("mover_color", observed=True)["blunder"].mean() * 100).round(2))


def process_all() -> None:
    stockfish_path = find_stockfish()
    print(f"using stockfish at {stockfish_path}, depth={ENGINE_DEPTH}")

    games_df = pd.read_parquet(GAMES_PATH)
    print(f"{len(games_df):,} games loaded")
    rows = games_df.to_dict("records")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    n_batches = len(batches)

    with mp.Pool(initializer=_init_worker, initargs=(stockfish_path,)) as pool:
        for b, batch_rows in enumerate(batches):
            ckpt_path = CHECKPOINT_DIR / f"batch_{b:04d}.parquet"
            if ckpt_path.exists():
                print(f"  batch {b + 1}/{n_batches}: already done, skipping", flush=True)
                continue

            t0 = time.time()
            per_game = pool.map(_parse_game_task, batch_rows, chunksize=1)
            move_rows = [r for g in per_game for r in g]
            pd.DataFrame(move_rows).to_parquet(ckpt_path, index=False)
            print(f"  batch {b + 1}/{n_batches}: {len(batch_rows)} games, "
                  f"{len(move_rows):,} move rows, {time.time() - t0:.0f}s", flush=True)

    all_moves = pd.concat(
        [pd.read_parquet(p) for p in sorted(CHECKPOINT_DIR.glob("batch_*.parquet"))],
        ignore_index=True,
    )
    print(f"\n{len(all_moves):,} total move rows (this player's own moves only)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "moves_gonzalopelotas.parquet"
    all_moves.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")

    print_verification(all_moves)

    for f in CHECKPOINT_DIR.glob("batch_*.parquet"):
        f.unlink()
    CHECKPOINT_DIR.rmdir()


def main() -> None:
    process_all()


if __name__ == "__main__":
    main()
