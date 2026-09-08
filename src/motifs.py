"""
Set 8 (part 1) -- group this player's blunders by the *type of mistake*, not the exact
position.

report.py already finds *which slices* leak rating points (middlegame, low clock, Queens
Gambit, ...). This module goes one level down: for every blunder that falls in one of those
leak slices, it re-analyses the position with a local Stockfish at MultiPV=3 and derives a
small set of heuristic tags -- what material was lost, how it was refuted (capture / check /
fork / back-rank / forced mate), and whether the player threw away a win or compounded a
worse position. Blunders are then bucketed into ~9 motif groups so the trainer (trainer.py)
can drill one pattern at a time.

Two honest limits, stated up front:

- **The tags are heuristics off a shallow line, not a tactics solver.** "fork" means the
  engine's refutation lands a piece attacking two of the player's pieces within the first
  couple of plies of one PV -- it will miss quieter forks and occasionally mislabel. They're
  useful for grouping, not for teaching tactics by themselves.
- **The base eval this all sits on is still parse.py's depth-14 single-PV pass.** This module
  re-evaluates only the ~few-hundred leak-category blunder positions, at depth 16 / MultiPV=3
  (from the pre-move position) plus a depth-14 / MultiPV=1 check of the position after the
  move actually played. That's enough to name the refutation and the "acceptable" replies;
  it is not a deep correspondence-grade analysis.

The engine is injected as a plain callable (same pattern as parse.py) so tests/test_motifs.py
can run against a deterministic fake instead of the real binary.
"""

import atexit
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

import chess
import chess.engine
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal import PLY_BUCKET_LABELS
from features import PIECE_VALUES
from parse import CP_CLAMP, MATE_CP, find_stockfish
from report import find_leaks, load_test_set

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
CHECKPOINT_DIR = PROCESSED_DIR / "_motifs_checkpoints"

USERNAME = "gonzalopelotas"

PRE_DEPTH = 16       # from the pre-move position -- deeper than parse.py's 14 (few positions)
POST_DEPTH = 14      # from the position after the move played -- only need the refutation
MULTIPV = 3
ENGINE_THREADS = 4   # parse.py got parallelism from a process pool; this module is one engine
ENGINE_HASH_MB = 256
ACCEPT_MARGIN_CP = 40   # a reply within this of the engine's best (mover POV) counts as "a best move"
PV_PLIES = 8            # how many plies of each line to keep for the reveal steppers
BATCH_SIZE = 40         # positions per checkpoint (mirrors parse.py's resumability)

LEAK_DIMENSIONS = ["opening_family", "mover_color", "ply_bucket", "clock_decile", "opp_band"]

PHASE_BY_BUCKET = {
    PLY_BUCKET_LABELS[0]: "opening",
    PLY_BUCKET_LABELS[1]: "middlegame",
    PLY_BUCKET_LABELS[2]: "late middlegame",
    PLY_BUCKET_LABELS[3]: "endgame",
}

# Highest-priority tag present wins the motif group. Phase is kept as a separate field
# (the trainer shows it as a secondary tag) rather than multiplied into the group name.
MOTIF_PRIORITY = [
    "allowed_mate", "back_rank", "fork",
    "hung a queen", "hung a rook", "hung a piece",
    "lost the exchange", "dropped a pawn",
]
MOTIF_GROUP_LABEL = {
    "allowed_mate": "Allowed forced mate",
    "back_rank": "Back-rank tactic",
    "fork": "Missed a fork",
    "hung a queen": "Hung the queen",
    "hung a rook": "Hung a rook",
    "hung a piece": "Hung a piece",
    "lost the exchange": "Lost the exchange",
    "dropped a pawn": "Dropped a pawn",
    "positional": "Positional slip (no material)",
}


class LineInfo(NamedTuple):
    """One MultiPV line, normalised. `score_cp` and `mate` are both from WHITE's point of
    view (matching parse.py's convention); `mate` is a signed move-count or None."""
    score_cp: float
    mate: int | None
    pv: list[chess.Move]


Analyzer = Callable[[chess.Board, int, int], list[LineInfo]]


# --------------------------------------------------------------------------------------------
# engine plumbing (skipped entirely by the tests, which inject a fake Analyzer)
# --------------------------------------------------------------------------------------------

_ENGINE: chess.engine.SimpleEngine | None = None


def _get_engine() -> chess.engine.SimpleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = chess.engine.SimpleEngine.popen_uci(find_stockfish())
        _ENGINE.configure({"Threads": ENGINE_THREADS, "Hash": ENGINE_HASH_MB})
        atexit.register(_ENGINE.quit)
    return _ENGINE


def _normalise_line(info: dict) -> LineInfo:
    score = info["score"].white()
    mate = score.mate()
    cp = score.score(mate_score=MATE_CP)
    cp = max(-CP_CLAMP, min(CP_CLAMP, float(cp)))
    return LineInfo(score_cp=cp, mate=mate, pv=list(info.get("pv", [])))


def stockfish_analyzer(board: chess.Board, multipv: int, depth: int) -> list[LineInfo]:
    engine = _get_engine()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    if isinstance(infos, dict):  # python-chess returns a bare dict when multipv == 1
        infos = [infos]
    return [_normalise_line(i) for i in infos]


# --------------------------------------------------------------------------------------------
# classification (pure -- unit-tested)
# --------------------------------------------------------------------------------------------

def _mover_cp(score_cp: float, mover: chess.Color) -> float:
    return score_cp if mover == chess.WHITE else -score_cp


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.color == color
    )


def _material_swing(fen_before: str, played: chess.Move, reply_pv: list[chess.Move]) -> int:
    """Most negative change in the mover's own material over the first few plies of the
    opponent's best line after the move played (a fork needs two plies to actually win the
    material, so we look a little way down the line, not just one capture)."""
    board = chess.Board(fen_before)
    mover = board.turn
    before = _material(board, mover)
    board.push(played)
    worst = _material(board, mover) - before
    for mv in reply_pv[:4]:
        if mv not in board.legal_moves:
            break
        board.push(mv)
        worst = min(worst, _material(board, mover) - before)
    return worst


def _material_label(swing: int) -> str:
    loss = -swing
    if loss >= 8:
        return "hung a queen"
    if loss >= 5:
        return "hung a rook"
    if loss >= 3:
        return "hung a piece"
    if loss >= 2:
        return "lost the exchange"
    if loss >= 1:
        return "dropped a pawn"
    return "positional"


def _is_fork(board_after_played: chess.Board, reply: chess.Move, victim: chess.Color) -> bool:
    """True if, after the opponent's reply, the piece that just moved attacks two or more of
    the blunderer's (`victim`'s) pieces worth a knight or more (king counts)."""
    b = board_after_played.copy()
    b.push(reply)
    targets = 0
    for sq in b.attacks(reply.to_square):
        piece = b.piece_at(sq)
        if piece is None or piece.color != victim:
            continue
        if piece.piece_type == chess.KING or PIECE_VALUES[piece.piece_type] >= 3:
            targets += 1
    return targets >= 2


def _is_back_rank(board_after_played: chess.Board, reply: chess.Move, victim: chess.Color) -> bool:
    king_sq = board_after_played.king(victim)
    if king_sq is None:
        return False
    back_rank = 0 if victim == chess.WHITE else 7
    if chess.square_rank(king_sq) != back_rank:
        return False
    moved = board_after_played.piece_at(reply.from_square)
    if moved is None or moved.piece_type not in (chess.ROOK, chess.QUEEN):
        return False
    return chess.square_rank(reply.to_square) == back_rank and board_after_played.gives_check(reply)


def _refutation_type(fen_before: str, played: chess.Move, lines_after: list[LineInfo]) -> str:
    if not lines_after or not lines_after[0].pv:
        return "unclear"
    board = chess.Board(fen_before)
    victim = board.turn
    board.push(played)
    reply = lines_after[0].pv[0]

    pov_mate = lines_after[0].mate
    if pov_mate is not None:
        # mate from white POV; victim gets mated if (white & mate<0) or (black & mate>0)
        victim_mated = (victim == chess.WHITE and pov_mate < 0) or (victim == chess.BLACK and pov_mate > 0)
        if victim_mated:
            return "allowed_mate"

    if reply not in board.legal_moves:
        return "unclear"
    if _is_back_rank(board, reply, victim):
        return "back_rank"
    if _is_fork(board, reply, victim):
        return "fork"
    if board.is_capture(reply):
        return "capture"
    if board.gives_check(reply):
        return "check"
    return "quiet"


def _advantage_state(wp_before: float) -> str:
    if wp_before >= 55:
        return "threw away a winning position"
    if wp_before >= 45:
        return "lost from an equal position"
    return "compounded a worse position"


def _line_sans(fen: str, pv: list[chess.Move], plies: int) -> str:
    board = chess.Board(fen)
    out = []
    for mv in pv[:plies]:
        if mv not in board.legal_moves:
            break
        out.append(board.san(mv))
        board.push(mv)
    return " ".join(out)


def _line_fens(fen: str, pv: list[chess.Move], plies: int) -> list[str]:
    board = chess.Board(fen)
    fens = [board.fen()]
    for mv in pv[:plies]:
        if mv not in board.legal_moves:
            break
        board.push(mv)
        fens.append(board.fen())
    return fens


def _eval_str(line: LineInfo, mover: chess.Color) -> str:
    if line.mate is not None:
        n = line.mate if mover == chess.WHITE else -line.mate
        return f"#{n}"
    cp = _mover_cp(line.score_cp, mover) / 100.0
    return f"{cp:+.1f}"


def _move_dict(board: chess.Board, move: chess.Move) -> dict:
    return {
        "uci": move.uci(),
        "san": board.san(move),
        "from": chess.square_name(move.from_square),
        "to": chess.square_name(move.to_square),
        "promotion": chess.piece_symbol(move.promotion) if move.promotion else None,
    }


def classify_blunder(
    fen_before: str,
    played_uci: str,
    wp_before: float,
    ply: int,
    lines_before: list[LineInfo],
    lines_after: list[LineInfo],
) -> dict:
    """Heuristic motif tags for one blunder. `lines_before` / `lines_after` are MultiPV
    lists, best-first for the side to move, from the pre-move position and from the position
    after the move actually played respectively."""
    board = chess.Board(fen_before)
    mover = board.turn
    played = chess.Move.from_uci(played_uci)

    best_mover_cp = _mover_cp(lines_before[0].score_cp, mover)
    acceptable = []
    for line in lines_before:
        if not line.pv:
            continue
        if _mover_cp(line.score_cp, mover) < best_mover_cp - ACCEPT_MARGIN_CP:
            break
        mv = line.pv[0]
        if mv in board.legal_moves:
            acceptable.append(_move_dict(board, mv))

    reply_pv = lines_after[0].pv if lines_after else []
    swing = _material_swing(fen_before, played, reply_pv)
    material_label = _material_label(swing)
    refutation = _refutation_type(fen_before, played, lines_after)

    tags = []
    if refutation in ("allowed_mate", "back_rank", "fork"):
        tags.append(refutation)
    if material_label != "positional":
        tags.append(material_label)
    motif_key = next((t for t in MOTIF_PRIORITY if t in tags), "positional")

    phase = PHASE_BY_BUCKET.get(_bucket_for_ply(ply), "middlegame")

    best_line = lines_before[0]
    top_lines = [
        {"san": _line_sans(fen_before, ln.pv, PV_PLIES), "eval": _eval_str(ln, mover)}
        for ln in lines_before if ln.pv
    ]

    return {
        "acceptable_moves": acceptable,
        "best_line_san": _line_sans(fen_before, best_line.pv, PV_PLIES),
        "best_line_fens": _line_fens(fen_before, best_line.pv, PV_PLIES),
        "top_lines": top_lines,
        "material_swing": swing,
        "material_label": material_label,
        "refutation_type": refutation,
        "advantage_state": _advantage_state(wp_before),
        "phase": phase,
        "motif_key": motif_key,
        "motif_group": MOTIF_GROUP_LABEL[motif_key],
    }


def _bucket_for_ply(ply: int) -> str:
    from causal import PLY_BUCKET_EDGES
    for label, lo, hi in zip(PLY_BUCKET_LABELS, PLY_BUCKET_EDGES[:-1], PLY_BUCKET_EDGES[1:]):
        if lo <= ply < hi:
            return label
    return PLY_BUCKET_LABELS[-1]


# --------------------------------------------------------------------------------------------
# selection + driver
# --------------------------------------------------------------------------------------------

def select_leak_blunders(df: pd.DataFrame, leaks: pd.DataFrame) -> pd.DataFrame:
    """Every blunder row that belongs to at least one of the ranked leak slices, tagged with
    which leak dimensions it fell in."""
    blunders = df[df["blunder"]].copy()
    leak_tags = {idx: [] for idx in blunders.index}
    for _, leak in leaks.iterrows():
        dim, val = leak["dimension"], leak["value"]
        hit = blunders[blunders[dim].astype(str) == val]
        for idx in hit.index:
            leak_tags[idx].append(f"{dim}={val}")
    blunders["leak_tags"] = [leak_tags[idx] for idx in blunders.index]
    return blunders[blunders["leak_tags"].str.len() > 0].reset_index(drop=True)


def analyse_one(row: pd.Series, analyzer: Analyzer) -> dict:
    board = chess.Board(row["fen_before"])
    lines_before = analyzer(board, MULTIPV, PRE_DEPTH)

    after = board.copy()
    after.push(chess.Move.from_uci(row["move_uci"]))
    lines_after = analyzer(after, 1, POST_DEPTH)

    tags = classify_blunder(
        row["fen_before"], row["move_uci"], float(row["wp_before"]), int(row["ply"]),
        lines_before, lines_after,
    )
    return {
        "game_id": row["game_id"],
        "ply": int(row["ply"]),
        "utc_date": row["utc_date"],
        "opening": row["opening"],
        "opening_family": row["opening_family"],
        "mover_color": row["mover_color"],
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "fen_before": row["fen_before"],
        "move_uci": row["move_uci"],
        "move_san": row["move_san"],
        "played": _move_dict(board, chess.Move.from_uci(row["move_uci"])),
        "wp_before": float(row["wp_before"]),
        "wp_loss": float(row["wp_loss"]),
        "cp_before": float(row["cp_before"]),
        "clock_before": float(row["clock_before"]),
        "increment": int(row["increment"]),
        "base_time": int(row["base_time"]),
        "leak_tags": list(row["leak_tags"]),
        **tags,
    }


def _flatten_for_csv(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["acceptable_moves"] = df["acceptable_moves"].apply(lambda ms: " / ".join(m["san"] for m in ms))
    df["top_lines"] = df["top_lines"].apply(lambda ls: " | ".join(f"{l['san']} ({l['eval']})" for l in ls))
    df["leak_tags"] = df["leak_tags"].apply("; ".join)
    df["played"] = df["played"].apply(lambda m: m["san"])
    df = df.drop(columns=["best_line_fens"])
    return df


def run(analyzer: Analyzer | None = None) -> pd.DataFrame:
    analyzer = analyzer or stockfish_analyzer

    df = load_test_set()
    leaks = find_leaks(df)
    targets = select_leak_blunders(df, leaks)
    print(f"{USERNAME}: {len(targets)} leak-category blunders to analyse "
          f"(MultiPV={MULTIPV}, depth {PRE_DEPTH}/{POST_DEPTH})")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    batches = [targets.iloc[i:i + BATCH_SIZE] for i in range(0, len(targets), BATCH_SIZE)]
    for b, batch in enumerate(batches):
        ckpt = CHECKPOINT_DIR / f"batch_{b:03d}.pkl"
        if ckpt.exists():
            print(f"  batch {b + 1}/{len(batches)}: done, skipping")
            continue
        t0 = time.time()
        records = [analyse_one(row, analyzer) for _, row in batch.iterrows()]
        with open(ckpt, "wb") as f:
            pickle.dump(records, f)
        print(f"  batch {b + 1}/{len(batches)}: {len(records)} positions, {time.time() - t0:.0f}s",
              flush=True)

    records = []
    for ckpt in sorted(CHECKPOINT_DIR.glob("batch_*.pkl")):
        with open(ckpt, "rb") as f:
            records.extend(pickle.load(f))

    records.sort(key=lambda r: r["wp_loss"], reverse=True)
    out = pd.DataFrame(records)

    # The full records carry nested lists-of-dicts (acceptable_moves, top_lines, ...), which
    # parquet type-inference handles badly -- JSON is the honest format here. trainer.py reads
    # this; the flat CSV alongside it is for eyeballing.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED_DIR / f"blunder_motifs_{USERNAME}.json"
    json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _flatten_for_csv(records).to_csv(TABLES_DIR / "blunder_motifs.csv", index=False)

    print(f"\nwrote {json_path} ({len(out)} rows)")
    print("\nmotif-group distribution:")
    dist = out.groupby("motif_group").agg(
        n=("motif_group", "size"),
        mean_wp_loss=("wp_loss", "mean"),
        pct_threw_away_win=("advantage_state", lambda s: (s == "threw away a winning position").mean() * 100),
    ).sort_values("n", ascending=False)
    print(dist.round(1).to_string())

    for f in CHECKPOINT_DIR.glob("batch_*.pkl"):
        f.unlink()
    CHECKPOINT_DIR.rmdir()
    return out


def main() -> None:
    run()


if __name__ == "__main__":
    main()
