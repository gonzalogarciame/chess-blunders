"""
Set 1 -- pull every one of gonzalopelotas's own chess.com games and keep the ones usable for a
clock-pressure blunder analysis: standard chess rules (no variants), real-time (not daily/
correspondence, where "time pressure" doesn't mean the same thing), rated (so Elo means
something), not abandoned, and carrying [%clk] annotations.

chess.com's public API never exposes engine [%eval] in the PGN it returns -- verified across
every one of this player's games, even ones "Game Review" was likely run on -- so unlike the
original Lichess-based design, eval isn't filtered on or extracted here at all. parse.py
generates it instead, with a local Stockfish instance.

Strategy: chess.com's API is one JSON array of games per player-month, unauthenticated, no
per-shard rate-limiting the way the HF pull had. At this player's scale (~1,000 games across
~36 months) that's cheap enough to just fetch every month directly -- no batching/resume
machinery needed, unlike the original ingest.py.

Per-game filter decisions use chess.com's own structured per-side `result` codes ("win",
"timeout", "resigned", "abandoned", ...) rather than parsing Lichess-style free-text
Termination strings -- more reliable, and chess.com doesn't give us the latter anyway.
"""

import json
import urllib.request
from pathlib import Path

import pandas as pd

USERNAME = "gonzalopelotas"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "rating-coach (personal chess self-analysis; contact via GitHub)"}

# The matched time-control pair (same base time, increment vs. none) causal.py's DiD design
# uses. NOT the most common time control (that's 180+2 by a wide margin) -- the point is a
# *balanced* pair, and most of this player's unlimited-increment 180 games turn out to be casual
# (unrated) and get dropped by the rated_ok filter, leaving only ~3 survivors at 180+0 vs. ~470
# at 180+2. 300/300+5 is what's actually balanced post-filter (see ingest.py's own printed
# breakdown) -- confirmed empirically, not assumed, per the project's usual practice of checking
# real survivor counts before picking a design.
TC_PAIR = ("300", "300+5")

STEP_NAMES = [
    "rules == chess (no variants)",
    "time_class != daily (real-time games only)",
    "rated",
    "movetext contains [%clk]",
    "not abandoned",
]


def list_archives() -> list[str]:
    url = f"https://api.chess.com/pub/player/{USERNAME}/games/archives"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.load(r)["archives"]


def fetch_month(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.load(r).get("games", [])


def game_id_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def opening_from_eco(eco_url: str | None) -> str | None:
    if not eco_url:
        return None
    slug = eco_url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ")


def result_string(g: dict) -> str:
    if g["white"]["result"] == "win":
        return "1-0"
    if g["black"]["result"] == "win":
        return "0-1"
    return "1/2-1/2"


def is_time_forfeit(g: dict) -> bool:
    return g["white"]["result"] == "timeout" or g["black"]["result"] == "timeout"


def is_abandoned(g: dict) -> bool:
    return g["white"]["result"] == "abandoned" or g["black"]["result"] == "abandoned"


def add_filter_flags(df: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    # Flags computed from the raw chess.com game dicts (games), not the flattened df -- to_row()
    # already collapsed white/black into scalar columns, so per-side result checks need the
    # original nested structure, aligned to df by position (same order as `games`).
    df["rules_ok"] = [g["rules"] == "chess" for g in games]
    df["not_daily"] = [g["time_class"] != "daily" for g in games]
    df["rated_ok"] = [bool(g["rated"]) for g in games]
    df["clk_ok"] = df["movetext"].str.contains("%clk", regex=False)
    df["not_abandoned"] = [not is_abandoned(g) for g in games]
    return df


def to_row(g: dict) -> dict:
    return {
        "game_id": game_id_from_url(g["url"]),
        "white": g["white"]["username"],
        "black": g["black"]["username"],
        "white_elo": g["white"]["rating"],
        "black_elo": g["black"]["rating"],
        "time_control": g["time_control"],
        "time_class": g["time_class"],
        "rules": g["rules"],
        "rated": g["rated"],
        "result": result_string(g),
        "termination": "Time forfeit" if is_time_forfeit(g) else "Normal",
        "end_time": g["end_time"],  # unix epoch, UTC -- causal.py's day-level grouping key
        "opening": opening_from_eco(g.get("eco")),
        "movetext": g["pgn"],
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    archives = list_archives()
    print(f"{len(archives)} monthly archives found for {USERNAME}")

    all_games = []
    for url in archives:
        games = fetch_month(url)
        all_games.extend(games)
        label = url.rsplit("/", 2)
        print(f"  {label[-2]}-{label[-1]}: {len(games)} games", flush=True)
    print(f"\ntotal games scanned: {len(all_games):,}")

    # A handful of games (e.g. still in progress, or missing PGN/result for other reasons) lack
    # fields this pipeline needs -- drop them here rather than letting to_row() crash on them.
    required = ("pgn", "white", "black", "rules", "time_control", "time_class", "rated", "end_time")
    usable = [g for g in all_games if all(g.get(k) is not None for k in required)]
    n_unusable = len(all_games) - len(usable)
    if n_unusable:
        print(f"dropped {n_unusable:,} games missing required fields (e.g. still in progress)")

    df = pd.DataFrame([to_row(g) for g in usable])
    df = add_filter_flags(df, usable)

    flag_cols = ["rules_ok", "not_daily", "rated_ok", "clk_ok", "not_abandoned"]
    print("\nsurvivors by filter step:")
    running = pd.Series(True, index=df.index)
    n = len(df)
    for name, col in zip(STEP_NAMES, flag_cols):
        running &= df[col]
        print(f"  scanned {n:,} -> {name}: {running.sum():,}")

    kept = df[running].copy()
    print(f"\nfinal kept: {len(kept):,} / {len(df):,}")

    kept["utc_date"] = pd.to_datetime(kept["end_time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")

    print("\ntime_control breakdown (kept games):")
    print(kept["time_control"].value_counts())

    tc_lo, tc_hi = TC_PAIR
    n_lo = (kept["time_control"] == tc_lo).sum()
    n_hi = (kept["time_control"] == tc_hi).sum()
    print(f"\nprimary matched pair {tc_lo} (no increment) / {tc_hi} (increment): "
          f"{n_lo:,} / {n_hi:,} games")

    out_cols = ["game_id", "white", "black", "white_elo", "black_elo", "time_control",
                "result", "termination", "utc_date", "end_time", "opening", "movetext"]
    out_path = RAW_DIR / f"games_{USERNAME}.parquet"
    kept[out_cols].to_parquet(out_path, index=False)
    print(f"\nwrote {len(kept):,} games to {out_path}")

    samples = kept["movetext"].head(2)
    for i, mv in enumerate(samples):
        print(f"  sample {i}: {mv[:300]}")


if __name__ == "__main__":
    main()
