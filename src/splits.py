"""
Set 3 -- train/val/test splits.

Train/val come from MONTH_A, split 80/20 by a hash of game_id -- never by move, since moves
within a game are heavily correlated and a move-level split would leak game context between
train and val. Test is all of MONTH_B, held out for temporal generalisation (see README).

test_unseen_players is the subset of test whose mover never appears in MONTH_A. The gap
between test and test_unseen_players metrics (computed downstream in evaluate.py) is a
direct measure of how much the model relies on having seen a specific player before.
"""

import hashlib
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MONTH_A = "2024-03"
MONTH_B = "2024-09"
VAL_FRACTION = 0.2


def game_id_hash_fraction(game_id: str) -> float:
    digest = hashlib.md5(game_id.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def make_splits() -> dict[str, pd.DataFrame]:
    df_a = pd.read_parquet(PROCESSED_DIR / f"features_{MONTH_A}.parquet")
    df_b = pd.read_parquet(PROCESSED_DIR / f"features_{MONTH_B}.parquet")

    is_val = df_a["game_id"].apply(game_id_hash_fraction) < VAL_FRACTION
    train, val = df_a[~is_val], df_a[is_val]

    known_players = set(df_a["mover"].unique())
    test = df_b
    test_unseen = df_b[~df_b["mover"].isin(known_players)]

    splits = {"train": train, "val": val, "test": test, "test_unseen_players": test_unseen}
    for name, split_df in splits.items():
        split_df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    print_verification(splits)
    return splits


def print_verification(splits: dict[str, pd.DataFrame]) -> None:
    for name, df in splits.items():
        print(f"{name}: shape={df.shape}, blunder rate={df['blunder'].mean() * 100:.2f}%")
    n_test, n_unseen = len(splits["test"]), len(splits["test_unseen_players"])
    print(f"test_unseen_players: {n_unseen:,} rows ({n_unseen / n_test * 100:.2f}% of test)")


if __name__ == "__main__":
    make_splits()
