"""
Set 3 -- train/val/test splits.

No population of players to hold out unseen players from anymore (single-player scope), so this
is a purely chronological split, exploiting a real ~9-month gap in this player's own history
(see ingest.py's printed per-month breakdown): train/val = every game through 2023 (a thin 2021
fragment plus the dense 2023-03..2023-11 block), test = every game from 2024-08 onward. That's a
longer, more real temporal-generalisation test than the original design's "MONTH_A vs. MONTH_B,
>=3 months apart" rule -- does a model (and the causal estimate) fit on old play generalise to
how this player plays 9+ months later, after whatever skill/style drift happened in between.

Train/val split within the train/val pool stays 80/20 by a hash of game_id -- never by move,
since moves within a game are heavily correlated and a move-level split would leak game context
between train and val (unchanged logic from the original design).
"""

import hashlib
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TRAIN_VAL_TEST_CUTOFF = "2024-08"  # test = utc_date >= this; train/val = everything before
VAL_FRACTION = 0.2


def game_id_hash_fraction(game_id: str) -> float:
    digest = hashlib.md5(game_id.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def make_splits() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(PROCESSED_DIR / "features_gonzalopelotas.parquet")

    is_test = df["utc_date"] >= TRAIN_VAL_TEST_CUTOFF
    train_val, test = df[~is_test], df[is_test]

    is_val = train_val["game_id"].apply(game_id_hash_fraction) < VAL_FRACTION
    train, val = train_val[~is_val], train_val[is_val]

    splits = {"train": train, "val": val, "test": test}
    for name, split_df in splits.items():
        split_df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    print_verification(splits)
    return splits


def print_verification(splits: dict[str, pd.DataFrame]) -> None:
    for name, df in splits.items():
        date_lo, date_hi = df["utc_date"].min(), df["utc_date"].max()
        print(f"{name}: shape={df.shape}, blunder rate={df['blunder'].mean() * 100:.2f}%, "
              f"dates [{date_lo}, {date_hi}]")


if __name__ == "__main__":
    make_splits()
