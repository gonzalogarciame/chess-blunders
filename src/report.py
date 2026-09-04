"""
Set 7 -- a single-player report: this player's blunder rate by clock decile against a
rating-matched baseline, their individual calibration, their time allocation profile, and a
counterfactual built from the Step 5 causal estimate. One matplotlib figure, one printed table,
no web app.

Uses data/processed/test.parquet (all of MONTH_B) throughout, deliberately -- since the model
was trained only on MONTH_A, every row here is out-of-sample for it regardless of whether this
player also appears in MONTH_A, which is what makes the calibration check meaningful.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal import PLY_BUCKET_EDGES, PLY_BUCKET_LABELS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

MIN_MOVES = 200
ELO_BAND_WIDTH = 100
N_DECILES = 10
OPENING_BUCKET = PLY_BUCKET_LABELS[0]   # "9-20"
MIDDLEGAME_BUCKET = PLY_BUCKET_LABELS[1]  # "21-40"
SHIFT_FRACTION = 0.20
MIN_RELIABLE_SECONDS = 1.0  # below this, the DiD-implied per-second rate is too noisy to trust


def load_test_set() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    df["ply_bucket"] = pd.cut(df["ply"], bins=PLY_BUCKET_EDGES, labels=PLY_BUCKET_LABELS, right=False)
    df["time_spent"] = df["clock_before"] - df["clock_after"] + df["increment"]
    return df


def load_model():
    with open(PROCESSED_DIR / "model_lightgbm.pkl", "rb") as f:
        return pickle.load(f)


def load_causal_artifacts() -> dict:
    with open(PROCESSED_DIR / "causal_did_artifacts.pkl", "rb") as f:
        return pickle.load(f)


def get_player_and_baseline(df: pd.DataFrame, username: str) -> tuple[pd.DataFrame, pd.DataFrame, tuple]:
    player = df[df["mover"] == username]
    if len(player) < MIN_MOVES:
        raise ValueError(f"{username} has {len(player)} moves in test.parquet, need >= {MIN_MOVES}")

    elo = player["mover_elo"].mean()
    band_lo = (elo // ELO_BAND_WIDTH) * ELO_BAND_WIDTH
    band_hi = band_lo + ELO_BAND_WIDTH
    baseline = df[(df["mover"] != username) & (df["mover_elo"] >= band_lo) & (df["mover_elo"] < band_hi)]
    return player, baseline, (band_lo, band_hi)


def blunder_by_clock_decile(player: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    _, edges = pd.qcut(baseline["clock_before"], N_DECILES, retbins=True, duplicates="drop")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf  # so the player's own min/max don't fall outside

    player_bins = pd.cut(player["clock_before"], bins=edges)
    baseline_bins = pd.cut(baseline["clock_before"], bins=edges)

    out = pd.DataFrame({
        "player": player.groupby(player_bins, observed=True)["blunder"].mean(),
        "baseline": baseline.groupby(baseline_bins, observed=True)["blunder"].mean(),
    })
    out.index = [f"decile {i + 1}" for i in range(len(out))]
    return out


def calibration_check(model, player: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict_proba(player)
    bins = pd.qcut(pred, N_DECILES, duplicates="drop")
    out = pd.DataFrame({"predicted": pred, "actual": player["blunder"].to_numpy()}, index=player.index)
    return out.groupby(bins, observed=True).mean()


def time_allocation_profile(player: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "player": player.groupby("ply_bucket", observed=True)["time_spent"].mean(),
        "baseline": baseline.groupby("ply_bucket", observed=True)["time_spent"].mean(),
    })
    return out.reindex(PLY_BUCKET_LABELS)


def counterfactual(player: pd.DataFrame, time_profile: pd.DataFrame, artifacts: dict) -> dict:
    """Extrapolates the Step 5 DiD estimate to a within-player time reallocation. Assumptions,
    stated plainly:
      1. The causal effect of increment on blunder rate operates through the extra clock
         seconds it makes available in a bucket, and scales roughly linearly with those
         seconds -- so a per-second rate estimated from the increment contrast can be applied
         to a much smaller, individually-chosen time shift. This is an extrapolation: the
         original estimate came from a fixed per-move bonus compounding over a whole game, not
         from a one-off reallocation between two buckets.
      2. Reducing opening time does not raise opening blunder risk, justified by the placebo/
         parallel-trends result: clock differences haven't yet had time to matter that early
         in the game, so the model has nothing to say about an opening-side cost and none is
         charged here.
      3. The player's own moves-per-bucket counts and current time-per-move are held fixed
         except for the reallocated seconds.
    """
    result = artifacts["did_result"]
    bucket_diff = artifacts["bucket_time_diff"]

    names = list(result.params.index)
    interaction_name = f"treat_x_bucket_{MIDDLEGAME_BUCKET}"
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(interaction_name)] = 1.0
    did_effect_middlegame = float(contrast @ result.params.values)

    extra_seconds_middlegame = float(bucket_diff.loc[MIDDLEGAME_BUCKET, True] - bucket_diff.loc[MIDDLEGAME_BUCKET, False])
    # This ratio's denominator is an empirical, noisy quantity -- if increment barely changes
    # measured time-spent-per-move in this bucket, the implied per-second rate (and everything
    # downstream of it) is not reliably identified, however large it looks.
    reliable = abs(extra_seconds_middlegame) >= MIN_RELIABLE_SECONDS
    per_second_effect = did_effect_middlegame / extra_seconds_middlegame if reliable else float("nan")

    n_opening_moves = int((player["ply_bucket"] == OPENING_BUCKET).sum())
    mean_opening_time = float(time_profile.loc[OPENING_BUCKET, "player"])
    total_opening_seconds = n_opening_moves * mean_opening_time
    shifted_seconds = SHIFT_FRACTION * total_opening_seconds

    n_middlegame_moves = int((player["ply_bucket"] == MIDDLEGAME_BUCKET).sum())
    expected_blunder_count_change = per_second_effect * shifted_seconds if reliable else float("nan")

    if not reliable:
        print(f"  WARNING: increment changes measured time-spent-per-move in the {MIDDLEGAME_BUCKET} "
              f"bucket by only {extra_seconds_middlegame:+.2f}s (< {MIN_RELIABLE_SECONDS}s) -- the "
              f"per-second conversion is not reliably identified, so the counterfactual is reported "
              f"as unavailable rather than as a number that looks precise but isn't.")

    return {
        "did_effect_middlegame_bucket": did_effect_middlegame,
        "extra_seconds_per_move_from_increment_middlegame": extra_seconds_middlegame,
        "implied_effect_per_extra_second": per_second_effect,
        "n_opening_moves": n_opening_moves,
        "shifted_seconds_total": shifted_seconds,
        "n_middlegame_moves": n_middlegame_moves,
        "expected_change_in_middlegame_blunder_count": expected_blunder_count_change,
        "reliable": reliable,
    }


def plot_report(username: str, decile_df: pd.DataFrame, calib_df: pd.DataFrame,
                 time_df: pd.DataFrame, cf: dict, elo_band: tuple, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Player report: {username}  (Elo band {elo_band[0]:.0f}-{elo_band[1]:.0f})")

    ax = axes[0, 0]
    x = np.arange(len(decile_df))
    ax.plot(x, decile_df["player"], marker="o", label=username)
    ax.plot(x, decile_df["baseline"], marker="o", label="rating-matched baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1}" for i in x])
    ax.set_xlabel("clock decile (1 = least time remaining)")
    ax.set_ylabel("blunder rate")
    ax.set_title("Blunder rate by clock decile")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(calib_df["predicted"], calib_df["actual"], marker="o", label=username)
    ax.plot([0, calib_df[["predicted", "actual"]].to_numpy().max()] * 1,
            [0, calib_df[["predicted", "actual"]].to_numpy().max()],
            linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed blunder rate")
    ax.set_title("Predicted vs. actual (individual calibration)")
    ax.legend()

    ax = axes[1, 0]
    x = np.arange(len(time_df))
    ax.plot(x, time_df["player"], marker="o", label=username)
    ax.plot(x, time_df["baseline"], marker="o", label="rating-matched baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(time_df.index.astype(str))
    ax.set_xlabel("ply bucket")
    ax.set_ylabel("mean seconds spent per move")
    ax.set_title("Time allocation profile")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"Counterfactual: shift {SHIFT_FRACTION:.0%} of opening\n"
        f"time into the middlegame\n\n"
        f"Opening moves ({OPENING_BUCKET}): {cf['n_opening_moves']}\n"
        f"Seconds shifted: {cf['shifted_seconds_total']:.1f}\n"
        f"DiD effect ({MIDDLEGAME_BUCKET} bucket):\n  {cf['did_effect_middlegame_bucket']:+.4f}\n"
        f"Extra sec/move from increment:\n  {cf['extra_seconds_per_move_from_increment_middlegame']:.2f}\n"
        f"Implied effect per extra second:\n  {cf['implied_effect_per_extra_second']:+.6f}\n\n"
        f"Expected change in middlegame\nblunder count:\n  {cf['expected_change_in_middlegame_blunder_count']:+.3f}\n\n"
        f"This is an extrapolation from the\n"
        f"Step 5 increment estimate, not a\n"
        f"direct measurement -- see README\n"
        f"for the assumptions it rests on."
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_report(username: str) -> None:
    df = load_test_set()
    player, baseline, elo_band = get_player_and_baseline(df, username)
    print(f"{username}: {len(player):,} moves in test set, Elo band [{elo_band[0]:.0f}, {elo_band[1]:.0f}), "
          f"baseline: {len(baseline):,} moves from {baseline['mover'].nunique():,} other players")

    model = load_model()
    artifacts = load_causal_artifacts()

    decile_df = blunder_by_clock_decile(player, baseline)
    calib_df = calibration_check(model, player)
    time_df = time_allocation_profile(player, baseline)
    cf = counterfactual(player, time_df, artifacts)

    out_path = FIGURES_DIR / f"player_report_{username}.png"
    plot_report(username, decile_df, calib_df, time_df, cf, elo_band, out_path)
    print(f"saved {out_path}")

    print("\nblunder rate by clock decile:")
    print(decile_df)
    print("\ncalibration (predicted vs actual, by predicted-probability bin):")
    print(calib_df)
    print("\ntime allocation profile (mean seconds/move):")
    print(time_df)
    print("\ncounterfactual:")
    for k, v in cf.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-player blunder report (Set 7).")
    parser.add_argument("username", help="Lichess username, as it appears in the 'mover' column")
    args = parser.parse_args()
    generate_report(args.username)


if __name__ == "__main__":
    main()
