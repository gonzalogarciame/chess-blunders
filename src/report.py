"""
Set 7 -- a rating-leak report: a ranked, self-relative list of what's actually costing this
player rating points, plus an individual calibration check, the increment counterfactual from
causal.py, and an LLM-written coaching narrative tying it together.

No peer/population comparison anywhere (explicit design choice) -- every number here is this
player's own moves sliced against their own overall average. That's what makes each finding
"theirs" and actionable without needing other players' data: "you blunder 3x your average rate
in the Sicilian as Black" doesn't need a baseline population to be useful.

Uses data/processed/test.parquet (2024-08 onward -- see splits.py) throughout, deliberately: the
model and causal estimate were both fit on earlier data, so every row here is out-of-sample,
which is what makes the calibration check and the leak-finder's numbers meaningful rather than
circular.
"""

import os
import pickle
from pathlib import Path

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal import PLY_BUCKET_EDGES, PLY_BUCKET_LABELS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

USERNAME = "gonzalopelotas"
MIN_SLICE_MOVES = 20  # floor below which a slice's rate is too noisy to call a "leak"
N_CLOCK_DECILES = 10
TOP_N_LEAKS = 8
OPENING_FAMILY_WORDS = 2  # how many words of the opening name define a "family"

OPPONENT_BAND_EDGES = [-10_000, -200, -50, 50, 200, 10_000]
OPPONENT_BAND_LABELS = ["much stronger opp", "stronger opp", "similar opp", "weaker opp", "much weaker opp"]

OPENING_BUCKET = PLY_BUCKET_LABELS[0]
MIDDLEGAME_BUCKET = PLY_BUCKET_LABELS[1]
SHIFT_FRACTION = 0.20
MIN_RELIABLE_SECONDS = 1.0

# Groq is the default (free tier, no credit card needed); ANTHROPIC_MODEL is only used if
# GROQ_API_KEY isn't set but ANTHROPIC_API_KEY is (see generate_coaching_narrative).
GROQ_MODEL = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-5"


def load_test_set() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    df["ply_bucket"] = pd.cut(df["ply"], bins=PLY_BUCKET_EDGES, labels=PLY_BUCKET_LABELS, right=False)
    df["time_spent"] = df["clock_before"] - df["clock_after"] + df["increment"]
    df["opening_family"] = df["opening"].fillna("(unknown)").str.split().str[:OPENING_FAMILY_WORDS].str.join(" ")
    df["opp_band"] = pd.cut(df["elo_diff"], bins=OPPONENT_BAND_EDGES, labels=OPPONENT_BAND_LABELS)
    df["clock_decile"] = pd.qcut(df["clock_before"], N_CLOCK_DECILES, labels=False, duplicates="drop") + 1
    return df


def load_model():
    with open(PROCESSED_DIR / "model_lightgbm.pkl", "rb") as f:
        return pickle.load(f)


def load_causal_artifacts() -> dict:
    with open(PROCESSED_DIR / "causal_did_artifacts.pkl", "rb") as f:
        return pickle.load(f)


def hangs_material(row: pd.Series) -> bool:
    """Coarse blunder-type tag, free from data parse.py already computed: if the engine's best
    reply to the position AFTER this move is itself a capture, the move most likely just hung a
    piece outright rather than losing to a subtler positional or deep-tactical idea."""
    if not row["blunder"] or pd.isna(row["engine_pv_after"]) or row["move_uci"] is None:
        return False
    try:
        board = chess.Board(row["fen_before"])
        board.push(chess.Move.from_uci(row["move_uci"]))
        reply = chess.Move.from_uci(row["engine_pv_after"])
        return board.is_capture(reply)
    except (ValueError, AssertionError):
        return False


RECOMMENDATIONS = {
    "opening_family": "study or avoid this opening -- it's costing you disproportionately.",
    "mover_color": "your prep or instincts differ noticeably by color; look at games in this "
                   "color specifically.",
    "ply_bucket": "this phase of the game is where you're leaking the most -- focus study time "
                  "here over the others.",
    "clock_decile": "you blunder disproportionately with this little time on the clock -- "
                     "practice faster calculation, or manage the clock earlier so you don't "
                     "arrive here this low.",
    "opp_band": "a skewed rate against this opponent-strength band suggests either nerves "
                "(vs. stronger opponents) or complacency (vs. weaker ones).",
}


def slice_leaks(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    overall_rate = df["blunder"].mean()
    g = df.groupby(dimension, observed=True)["blunder"].agg(["mean", "size"])
    g = g[g["size"] >= MIN_SLICE_MOVES].copy()
    g["dimension"] = dimension
    g["value"] = g.index.astype(str)
    g["blunder_rate"] = g["mean"]
    g["n_moves"] = g["size"]
    g["excess_rate"] = g["blunder_rate"] - overall_rate
    g["excess_blunders"] = g["excess_rate"] * g["n_moves"]
    return g[["dimension", "value", "n_moves", "blunder_rate", "excess_rate", "excess_blunders"]]


def find_leaks(df: pd.DataFrame) -> pd.DataFrame:
    dimensions = ["opening_family", "mover_color", "ply_bucket", "clock_decile", "opp_band"]
    all_slices = pd.concat([slice_leaks(df, d) for d in dimensions], ignore_index=True)
    leaks = all_slices[all_slices["excess_blunders"] > 0].sort_values("excess_blunders", ascending=False)
    return leaks.head(TOP_N_LEAKS).reset_index(drop=True)


def add_leak_recommendations(leaks: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    leaks = leaks.copy()
    hangs_pct = []
    for _, row in leaks.iterrows():
        slice_df = df[df[row["dimension"]].astype(str) == row["value"]]
        blunders = slice_df[slice_df["blunder"]]
        if len(blunders) == 0:
            hangs_pct.append(np.nan)
            continue
        tagged = blunders.apply(hangs_material, axis=1)
        hangs_pct.append(tagged.mean() * 100)
    leaks["pct_blunders_hang_material"] = hangs_pct
    leaks["recommendation"] = leaks["dimension"].map(RECOMMENDATIONS)
    return leaks


def calibration_check(model, df: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict_proba(df)
    bins = pd.qcut(pred, 10, duplicates="drop")
    out = pd.DataFrame({"predicted": pred, "actual": df["blunder"].to_numpy()}, index=df.index)
    return out.groupby(bins, observed=True).mean()


def counterfactual(player: pd.DataFrame, artifacts: dict) -> dict:
    """Extrapolates the causal.py estimate to a within-player time reallocation. Assumptions,
    stated plainly (unchanged from the original design's logic, now applied to the regression-
    adjusted effect instead of a DiD-with-fixed-effects one):
      1. The effect of increment on blunder rate operates through extra clock seconds and
         scales roughly linearly with them -- an extrapolation from a fixed per-move bonus
         compounding over a whole game to a one-off reallocation.
      2. Reducing opening time doesn't raise opening blunder risk (justified by the placebo
         check in causal.py: clock differences haven't had time to matter that early).
      3. This player's own moves-per-bucket counts are held fixed except for the reallocation.
    """
    result = artifacts["did_result"]
    bucket_diff = artifacts["bucket_time_diff"]

    names = list(result.params.index)
    interaction_name = next(n for n in names if n.startswith("treatment:") and f".{MIDDLEGAME_BUCKET}]" in n)
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(interaction_name)] = 1.0
    did_effect_middlegame = float(contrast @ result.params.values)

    # bucket_diff is causal.py's regression-adjusted (for base_time) seconds-per-move effect,
    # already the treatment-vs-control difference -- not a pair of group means to subtract.
    extra_seconds_middlegame = float(bucket_diff[MIDDLEGAME_BUCKET])
    reliable = abs(extra_seconds_middlegame) >= MIN_RELIABLE_SECONDS
    per_second_effect = did_effect_middlegame / extra_seconds_middlegame if reliable else float("nan")

    time_profile = player.groupby("ply_bucket", observed=True)["time_spent"].mean()
    n_opening_moves = int((player["ply_bucket"] == OPENING_BUCKET).sum())
    mean_opening_time = float(time_profile.get(OPENING_BUCKET, np.nan))
    total_opening_seconds = n_opening_moves * mean_opening_time
    shifted_seconds = SHIFT_FRACTION * total_opening_seconds

    expected_blunder_count_change = per_second_effect * shifted_seconds if reliable else float("nan")

    if not reliable:
        print(f"  WARNING: increment changes measured time-spent-per-move in the {MIDDLEGAME_BUCKET} "
              f"bucket by only {extra_seconds_middlegame:+.2f}s (< {MIN_RELIABLE_SECONDS}s) -- the "
              f"counterfactual is reported as unavailable rather than as a number that looks "
              f"precise but isn't.")

    return {
        "did_effect_middlegame_bucket": did_effect_middlegame,
        "extra_seconds_per_move_from_increment_middlegame": extra_seconds_middlegame,
        "implied_effect_per_extra_second": per_second_effect,
        "n_opening_moves": n_opening_moves,
        "shifted_seconds_total": shifted_seconds,
        "expected_change_in_middlegame_blunder_count": expected_blunder_count_change,
        "reliable": reliable,
    }


def plot_report(leaks: pd.DataFrame, calib_df: pd.DataFrame, cf: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Rating-leak report: {USERNAME}")

    ax = axes[0]
    plot_leaks = leaks.iloc[::-1]  # largest at top
    labels = [f"{r['dimension']}={r['value']}" for _, r in plot_leaks.iterrows()]
    ax.barh(labels, plot_leaks["excess_blunders"], color="tab:red")
    ax.set_xlabel("estimated excess blunders (excess rate x moves seen)")
    ax.set_title("Top rating leaks")

    ax = axes[1]
    ax.plot(calib_df["predicted"], calib_df["actual"], marker="o")
    lim = float(calib_df[["predicted", "actual"]].to_numpy().max())
    ax.plot([0, lim], [0, lim], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed blunder rate")
    ax.set_title("Individual calibration (test set)")
    ax.legend()

    ax = axes[2]
    ax.axis("off")
    text = (
        f"Counterfactual: shift {SHIFT_FRACTION:.0%} of opening\n"
        f"time into the middlegame\n\n"
        f"Opening moves ({OPENING_BUCKET}): {cf['n_opening_moves']}\n"
        f"Seconds shifted: {cf['shifted_seconds_total']:.1f}\n"
        f"Regression effect ({MIDDLEGAME_BUCKET} bucket):\n  {cf['did_effect_middlegame_bucket']:+.4f}\n"
        f"Extra sec/move from increment:\n  {cf['extra_seconds_per_move_from_increment_middlegame']:.2f}\n"
        f"Implied effect per extra second:\n  {cf['implied_effect_per_extra_second']:+.6f}\n\n"
        f"Expected change in middlegame\nblunder count:\n  {cf['expected_change_in_middlegame_blunder_count']:+.3f}\n\n"
        f"Extrapolation from causal.py's\n"
        f"regression-adjusted estimate, not a\n"
        f"direct measurement -- see README\n"
        f"for the assumptions it rests on."
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9.5, family="monospace")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_llm_prompt(leaks: pd.DataFrame, cf: dict, overall_rate: float) -> str:
    leak_lines = "\n".join(
        f"- {r['dimension']}={r['value']}: blunder rate {r['blunder_rate']*100:.1f}% vs your "
        f"{overall_rate*100:.1f}% overall average, over {int(r['n_moves'])} moves "
        f"({r['pct_blunders_hang_material']:.0f}% of these blunders hang material outright). "
        f"{r['recommendation']}"
        for _, r in leaks.iterrows()
    )
    cf_line = (
        f"Playing with increment is estimated to change your middlegame blunder count by "
        f"{cf['expected_change_in_middlegame_blunder_count']:+.2f} per game if you shifted "
        f"{SHIFT_FRACTION:.0%} of your opening time into the middlegame (caveat: this is an "
        f"extrapolation from a regression-adjusted estimate, not a controlled experiment)."
        if cf["reliable"] else
        "The increment counterfactual couldn't be reliably estimated from this player's data."
    )
    return f"""You are a chess coach writing a short, direct, encouraging improvement plan for
an intermediate club player based on real statistics from their own recent games (chess.com
username: {USERNAME}). Their overall blunder rate (moves losing >=20 win-probability points) on
these games is {overall_rate*100:.1f}%.

Their top rating leaks, ranked by estimated cost (most costly first):
{leak_lines}

Causal note on time management:
{cf_line}

Write 3-4 short paragraphs: (1) a one-line honest summary of where they stand, (2) the 2-3
leaks worth fixing first and concretely how (specific enough to act on this week, not generic
chess advice), (3) one thing they're doing fine that doesn't need more attention (to keep this
useful rather than just a list of flaws), (4) a closing note on the time-management finding.
Address the player directly ("you"). No headers, no bullet points, plain prose."""


def _call_groq(prompt: str, api_key: str) -> str:
    import groq
    client = groq.Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_coaching_narrative(leaks: pd.DataFrame, cf: dict, overall_rate: float) -> str | None:
    """Groq first (free tier, no credit card needed -- console.groq.com) since that's this
    project's default; ANTHROPIC_API_KEY works too if you'd rather use that (paid) instead."""
    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if groq_key:
        narrative = _call_groq(build_llm_prompt(leaks, cf, overall_rate), groq_key)
    elif anthropic_key:
        narrative = _call_anthropic(build_llm_prompt(leaks, cf, overall_rate), anthropic_key)
    else:
        print("\nNeither GROQ_API_KEY nor ANTHROPIC_API_KEY is set -- skipping the LLM coaching "
              "narrative (numeric leak table/figure are unaffected). Get a free key at "
              "console.groq.com, set GROQ_API_KEY, and rerun report.py to generate "
              "outputs/coaching_narrative.md.")
        return None

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "coaching_narrative.md"
    out_path.write_text(narrative, encoding="utf-8")
    print(f"wrote {out_path}")
    return narrative


def generate_report() -> None:
    df = load_test_set()
    overall_rate = df["blunder"].mean()
    print(f"{USERNAME}: {len(df):,} moves in test set (2024-08 onward), "
          f"overall blunder rate {overall_rate*100:.2f}%")

    model = load_model()
    artifacts = load_causal_artifacts()

    leaks = find_leaks(df)
    leaks = add_leak_recommendations(leaks, df)
    leaks.to_csv(TABLES_DIR / "rating_leaks.csv", index=False)
    print(f"\nwrote {TABLES_DIR / 'rating_leaks.csv'}")
    print("\ntop rating leaks:")
    print(leaks.to_string(index=False))

    calib_df = calibration_check(model, df)
    cf = counterfactual(df, artifacts)

    out_path = FIGURES_DIR / f"player_report_{USERNAME}.png"
    plot_report(leaks, calib_df, cf, out_path)
    print(f"\nsaved {out_path}")

    print("\ncalibration (predicted vs actual, by predicted-probability bin):")
    print(calib_df)
    print("\ncounterfactual:")
    for k, v in cf.items():
        print(f"  {k}: {v}")

    generate_coaching_narrative(leaks, cf, overall_rate)


def main() -> None:
    generate_report()


if __name__ == "__main__":
    main()
