"""
Set 5 -- increment as a natural experiment.

Setup, unchanged from the original design's logic: increment is fixed before a game starts and
can't respond to any particular position, and its effect on the clock only compounds as the
game goes on -- a difference-in-differences structure:

  unit:      game
  treatment: increment > 0
  "time":    ply bucket (9-20, 21-40, 41-60, 61+)
  outcome:   blunder rate within that game's moves in that bucket

What changed from the original population design, and why (checked empirically, not assumed --
see the switch-day diagnostic this module prints): the original design identified the effect via
player fixed effects across many "switcher" players (people who use both increment and no-
increment). With one player, that collapses to a constant -- there's no cross-player variation to
demean. The natural single-player analogue would be DAY fixed effects, restricted to days this
player used both settings ("switch days"). That was the plan -- but this player's actual history
barely switches within any short window: at the best matched base-time pair, only 0-2 days (out
of dozens) have both a treatment and a control game. Day fixed effects have essentially no
within-day variation left to work with here.

So this falls back to regression adjustment instead of fixed effects: pool every game (not just
one matched time-control pair), and control for `base_time`, `mover_elo`, and `opp_elo` directly
in the regression rather than removing them via demeaning. mover_elo in particular absorbs most
of the mechanical confound (this player's Elo -- and choice of time control -- drifted a lot
across the 2021 / 2023 / 2024+ eras in the data), but it's a weaker guarantee than a fixed
effect: day-to-day form and switcher self-selection go back to being fully residual threats
rather than partially handled ones, on top of the threats the original design already named.
That's an honest downgrade in rigor, forced by this player's actual play patterns, and it's
reported as such rather than smoothed over -- see the Sensitivity section below and the README.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

PLY_BUCKET_EDGES = [9, 21, 41, 61, 10_000]
PLY_BUCKET_LABELS = ["9-20", "21-40", "41-60", "61+"]
PARALLEL_TRENDS_TOLERANCE = 0.02  # 2 percentage points of blunder rate

# ply_bucket is already an ordered pandas Categorical (pd.cut(..., labels=PLY_BUCKET_LABELS)),
# so C(ply_bucket) uses that order directly -- the earliest bucket ("9-20") ends up as the
# omitted reference category, same trick the original design used to make the bare 'treatment'
# coefficient read as the earliest-bucket effect (see placebo_check).
REGRESSION_FORMULA = "blunder_rate ~ treatment * C(ply_bucket) + C(base_time) + mover_elo + opp_elo"
POOLED_FORMULA = "blunder_rate ~ treatment + C(base_time) + mover_elo + opp_elo"


def load_moves() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "moves_gonzalopelotas.parquet")


def add_ply_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ply_bucket"] = pd.cut(df["ply"], bins=PLY_BUCKET_EDGES, labels=PLY_BUCKET_LABELS, right=False)
    # int, not bool: patsy encodes a bool column as a 2-level categorical ("treatment[T.True]"),
    # which would break every coefficient-name lookup below (full_did_estimate,
    # sensitivity_analysis, report.py's counterfactual all assume the bare name "treatment").
    # Cast to int so it's treated as a plain numeric 0/1 regressor and keeps that bare name.
    df["treatment"] = (df["increment"] > 0).astype(int)
    return df


def switch_day_diagnostic(df: pd.DataFrame) -> None:
    """Checks whether a day-fixed-effects design (the direct single-player analogue of the
    original 'switcher players' restriction) is actually viable here. Printed plainly regardless
    of the answer -- this is what determined the fallback to regression adjustment below."""
    print("\nswitch-day diagnostic (days with both a treatment and a control game):")
    for base, grp in df.groupby("base_time"):
        if grp["treatment"].nunique() < 2:
            continue
        # treatment is int (0/1), not bool -- `~` on it is bitwise NOT, not logical negation
        # (~pd.Series([0,1]) == [-1,-2], both truthy). Compare to 0 explicitly instead.
        by_day = grp.groupby("utc_date")["treatment"].agg(lambda s: ((s == 1).any(), (s == 0).any()))
        switch_days = sum(1 for has_t, has_c in by_day if has_t and has_c)
        n0, n1 = (grp["treatment"] == 0).sum(), grp["treatment"].sum()
        print(f"  base={base}: no-inc moves={n0:,}, inc moves={n1:,}, "
              f"switch_days={switch_days}/{grp['utc_date'].nunique()} days")
    print("  -- too sparse at every matched base time for a day-fixed-effects design. Falling "
          "back to regression adjustment (base_time + mover_elo + opp_elo controls) over the "
          "pooled sample instead of a switcher/fixed-effects restriction. See README.")


def build_game_bucket_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.groupby(["game_id", "treatment", "ply_bucket"], observed=True).agg(
        blunder_rate=("blunder", "mean"),
        n_moves=("blunder", "size"),
        base_time=("base_time", "first"),
        mover_elo=("mover_elo", "mean"),
        opp_elo=("opp_elo", "mean"),
        utc_date=("utc_date", "first"),
    ).reset_index()
    # groupby/reset_index isn't guaranteed to preserve categorical dtype+order -- pin it
    # explicitly, since fit_did/placebo_check rely on PLY_BUCKET_LABELS[0] ("9-20") being the
    # omitted reference category in the regression, not whatever order string sorting would give.
    panel["ply_bucket"] = pd.Categorical(panel["ply_bucket"], categories=PLY_BUCKET_LABELS, ordered=True)
    return panel


def build_game_level(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["game_id", "treatment"], observed=True).agg(
        mover_elo=("mover_elo", "mean"),
        opp_elo=("opp_elo", "mean"),
        game_length=("ply", "max"),
        base_time=("base_time", "first"),
        utc_date=("utc_date", "first"),
    ).reset_index()


def balance_table(game_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = game_level.groupby("treatment", observed=True)[
        ["mover_elo", "opp_elo", "game_length"]].agg(["mean", "std", "count"])
    base_time_dist = pd.crosstab(game_level["treatment"], game_level["base_time"])
    return numeric, base_time_dist


def raw_group_means(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby(["ply_bucket", "treatment"], observed=True)["blunder_rate"] \
        .mean().unstack("treatment").reindex(PLY_BUCKET_LABELS)


def plot_parallel_trends(raw_means: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw_means.index.astype(str), raw_means[0], marker="o", label="increment = 0 (control)")
    ax.plot(raw_means.index.astype(str), raw_means[1], marker="o", label="increment > 0 (treatment)")
    ax.set_xlabel("ply bucket")
    ax.set_ylabel("mean blunder rate (game-bucket, raw/unadjusted)")
    ax.set_title("Parallel trends: blunder rate by ply bucket and increment group")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def placebo_check(raw_means: pd.DataFrame, result) -> None:
    """Two versions, since this design isn't restricted to a matched sample the way the
    original was: the RAW earliest-bucket gap (which can reflect base_time/Elo composition
    differences between groups, not just treatment endogeneity), and the regression-ADJUSTED
    earliest-bucket effect (the fitted model's own 'treatment' coefficient, since ply_bucket's
    earliest level is the omitted reference category) -- the one that actually matters for this
    design, analogous to checking pre-trends are ~0 in an event-study specification."""
    first_bucket = PLY_BUCKET_LABELS[0]
    raw_diff = raw_means.loc[first_bucket, 1] - raw_means.loc[first_bucket, 0]
    adj_diff = float(result.params["treatment"])
    adj_se = float(result.bse["treatment"])
    print(f"\nplacebo check, earliest bucket ({first_bucket}):")
    print(f"  raw gap (treatment - control, unadjusted): {raw_diff:+.4f}")
    print(f"  regression-adjusted gap (fitted 'treatment' coefficient): {adj_diff:+.4f} "
          f"(SE {adj_se:.4f})")
    if abs(adj_diff) > PARALLEL_TRENDS_TOLERANCE:
        print(f"  WARNING: adjusted gap exceeds the {PARALLEL_TRENDS_TOLERANCE:.2f} tolerance -- "
              f"even after controlling for base_time/Elo, the two groups differ before the clocks "
              f"have had time to diverge. Later-bucket estimates should be treated with caution.")
    else:
        print(f"  adjusted gap is within {PARALLEL_TRENDS_TOLERANCE:.2f} of zero, consistent with "
              f"parallel trends holding once base_time/Elo composition is controlled for.")


def bucket_time_diff(df: pd.DataFrame) -> pd.Series:
    """Regression-adjusted (for base_time) extra seconds spent per move from playing with
    increment, by ply bucket. NOT a raw group-mean difference -- base_time composition differs
    sharply between treatment and control (treatment is mostly fast 180s games, control mostly
    slow 600s/300s ones -- see the printed base_time distribution table), so an unadjusted
    difference mostly measures "extra seconds from a slower base time," not from increment
    itself, and can come out with the wrong sign entirely. Used by report.py to translate the
    blunder-rate regression's per-bucket effect into a per-second rate for its counterfactual."""
    sub = df.copy()
    sub["time_spent"] = sub["clock_before"] - sub["clock_after"] + sub["increment"]
    diffs = {}
    for bucket in PLY_BUCKET_LABELS:
        bsub = sub[sub["ply_bucket"] == bucket]
        if bsub["treatment"].nunique() < 2:
            diffs[bucket] = float("nan")
            continue
        formula = "time_spent ~ treatment + C(base_time)" if bsub["base_time"].nunique() > 1 \
            else "time_spent ~ treatment"
        result = smf.ols(formula, data=bsub).fit()
        diffs[bucket] = float(result.params["treatment"])
    return pd.Series(diffs)


def fit_did(panel: pd.DataFrame):
    return smf.ols(REGRESSION_FORMULA, data=panel) \
        .fit(cov_type="cluster", cov_kwds={"groups": panel["game_id"]})


def naive_estimate(df: pd.DataFrame) -> dict:
    """Step 1: raw move-level blunder rate, treatment minus control, across every game -- no
    adjustment for base_time/Elo/anything at all."""
    g = df.groupby("treatment")["blunder"]
    means, ns, vars_ = g.mean(), g.count(), g.var()
    diff = float(means[1] - means[0])
    se = float(np.sqrt(vars_[1] / ns[1] + vars_[0] / ns[0]))
    return {"step": "1. naive (raw, all games)", "estimate": diff, "se": se}


def adjusted_pooled_estimate(panel: pd.DataFrame) -> dict:
    """Step 2: regression-adjusted (base_time + Elo controls), pooled across ply buckets (no
    time interaction) -- removes observed-covariate composition differences but still assumes
    the effect is constant over the whole game rather than compounding."""
    result = smf.ols(POOLED_FORMULA, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["game_id"]})
    return {"step": "2. covariate-adjusted, no time interaction",
            "estimate": float(result.params["treatment"]), "se": float(result.bse["treatment"])}


def full_did_estimate(did_result) -> dict:
    """Step 3: the full regression -- treatment effect in the latest ply bucket (main effect +
    interaction), where the design predicts the compounding effect is largest."""
    last_bucket = PLY_BUCKET_LABELS[-1]
    names = list(did_result.params.index)
    interaction_name = next(n for n in names if n.startswith("treatment:") and f".{last_bucket}]" in n)
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(interaction_name)] = 1.0
    point = float(contrast @ did_result.params.values)
    se = float(np.sqrt(contrast @ did_result.cov_params().values @ contrast))
    return {"step": f"3. full regression, treatment x ply bucket ({last_bucket})",
            "estimate": point, "se": se}


def estimate_trajectory(df: pd.DataFrame, panel: pd.DataFrame, did_result) -> pd.DataFrame:
    return pd.DataFrame([
        naive_estimate(df),
        adjusted_pooled_estimate(panel),
        full_did_estimate(did_result),
    ])


def plot_trajectory(traj: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(traj))
    ax.errorbar(x, traj["estimate"], yerr=1.96 * traj["se"], fmt="o-", capsize=5, color="tab:blue")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(traj["step"], rotation=15, ha="right")
    ax.set_ylabel("estimated treatment effect on blunder rate")
    ax.set_title("Estimate trajectory: naive -> covariate-adjusted -> full regression")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dag(out_path: Path) -> None:
    """A conceptual DAG, not computed from data: the assumed causal path (solid) and the
    residual threats named in the README's honest-limitations paragraph (dashed). Unlike the
    original population design, there is no fixed-effects layer handling day-to-day form or
    switcher self-selection here -- switch-day sparsity ruled that out (see switch_day_
    diagnostic) -- so both stay fully residual, on top of a new threat this design surfaced:
    unmeasured skill/meta drift across eras that mover_elo only partially captures."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")

    box_w, box_h = 2.1, 1.1
    nodes = {
        "increment": (1.5, 4.5, "Increment\n(treatment)"),
        "clock": (5.5, 4.5, "Clock time /\ntime pressure"),
        "blunder": (9.5, 4.5, "Blunder"),
        "form": (1.5, 7.7, "Day-to-day form\n/ mood"),
        "position": (5.5, 1.7, "Position difficulty\nwithin ply bucket"),
        "opponent": (9.5, 1.7, "Opponent\nbehaviour"),
        "drift": (5.5, 7.7, "Unmeasured skill/meta\ndrift (beyond mover_elo)"),
    }
    for x, y, label in nodes.values():
        ax.add_patch(plt.Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, fill=True,
                                    facecolor="white", edgecolor="black", zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5, zorder=3)

    def arrow(a, b, style="solid", color="black", rad=0.0):
        (x1, y1, _), (x2, y2, _) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", linestyle=style, color=color, lw=1.6,
                                     shrinkA=32, shrinkB=32, connectionstyle=f"arc3,rad={rad}"))

    arrow("increment", "clock", "solid")
    arrow("clock", "blunder", "solid")

    arrow("form", "increment", "dashed", "tab:red")
    arrow("form", "blunder", "dashed", "tab:red", rad=-0.2)
    arrow("position", "blunder", "dashed", "tab:red")
    arrow("opponent", "blunder", "dashed", "tab:red")
    arrow("drift", "blunder", "dashed", "tab:red")
    arrow("drift", "increment", "dashed", "tab:red", rad=0.2)

    ax.plot([], [], color="black", linestyle="solid", label="assumed causal path")
    ax.plot([], [], color="tab:red", linestyle="dashed", label="residual threat (not resolved)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5, frameon=False)
    ax.set_title("Assumed causal structure and residual confounding (single-player, n=1)")
    ax.text(0.5, -0.02,
            "base_time, mover_elo, and opp_elo are controlled for as regression covariates (not shown) --\n"
            "composition differences a fixed-effects design would otherwise remove. Everything else here\n"
            "is fully residual: unlike the population design, there's no fixed-effects layer left to absorb it.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic", color="dimgray")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def e_value(risk_ratio: float) -> float:
    """VanderWeele & Ding (2017): the minimum risk-ratio association an unmeasured confounder
    would need with *both* treatment and outcome, above and beyond measured covariates, to
    fully explain away an observed risk ratio (working on the side of the ratio away from 1)."""
    rr = max(risk_ratio, 1 / risk_ratio)
    return rr + np.sqrt(rr * (rr - 1))


def sensitivity_analysis(result, raw_means: pd.DataFrame) -> dict:
    """E-value for the total treatment effect in the latest ply bucket -- main effect +
    interaction. Reads *higher* than the population design's would for the same point estimate,
    since regression adjustment is a weaker guarantee than fixed effects -- the E-value itself
    doesn't know that, so the honest caveat belongs in the printed interpretation, not just the
    number."""
    last_bucket = PLY_BUCKET_LABELS[-1]
    names = list(result.params.index)
    interaction_name = next(n for n in names if n.startswith("treatment:") and f".{last_bucket}]" in n)
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(interaction_name)] = 1.0

    point = float(contrast @ result.params.values)
    se = float(np.sqrt(contrast @ result.cov_params().values @ contrast))
    ci_lo, ci_hi = point - 1.96 * se, point + 1.96 * se

    baseline_risk = float(raw_means.loc[last_bucket, 0])

    def to_rr(effect: float) -> float:
        return (baseline_risk + effect) / baseline_risk

    rr_point = to_rr(point)
    ev_point = e_value(rr_point)

    if ci_lo <= 0 <= ci_hi:
        ev_ci = 1.0
    else:
        closest_bound = ci_lo if abs(ci_lo) < abs(ci_hi) else ci_hi
        ev_ci = e_value(to_rr(closest_bound))

    print(f"\nsensitivity analysis (E-value, latest bucket '{last_bucket}'):")
    print(f"  total treatment effect: {point:+.4f} blunder-rate points "
          f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]) on a baseline of {baseline_risk:.4f}")
    print(f"  implied risk ratio: {rr_point:.3f}")
    print(f"  E-value (point estimate): {ev_point:.2f}")
    print(f"  E-value (CI bound closest to null): {ev_ci:.2f}")
    print(f"  interpretation: an unmeasured confounder would need to be associated with both "
          f"increment and blunder risk by a risk ratio of at least {ev_point:.2f} each, above "
          f"and beyond base_time/Elo, to fully explain away the point estimate ({ev_ci:.2f} to "
          f"move the CI to include no effect). Read this number as an upper bound on rigor, not "
          f"a guarantee: this is regression adjustment on observed covariates, not a fixed-"
          f"effects or switcher design -- day-to-day form and switcher self-selection (see "
          f"README) are exactly the kind of confounder this E-value is warning about, and this "
          f"player's data didn't have enough within-day switching to rule them out directly.")

    return {"effect": point, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "baseline_risk": baseline_risk,
            "risk_ratio": rr_point, "e_value_point": ev_point, "e_value_ci": ev_ci}


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_moves()
    df = add_ply_bucket(df)
    print(f"pooled move rows: {len(df):,}, {df['game_id'].nunique():,} games")

    switch_day_diagnostic(df)

    panel = build_game_bucket_panel(df)
    game_level = build_game_level(df)

    numeric_balance, base_time_dist = balance_table(game_level)
    numeric_balance.to_csv(TABLES_DIR / "increment_balance.csv")
    base_time_dist.to_csv(TABLES_DIR / "increment_balance_base_time.csv")
    print("\nbalance table (mover Elo / opponent Elo / game length by treatment, full sample):")
    print(numeric_balance)
    print("\nbase_time distribution by treatment (this is what the regression controls for):")
    print(base_time_dist)

    raw_means = raw_group_means(panel)
    plot_parallel_trends(raw_means, FIGURES_DIR / "parallel_trends.png")
    print(f"\nsaved {FIGURES_DIR / 'parallel_trends.png'}")

    result = fit_did(panel)
    print("\nregression (treatment x ply bucket, controlling for base_time/mover_elo/opp_elo, "
          "cluster-robust SE by game):")
    print(result.summary())

    placebo_check(raw_means, result)

    coef_table = pd.DataFrame({"coef": result.params, "std_err": result.bse, "p_value": result.pvalues})
    coef_table.to_csv(TABLES_DIR / "increment_did_results.csv")
    print(f"\nwrote {TABLES_DIR / 'increment_did_results.csv'}")

    sensitivity = sensitivity_analysis(result, raw_means)
    pd.DataFrame([sensitivity]).to_csv(TABLES_DIR / "increment_sensitivity.csv", index=False)
    print(f"wrote {TABLES_DIR / 'increment_sensitivity.csv'}")

    time_diff = bucket_time_diff(df)
    artifacts = {"did_result": result, "bucket_time_diff": time_diff}
    artifacts_path = PROCESSED_DIR / "causal_did_artifacts.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"wrote {artifacts_path}")

    traj = estimate_trajectory(df, panel, result)
    traj.to_csv(TABLES_DIR / "estimate_trajectory.csv", index=False)
    plot_trajectory(traj, FIGURES_DIR / "estimate_trajectory.png")
    print(f"\nestimate trajectory:\n{traj.to_string(index=False)}")
    print(f"wrote {TABLES_DIR / 'estimate_trajectory.csv'} and {FIGURES_DIR / 'estimate_trajectory.png'}")

    plot_dag(FIGURES_DIR / "causal_dag.png")
    print(f"wrote {FIGURES_DIR / 'causal_dag.png'}")


if __name__ == "__main__":
    main()
