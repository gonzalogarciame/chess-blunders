"""
Set 5 -- increment as a natural experiment.

Setup: within a matched pair (identical base time, different increment -- see ingest.py's
TC_PAIRS), increment is fixed before the game starts and cannot respond to any particular
position. Its effect on the clock only compounds as the game goes on, which gives a
difference-in-differences structure:

  unit:      player-game
  treatment: increment > 0
  "time":    ply bucket (9-20, 21-40, 41-60, 61+)
  outcome:   blunder rate within that player-game's moves in that bucket

Fixed effects are player, not player-game, so within one player the same game's treatment
status still varies across their games -- the treatment main effect is identified, not
absorbed. Restricted to "switchers" (players who appear in both increment groups across
either month) to remove the selection problem of who chooses which time control: every
comparison is then within-player.

Fixed effects are implemented by demeaning (the "within" estimator) rather than dummy
variables, since player-count here is too large for a dummy design matrix to be practical.
Caveat: plain OLS on demeaned data doesn't adjust residual degrees of freedom for the number
of player means absorbed, so standard errors are a close approximation, not textbook-exact --
acceptable here since cluster-robust SEs (by player) are the thing that matters most for
validity, and those are computed properly.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

MONTHS = ["2024-03", "2024-09"]

PLY_BUCKET_EDGES = [9, 21, 41, 61, 10_000]
PLY_BUCKET_LABELS = ["9-20", "21-40", "41-60", "61+"]
PARALLEL_TRENDS_TOLERANCE = 0.02  # 2 percentage points of blunder rate


def load_pooled_moves() -> pd.DataFrame:
    frames = [pd.read_parquet(PROCESSED_DIR / f"moves_{m}.parquet") for m in MONTHS]
    return pd.concat(frames, ignore_index=True)


def add_ply_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ply_bucket"] = pd.cut(df["ply"], bins=PLY_BUCKET_EDGES, labels=PLY_BUCKET_LABELS, right=False)
    df["treatment"] = df["increment"] > 0
    return df


def build_player_game_bucket_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["mover", "game_id", "treatment", "ply_bucket"], observed=True).agg(
        blunder_rate=("blunder", "mean"),
        n_moves=("blunder", "size"),
    ).reset_index()


def build_player_game_level(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["mover", "game_id", "treatment"], observed=True).agg(
        mover_elo=("mover_elo", "mean"),
        opp_elo=("opp_elo", "mean"),
        game_length=("ply", "max"),
    ).reset_index()


def restrict_to_switchers(panel: pd.DataFrame) -> pd.DataFrame:
    n_groups = panel.groupby("mover")["treatment"].nunique()
    switchers = n_groups[n_groups == 2].index
    return panel[panel["mover"].isin(switchers)].copy()


def balance_table(pg_switchers: pd.DataFrame) -> pd.DataFrame:
    return pg_switchers.groupby("treatment", observed=True)[["mover_elo", "opp_elo", "game_length"]] \
        .agg(["mean", "std", "count"])


def raw_group_means(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby(["ply_bucket", "treatment"], observed=True)["blunder_rate"] \
        .mean().unstack("treatment").reindex(PLY_BUCKET_LABELS)


def plot_parallel_trends(raw_means: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(raw_means.index.astype(str), raw_means[False], marker="o", label="increment = 0 (control)")
    ax.plot(raw_means.index.astype(str), raw_means[True], marker="o", label="increment > 0 (treatment)")
    ax.set_xlabel("ply bucket")
    ax.set_ylabel("mean blunder rate (player-game-bucket)")
    ax.set_title("Parallel trends: blunder rate by ply bucket and increment group")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def placebo_check(raw_means: pd.DataFrame) -> float:
    first_bucket = PLY_BUCKET_LABELS[0]
    diff = raw_means.loc[first_bucket, True] - raw_means.loc[first_bucket, False]
    print(f"\nplacebo / parallel-trends check: earliest-bucket ({first_bucket}) raw gap "
          f"(treatment - control) = {diff:+.4f}")
    if abs(diff) > PARALLEL_TRENDS_TOLERANCE:
        print(f"  WARNING: gap exceeds the {PARALLEL_TRENDS_TOLERANCE:.2f} tolerance -- parallel "
              f"trends may be violated, and the DiD design is compromised for this sample. "
              f"Later-bucket estimates should be treated with caution.")
    else:
        print(f"  gap is within {PARALLEL_TRENDS_TOLERANCE:.2f} of zero, consistent with parallel "
              f"trends holding before the clocks have had time to diverge.")
    return diff


def bucket_time_diff(df: pd.DataFrame, switcher_movers: pd.Series) -> pd.DataFrame:
    """Mean seconds spent per move, by treatment and ply bucket, restricted to switchers --
    the empirical "extra seconds increment buys you" in each bucket. Used by report.py to
    translate the DiD's per-bucket blunder-rate effect into a per-second rate for its
    counterfactual (see README)."""
    sub = df[df["mover"].isin(switcher_movers)].copy()
    sub["time_spent"] = sub["clock_before"] - sub["clock_after"] + sub["increment"]
    return sub.groupby(["ply_bucket", "treatment"], observed=True)["time_spent"] \
        .mean().unstack("treatment").reindex(PLY_BUCKET_LABELS)


def within_transform(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    return X - X.groupby(groups).transform("mean")


def fit_did(panel: pd.DataFrame):
    df = panel.copy()
    df["treatment"] = df["treatment"].astype(float)
    bucket_dummies = pd.get_dummies(df["ply_bucket"], prefix="bucket", drop_first=True).astype(float)
    interaction = bucket_dummies.mul(df["treatment"], axis=0)
    interaction.columns = [f"treat_x_{c}" for c in bucket_dummies.columns]

    X = pd.concat([df[["treatment"]], bucket_dummies, interaction], axis=1)
    y = df["blunder_rate"]

    X_within = within_transform(X, df["mover"])
    y_within = y - y.groupby(df["mover"]).transform("mean")

    model = sm.OLS(y_within, X_within)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df["mover"]})


def naive_estimate(df: pd.DataFrame) -> dict:
    """Step 1 of the trajectory: raw move-level blunder rate, treatment minus control, across
    every player (switchers and non-switchers alike) -- no adjustment for who selects into
    which time control at all."""
    g = df.groupby("treatment")["blunder"]
    means, ns, vars_ = g.mean(), g.count(), g.var()
    diff = float(means[True] - means[False])
    se = float(np.sqrt(vars_[True] / ns[True] + vars_[False] / ns[False]))
    return {"step": "1. naive (raw, all players)", "estimate": diff, "se": se}


def fe_pooled_estimate(switcher_panel: pd.DataFrame) -> dict:
    """Step 2: player fixed effects, switchers only, but pooled across ply buckets (no time
    interaction) -- removes between-player selection but still assumes the effect is constant
    over the whole game rather than compounding."""
    df = switcher_panel.copy()
    df["treatment"] = df["treatment"].astype(float)
    X = df[["treatment"]]
    y = df["blunder_rate"]
    X_within = within_transform(X, df["mover"])
    y_within = y - y.groupby(df["mover"]).transform("mean")
    result = sm.OLS(y_within, X_within).fit(cov_type="cluster", cov_kwds={"groups": df["mover"]})
    return {"step": "2. player FE, switchers, no time interaction",
            "estimate": float(result.params["treatment"]), "se": float(result.bse["treatment"])}


def full_did_estimate(did_result) -> dict:
    """Step 3: the full DiD -- treatment effect in the latest ply bucket (main effect +
    interaction), where the design predicts the compounding effect is largest."""
    last_bucket = PLY_BUCKET_LABELS[-1]
    names = list(did_result.params.index)
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(f"treat_x_bucket_{last_bucket}")] = 1.0
    point = float(contrast @ did_result.params.values)
    se = float(np.sqrt(contrast @ did_result.cov_params().values @ contrast))
    return {"step": f"3. full DiD, switchers, player FE x ply bucket ({last_bucket})",
            "estimate": point, "se": se}


def estimate_trajectory(df: pd.DataFrame, switcher_panel: pd.DataFrame, did_result) -> pd.DataFrame:
    return pd.DataFrame([
        naive_estimate(df),
        fe_pooled_estimate(switcher_panel),
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
    ax.set_title("Estimate trajectory: naive -> fixed effects -> full DiD")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dag(out_path: Path) -> None:
    """A conceptual DAG, not computed from data: the assumed causal path (solid) and the three
    residual threats named in the README's honest-limitations paragraph (dashed) that player
    fixed effects and the switcher restriction do not rule out. Stable, between-player skill
    differences are the one confound player FE does handle -- noted as a caption rather than a
    graph node, since drawing it would need long arrows crossing the whole diagram for a point
    that's the *absence* of a threat, not a residual one."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_w, box_h = 2.0, 1.1
    nodes = {
        "increment": (1.5, 4.0, "Increment\n(treatment)"),
        "clock": (5.5, 4.0, "Clock time /\ntime pressure"),
        "blunder": (9.5, 4.0, "Blunder"),
        "form": (1.5, 7.0, "Day-to-day form\n/ mood"),
        "position": (5.5, 1.7, "Position difficulty\nwithin ply bucket"),
        "opponent": (9.5, 1.7, "Opponent\nbehaviour"),
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

    # assumed causal path (the thing the DiD tries to estimate)
    arrow("increment", "clock", "solid")
    arrow("clock", "blunder", "solid")

    # residual threats named in the README -- each fans straight into the path, no crossing
    arrow("form", "increment", "dashed", "tab:red")           # self-selection into time control
    arrow("form", "blunder", "dashed", "tab:red", rad=-0.25)  # direct confound, arcs above "clock"
    arrow("position", "blunder", "dashed", "tab:red")         # unobserved within ply bucket
    arrow("opponent", "blunder", "dashed", "tab:red")

    ax.plot([], [], color="black", linestyle="solid", label="assumed causal path")
    ax.plot([], [], color="tab:red", linestyle="dashed", label="residual threat (not resolved)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9.5, frameon=False)
    ax.set_title("Assumed causal structure and residual confounding")
    ax.text(0.5, -0.03,
            "Stable, between-player skill differences are the one confound player fixed\n"
            "effects do control for -- not shown, since it's handled rather than residual.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic",
            color="dimgray")

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
    interaction, the bucket where the design predicts the compounding effect is largest.
    Converts the linear-probability effect to an approximate risk ratio using the control
    group's raw blunder rate in that bucket as the reference risk, then reports the E-value
    for both the point estimate and the CI bound closest to the null."""
    last_bucket = PLY_BUCKET_LABELS[-1]
    interaction_name = f"treat_x_bucket_{last_bucket}"

    names = list(result.params.index)
    contrast = np.zeros(len(names))
    contrast[names.index("treatment")] = 1.0
    contrast[names.index(interaction_name)] = 1.0

    point = float(contrast @ result.params.values)
    se = float(np.sqrt(contrast @ result.cov_params().values @ contrast))
    ci_lo, ci_hi = point - 1.96 * se, point + 1.96 * se

    baseline_risk = float(raw_means.loc[last_bucket, False])

    def to_rr(effect: float) -> float:
        return (baseline_risk + effect) / baseline_risk

    rr_point = to_rr(point)
    ev_point = e_value(rr_point)

    if ci_lo <= 0 <= ci_hi:
        ev_ci = 1.0  # CI already includes the null; no confounding needed to reach it
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
          f"and beyond player fixed effects, to fully explain away the point estimate "
          f"({ev_ci:.2f} to move the CI to include no effect).")

    return {"effect": point, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "baseline_risk": baseline_risk,
            "risk_ratio": rr_point, "e_value_point": ev_point, "e_value_ci": ev_ci}


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_pooled_moves()
    df = add_ply_bucket(df)

    panel = build_player_game_bucket_panel(df)
    switcher_panel = restrict_to_switchers(panel)
    print(f"player-game-bucket panel: {len(panel):,} rows, {panel['mover'].nunique():,} players")
    print(f"switcher subsample: {len(switcher_panel):,} rows, "
          f"{switcher_panel['mover'].nunique():,} players")

    pg = build_player_game_level(df)
    switcher_pg = pg[pg["mover"].isin(switcher_panel["mover"].unique())]
    balance = balance_table(switcher_pg)
    balance.to_csv(TABLES_DIR / "increment_balance.csv")
    print("\nbalance table (switcher sample, Elo / opponent Elo / game length by treatment):")
    print(balance)

    raw_means = raw_group_means(switcher_panel)
    plot_parallel_trends(raw_means, FIGURES_DIR / "parallel_trends.png")
    print(f"\nsaved {FIGURES_DIR / 'parallel_trends.png'}")
    placebo_check(raw_means)

    result = fit_did(switcher_panel)
    print("\nDiD regression (player fixed effects via demeaning, cluster-robust SE by player):")
    print(result.summary())

    coef_table = pd.DataFrame({"coef": result.params, "std_err": result.bse, "p_value": result.pvalues})
    coef_table.to_csv(TABLES_DIR / "increment_did_results.csv")
    print(f"\nwrote {TABLES_DIR / 'increment_did_results.csv'}")

    sensitivity = sensitivity_analysis(result, raw_means)
    pd.DataFrame([sensitivity]).to_csv(TABLES_DIR / "increment_sensitivity.csv", index=False)
    print(f"wrote {TABLES_DIR / 'increment_sensitivity.csv'}")

    time_diff = bucket_time_diff(df, switcher_panel["mover"])
    artifacts = {"did_result": result, "bucket_time_diff": time_diff}
    artifacts_path = PROCESSED_DIR / "causal_did_artifacts.pkl"
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"wrote {artifacts_path}")

    traj = estimate_trajectory(df, switcher_panel, result)
    traj.to_csv(TABLES_DIR / "estimate_trajectory.csv", index=False)
    plot_trajectory(traj, FIGURES_DIR / "estimate_trajectory.png")
    print(f"\nestimate trajectory:\n{traj.to_string(index=False)}")
    print(f"wrote {TABLES_DIR / 'estimate_trajectory.csv'} and {FIGURES_DIR / 'estimate_trajectory.png'}")

    plot_dag(FIGURES_DIR / "causal_dag.png")
    print(f"wrote {FIGURES_DIR / 'causal_dag.png'}")


if __name__ == "__main__":
    main()
