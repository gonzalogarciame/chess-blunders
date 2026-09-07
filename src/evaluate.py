"""
Set 4 -- metrics, calibration, and reporting for the model ladder fit in models.py.

Reports PR-AUC (primary, always printed alongside the base rate it's relative to), ROC-AUC,
Brier score, log loss, and top-decile lift, on validation and test. No test_unseen_players
split anymore -- "generalises to unseen players" isn't a meaningful question for one player's
own games. Also reruns the whole ladder at blunder thresholds of 10/15/20/30 win-percentage
points (see README for the naming-a-flip-if-one-happens rule).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_recall_curve, roc_auc_score,
)

from causal import PLY_BUCKET_EDGES, PLY_BUCKET_LABELS
from models import fit_ladder, save_models

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

THRESHOLDS = [10, 15, 20, 30]
# Wider/lower than the original population design's 800-2600 -- this player's own rating (across
# chess.com's per-time-class scale, from their 2021 games through now) starts well below 800.
ELO_BAND_EDGES = list(range(0, 2001, 200))


def lift_top_decile(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n = len(y_true)
    top_n = max(1, n // 10)
    order = np.argsort(-y_score)
    top_rate = y_true[order[:top_n]].mean()
    base_rate = y_true.mean()
    return top_rate / base_rate if base_rate > 0 else np.nan


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, train_base_rate: float) -> dict:
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "base_rate": train_base_rate,
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "log_loss": log_loss(y_true, y_score, labels=[0, 1]),
        "lift_top_decile": lift_top_decile(y_true, y_score),
    }


def evaluate_ladder(models: dict, splits: dict, label_col: str, train_base_rate: float) -> pd.DataFrame:
    rows = []
    for split_name, df in splits.items():
        y_true = df[label_col].astype(int).to_numpy()
        for name, model in models.items():
            y_score = model.predict_proba(df)
            rows.append({"split": split_name, "model": name,
                         **compute_metrics(y_true, y_score, train_base_rate)})
    return pd.DataFrame(rows)


def slice_heatmap(model, test_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Calibration gap (mean predicted - actual blunder rate) for the given model, sliced by
    Elo band x ply bucket -- the two dimensions most likely to expose where the model is
    over- or under-confident, since both interact with how much signal is actually available
    (weak players in the opening vs. strong players deep in an endgame, etc.)."""
    df = test_df.copy()
    df["elo_band"] = pd.cut(df["mover_elo"], bins=ELO_BAND_EDGES, right=False)
    df["ply_bucket"] = pd.cut(df["ply"], bins=PLY_BUCKET_EDGES, labels=PLY_BUCKET_LABELS, right=False)
    df["predicted"] = model.predict_proba(df)
    df["actual"] = df[label_col].astype(int)
    df["gap"] = df["predicted"] - df["actual"]
    return df.groupby(["elo_band", "ply_bucket"], observed=True)["gap"].mean().unstack("ply_bucket") \
        .reindex(columns=PLY_BUCKET_LABELS)


def plot_slice_heatmap(heatmap: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = np.nanmax(np.abs(heatmap.to_numpy()))
    im = ax.imshow(heatmap.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns.astype(str))
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.astype(str))
    ax.set_xlabel("ply bucket")
    ax.set_ylabel("mover Elo band")
    ax.set_title("Calibration gap (predicted - actual blunder rate), LightGBM, test")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="predicted - actual")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(models: dict, test_df: pd.DataFrame, label_col: str, out_path: Path) -> None:
    y_true = test_df[label_col].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        y_score = model.predict_proba(test_df)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.axhline(y_true.mean(), linestyle="--", color="gray", label=f"base rate ({y_true.mean():.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def calibrate_and_plot(models: dict, val_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str,
                        out_path: Path) -> tuple[IsotonicRegression, float, float]:
    model = models["lightgbm"]
    val_raw = model.predict_proba(val_df)
    test_raw = model.predict_proba(test_df)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_raw, val_df[label_col].astype(int))
    test_cal = iso.predict(test_raw)

    y_test = test_df[label_col].astype(int).to_numpy()
    brier_before = brier_score_loss(y_test, test_raw)
    brier_after = brier_score_loss(y_test, test_cal)
    print(f"calibration (LightGBM, test): Brier before={brier_before:.5f}, after={brier_after:.5f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, preds in [("raw", test_raw), ("isotonic-calibrated", test_cal)]:
        frac_pos, mean_pred = calibration_curve(y_test, preds, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed blunder rate")
    ax.set_title("Reliability curve (LightGBM, test, 10 quantile bins)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

    return iso, brier_before, brier_after


def threshold_sensitivity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Rerun the whole ladder at each threshold. LightGBM uses fixed default params here (no
    grid search) -- the point is threshold sensitivity, not re-tuning at every threshold."""
    rows = []
    for threshold in THRESHOLDS:
        print(f"\n--- threshold sensitivity: wp_loss >= {threshold} ---")
        train_t = train_df.assign(blunder_t=train_df["wp_loss"] >= threshold)
        val_t = val_df.assign(blunder_t=val_df["wp_loss"] >= threshold)
        test_t = test_df.assign(blunder_t=test_df["wp_loss"] >= threshold)

        train_base_rate = train_t["blunder_t"].mean()
        models = fit_ladder(train_t, val_t, label_col="blunder_t", search_lgb=False)

        y_test = test_t["blunder_t"].astype(int).to_numpy()
        for name, model in models.items():
            y_score = model.predict_proba(test_t)
            rows.append({
                "threshold": threshold,
                "model": name,
                "pr_auc": average_precision_score(y_test, y_score),
                "base_rate": train_base_rate,
            })
    return pd.DataFrame(rows)


def load_splits() -> dict[str, pd.DataFrame]:
    names = ["train", "val", "test"]
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in names}


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]
    train_base_rate = train_df["blunder"].mean()

    models = fit_ladder(train_df, val_df, label_col="blunder", search_lgb=True)
    save_models(models)

    eval_splits = {"val": val_df, "test": splits["test"]}
    comparison = evaluate_ladder(models, eval_splits, "blunder", train_base_rate)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print(f"\nwrote {TABLES_DIR / 'model_comparison.csv'}")
    print(comparison.to_string(index=False))

    base_pr = comparison.query("split == 'test' and model == 'base_rate'")["pr_auc"].iloc[0]
    elo_pr = comparison.query("split == 'test' and model == 'elo_only'")["pr_auc"].iloc[0]
    verdict = "better than" if elo_pr > base_pr else "NOT better than (investigate!)"
    print(f"\nElo-only test PR-AUC {elo_pr:.4f} is {verdict} base rate {base_pr:.4f}")

    lgb_pr = comparison.query("split == 'test' and model == 'lightgbm'")["pr_auc"].iloc[0]
    logit_pr = comparison.query("split == 'test' and model == 'full_logistic'")["pr_auc"].iloc[0]
    if lgb_pr > logit_pr:
        print(f"LightGBM beats full logistic on test PR-AUC: {lgb_pr:.4f} vs {logit_pr:.4f}")
    else:
        print(f"LightGBM does NOT beat full logistic on test PR-AUC: {lgb_pr:.4f} vs {logit_pr:.4f} "
              f"-- reporting plainly rather than tuning further")

    plot_pr_curve(models, splits["test"], "blunder", FIGURES_DIR / "pr_curve.png")
    calibrate_and_plot(models, val_df, splits["test"], "blunder", FIGURES_DIR / "calibration.png")

    heatmap = slice_heatmap(models["lightgbm"], splits["test"], "blunder")
    heatmap.to_csv(TABLES_DIR / "slice_heatmap.csv")
    plot_slice_heatmap(heatmap, FIGURES_DIR / "slice_heatmap.png")
    print(f"\nwrote {TABLES_DIR / 'slice_heatmap.csv'} and {FIGURES_DIR / 'slice_heatmap.png'}")
    print(heatmap)

    sensitivity = threshold_sensitivity(train_df, val_df, splits["test"])
    sensitivity.to_csv(TABLES_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\nwrote {TABLES_DIR / 'threshold_sensitivity.csv'}")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
