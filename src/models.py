"""
Set 4 -- the model ladder: base rate, Elo-only, Elo+position, full logistic (standardised, L2),
LightGBM. All models are trained on the natural class distribution -- no SMOTE, no under/over-
sampling. Blunders are genuinely rare, and resampling would distort the predicted probabilities
away from true blunder frequency, which defeats the point of a *calibrated* output (see README).

Rolling-history features (wp_volatility_3, wp_swing_last, time_spent_prev) are null for a mover's
first tracked move in a game -- filled with 0 for the logistic models (no history = no signal),
left as native nulls for LightGBM, which splits on missingness directly.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ELO_ONLY_FEATURES = ["mover_elo"]
ELO_POSITION_FEATURES = ["mover_elo", "wp_before", "legal_move_count", "material_total"]

LGB_PARAM_GRID = [
    {"num_leaves": 15, "min_child_samples": 100, "learning_rate": 0.1},
    {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.05},
    {"num_leaves": 63, "min_child_samples": 100, "learning_rate": 0.05},
]
LGB_DEFAULT_PARAMS = LGB_PARAM_GRID[1]
LGB_NUM_BOOST_ROUND = 1000
LGB_EARLY_STOPPING_ROUNDS = 30


class BaseRateModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRateModel":
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


class LogisticModel:
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.scaler = StandardScaler()

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columns].fillna(0).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        Xt = self.scaler.fit_transform(self._matrix(X))
        self.clf = LogisticRegression(C=1.0, max_iter=2000)  # penalty="l2" is the sklearn default
        self.clf.fit(Xt, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.scaler.transform(self._matrix(X))
        return self.clf.predict_proba(Xt)[:, 1]


def _pr_auc_feval(preds: np.ndarray, dataset: lgb.Dataset):
    return "pr_auc", average_precision_score(dataset.get_label(), preds), True


class LightGBMModel:
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X_train, y_train, X_val, y_val, search: bool = True) -> "LightGBMModel":
        param_grid = LGB_PARAM_GRID if search else [LGB_DEFAULT_PARAMS]
        train_set = lgb.Dataset(X_train[self.columns], label=y_train)
        val_set = lgb.Dataset(X_val[self.columns], label=y_val, reference=train_set)

        best_score, best_booster, best_params = -1.0, None, None
        for params in param_grid:
            booster = lgb.train(
                {**params, "objective": "binary", "metric": "None", "verbosity": -1,
                 "feature_pre_filter": False},
                train_set,
                num_boost_round=LGB_NUM_BOOST_ROUND,
                valid_sets=[val_set],
                feval=_pr_auc_feval,
                callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=False,
                                               first_metric_only=True)],
            )
            score = booster.best_score["valid_0"]["pr_auc"]
            if score > best_score:
                best_score, best_booster, best_params = score, booster, params

        print(f"  LightGBM: best params {best_params}, val PR-AUC={best_score:.4f}, "
              f"best_iteration={best_booster.best_iteration}")
        self.booster = best_booster
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.columns], num_iteration=self.booster.best_iteration)


MODEL_SPECS = [
    ("base_rate", lambda: BaseRateModel()),
    ("elo_only", lambda: LogisticModel(ELO_ONLY_FEATURES)),
    ("elo_position", lambda: LogisticModel(ELO_POSITION_FEATURES)),
    ("full_logistic", lambda: LogisticModel(FEATURE_COLUMNS)),
    ("lightgbm", lambda: LightGBMModel(FEATURE_COLUMNS)),
]


def fit_ladder(train_df: pd.DataFrame, val_df: pd.DataFrame, label_col: str = "blunder",
               search_lgb: bool = True) -> dict:
    y_train = train_df[label_col].astype(int)
    y_val = val_df[label_col].astype(int)

    models = {}
    print("fitting model ladder:")
    for name, ctor in MODEL_SPECS:
        model = ctor()
        if isinstance(model, LightGBMModel):
            model.fit(train_df, y_train, val_df, y_val, search=search_lgb)
        else:
            model.fit(train_df, y_train)
        models[name] = model
        print(f"  fit {name}")
    return models


def save_models(models: dict, suffix: str = "") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = PROCESSED_DIR / f"model_{name}{suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  saved {path}")
