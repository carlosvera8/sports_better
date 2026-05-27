"""
Poisson GLM for MLB total runs prediction.

Why Poisson: run scoring in baseball is a count process with low mean (~4.5 runs/team/game).
We model total runs as the sum of two independent Poisson processes (home + away scoring).

Model structure:
  log(E[runs]) = β0 + β1*offense_rate + β2*pitcher_quality_opponent
               + β3*park_factor + β4*wind_effect + β5*temperature_effect
               + β6*is_dome

We fit two separate models: home_runs ~ f(away_pitcher, home_offense, park, weather)
                           away_runs ~ f(home_pitcher, away_offense, park, weather)
Then total = home + away, and P(total > line) from the convolution of two Poissons.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from scipy.special import gammaln
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PROCESSED_DIR


FEATURE_COLS_HOME = [
    "home_offense_rate",
    "away_pitcher_quality",   # opponent pitcher quality
    "park_factor",
    "wind_effect",
    "temperature_effect",
    "is_dome",
]

FEATURE_COLS_AWAY = [
    "away_offense_rate",
    "home_pitcher_quality",   # opponent pitcher quality
    "park_factor",
    "wind_effect",
    "temperature_effect",
    "is_dome",
]


def temperature_effect(temp_f: float) -> float:
    """
    Converts temperature to a centered effect on run scoring.
    Ball travels farther in warm air. Effect is roughly linear:
    ~0.4% more distance per 10°F above 70°F.
    We normalize: 70°F = 0, 90°F = +0.2, 50°F = -0.2
    """
    return (temp_f - 70.0) / 100.0


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temperature_effect"] = df["temperature_f"].apply(temperature_effect)
    return df


class PoissonTotalsModel:
    """
    Fits two Poisson GLMs (home runs, away runs) and predicts total run distributions.
    """

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self.home_scaler_mean = None
        self.home_scaler_std = None
        self.away_scaler_mean = None
        self.away_scaler_std = None

    def _prepare_X(self, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        X = df[feature_cols].values.astype(float)
        return np.column_stack([np.ones(len(X)), X])  # intercept + features

    def fit(self, df: pd.DataFrame) -> "PoissonTotalsModel":
        df = add_derived_features(df)

        # Home runs model: home offense vs away pitcher
        X_home = self._prepare_X(df, FEATURE_COLS_HOME)
        y_home = df["home_runs"].values.astype(float)
        self.home_model = sm.GLM(
            y_home, X_home,
            family=sm.families.Poisson(link=sm.families.links.Log())
        ).fit()

        # Away runs model: away offense vs home pitcher
        X_away = self._prepare_X(df, FEATURE_COLS_AWAY)
        y_away = df["away_runs"].values.astype(float)
        self.away_model = sm.GLM(
            y_away, X_away,
            family=sm.families.Poisson(link=sm.families.links.Log())
        ).fit()

        print("Home runs model:")
        print(self.home_model.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])
        print("\nAway runs model:")
        print(self.away_model.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])

        return self

    def predict_lambda(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lambda_home, lambda_away) — expected runs for each team."""
        df = add_derived_features(df)
        X_home = self._prepare_X(df, FEATURE_COLS_HOME)
        X_away = self._prepare_X(df, FEATURE_COLS_AWAY)
        lambda_home = self.home_model.predict(X_home)
        lambda_away = self.away_model.predict(X_away)
        return np.array(lambda_home), np.array(lambda_away)

    def predict_total_distribution(
        self, df: pd.DataFrame, max_runs: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the PMF of total runs for each game via convolution of two Poissons.
        Shape: (n_games, max_runs+1) where column k = P(total = k)
        Also returns expected total runs.
        """
        lambda_home, lambda_away = self.predict_lambda(df)
        n = len(df)

        # Convolve two independent Poisson PMFs
        pmf_matrix = np.zeros((n, max_runs + 1))
        for i in range(n):
            pmf_h = poisson.pmf(np.arange(max_runs + 1), lambda_home[i])
            pmf_a = poisson.pmf(np.arange(max_runs + 1), lambda_away[i])
            conv = np.convolve(pmf_h, pmf_a)[:max_runs + 1]
            pmf_matrix[i] = conv / conv.sum()  # renormalize to handle truncation

        expected_total = lambda_home + lambda_away
        return pmf_matrix, expected_total

    def predict_over_under_probs(
        self, df: pd.DataFrame, lines: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given total run lines for each game, returns (P(over), P(under)).
        Lines are typically half-integers (8.5, 9.0, etc.).
        P(over line) = P(total > line) = P(total >= ceil(line))
        P(under line) = P(total < line) = P(total <= floor(line))
        For whole-number lines (push possible), P(push) is split evenly.
        """
        pmf_matrix, _ = self.predict_total_distribution(df)
        n = len(df)
        p_over = np.zeros(n)
        p_under = np.zeros(n)

        for i in range(n):
            line = lines[i]
            pmf = pmf_matrix[i]
            totals = np.arange(len(pmf))

            p_over[i] = pmf[totals > line].sum()
            p_under[i] = pmf[totals < line].sum()

            # Push probability (only on whole-number lines)
            p_push = pmf[totals == line].sum() if line == int(line) else 0
            # Allocate push 50/50
            p_over[i] += p_push / 2
            p_under[i] += p_push / 2

        return p_over, p_under

    def save(self, path: Path = None):
        if path is None:
            path = PROCESSED_DIR / "poisson_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path = None) -> "PoissonTotalsModel":
        if path is None:
            path = PROCESSED_DIR / "poisson_model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Run src/features/build_features.py first to generate features.parquet")
        sys.exit(1)

    df = pd.read_parquet(features_path)
    train = df[df["season"] <= 2021]

    print(f"Training on {len(train)} games ({train['season'].min()}–{train['season'].max()})")
    model = PoissonTotalsModel()
    model.fit(train)
    model.save()

    # Quick sanity check
    sample = df[df["season"] == 2022].head(10).copy()
    lines = np.full(len(sample), 8.5)
    p_over, p_under = model.predict_over_under_probs(sample, lines)
    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"{row['away_team']} @ {row['home_team']}: "
              f"P(O8.5)={p_over[i]:.3f}  P(U8.5)={p_under[i]:.3f}  "
              f"actual={row['total_runs']:.0f}")
