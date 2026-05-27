"""
Statistical validation tools for backtest results.

The central question: is our edge real, or could these results be explained by variance?
We use a two-sided binomial test and bootstrapped confidence intervals on ROI.

Also computes CLV (Closing Line Value) — the gold standard for edge validation.
CLV > 0 means our model systematically agreed with where the sharp money moved the line,
which is a stronger predictor of continued edge than historical ROI.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PROCESSED_DIR


def binomial_edge_test(bets: pd.DataFrame) -> dict:
    """
    Tests H0: win rate = breakeven win rate (given the average odds).
    Uses a binomial z-test.
    """
    n_bets = len(bets)
    n_wins = bets["won"].sum()
    win_rate = n_wins / n_bets

    # Breakeven win rate given the average odds
    # At -110: you need to win 52.38% to break even
    avg_odds = bets["odds"].mean()
    if avg_odds < 0:
        breakeven = abs(avg_odds) / (abs(avg_odds) + 100)
    else:
        breakeven = 100 / (avg_odds + 100)

    # z-test
    se = np.sqrt(breakeven * (1 - breakeven) / n_bets)
    z_stat = (win_rate - breakeven) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "n_bets": n_bets,
        "win_rate": round(win_rate, 4),
        "breakeven_rate": round(breakeven, 4),
        "z_statistic": round(z_stat, 3),
        "p_value": round(p_value, 4),
        "significant_at_5pct": p_value < 0.05,
        "sample_needed_for_significance": int(
            (stats.norm.ppf(0.975) / (win_rate - breakeven + 1e-9)) ** 2
            * breakeven * (1 - breakeven)
        ) if win_rate > breakeven else None,
    }


def bootstrap_roi_ci(bets: pd.DataFrame, n_bootstrap: int = 10_000, alpha: float = 0.05) -> dict:
    """
    Bootstrap confidence interval on ROI.
    Returns the (alpha/2, 1-alpha/2) percentile interval.
    """
    per_bet_roi = (bets["pnl"] / bets["stake"]).values
    n = len(per_bet_roi)

    boot_means = np.empty(n_bootstrap)
    rng = np.random.default_rng(42)
    for i in range(n_bootstrap):
        sample = rng.choice(per_bet_roi, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_lo = np.percentile(boot_means, 100 * alpha / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    point_est = per_bet_roi.mean()

    return {
        "roi_point_estimate": round(point_est, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "ci_level": f"{100*(1-alpha):.0f}%",
        "profitable_with_confidence": ci_lo > 0,
    }


def compute_clv(bets: pd.DataFrame, closing_odds_col: str = "closing_odds") -> dict | None:
    """
    Computes Closing Line Value (CLV).
    CLV = model's implied probability - closing market's no-vig probability.
    Positive CLV = model was consistently sharper than the line when it moved.

    Requires a 'closing_odds' column in bets DataFrame.
    If not present, returns None (CLV requires historical closing line data).
    """
    if closing_odds_col not in bets.columns:
        return None

    from src.data.fetch_odds import american_to_implied_prob, remove_vig

    def closing_fair_prob(row) -> float:
        # For simplicity, assume closing over/under are symmetric (-vig adjustment)
        # In production, you'd have both sides of the closing line
        raw_prob = american_to_implied_prob(row[closing_odds_col])
        # Approximate vig removal assuming symmetric market
        fair = raw_prob / (raw_prob + (1 - raw_prob * 1.045))  # approximate
        return fair

    bets = bets.copy()
    bets["closing_fair_prob"] = bets.apply(closing_fair_prob, axis=1)
    bets["clv"] = bets["model_prob"] - bets["closing_fair_prob"]

    return {
        "mean_clv": round(bets["clv"].mean(), 4),
        "pct_positive_clv": round((bets["clv"] > 0).mean(), 4),
        "clv_t_stat": round(stats.ttest_1samp(bets["clv"], 0).statistic, 3),
        "clv_p_value": round(stats.ttest_1samp(bets["clv"], 0).pvalue, 4),
    }


def full_validation_report(bets: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  STATISTICAL VALIDATION")
    print(f"{'='*60}")

    binom = binomial_edge_test(bets)
    print(f"\n  Binomial Edge Test")
    print(f"  ------------------")
    print(f"  Win rate:          {binom['win_rate']:.3%}")
    print(f"  Breakeven rate:    {binom['breakeven_rate']:.3%}")
    print(f"  z-statistic:       {binom['z_statistic']:.3f}")
    print(f"  p-value:           {binom['p_value']:.4f}")
    print(f"  Significant (5%):  {binom['significant_at_5pct']}")
    if binom.get("sample_needed_for_significance"):
        print(f"  Bets needed:       {binom['sample_needed_for_significance']:,}")

    boot = bootstrap_roi_ci(bets)
    print(f"\n  Bootstrap ROI Confidence Interval ({boot['ci_level']})")
    print(f"  ------------------")
    print(f"  Point estimate:    {boot['roi_point_estimate']:+.3%}")
    print(f"  CI lower:          {boot['ci_lower']:+.3%}")
    print(f"  CI upper:          {boot['ci_upper']:+.3%}")
    print(f"  Profitable w/ CI:  {boot['profitable_with_confidence']}")

    clv = compute_clv(bets)
    if clv:
        print(f"\n  Closing Line Value (CLV)")
        print(f"  ------------------")
        print(f"  Mean CLV:          {clv['mean_clv']:+.4f}")
        print(f"  Pct positive CLV:  {clv['pct_positive_clv']:.2%}")
        print(f"  CLV t-stat:        {clv['clv_t_stat']:.3f}")
        print(f"  CLV p-value:       {clv['clv_p_value']:.4f}")
    else:
        print("\n  CLV: Not available (no closing_odds column in bet data)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    bets_path = PROCESSED_DIR / "backtest_results.parquet"
    if not bets_path.exists():
        print("Run src/backtest/walk_forward.py first")
        sys.exit(1)

    bets = pd.read_parquet(bets_path)
    full_validation_report(bets)
