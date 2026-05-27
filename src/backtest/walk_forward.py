"""
Walk-forward backtesting harness.

Strategy:
  - Train on seasons 2015–N, test on season N+1
  - Repeat for N = 2021, 2022, 2023 (testing 2022, 2023, 2024)
  - Never allow future data to leak into training

For each test game we simulate:
  1. Model predicts P(over) and P(under) given the features
  2. We compare to the historical odds line
  3. If EV >= threshold, we record the bet and its outcome

Key metric: ROI and CLV (Closing Line Value).
CLV > 0 is a stronger signal of real edge than ROI, because it shows your model
was consistently more accurate than the closing market — which is the true test.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PROCESSED_DIR, TRAIN_SEASONS, TEST_SEASONS, MIN_EV_THRESHOLD, KELLY_FRACTION
from src.models.poisson_model import PoissonTotalsModel
from src.models.ev_calculator import expected_value, kelly_stake, american_to_decimal


def simulate_bet_outcome(
    side: str,
    line: float,
    actual_total: float,
    odds: int,
    stake: float,
) -> float:
    """
    Returns profit/loss for a single bet.
    side: 'OVER' or 'UNDER'
    """
    decimal_odds = american_to_decimal(odds)
    payout = decimal_odds - 1  # net profit per unit

    if side == "OVER":
        won = actual_total > line
        push = (line == int(line)) and (actual_total == line)
    else:
        won = actual_total < line
        push = (line == int(line)) and (actual_total == line)

    if push:
        return 0.0
    elif won:
        return stake * payout
    else:
        return -stake


def run_walk_forward_backtest(
    features: pd.DataFrame,
    historical_odds: pd.DataFrame = None,
    train_seasons: list[int] = None,
    test_seasons: list[int] = None,
    ev_threshold: float = MIN_EV_THRESHOLD,
    kelly_fraction: float = KELLY_FRACTION,
    default_line: float = 8.5,
    default_over_odds: int = -110,
    default_under_odds: int = -110,
) -> pd.DataFrame:
    """
    Runs a walk-forward backtest.

    historical_odds: optional DataFrame from fetch_historical_odds.py with columns
        (game_date, home_team, away_team, total_line, over_odds, under_odds).
        When provided, each game uses its real market line and odds instead of the
        fixed defaults. Games without a matched line fall back to the defaults.

    Returns a DataFrame of all simulated bets with outcomes.
    """
    from src.data.fetch_historical_odds import merge_historical_odds

    if train_seasons is None:
        train_seasons = TRAIN_SEASONS
    if test_seasons is None:
        test_seasons = TEST_SEASONS

    all_bets = []
    bankroll = 1000.0
    bankroll_history = [bankroll]

    for test_season in test_seasons:
        current_train_seasons = [s for s in train_seasons if s < test_season]
        if not current_train_seasons:
            print(f"No training data before {test_season}, skipping")
            continue

        train_df = features[features["season"].isin(current_train_seasons)].copy()
        test_df = features[features["season"] == test_season].copy()

        if train_df.empty or test_df.empty:
            print(f"Empty split for {test_season}, skipping")
            continue

        print(f"\nTraining on {current_train_seasons[0]}–{current_train_seasons[-1]} "
              f"({len(train_df)} games), testing on {test_season} ({len(test_df)} games)...")

        # Merge real odds into test data if available
        season_odds = pd.DataFrame()
        if historical_odds is not None and not historical_odds.empty:
            season_odds = historical_odds[
                historical_odds["game_date"].dt.year == test_season
            ].copy()
        test_df = merge_historical_odds(test_df, season_odds)

        model = PoissonTotalsModel()
        model.fit(train_df)

        lines = test_df["total_line"].values
        p_over, p_under = model.predict_over_under_probs(test_df, lines)

        for i, (_, row) in enumerate(test_df.iterrows()):
            line = row["total_line"]
            over_odds = int(row["over_odds"])
            under_odds = int(row["under_odds"])
            actual_total = row["total_runs"]

            for side, model_p, odds in [("OVER", p_over[i], over_odds),
                                         ("UNDER", p_under[i], under_odds)]:
                ev = expected_value(model_p, odds)
                if ev < ev_threshold:
                    continue

                stake = kelly_stake(model_p, odds, bankroll, kelly_fraction)
                if stake < 1.0:
                    continue

                pnl = simulate_bet_outcome(side, line, actual_total, odds, stake)
                bankroll += pnl
                bankroll_history.append(bankroll)

                all_bets.append({
                    "season": test_season,
                    "game_date": row.get("game_date"),
                    "matchup": f"{row['away_team']} @ {row['home_team']}",
                    "side": side,
                    "line": line,
                    "odds": odds,
                    "model_prob": round(model_p, 4),
                    "ev": round(ev, 4),
                    "stake": round(stake, 2),
                    "actual_total": actual_total,
                    "won": (pnl > 0),
                    "pnl": round(pnl, 2),
                    "bankroll_after": round(bankroll, 2),
                })

    results = pd.DataFrame(all_bets)
    return results


def compute_backtest_metrics(bets: pd.DataFrame) -> dict:
    """
    Computes standard betting performance metrics.
    """
    if bets.empty:
        return {"error": "No bets placed"}

    total_bets = len(bets)
    total_wagered = bets["stake"].sum()
    total_pnl = bets["pnl"].sum()
    win_rate = bets["won"].mean()
    roi = total_pnl / total_wagered if total_wagered > 0 else 0

    avg_ev = bets["ev"].mean()
    avg_odds = bets["odds"].mean()

    # Sharpe-like ratio: mean return / std of returns (per bet)
    per_bet_roi = bets["pnl"] / bets["stake"]
    sharpe = per_bet_roi.mean() / per_bet_roi.std() if per_bet_roi.std() > 0 else 0

    # Maximum drawdown
    bankroll_series = bets["bankroll_after"].values
    peak = np.maximum.accumulate(bankroll_series)
    drawdown = (bankroll_series - peak) / peak
    max_drawdown = drawdown.min()

    # By season breakdown
    by_season = bets.groupby("season").agg(
        bets_placed=("pnl", "count"),
        total_wagered=("stake", "sum"),
        total_pnl=("pnl", "sum"),
        win_rate=("won", "mean"),
    ).assign(roi=lambda x: x["total_pnl"] / x["total_wagered"]).round(4)

    return {
        "total_bets": total_bets,
        "total_wagered": round(total_wagered, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 4),
        "win_rate": round(win_rate, 4),
        "avg_ev": round(avg_ev, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "by_season": by_season,
    }


def print_backtest_report(metrics: dict) -> None:
    print(f"\n{'='*60}")
    print("  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total bets:       {metrics['total_bets']}")
    print(f"  Total wagered:    ${metrics['total_wagered']:,.2f}")
    print(f"  Total P&L:        ${metrics['total_pnl']:+,.2f}")
    print(f"  ROI:              {metrics['roi']:+.2%}")
    print(f"  Win rate:         {metrics['win_rate']:.2%}")
    print(f"  Avg model EV:     {metrics['avg_ev']:+.2%}")
    print(f"  Sharpe ratio:     {metrics['sharpe']:.3f}")
    print(f"  Max drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"\n  By Season:")
    print(metrics["by_season"].to_string())
    print(f"{'='*60}\n")


if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Run src/features/build_features.py first")
        sys.exit(1)

    features = pd.read_parquet(features_path)
    print(f"Loaded {len(features)} games")

    bets = run_walk_forward_backtest(
        features,
        train_seasons=list(range(2015, 2022)),
        test_seasons=[2022, 2023, 2024],
        ev_threshold=0.03,
    )

    if not bets.empty:
        metrics = compute_backtest_metrics(bets)
        print_backtest_report(metrics)
        bets.to_parquet(PROCESSED_DIR / "backtest_results.parquet", index=False)
        print(f"Saved {len(bets)} bet records to backtest_results.parquet")
    else:
        print("No bets met the EV threshold.")
