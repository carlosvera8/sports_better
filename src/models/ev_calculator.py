"""
Expected Value and Kelly Criterion calculator.

EV = (model_prob * payout) - (1 - model_prob)
   where payout = (100 / |odds|) for favorites or (odds / 100) for underdogs

Kelly fraction = (bp - q) / b
   where b = net odds (payout per unit), p = P(win), q = 1 - p
We use fractional Kelly (default 0.25) to reduce variance.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import MIN_EV_THRESHOLD, KELLY_FRACTION


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1.0
    else:
        return (100 / abs(american_odds)) + 1.0


def expected_value(model_prob: float, american_odds: int) -> float:
    """
    Returns EV per unit wagered.
    e.g., EV=0.05 means the bet returns $1.05 per $1 wagered in expectation.
    A positive EV means the bet has edge.
    """
    decimal_odds = american_to_decimal(american_odds)
    payout = decimal_odds - 1  # net profit per unit
    ev = (model_prob * payout) - (1 - model_prob)
    return ev


def kelly_stake(model_prob: float, american_odds: int, bankroll: float = 1.0,
                fraction: float = KELLY_FRACTION) -> float:
    """
    Returns the recommended bet size using fractional Kelly criterion.
    Result is in the same units as bankroll.

    Full Kelly maximizes long-run geometric growth but is very aggressive.
    Fractional Kelly (0.25) is the practical standard among professional bettors.
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1  # net odds
    p = model_prob
    q = 1 - p

    if b <= 0 or p <= 0:
        return 0.0

    kelly_full = (b * p - q) / b
    kelly_full = max(0.0, kelly_full)  # never bet negative

    return bankroll * fraction * kelly_full


def compute_ev_table(
    games: pd.DataFrame,
    p_over: np.ndarray,
    p_under: np.ndarray,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges model probabilities with market lines to produce a bet recommendation table.

    games: feature table rows (one row per game)
    p_over/p_under: model probabilities from PoissonTotalsModel
    odds_df: consensus odds with columns [home_team, away_team, total_line, over_odds, under_odds, fair_prob_over, fair_prob_under]

    Returns a DataFrame with EV, Kelly stake, and recommendation for each game/side.
    """
    results = []

    for i, (_, game_row) in enumerate(games.iterrows()):
        # Find matching odds row
        match = odds_df[
            (odds_df["home_team"].str.upper() == game_row["home_team"].upper()) |
            (odds_df["away_team"].str.upper() == game_row["away_team"].upper())
        ]

        if match.empty:
            continue

        odds_row = match.iloc[0]
        line = odds_row["total_line"]
        over_odds = int(odds_row.get("over_odds", -110))
        under_odds = int(odds_row.get("under_odds", -110))

        model_p_over = p_over[i]
        model_p_under = p_under[i]

        ev_over = expected_value(model_p_over, over_odds)
        ev_under = expected_value(model_p_under, under_odds)

        # Market implied probabilities (no-vig)
        market_p_over = odds_row.get("fair_prob_over", 0.5)
        market_p_under = odds_row.get("fair_prob_under", 0.5)

        # Closing Line Value: if our model prob > market no-vig prob, we have CLV
        clv_over = model_p_over - market_p_over
        clv_under = model_p_under - market_p_under

        for side, ev, kelly, model_p, market_p, clv, odds in [
            ("OVER",  ev_over,  kelly_stake(model_p_over, over_odds),   model_p_over,  market_p_over,  clv_over,  over_odds),
            ("UNDER", ev_under, kelly_stake(model_p_under, under_odds), model_p_under, market_p_under, clv_under, under_odds),
        ]:
            results.append({
                "game_date": game_row.get("game_date"),
                "matchup": f"{game_row['away_team']} @ {game_row['home_team']}",
                "home_team": game_row["home_team"],
                "away_team": game_row["away_team"],
                "side": side,
                "line": line,
                "odds": odds,
                "model_prob": round(model_p, 4),
                "market_prob": round(market_p, 4),
                "ev": round(ev, 4),
                "kelly_fraction_of_bankroll": round(kelly, 4),
                "clv": round(clv, 4),
                "flag_bet": ev >= MIN_EV_THRESHOLD,
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df


def summarize_edge(ev_table: pd.DataFrame) -> None:
    """Prints a summary of flagged bets."""
    flagged = ev_table[ev_table["flag_bet"]]
    if flagged.empty:
        print("No bets above EV threshold today.")
        return

    print(f"\n{'='*70}")
    print(f"  BETS WITH EDGE (EV >= {MIN_EV_THRESHOLD:.0%})")
    print(f"{'='*70}")
    for _, row in flagged.iterrows():
        print(
            f"  {row['matchup']:<30} {row['side']} {row['line']:<6} "
            f"Odds: {row['odds']:>5}  Model: {row['model_prob']:.3f}  "
            f"EV: {row['ev']:+.3f}  Kelly: {row['kelly_fraction_of_bankroll']:.3f}"
        )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Sanity check
    print("EV at -110 with 55% win probability:")
    print(f"  EV = {expected_value(0.55, -110):.4f}")  # should be ~0.056

    print("Kelly stake on $1000 bankroll:")
    print(f"  Stake = ${kelly_stake(0.55, -110, bankroll=1000):.2f}")

    print("\nEV at -110 with 50% win probability (break-even at -105):")
    print(f"  EV = {expected_value(0.50, -110):.4f}")  # should be slightly negative
