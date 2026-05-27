"""
Daily signal generator — the thing you run each morning to get today's bet recommendations.

Workflow:
  1. Load the trained model (or retrain if stale)
  2. Fetch today's MLB games and announced starters from Baseball Reference / pybaseball
  3. Fetch today's odds from The Odds API
  4. Fetch today's weather forecast (Open-Meteo)
  5. Build features for today's games
  6. Compute EV and Kelly stakes
  7. Print ranked bet recommendations

Usage:
  python -m src.signals.daily_picks
  python -m src.signals.daily_picks --date 2024-07-15   (for a specific date)
"""

import argparse
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PROCESSED_DIR, RAW_DIR, MIN_EV_THRESHOLD, KELLY_FRACTION
from src.data.fetch_odds import fetch_live_odds, build_consensus_line
from src.data.fetch_weather import fetch_forecast_weather, STADIUMS
from src.features.park_factors import FALLBACK_PARK_FACTORS
from src.models.poisson_model import temperature_effect as _temp_effect
from src.models.poisson_model import PoissonTotalsModel
from src.models.ev_calculator import compute_ev_table, summarize_edge


# Map Odds API team names to our abbreviations
ODDS_API_TEAM_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
    # Athletics moved to Sacramento; update as needed
    "Athletics": "OAK",
}


def get_season_averages(features: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Returns team-level averages for the most recent available season.
    Used to populate features for today's games when rolling window isn't available.
    """
    available_seasons = sorted(features["season"].unique())
    use_season = max([s for s in available_seasons if s <= season], default=available_seasons[-1])
    season_data = features[features["season"] == use_season]

    home_avg = season_data.groupby("home_team")["home_offense_rate"].mean().rename("offense_rate")
    away_avg = season_data.groupby("away_team")["away_offense_rate"].mean().rename("offense_rate")
    team_avg = pd.concat([home_avg, away_avg]).groupby(level=0).mean()

    return team_avg


def fetch_todays_starters(target_date: str = None) -> dict[str, str]:
    """
    Returns {team_abbr: pitcher_fullname} for today's probable starters.
    Uses the free MLB Stats API (no auth required).
    Falls back to empty dict — model will use team season average.
    """
    if target_date is None:
        target_date = date.today().isoformat()
    try:
        import requests
        from src.data.fetch_games import TEAM_ID_TO_ABBR

        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "date": target_date,
                "gameType": "R",
                "hydrate": "probablePitchers",
            },
            timeout=15,
        )
        resp.raise_for_status()

        starters = {}
        for date_entry in resp.json().get("dates", []):
            for game in date_entry.get("games", []):
                for side in ["home", "away"]:
                    team_block = game.get("teams", {}).get(side, {})
                    team_id = team_block.get("team", {}).get("id")
                    pitcher = team_block.get("probablePitcher")
                    if team_id and pitcher:
                        abbr = TEAM_ID_TO_ABBR.get(team_id)
                        if abbr:
                            starters[abbr] = pitcher.get("fullName", "")

        n = len(starters)
        if n:
            print(f"  Probable starters announced for {n} teams")
        else:
            print("  No starters announced yet — using team season averages")
        return starters
    except Exception as e:
        print(f"  Could not fetch starters: {e}")
        return {}


def build_today_features(
    odds_df: pd.DataFrame,
    historical_features: pd.DataFrame,
    pitcher_data: pd.DataFrame,
    target_date: str = None,
    starters: dict = None,
) -> pd.DataFrame:
    """
    Constructs the feature row for each game happening today.
    Merges odds, weather, historical averages, and announced starters.

    starters: {team_abbr: pitcher_fullname} from fetch_todays_starters().
              When provided, uses the announced starter's individual quality score
              instead of the team's season-average top-2 starters.
    """
    from src.data.fetch_pitchers import get_pitcher_quality_features
    from src.models.poisson_model import add_derived_features

    if target_date is None:
        target_date = date.today().isoformat()
    if starters is None:
        starters = {}

    current_season = int(target_date[:4])
    team_offense_avg = get_season_averages(historical_features, current_season)

    pitcher_quality = get_pitcher_quality_features(pitcher_data)
    league_avg_pq = pitcher_quality[pitcher_quality["Season"] == current_season]["pitcher_quality"].median()
    if pd.isna(league_avg_pq):
        league_avg_pq = 4.0

    # Team-season average (fallback when no announced starter or starter not in DB)
    sp_lookup = (
        pitcher_quality[
            (pitcher_quality["Season"] == current_season) & (pitcher_quality["GS"] >= 5)
        ]
        .sort_values("pitcher_quality")
        .groupby("team")
        .head(2)
        .groupby("team")["pitcher_quality"]
        .mean()
        .to_dict()
    )

    # Individual pitcher lookup: (season, name) → quality score
    # Also check prior season for pitchers who changed teams or aren't in current-year DB yet
    pq_by_name = pitcher_quality.set_index(["Season", "Name"])["pitcher_quality"].to_dict()

    # Deduplicate to one row per game (consensus across books) for feature building
    from src.data.fetch_odds import build_consensus_line
    consensus = build_consensus_line(odds_df)
    # Re-attach per-game odds columns needed downstream
    best_odds = (
        odds_df.sort_values("vig")  # lowest vig = best line
        .groupby(["home_team", "away_team"])
        .first()
        .reset_index()[["home_team", "away_team", "total_line", "over_odds", "under_odds",
                         "fair_prob_over", "fair_prob_under"]]
    )
    consensus = consensus.merge(
        best_odds[["home_team", "away_team", "over_odds", "under_odds"]],
        on=["home_team", "away_team"], how="left"
    )

    rows = []
    for _, odds_row in consensus.iterrows():
        raw_home = odds_row.get("home_team", "")
        raw_away = odds_row.get("away_team", "")
        home = ODDS_API_TEAM_MAP.get(raw_home, raw_home)
        away = ODDS_API_TEAM_MAP.get(raw_away, raw_away)

        # Offense rates from historical averages
        home_offense = team_offense_avg.get(home, 4.5)
        away_offense = team_offense_avg.get(away, 4.5)

        # Pitcher quality — announced starter takes priority over team average
        home_starter = starters.get(home, "")
        away_starter = starters.get(away, "")
        home_pq = (
            pq_by_name.get((current_season, home_starter))
            or pq_by_name.get((current_season - 1, home_starter))
            or sp_lookup.get(home, league_avg_pq)
        )
        away_pq = (
            pq_by_name.get((current_season, away_starter))
            or pq_by_name.get((current_season - 1, away_starter))
            or sp_lookup.get(away, league_avg_pq)
        )

        # Park factor
        park_factor = FALLBACK_PARK_FACTORS.get(home, 100) / 100.0

        # Weather forecast
        weather = fetch_forecast_weather(home, target_date) or {
            "wind_effect": 0.0, "temperature_f": 70.0, "is_dome": False
        }

        rows.append({
            "game_date": pd.Timestamp(target_date),
            "season": current_season,
            "home_team": home,
            "away_team": away,
            "home_starter": home_starter or "(team avg)",
            "away_starter": away_starter or "(team avg)",
            "home_offense_rate": home_offense,
            "away_offense_rate": away_offense,
            "home_pitcher_quality": home_pq,
            "away_pitcher_quality": away_pq,
            "park_factor": park_factor,
            "wind_effect": weather["wind_effect"],
            "temperature_f": weather["temperature_f"],
            "is_dome": int(weather["is_dome"]),
            # carry odds info for EV calc
            "total_line": odds_row.get("total_line", 8.5),
            "over_odds": odds_row.get("over_odds", -110),
            "under_odds": odds_row.get("under_odds", -110),
            "fair_prob_over": odds_row.get("fair_prob_over", 0.5),
            "fair_prob_under": odds_row.get("fair_prob_under", 0.5),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["temperature_effect"] = df["temperature_f"].apply(_temp_effect)
    return df


def run_daily_picks(target_date: str = None, bankroll: float = 1000.0):
    if target_date is None:
        target_date = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"  MLB TOTALS SIGNAL GENERATOR — {target_date}")
    print(f"{'='*60}")

    # Load model
    model_path = PROCESSED_DIR / "poisson_model.pkl"
    if not model_path.exists():
        print("ERROR: No trained model found. Run src/models/poisson_model.py first.")
        return

    print("Loading model...")
    model = PoissonTotalsModel.load(model_path)

    # Load historical features for context
    features_path = PROCESSED_DIR / "features.parquet"
    historical = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()

    # Load pitcher data
    pitchers_path = RAW_DIR / "pitchers_all.parquet"
    pitchers = pd.read_parquet(pitchers_path) if pitchers_path.exists() else pd.DataFrame()

    # Fetch live odds
    print("Fetching today's odds...")
    try:
        odds_raw = fetch_live_odds()
        if odds_raw.empty:
            print("No odds available yet for today. Try again closer to game time.")
            return
        odds = build_consensus_line(odds_raw)
    except Exception as e:
        print(f"ERROR fetching odds: {e}")
        print("Make sure ODDS_API_KEY is set in your .env file.")
        return

    print(f"Found {len(odds)} games with odds.")

    # Fetch today's announced starters
    print("Fetching today's probable starters...")
    starters = fetch_todays_starters(target_date)

    # Build features for today's games
    print("Building features + fetching weather...")
    today_features = build_today_features(odds_raw, historical, pitchers, target_date, starters)

    if today_features.empty:
        print("Could not build features for today's games.")
        return

    # Archive today's odds — builds real historical dataset over time (free tier)
    from src.data.fetch_historical_odds import archive_todays_odds
    archive_todays_odds(today_features)

    # Get model probabilities
    lines = today_features["total_line"].values
    p_over, p_under = model.predict_over_under_probs(today_features, lines)

    # Build EV table directly from today_features (which already has odds embedded)
    # This avoids the team-name mismatch between odds_raw (full names) and features (abbrs)
    from src.models.ev_calculator import expected_value, kelly_stake
    ev_rows = []
    for i, (_, row) in enumerate(today_features.iterrows()):
        for side, model_p, odds_col in [
            ("OVER",  p_over[i],  "over_odds"),
            ("UNDER", p_under[i], "under_odds"),
        ]:
            odds = int(row.get(odds_col, -110))
            market_p = row.get(f"fair_prob_{side.lower()}", 0.5)
            ev = expected_value(model_p, odds)
            stake = kelly_stake(model_p, odds, bankroll, KELLY_FRACTION)
            ev_rows.append({
                "game_date":  row.get("game_date"),
                "matchup":    f"{row['away_team']} @ {row['home_team']}",
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "side":       side,
                "line":       row.get("total_line", 0),
                "odds":       odds,
                "model_prob": round(model_p, 4),
                "market_prob": round(market_p, 4),
                "ev":         round(ev, 4),
                "kelly_fraction_of_bankroll": round(stake, 4),
                "clv":        round(model_p - market_p, 4),
                "flag_bet":   ev >= MIN_EV_THRESHOLD,
            })
    ev_table = pd.DataFrame(ev_rows).sort_values("ev", ascending=False).reset_index(drop=True)

    if ev_table.empty:
        print("No EV data computed.")
        return

    # Show all games
    print(f"\n  ALL GAMES TODAY")
    print(f"  {'Matchup':<30} {'Line':<6} {'P(O)':<8} {'P(U)':<8} {'EV_O':>7} {'EV_U':>7}  Starters (away / home)")
    print(f"  {'-'*95}")
    for _, g_row in today_features.iterrows():
        home = g_row["home_team"]
        away = g_row["away_team"]
        matchup = f"{away} @ {home}"
        ev_over_row = ev_table[(ev_table["home_team"] == home) & (ev_table["side"] == "OVER")]
        ev_under_row = ev_table[(ev_table["home_team"] == home) & (ev_table["side"] == "UNDER")]
        if ev_over_row.empty or ev_under_row.empty:
            continue
        line = ev_over_row.iloc[0]["line"]
        p_o = ev_over_row.iloc[0]["model_prob"]
        p_u = ev_under_row.iloc[0]["model_prob"]
        ev_o = ev_over_row.iloc[0]["ev"]
        ev_u = ev_under_row.iloc[0]["ev"]
        away_sp = g_row.get("away_starter", "(team avg)")
        home_sp = g_row.get("home_starter", "(team avg)")
        print(f"  {matchup:<30} {line:<6.1f} {p_o:<8.3f} {p_u:<8.3f} {ev_o:>+7.3f} {ev_u:>+7.3f}  {away_sp} / {home_sp}")

    # Show flagged bets
    summarize_edge(ev_table)

    # Save to file
    output_path = PROCESSED_DIR / f"picks_{target_date}.parquet"
    ev_table.to_parquet(output_path, index=False)
    print(f"Full picks table saved to {output_path}")

    return ev_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Current bankroll")
    args = parser.parse_args()

    run_daily_picks(target_date=args.date, bankroll=args.bankroll)
