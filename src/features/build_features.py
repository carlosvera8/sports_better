"""
Assembles the master feature table used by the Poisson model.
One row per game with all features needed to predict total runs.

Features:
  - home_offense_rate: park-neutral home team runs/game (rolling 15-game window)
  - away_offense_rate: park-neutral away team runs/game (rolling 15-game window)
  - home_pitcher_quality: xFIP-based quality score for home starter
  - away_pitcher_quality: xFIP-based quality score for away starter
  - park_factor: run factor for the home stadium (1.0 = neutral)
  - wind_effect: wind score (-1 to +1, positive = blowing out)
  - temperature_f: game time temperature
  - is_dome: binary flag for dome stadiums
  - total_runs: target variable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PROCESSED_DIR, RAW_DIR
from src.features.park_factors import build_park_factor_table, FALLBACK_PARK_FACTORS
from src.data.fetch_weather import fetch_historical_weather, STADIUMS


ROLLING_WINDOW = 15  # games for rolling offense rate


def compute_rolling_offense(games: pd.DataFrame) -> pd.DataFrame:
    """
    For each team and each game, compute the park-neutral run scoring rate
    over the prior ROLLING_WINDOW games.

    Returns a DataFrame indexed by (game_date, team) with column 'offense_rate'.
    """
    # Build a long-form table: one row per team per game
    home_side = games[["game_date", "home_team", "home_runs", "away_runs", "season"]].copy()
    home_side = home_side.rename(columns={"home_team": "team", "home_runs": "runs_scored", "away_runs": "runs_allowed"})
    home_side["is_home"] = True

    away_side = games[["game_date", "away_team", "away_runs", "home_runs", "season"]].copy()
    away_side = away_side.rename(columns={"away_team": "team", "away_runs": "runs_scored", "home_runs": "runs_allowed"})
    away_side["is_home"] = False

    long = pd.concat([home_side, away_side], ignore_index=True)
    long = long.sort_values(["team", "game_date"]).reset_index(drop=True)

    # Rolling mean of runs_scored over prior N games (exclude current game)
    long["offense_rate"] = (
        long.groupby("team")["runs_scored"]
        .transform(lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=5).mean())
    )

    # Fill NaN at season start with season average
    season_avg = long.groupby(["team", "season"])["runs_scored"].transform("mean")
    long["offense_rate"] = long["offense_rate"].fillna(season_avg)

    return long[["game_date", "team", "offense_rate", "is_home"]].copy()


def build_feature_table(
    games: pd.DataFrame,
    pitchers: pd.DataFrame,
    seasons: list[int],
    include_weather: bool = True,
) -> pd.DataFrame:
    """
    Main feature assembly pipeline.
    games: output of fetch_games
    pitchers: output of fetch_pitchers (season-level stats)
    seasons: list of seasons to include
    include_weather: if True, fetches weather for each game (slow on first run, then cached)
    """
    games = games[games["season"].isin(seasons)].copy()
    games = games.sort_values("game_date").reset_index(drop=True)

    # --- Park factors ---
    park_df = build_park_factor_table(seasons)
    park_lookup = park_df.set_index(["season", "team"])["park_factor"].to_dict()

    games["park_factor"] = games.apply(
        lambda r: park_lookup.get((r["season"], r["home_team"]),
                  FALLBACK_PARK_FACTORS.get(r["home_team"], 100) / 100.0),
        axis=1,
    )

    # --- Pitcher quality ---
    from src.data.fetch_pitchers import get_pitcher_quality_features
    pitcher_quality = get_pitcher_quality_features(pitchers)
    # Pivot to lookup: (Season, Name) -> pitcher_quality score
    pq_lookup = pitcher_quality.set_index(["Season", "Name"])["pitcher_quality"].to_dict()
    # League-average fallback per season
    league_avg_pq = pitcher_quality.groupby("Season")["pitcher_quality"].median().to_dict()

    # NOTE: We don't have per-game starter assignment in the game logs from Baseball Reference
    # schedule_and_record. The 'Win'/'Loss' columns have starter names, but assignment is ambiguous.
    # We use a team-season average of their top-2 starters as a proxy for backtesting.
    # For live signals, the daily generator will use the announced starter directly.
    top2_starters = (
        pitcher_quality[pitcher_quality["GS"] >= 5]
        .sort_values("pitcher_quality")
        .groupby(["Season", "team"])
        .head(2)
        .groupby(["Season", "team"])["pitcher_quality"]
        .mean()
        .reset_index()
        .rename(columns={"pitcher_quality": "team_sp_quality"})
    )
    sp_lookup = top2_starters.set_index(["Season", "team"])["team_sp_quality"].to_dict()

    games["home_pitcher_quality"] = games.apply(
        lambda r: sp_lookup.get((r["season"], r["home_team"]),
                  league_avg_pq.get(r["season"], 4.0)),
        axis=1,
    )
    games["away_pitcher_quality"] = games.apply(
        lambda r: sp_lookup.get((r["season"], r["away_team"]),
                  league_avg_pq.get(r["season"], 4.0)),
        axis=1,
    )

    # --- Rolling offense rates ---
    offense = compute_rolling_offense(games)
    home_off = offense[offense["is_home"]].rename(columns={"team": "home_team", "offense_rate": "home_offense_rate"})
    away_off = offense[~offense["is_home"]].rename(columns={"team": "away_team", "offense_rate": "away_offense_rate"})

    games = games.merge(
        home_off[["game_date", "home_team", "home_offense_rate"]],
        on=["game_date", "home_team"], how="left"
    )
    games = games.merge(
        away_off[["game_date", "away_team", "away_offense_rate"]],
        on=["game_date", "away_team"], how="left"
    )

    # --- Weather ---
    if include_weather:
        weather_cache_path = PROCESSED_DIR / "weather_cache.parquet"
        if weather_cache_path.exists():
            weather_cache = pd.read_parquet(weather_cache_path)
            cached_keys = set(zip(weather_cache["game_date"].astype(str), weather_cache["home_team"]))
        else:
            weather_cache = pd.DataFrame()
            cached_keys = set()

        new_weather_rows = []
        for _, row in games.iterrows():
            date_str = str(row["game_date"])[:10]
            key = (date_str, row["home_team"])
            if key not in cached_keys:
                weather = fetch_historical_weather(row["home_team"], date_str)
                if weather:
                    new_weather_rows.append({
                        "game_date": row["game_date"],
                        "home_team": row["home_team"],
                        **weather,
                    })

        if new_weather_rows:
            new_df = pd.DataFrame(new_weather_rows)
            weather_cache = pd.concat([weather_cache, new_df], ignore_index=True)
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            weather_cache.to_parquet(weather_cache_path, index=False)
            print(f"  Added {len(new_weather_rows)} weather records")

        if not weather_cache.empty:
            games = games.merge(
                weather_cache[["game_date", "home_team", "wind_effect", "temperature_f", "is_dome"]],
                on=["game_date", "home_team"], how="left"
            )
        else:
            games["wind_effect"] = 0.0
            games["temperature_f"] = 70.0
            games["is_dome"] = games["home_team"].apply(
                lambda t: STADIUMS.get(t, {}).get("dome", False)
            )
    else:
        games["wind_effect"] = 0.0
        games["temperature_f"] = 70.0
        games["is_dome"] = games["home_team"].apply(
            lambda t: STADIUMS.get(t, {}).get("dome", False)
        )

    # Fill remaining NaNs with sensible defaults
    games["wind_effect"] = games["wind_effect"].fillna(0.0)
    games["temperature_f"] = games["temperature_f"].fillna(70.0)
    games["is_dome"] = games["is_dome"].fillna(False).astype(int)

    # Drop rows with missing offense rates (early season without enough history)
    games = games.dropna(subset=["home_offense_rate", "away_offense_rate", "total_runs"])

    feature_cols = [
        "game_date", "season", "home_team", "away_team",
        "home_offense_rate", "away_offense_rate",
        "home_pitcher_quality", "away_pitcher_quality",
        "park_factor", "wind_effect", "temperature_f", "is_dome",
        "total_runs", "home_runs", "away_runs",
    ]
    result = games[[c for c in feature_cols if c in games.columns]].copy()
    result = result.sort_values("game_date").reset_index(drop=True)

    # Dedup: doubleheader games with identical scores produce multiple rows from the
    # rolling-offense merge. Keep one row per (date, teams, score).
    before = len(result)
    result = result.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "home_runs", "away_runs"]
    ).reset_index(drop=True)
    if len(result) < before:
        print(f"  Removed {before - len(result)} duplicate rows (doubleheader same-score collisions)")

    print(f"Built feature table: {len(result)} games, {result['season'].nunique()} seasons")
    return result


if __name__ == "__main__":
    from src.data.fetch_games import fetch_and_save_seasons
    from src.data.fetch_pitchers import fetch_and_save_pitcher_seasons
    from config import TRAIN_SEASONS, TEST_SEASONS

    seasons = TRAIN_SEASONS + TEST_SEASONS
    print("Loading game data...")
    games = pd.read_parquet(RAW_DIR / "games_all.parquet") if (RAW_DIR / "games_all.parquet").exists() \
        else fetch_and_save_seasons(seasons)

    print("Loading pitcher data...")
    pitchers = pd.read_parquet(RAW_DIR / "pitchers_all.parquet") if (RAW_DIR / "pitchers_all.parquet").exists() \
        else fetch_and_save_pitcher_seasons(seasons)

    print("Building features (this will fetch weather for all games — slow on first run)...")
    features = build_feature_table(games, pitchers, seasons, include_weather=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(features.describe())
