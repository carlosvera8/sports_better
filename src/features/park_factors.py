"""
Park run factors sourced from FanGraphs via pybaseball.
A park factor of 105 means 5% more runs scored at that park vs. a neutral one.
Used to adjust team offensive rates before modeling.
"""

import pandas as pd
import pybaseball as pb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_DIR

# Multi-year average park factors (FanGraphs, basic runs factor, 100 = neutral)
# Refreshed periodically; hardcoded here as a stable fallback when pybaseball scraping fails.
# Source: FanGraphs Park Factors, 3-year rolling average circa 2023.
FALLBACK_PARK_FACTORS = {
    "ARI": 104, "ATL": 97,  "BAL": 102, "BOS": 104, "CHC": 102,
    "CHW": 98,  "CIN": 105, "CLE": 97,  "COL": 119, "DET": 96,
    "HOU": 97,  "KCR": 97,  "LAA": 98,  "LAD": 97,  "MIA": 93,
    "MIL": 97,  "MIN": 101, "NYM": 98,  "NYY": 105, "OAK": 96,
    "PHI": 101, "PIT": 98,  "SDP": 95,  "SFG": 93,  "SEA": 96,
    "STL": 98,  "TBR": 96,  "TEX": 104, "TOR": 102, "WSN": 100,
}

# Map FanGraphs team names to Baseball Reference abbreviations
FANGRAPHS_TO_BBR = {
    "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS",
    "Cubs": "CHC", "White Sox": "CHW", "Reds": "CIN", "Guardians": "CLE",
    "Indians": "CLE", "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KCR", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA",
    "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY",
    "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SDP",
    "Giants": "SFG", "Mariners": "SEA", "Cardinals": "STL", "Rays": "TBR",
    "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSN",
}


def fetch_park_factors(season: int) -> dict[str, float]:
    """
    Returns a dict mapping team abbreviation -> park run factor (100 = neutral).
    Tries pybaseball first, falls back to hardcoded values.
    """
    try:
        df = pb.park_factors(season)
        # pybaseball returns a DataFrame with 'Team' and 'Basic' (runs factor) columns
        pf = {}
        for _, row in df.iterrows():
            team_name = str(row.get("Team", ""))
            abbr = FANGRAPHS_TO_BBR.get(team_name)
            if abbr:
                pf[abbr] = float(row.get("Basic", 100))
        if pf:
            return pf
    except Exception as e:
        print(f"  Park factors fetch failed for {season}: {e}, using fallback")

    return FALLBACK_PARK_FACTORS.copy()


def get_park_factor(team: str, season: int, pf_cache: dict[int, dict] = None) -> float:
    """Returns park factor for a given team and season, normalized to 1.0 (not 100)."""
    if pf_cache and season in pf_cache:
        pf = pf_cache[season].get(team, 100)
    else:
        pf = FALLBACK_PARK_FACTORS.get(team, 100)
    return pf / 100.0


def build_park_factor_table(seasons: list[int]) -> pd.DataFrame:
    """
    Returns a tidy DataFrame: (season, team, park_factor_normalized)
    park_factor_normalized: 1.0 = neutral, 1.05 = 5% more runs
    """
    rows = []
    for season in seasons:
        pf = fetch_park_factors(season)
        for team, factor in pf.items():
            rows.append({"season": season, "team": team, "park_factor": factor / 100.0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pf = fetch_park_factors(2023)
    print("2023 Park Factors:")
    for team, factor in sorted(pf.items(), key=lambda x: -x[1]):
        print(f"  {team}: {factor:.0f}")
