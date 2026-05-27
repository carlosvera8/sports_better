import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ODDS_DIR = DATA_DIR / "odds"

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")

# Seasons to use for training/testing.
# 2020 excluded: COVID-shortened 60-game season played in empty stadiums.
# It's a structural outlier (no home field effect, compressed rosters) that hurts model generalization.
TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021]
TEST_SEASONS = [2022, 2023, 2024]

# Minimum EV threshold to flag a bet (3%)
MIN_EV_THRESHOLD = 0.03

# Kelly fraction — full Kelly is aggressive; 0.25 is quarter-Kelly (recommended)
KELLY_FRACTION = 0.25

# Sportsbook vig (typical -110 on totals = 4.55% vig)
STANDARD_VIG = -110
