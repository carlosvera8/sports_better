"""
Main pipeline runner. Runs steps in order:
  1. fetch    — Download game logs and pitcher stats
  2. features — Build feature table
  3. train    — Fit the Poisson model
  4. backtest — Run walk-forward backtest + print metrics
  5. picks    — Generate today's bet recommendations (requires API keys)

Usage:
  python main.py fetch
  python main.py features
  python main.py train
  python main.py backtest
  python main.py picks [--date YYYY-MM-DD]
  python main.py all       (runs fetch → features → train → backtest)
"""

import argparse
import sys
from pathlib import Path
from config import TRAIN_SEASONS, TEST_SEASONS, RAW_DIR, PROCESSED_DIR


def cmd_fetch(args):
    from src.data.fetch_games import fetch_and_save_seasons
    from src.data.fetch_pitchers import fetch_and_save_pitcher_seasons
    seasons = TRAIN_SEASONS + TEST_SEASONS
    print(f"Fetching game logs for seasons {seasons[0]}–{seasons[-1]}...")
    fetch_and_save_seasons(seasons)
    print(f"\nFetching pitcher stats...")
    fetch_and_save_pitcher_seasons(seasons)
    print("\nDone. Data saved to data/raw/")


def cmd_features(args):
    import pandas as pd
    from src.features.build_features import build_feature_table
    seasons = TRAIN_SEASONS + TEST_SEASONS

    print("Loading raw data...")
    games = pd.read_parquet(RAW_DIR / "games_all.parquet")
    pitchers = pd.read_parquet(RAW_DIR / "pitchers_all.parquet")

    include_weather = not args.no_weather
    if include_weather:
        print("Weather fetching enabled (slow on first run — fetches each game individually).")
        print("Use --no-weather to skip and run much faster.")
    else:
        print("Weather disabled (--no-weather flag set).")

    features = build_feature_table(games, pitchers, seasons, include_weather=include_weather)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print(f"\nFeature table saved: {len(features)} rows to data/processed/features.parquet")


def cmd_train(args):
    import pandas as pd
    from src.models.poisson_model import PoissonTotalsModel
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train = features[features["season"].isin(TRAIN_SEASONS)]
    print(f"Training on {len(train)} games ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})...")
    model = PoissonTotalsModel()
    model.fit(train)
    model.save()
    print("Model saved.")


def cmd_backtest(args):
    import pandas as pd
    from src.backtest.walk_forward import run_walk_forward_backtest, compute_backtest_metrics, print_backtest_report
    from src.backtest.metrics import full_validation_report
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    historical_odds = pd.DataFrame()
    if args.real_odds:
        from src.data.fetch_historical_odds import load_historical_odds_for_seasons
        print("Loading real historical odds...")
        historical_odds = load_historical_odds_for_seasons(TEST_SEASONS)
        if historical_odds.empty:
            print("No historical odds available — falling back to synthetic lines.")
            print("To enable: subscribe to The Odds API Starter plan (~$19/month)")
            print("           or run: python main.py import-odds --csv <path>")

    print("Running walk-forward backtest...")
    bets = run_walk_forward_backtest(features, historical_odds=historical_odds)
    if not bets.empty:
        metrics = compute_backtest_metrics(bets)
        print_backtest_report(metrics)
        full_validation_report(bets)
        bets.to_parquet(PROCESSED_DIR / "backtest_results.parquet", index=False)
    else:
        print("No bets met the EV threshold — try lowering MIN_EV_THRESHOLD in config.py")


def cmd_import_odds(args):
    """Imports historical odds from a CSV file and caches as parquet."""
    import pandas as pd
    from src.data.fetch_historical_odds import load_odds_from_csv, HISTORICAL_ODDS_DIR

    HISTORICAL_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    odds = load_odds_from_csv(args.csv)
    if odds.empty:
        print("No odds loaded — check CSV format.")
        return

    # Split by season and cache each year
    odds["season"] = odds["game_date"].dt.year
    for season, group in odds.groupby("season"):
        out = HISTORICAL_ODDS_DIR / f"odds_{season}.parquet"
        group.drop(columns=["season"]).to_parquet(out, index=False)
        print(f"  Cached {len(group)} rows → {out.name}")
    print(f"Done. Run 'python main.py backtest --real-odds' to use these lines.")


def cmd_picks(args):
    from src.signals.daily_picks import run_daily_picks
    run_daily_picks(target_date=args.date, bankroll=args.bankroll)


def main():
    parser = argparse.ArgumentParser(description="MLB Totals Betting System")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("fetch", help="Download historical game logs and pitcher stats")

    feat_p = sub.add_parser("features", help="Build feature table from raw data")
    feat_p.add_argument("--no-weather", action="store_true",
                        help="Skip weather fetching (much faster, less accurate)")

    sub.add_parser("train", help="Train the Poisson GLM on training seasons")

    bt_p = sub.add_parser("backtest", help="Run walk-forward backtest")
    bt_p.add_argument(
        "--real-odds", action="store_true",
        help="Use real historical closing lines (requires The Odds API paid plan or prior import-odds run)"
    )

    import_p = sub.add_parser("import-odds", help="Import historical odds from a CSV file")
    import_p.add_argument("--csv", required=True, help="Path to CSV with historical odds")

    picks_p = sub.add_parser("picks", help="Generate today's bet recommendations")
    picks_p.add_argument("--date", type=str, default=None, help="Date: YYYY-MM-DD (default: today)")
    picks_p.add_argument("--bankroll", type=float, default=1000.0, help="Current bankroll")

    all_p = sub.add_parser("all", help="Run full pipeline: fetch → features → train → backtest")
    all_p.add_argument("--no-weather", action="store_true")

    args = parser.parse_args()

    dispatch = {
        "fetch": cmd_fetch,
        "features": cmd_features,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "picks": cmd_picks,
        "import-odds": cmd_import_odds,
    }

    if args.command == "all":
        cmd_fetch(args)
        cmd_features(args)
        cmd_train(args)
        cmd_backtest(args)
    elif args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
