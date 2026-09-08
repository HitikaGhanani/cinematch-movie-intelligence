from __future__ import annotations

import argparse
import json

from .analytics import train_historical_baseline
from .config import CREDITS_FILE, MOVIES_FILE
from .data import load_tmdb_data
from .evaluation import genre_overlap_evaluation
from .recommender import CineMatchRecommender


def load_movies():
    if not MOVIES_FILE.exists() or not CREDITS_FILE.exists():
        raise FileNotFoundError("Dataset files were not found. Add both TMDB CSV files to the data/ directory.")
    return load_tmdb_data(str(MOVIES_FILE), str(CREDITS_FILE))


def build_model() -> CineMatchRecommender:
    return CineMatchRecommender().fit(load_movies())


def main() -> None:
    parser = argparse.ArgumentParser(description="CineMatch movie discovery and analytics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    recommendation_parser = subparsers.add_parser("recommend", help="Recommend similar movies")
    recommendation_parser.add_argument("--title", required=True)
    recommendation_parser.add_argument("--top-k", type=int, default=10)
    recommendation_parser.add_argument("--min-vote-count", type=int, default=0)
    recommendation_parser.add_argument("--no-diversify", action="store_true")
    evaluation_parser = subparsers.add_parser("evaluate", help="Evaluate genre-overlap precision")
    evaluation_parser.add_argument("--top-k", type=int, default=10)
    evaluation_parser.add_argument("--sample-size", type=int, default=300)
    analysis_parser = subparsers.add_parser("analyze", help="Run leakage-aware historical baseline")
    analysis_parser.add_argument("--split", choices=["chronological", "random"], default="chronological")
    args = parser.parse_args()

    if args.command == "analyze":
        _, metrics = train_historical_baseline(load_movies(), split_strategy=args.split)
        print(json.dumps(metrics, indent=2))
        return

    model = build_model()
    if args.command == "recommend":
        results = model.recommend(
            args.title,
            top_k=args.top_k,
            min_vote_count=args.min_vote_count,
            diversify=not args.no_diversify,
        )
        print(results.to_string(index=False))
    elif args.command == "evaluate":
        print(json.dumps(genre_overlap_evaluation(model, top_k=args.top_k, sample_size=args.sample_size), indent=2))


if __name__ == "__main__":
    main()
