from __future__ import annotations

import pandas as pd

from .recommender import CineMatchRecommender


def genre_overlap_evaluation(
    model: CineMatchRecommender,
    top_k: int = 10,
    sample_size: int | None = 300,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estimate broad content consistency using shared genre as a relevance proxy."""
    model._require_fit()
    assert model.items is not None
    queries = model.items.dropna(subset=["title"])
    if sample_size and len(queries) > sample_size:
        queries = queries.sample(sample_size, random_state=random_state)

    relevant = 0
    total = 0
    for _, query in queries.iterrows():
        recommendations = model.recommend(str(query["title"]), top_k=top_k, diversify=False)
        query_genres = set(query["genres_list"])
        for _, candidate in recommendations.iterrows():
            candidate_genres = set(candidate["genres"].split(", ")) if candidate["genres"] else set()
            relevant += int(bool(query_genres & candidate_genres))
            total += 1

    return {
        "queries_evaluated": int(len(queries)),
        "top_k": int(top_k),
        "recommendations_evaluated": int(total),
        "genre_overlap_precision_at_k": round(relevant / total if total else 0.0, 4),
    }
