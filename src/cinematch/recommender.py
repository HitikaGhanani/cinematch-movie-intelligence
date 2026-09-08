from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import prepare_recommendation_frame


@dataclass
class CineMatchRecommender:
    """Content-based movie retrieval with configurable metadata components."""

    max_features: int = 20_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    items: pd.DataFrame | None = field(default=None, init=False)
    feature_matrix: object | None = field(default=None, init=False)

    def fit(self, movies: pd.DataFrame) -> "CineMatchRecommender":
        self.items = prepare_recommendation_frame(movies)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        self.feature_matrix = self.vectorizer.fit_transform(self.items["metadata"])
        return self

    def _require_fit(self) -> None:
        if self.items is None or self.feature_matrix is None:
            raise RuntimeError("Fit the recommender before requesting recommendations.")

    def resolve_title(self, title: str) -> str:
        self._require_fit()
        assert self.items is not None
        normalized = title.casefold().strip()
        lookup = {value.casefold(): value for value in self.items["title"].astype(str)}
        if normalized in lookup:
            return lookup[normalized]
        matches = get_close_matches(normalized, list(lookup), n=1, cutoff=0.65)
        if matches:
            return lookup[matches[0]]
        raise ValueError(f"No movie title matched '{title}'. Try a more specific title.")

    def recommend(
        self,
        title: str,
        top_k: int = 10,
        min_vote_count: int = 0,
        diversify: bool = True,
        candidate_multiplier: int = 8,
    ) -> pd.DataFrame:
        """Return similar movies with optional quality filtering and genre-aware diversity.

        Diversity is a lightweight greedy re-ranker that penalizes candidates with an
        identical genre signature to already-selected recommendations.
        """
        self._require_fit()
        assert self.items is not None and self.feature_matrix is not None
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        resolved_title = self.resolve_title(title)
        source_idx = self.items.index[self.items["title"] == resolved_title][0]
        scores = cosine_similarity(self.feature_matrix[source_idx], self.feature_matrix).ravel()
        ordered = [idx for idx in np.argsort(scores)[::-1] if idx != source_idx]
        if min_vote_count > 0:
            ordered = [idx for idx in ordered if self.items.loc[idx, "vote_count"] >= min_vote_count]
        candidates = ordered[: max(top_k * candidate_multiplier, top_k)]

        selected: list[int] = []
        selected_signatures: set[tuple[str, ...]] = set()
        for idx in candidates:
            signature = tuple(sorted(self.items.loc[idx, "genres_list"]))
            if diversify and signature in selected_signatures and len(candidates) > top_k:
                continue
            selected.append(idx)
            selected_signatures.add(signature)
            if len(selected) == top_k:
                break
        if len(selected) < top_k:
            selected.extend(idx for idx in candidates if idx not in selected)
            selected = selected[:top_k]

        columns = ["title", "release_date", "vote_average", "vote_count", "popularity", "genres_list"]
        result = self.items.loc[selected, columns].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result.insert(1, "similarity", scores[selected].round(4))
        result = result.rename(columns={"genres_list": "genres"})
        result["genres"] = result["genres"].apply(lambda genres: ", ".join(genres))
        return result.reset_index(drop=True)

    def save(self, path: str) -> None:
        self._require_fit()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "CineMatchRecommender":
        return joblib.load(path)
