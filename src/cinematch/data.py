from __future__ import annotations

from ast import literal_eval
from typing import Iterable

import pandas as pd


def parse_json_list(value: object) -> list[dict]:
    """Safely convert one TMDB JSON-like cell into a list of dictionaries."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def extract_names(value: object, limit: int | None = None) -> list[str]:
    names = [str(item.get("name", "")).strip() for item in parse_json_list(value) if isinstance(item, dict)]
    names = [name for name in names if name]
    return names[:limit] if limit else names


def extract_director(value: object) -> str:
    for person in parse_json_list(value):
        if isinstance(person, dict) and person.get("job") == "Director":
            return str(person.get("name", "")).strip()
    return ""


def normalize_tokens(tokens: Iterable[str]) -> str:
    return " ".join(str(token).replace(" ", "").strip() for token in tokens if str(token).strip())


def load_tmdb_data(movies_path: str, credits_path: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path).rename(columns={"movie_id": "id"})
    merged = movies.merge(credits[["id", "cast", "crew"]], on="id", how="inner")
    return merged.dropna(subset=["id", "title"]).drop_duplicates(subset=["id"]).reset_index(drop=True)


def prepare_recommendation_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["overview"] = frame["overview"].fillna("")
    frame["genres_list"] = frame["genres"].apply(extract_names)
    frame["keywords_list"] = frame["keywords"].apply(extract_names)
    frame["cast_list"] = frame["cast"].apply(lambda value: extract_names(value, limit=3))
    frame["director"] = frame["crew"].apply(extract_director)
    frame["metadata"] = (
        frame["genres_list"].apply(normalize_tokens)
        + " " + frame["keywords_list"].apply(normalize_tokens)
        + " " + frame["cast_list"].apply(normalize_tokens)
        + " " + frame["director"].astype(str).str.replace(" ", "", regex=False)
        + " " + frame["overview"].astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return frame[frame["metadata"].str.len() > 0].reset_index(drop=True)
