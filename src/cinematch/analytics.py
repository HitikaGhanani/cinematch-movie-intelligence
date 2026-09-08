from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import extract_names


FEATURE_COLUMNS = [
    "budget",
    "runtime",
    "release_year",
    "genre_count",
    "keyword_count",
    "primary_genre",
    "original_language",
]
POST_RELEASE_EXCLUSIONS = ["revenue", "popularity", "vote_count", "vote_average"]


def prepare_historical_performance_data(movies: pd.DataFrame) -> pd.DataFrame:
    """Prepare a documented retrospective dataset without post-release model inputs.

    Revenue creates the historical outcome label but is never passed into the model.
    """
    frame = movies.copy()
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame["release_year"] = frame["release_date"].dt.year
    frame["runtime"] = pd.to_numeric(frame["runtime"], errors="coerce")
    frame["budget"] = pd.to_numeric(frame["budget"], errors="coerce")
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce")
    frame["genre_count"] = frame["genres"].apply(lambda value: len(extract_names(value)))
    frame["keyword_count"] = frame["keywords"].apply(lambda value: len(extract_names(value)))
    frame["primary_genre"] = frame["genres"].apply(
        lambda value: (extract_names(value) or ["Unknown"])[0]
    )
    frame = frame[(frame["budget"] > 0) & (frame["revenue"] > 0)].copy()
    frame = frame.dropna(subset=["release_year"])
    frame["profitable"] = (frame["revenue"] >= frame["budget"]).astype(int)
    return frame.reset_index(drop=True)


def _build_pipeline() -> Pipeline:
    numeric = ["budget", "runtime", "release_year", "genre_count", "keyword_count"]
    categorical = ["primary_genre", "original_language"]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the most recent records for testing to reduce temporal leakage."""
    ordered = frame.sort_values(["release_year", "id"], kind="stable").reset_index(drop=True)
    cutoff = max(1, int(len(ordered) * (1 - test_fraction)))
    return ordered.iloc[:cutoff].copy(), ordered.iloc[cutoff:].copy()


def train_historical_baseline(
    movies: pd.DataFrame,
    random_state: int = 42,
    split_strategy: str = "chronological",
) -> tuple[Pipeline, dict[str, float | int | str]]:
    """Fit a leakage-aware retrospective baseline and report holdout performance."""
    frame = prepare_historical_performance_data(movies)
    if split_strategy == "chronological":
        train_frame, test_frame = chronological_split(frame)
    elif split_strategy == "random":
        train_frame, test_frame = train_test_split(
            frame,
            test_size=0.2,
            random_state=random_state,
            stratify=frame["profitable"],
        )
    else:
        raise ValueError("split_strategy must be 'chronological' or 'random'.")

    X_train, y_train = train_frame[FEATURE_COLUMNS], train_frame["profitable"]
    X_test, y_test = test_frame[FEATURE_COLUMNS], test_frame["profitable"]
    search = GridSearchCV(
        _build_pipeline(),
        {"model__C": [0.1, 1.0, 5.0, 10.0]},
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    probabilities = search.predict_proba(X_test)[:, 1]

    metrics: dict[str, float | int | str] = {
        "split_strategy": split_strategy,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_end_year": int(train_frame["release_year"].max()),
        "test_start_year": int(test_frame["release_year"].min()),
        "best_C": float(search.best_params_["model__C"]),
    }
    return search.best_estimator_, metrics
