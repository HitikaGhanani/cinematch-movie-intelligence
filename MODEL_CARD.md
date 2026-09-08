# CineMatch Model Card

## Model details

CineMatch uses a content-based retrieval model. Each movie is represented by a TF-IDF vector built from its genres, keywords, overview, top-billed cast, and director. Recommendations are ranked by cosine similarity.

The repository also includes an optional logistic-regression baseline for retrospective historical-performance analysis.

## Intended use

- Demonstrating metadata cleaning, text feature engineering, and nearest-neighbor retrieval.
- Helping users explore movies with related content characteristics.
- Demonstrating careful baseline construction and evaluation practices in a portfolio setting.

## Out of scope

- Personalized recommendations based on individual taste.
- Investment, production, or box-office decisions.
- Pre-release success prediction.
- Claims of causal relationships between metadata and commercial outcomes.

## Data

The project expects the TMDB 5000 Movie Dataset. The data are not included in this repository. Source records can contain missing, zero, stale, or inconsistent values.

## Recommendation evaluation

The included offline metric is genre-overlap precision@K: a recommendation is considered relevant if it shares at least one genre with the query title. This tests broad content consistency only. It does not validate user satisfaction, personalization, novelty, diversity, fairness, or commercial usefulness.

## Historical baseline design

The performance baseline defines a historical label using revenue and budget, then excludes final revenue, final popularity, vote count, and vote average from the model inputs. It uses a chronological holdout when release dates allow it; this better resembles future deployment than a purely random split. Its results remain retrospective and should not be treated as real-world forecasting performance.

## Known limitations

- TF-IDF underrepresents semantic similarity when titles use different vocabulary.
- Popular metadata patterns can dominate retrieval.
- Duplicate titles and remakes can create ambiguous lookup behavior.
- Genre overlap favors broad genre consistency and can overstate retrieval quality.
- The data catalog is limited and not necessarily current.

## Monitoring and future work

Future versions should evaluate diversity, catalog coverage, NDCG with relevance labels, user satisfaction, and performance by genre/language. A deployed version should log only consented, privacy-preserving interaction data.
