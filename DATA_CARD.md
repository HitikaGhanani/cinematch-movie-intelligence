# Data Card

## Dataset

CineMatch expects the TMDB 5000 Movie Dataset:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

The project does not redistribute these files. Download them from the original source and review its license and terms before use.

## Fields used

### Recommendation engine

- Title and identifier
- Genres and keywords
- Plot overview
- Cast and crew metadata
- Release date and selected display metadata

### Historical analysis

- Budget, runtime, release date, genre/keyword counts
- Primary genre and original language
- Revenue only for constructing a retrospective outcome label

## Data quality considerations

- Financial values may be missing or recorded as zero.
- Metadata completeness varies by title, language, and era.
- Cast and crew data can be incomplete.
- The catalog is a snapshot and may not represent current movie production or viewing behavior.

## Privacy and sensitive data

The dataset contains public movie metadata, not user-level behavior. Do not add personally identifiable user ratings, watch histories, or credentials to the repository.
