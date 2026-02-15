"""
Phase 1 (Enhanced): Feature Engineering

Goal: Transform raw TMDB data into high-value ML features.

Why Feature Engineering Matters:
    Raw data (like a JSON list of actors) can't be fed directly into a model.
    We need to convert domain knowledge into numerical signals. For example,
    instead of feeding "Christopher Nolan" as text, we feed his AVERAGE
    HISTORICAL REVENUE ($240M) — a number the model can learn from.

    This module engineers 12+ features across 5 categories:
    1. Temporal   — release timing patterns (summer, holidays)
    2. Personnel  — director track record, actor star power
    3. Financial  — budget, vote metrics
    4. Production — studio tier (Major/Mid/Indie)
    5. Content    — genre encoding, runtime

Methods Used:
    - pd.to_datetime: Converts date strings to datetime objects for extraction
    - groupby().agg(): Calculates aggregate statistics per group (e.g., avg revenue per director)
    - pd.get_dummies: One-hot encodes categorical variables into binary columns
    - .map(): Maps a lookup dictionary to a Series (fast vectorized operation)
    - np.log1p: Log-transform for skewed distributions (revenue, budget)
"""

import pandas as pd
import numpy as np


def _add_temporal_features(df):
    """
    Extract time-based features from release_date.

    Why: Movie release timing is a strategic business decision:
        - Summer blockbusters (May-Aug) benefit from school holidays globally
        - Holiday releases (Nov-Dec) ride the Christmas/awards season wave
        - January ("dump month") is where studios release expected flops

    Method: pd.to_datetime() converts strings like '2009-12-10' into datetime
    objects, then .dt.month extracts the integer month (1-12). We then create
    binary flags for known high-earning windows.

    Features Created:
        - release_month (int 1-12): Month of release
        - is_summer_release (bool): May through August
        - is_holiday_release (bool): November or December
        - release_quarter (int 1-4): Fiscal quarter

    Args:
        df (pd.DataFrame): DataFrame with 'release_date' column

    Returns:
        pd.DataFrame: DataFrame with added temporal columns
    """
    if 'release_date' not in df.columns:
        return df

    df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date_parsed'].dt.month.fillna(0).astype(int)
    df['release_year'] = df['release_date_parsed'].dt.year.fillna(0).astype(int)
    df['is_summer_release'] = df['release_month'].isin([5, 6, 7, 8]).astype(int)
    df['is_holiday_release'] = df['release_month'].isin([11, 12]).astype(int)
    df['release_quarter'] = ((df['release_month'] - 1) // 3 + 1).clip(1, 4)

    # Drop the intermediate parsed column
    df = df.drop(columns=['release_date_parsed'], errors='ignore')

    return df


def _add_director_track_record(df):
    """
    Calculate each director's historical average revenue.

    Why: A director's past performance is one of the strongest predictors
    of future revenue. Christopher Nolan's average gross is ~$500M; a first-time
    director averages ~$10M. This encoding captures that signal.

    Method:
        1. groupby('director')['Revenue_Millions'].mean() — calculates average
           revenue for each director across all their movies in the dataset
        2. .map(director_avg) — maps each movie's director to their average
        3. fillna(global_median) — for directors with only one movie or
           unknown directors, use the dataset median (prevents leakage)

    Why we use MEDIAN for unknown directors (not mean):
        Revenue is heavily right-skewed (a few movies earn billions while
        most earn under $100M). The median is more robust to outliers.

    Args:
        df (pd.DataFrame): DataFrame with 'director' and 'Revenue_Millions'

    Returns:
        pd.DataFrame: DataFrame with 'director_avg_revenue' column
    """
    if 'director' not in df.columns:
        df['director_avg_revenue'] = df['Revenue_Millions'].median()
        return df

    # Calculate mean revenue per director
    director_revenue = df.groupby('director')['Revenue_Millions'].mean()
    global_median = df['Revenue_Millions'].median()

    # Map back to each movie
    df['director_avg_revenue'] = df['director'].map(director_revenue)
    df['director_avg_revenue'] = df['director_avg_revenue'].fillna(global_median)

    # Also count how many movies each director has (experience proxy)
    director_count = df.groupby('director')['title'].count()
    df['director_movie_count'] = df['director'].map(director_count).fillna(1)

    return df


def _add_actor_star_power(df):
    """
    Calculate lead actor's average historical revenue.

    Why: Star power sells tickets. Will Smith in a movie changes its
    revenue trajectory compared to an unknown actor. We quantify this
    by calculating each actor's average box office across the dataset.

    Method:
        1. Extract lead_actor (already done in tmdb_loader)
        2. groupby('lead_actor').Revenue_Millions.mean() — average per actor
        3. .map() — fast lookup to assign star power score to each movie
        4. fillna(median) — unknown actors get the dataset median

    Args:
        df (pd.DataFrame): DataFrame with 'lead_actor' and 'Revenue_Millions'

    Returns:
        pd.DataFrame: DataFrame with 'lead_actor_avg_revenue' column
    """
    if 'lead_actor' not in df.columns:
        df['lead_actor_avg_revenue'] = df['Revenue_Millions'].median()
        return df

    actor_revenue = df.groupby('lead_actor')['Revenue_Millions'].mean()
    global_median = df['Revenue_Millions'].median()

    df['lead_actor_avg_revenue'] = df['lead_actor'].map(actor_revenue)
    df['lead_actor_avg_revenue'] = df['lead_actor_avg_revenue'].fillna(global_median)

    return df


def _add_genre_features(df):
    """
    Multi-label one-hot encode ALL genres for each movie.

    Why: The old code only used the FIRST genre (e.g., "Action" for an
    "Action, Adventure, Sci-Fi" movie). This lost information. A movie
    tagged "Action + Comedy" earns differently than pure "Action".

    Method:
        1. df['genres_list'].explode() — transforms [[Action, Comedy], [Drama]]
           into [Action, Comedy, Drama] (one row per genre-movie pair)
        2. pd.get_dummies() — creates binary columns: Genre_Action, Genre_Comedy, etc.
        3. groupby().max() — recombines: each movie gets 1 in ALL its genre columns

    Why explode + groupby instead of a loop:
        Vectorized pandas operations are 100x faster than Python loops on
        large datasets. For 4000 movies × 20 genres, this matters.

    Also creates:
        - num_genres: Count of genres per movie (wider appeal indicator)

    Args:
        df (pd.DataFrame): DataFrame with 'genres_list' column

    Returns:
        pd.DataFrame: DataFrame with Genre_* one-hot columns + num_genres
    """
    if 'genres_list' not in df.columns:
        return df

    # Count of genres per movie
    df['num_genres'] = df['genres_list'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Explode genres and create dummies
    genres_exploded = df[['genres_list']].explode('genres_list')
    genres_exploded = genres_exploded.dropna(subset=['genres_list'])

    if len(genres_exploded) > 0:
        genre_dummies = pd.get_dummies(genres_exploded['genres_list'], prefix='Genre')
        # Group by original index and take max (1 if movie has that genre, 0 otherwise)
        genre_dummies = genre_dummies.groupby(genre_dummies.index).max()
        # Join back to original df
        df = df.join(genre_dummies, how='left')
        # Fill any NaN values (movies with no genres) with 0
        genre_cols = [c for c in df.columns if c.startswith('Genre_')]
        df[genre_cols] = df[genre_cols].fillna(0).astype(int)

    return df


def _add_studio_features(df):
    """
    One-hot encode the studio_tier classification.

    Why: Major studios (Disney, Warner) have fundamentally different
    economics — massive marketing budgets, global distribution, franchise
    IP. This structural advantage should be captured as features.

    Method: pd.get_dummies() — creates binary columns:
        Studio_Major, Studio_Mid (Indie is the dropped reference category,
        via drop_first=True, to avoid multicollinearity).

    Why drop_first=True:
        If a movie is NOT Major and NOT Mid, it MUST be Indie. Including
        all three creates perfect multicollinearity (they always sum to 1),
        which breaks Linear Regression's math (singular matrix).

    Args:
        df (pd.DataFrame): DataFrame with 'studio_tier' column

    Returns:
        pd.DataFrame: DataFrame with Studio_* one-hot columns
    """
    if 'studio_tier' not in df.columns:
        return df

    studio_dummies = pd.get_dummies(df['studio_tier'], prefix='Studio', drop_first=True)
    df = pd.concat([df, studio_dummies], axis=1)

    return df


def _add_financial_features(df):
    """
    Create derived financial features.

    Why: Raw budget in millions is useful, but log-transformed budget
    captures the DIMINISHING RETURNS of spending. Going from $10M to $20M
    doubles your production value; going from $200M to $210M barely changes
    anything. Log-transform captures this non-linear relationship.

    Method: np.log1p(x) = ln(x + 1)
        The +1 prevents log(0) = -infinity for ultra-low-budget films.
        This is a standard data science technique for right-skewed distributions.

    Features Created:
        - log_budget: Logarithm of budget (captures diminishing returns)
        - budget_to_vote_ratio: Budget per vote count (spending efficiency)

    Args:
        df (pd.DataFrame): DataFrame with Budget_Millions, vote_average, vote_count

    Returns:
        pd.DataFrame: DataFrame with derived financial columns
    """
    if 'Budget_Millions' in df.columns:
        df['log_budget'] = np.log1p(df['Budget_Millions'])

    if 'vote_count' in df.columns and 'Budget_Millions' in df.columns:
        # Budget efficiency: how much is spent per unit of audience engagement
        df['budget_per_vote'] = df['Budget_Millions'] / (df['vote_count'] + 1)

    return df


def engineer_features(df):
    """
    Master orchestrator: applies all feature engineering steps.

    This function calls each sub-function in the correct order:
    1. Temporal → depends only on release_date
    2. Director track record → depends on Revenue_Millions
    3. Actor star power → depends on Revenue_Millions
    4. Genre encoding → depends on genres_list
    5. Studio encoding → depends on studio_tier
    6. Financial derivations → depends on Budget_Millions, vote data

    The ORDER matters because some features rely on others being present
    (e.g., director track record needs Revenue_Millions to exist first).

    Args:
        df (pd.DataFrame): Raw DataFrame from load_tmdb_data()

    Returns:
        pd.DataFrame: Feature-enriched DataFrame ready for ML modeling
        list: Names of all feature columns created
    """
    print("      Engineering features...")
    initial_cols = set(df.columns)

    df = _add_temporal_features(df)
    df = _add_director_track_record(df)
    df = _add_actor_star_power(df)
    df = _add_genre_features(df)
    df = _add_studio_features(df)
    df = _add_financial_features(df)

    new_cols = set(df.columns) - initial_cols
    print(f"      Created {len(new_cols)} new features: {sorted(new_cols)}")

    # Define the feature columns for modeling
    # (everything we engineered, excluding raw text, lists, and target variables)
    exclude_cols = {
        'Revenue_Millions', 'revenue', 'title', 'id', 'homepage', 'tagline',
        'overview', 'User_Review', 'original_title', 'original_language',
        'status', 'release_date', 'spoken_languages', 'production_countries',
        'keywords', 'director', 'lead_actor', 'top_cast', 'genres_list',
        'Genre', 'studio_tier', 'Sentiment_Category',
        'budget',  # we use Budget_Millions instead
    }

    feature_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and not col.startswith('Sentiment_')
                    # keep Sentiment_Score but drop Sentiment_Positive etc
                    or col == 'Sentiment_Score']

    # Ensure all feature columns are numeric
    for col in feature_cols:
        if df[col].dtype == 'object':
            feature_cols.remove(col)

    print(f"      Final feature set ({len(feature_cols)} features): {sorted(feature_cols)}")

    return df, feature_cols


# ──────────────────────────────────────────────────────────────────
#  Standalone Test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase1_tmdb_loader import load_tmdb_data

    print("=" * 60)
    print("  Feature Engineering — Standalone Test")
    print("=" * 60)

    df = load_tmdb_data()
    df, feature_cols = engineer_features(df)

    print(f"\n--- Feature Summary ---")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nFeature list:")
    for col in sorted(feature_cols):
        print(f"  • {col}: {df[col].dtype} | mean={df[col].mean():.2f} | nulls={df[col].isnull().sum()}")
