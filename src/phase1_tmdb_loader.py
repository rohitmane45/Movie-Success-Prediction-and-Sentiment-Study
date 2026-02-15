"""
Phase 1 (Enhanced): TMDB Real Data Loader

Goal: Replace the 200-sample synthetic dataset with real TMDB 5000 movies data.

Why This Matters:
    The project previously trained on synthetic (fake) data. This module loads
    the REAL TMDB 5000 dataset — 4803 movies with actual budgets, revenues,
    cast, crew, and audience ratings — transforming the project from a demo
    into genuine data science.

Data Sources:
    - tmdb_5000_movies.csv  (4803 rows × 20 columns: budget, revenue, genres, etc.)
    - tmdb_5000_credits.csv (4803 rows × 4 columns: movie_id, title, cast JSON, crew JSON)

Methods Used:
    - ast.literal_eval: Safely parses JSON-like strings into Python dicts/lists
      (safer than eval() which can execute arbitrary code)
    - pd.merge: SQL-style JOIN to combine movies with their cast/crew data
    - json.loads: Parses strictly valid JSON strings
"""

import pandas as pd
import numpy as np
import os
import ast
import json


def _parse_json_column(data_str):
    """
    Safely parse a JSON-like string column into a Python list of dicts.

    Why: TMDB stores genres, cast, crew, and production_companies as JSON
    strings inside CSV cells. For example, a genres cell looks like:
        '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    We need to convert this string into an actual Python list so we can
    extract individual genre names, director names, etc.

    Method: ast.literal_eval() — evaluates a string as a Python literal.
    It's SAFE (unlike eval()) because it only allows basic types (lists,
    dicts, strings, numbers) and cannot execute arbitrary code.

    Args:
        data_str: A JSON-like string from a CSV cell

    Returns:
        list: Parsed list of dictionaries, or empty list on failure
    """
    if pd.isna(data_str) or data_str == '' or data_str == '[]':
        return []
    try:
        return ast.literal_eval(data_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            return []


def _extract_director(crew_list):
    """
    Find the director's name from the crew JSON list.

    Why: The director is one of the strongest predictors of box office success.
    A Christopher Nolan film will earn differently than a first-time director's
    film. The crew JSON contains ALL crew members (camera, sound, VFX, etc.),
    so we filter specifically for job == 'Director'.

    Method: List comprehension with conditional filter — iterates through
    all crew members and returns only those whose 'job' field is 'Director'.

    Args:
        crew_list (list): Parsed list of crew member dicts

    Returns:
        str: Director's name, or 'Unknown' if not found
    """
    if not isinstance(crew_list, list):
        return 'Unknown'
    directors = [member['name'] for member in crew_list
                 if isinstance(member, dict) and member.get('job') == 'Director']
    return directors[0] if directors else 'Unknown'


def _extract_top_cast(cast_list, n=3):
    """
    Extract the top N billed actors from the cast JSON.

    Why: Star power is a major revenue driver. The cast JSON is ordered by
    billing position (the 'order' field: 0 = lead, 1 = second lead, etc.).
    We extract the top 3 because they appear on posters and drive ticket sales.

    Method: Slicing (cast_list[:n]) — takes the first N elements from the
    already-sorted-by-billing list. Much more efficient than sorting since
    the data is pre-sorted by TMDB.

    Args:
        cast_list (list): Parsed list of cast member dicts
        n (int): Number of top actors to extract (default: 3)

    Returns:
        list: Names of top N actors, e.g. ['Sam Worthington', 'Zoe Saldana', ...]
    """
    if not isinstance(cast_list, list):
        return []
    return [member['name'] for member in cast_list[:n]
            if isinstance(member, dict) and 'name' in member]


def _extract_all_genres(genres_list):
    """
    Extract all genre names from the genres JSON.

    Why: Movies often span multiple genres (e.g., "Action, Adventure, Sci-Fi").
    The old code only took the FIRST genre, losing valuable information.
    Multi-genre encoding helps the model understand genre combinations
    (e.g., "Action + Comedy" behaves differently than pure "Action").

    Args:
        genres_list (list): Parsed list of genre dicts

    Returns:
        list: Genre names, e.g. ['Action', 'Adventure', 'Science Fiction']
    """
    if not isinstance(genres_list, list):
        return []
    return [g['name'] for g in genres_list if isinstance(g, dict) and 'name' in g]


def _classify_studio(companies_list):
    """
    Classify the production company into Major / Mid-tier / Indie.

    Why: A Disney blockbuster has fundamentally different economics than an
    indie film. Major studios have massive marketing budgets, global
    distribution networks, and franchise IP — all of which drive revenue
    independently of film quality. This feature captures that structural
    advantage.

    Method: Set intersection — we check if ANY of the movie's production
    companies appear in our predefined set of major studios. Using sets
    allows O(1) lookup instead of O(n) list scanning.

    Args:
        companies_list (list): Parsed list of production company dicts

    Returns:
        str: 'Major', 'Mid', or 'Indie'
    """
    MAJOR_STUDIOS = {
        'Walt Disney Pictures', 'Warner Bros.', 'Warner Bros. Pictures',
        'Universal Pictures', 'Paramount Pictures', 'Columbia Pictures',
        'Twentieth Century Fox Film Corporation', '20th Century Fox',
        'Sony Pictures', 'Metro-Goldwyn-Mayer (MGM)', 'Lionsgate',
        'New Line Cinema', 'DreamWorks SKG', 'DreamWorks Animation',
        'Marvel Studios', 'Lucasfilm', 'Pixar Animation Studios',
        'Touchstone Pictures', 'Walt Disney Animation Studios',
        'Columbia Pictures Corporation'
    }

    MID_STUDIOS = {
        'Relativity Media', 'Summit Entertainment', 'Focus Features',
        'Miramax Films', 'Miramax', 'The Weinstein Company',
        'Legendary Pictures', 'Village Roadshow Pictures',
        'Amblin Entertainment', 'StudioCanal', 'Entertainment One',
        'Lionsgate Films', 'Screen Gems', 'TriStar Pictures',
        'Revolution Studios', 'Millennium Films', 'EuropaCorp'
    }

    if not isinstance(companies_list, list):
        return 'Indie'

    company_names = {c.get('name', '') for c in companies_list if isinstance(c, dict)}

    if company_names & MAJOR_STUDIOS:
        return 'Major'
    elif company_names & MID_STUDIOS:
        return 'Mid'
    else:
        return 'Indie'


def load_tmdb_data(movies_path=None, credits_path=None, min_revenue=10000, min_budget=10000):
    """
    Load, merge, and clean TMDB 5000 movies dataset.

    This is the main entry point. It performs a complete ETL
    (Extract-Transform-Load) pipeline:

    1. EXTRACT: Read both CSV files from disk
    2. TRANSFORM: Parse JSON columns, extract features, filter bad data
    3. LOAD: Return a clean, ML-ready DataFrame

    Why we filter zero revenue/budget:
        ~1300 movies in TMDB have $0 revenue or $0 budget. These are either
        undisclosed data or non-theatrical releases (documentaries, TV movies).
        Including them would poison the model — it'd learn that some movies
        earn $0, which skews predictions for real theatrical releases.

    Method: pd.merge(on='id', how='inner') — SQL-style INNER JOIN on movie ID.
    Only movies present in BOTH CSVs are kept. 'inner' is chosen over 'left'
    to ensure every row has complete data (no missing cast/crew).

    Args:
        movies_path (str): Path to tmdb_5000_movies.csv (auto-detected if None)
        credits_path (str): Path to tmdb_5000_credits.csv (auto-detected if None)
        min_revenue (int): Minimum revenue to include (filters non-theatrical releases)
        min_budget (int): Minimum budget to include (filters entries with missing data)

    Returns:
        pd.DataFrame: Merged and cleaned DataFrame with columns:
            - title, budget, revenue, runtime, release_date, vote_average, etc.
            - director (str), top_cast (list), genres_list (list)
            - studio_tier ('Major'/'Mid'/'Indie')
            - Genre (primary genre for backward compatibility)
            - Budget_Millions, Revenue_Millions (in millions, for consistency)
    """
    # ── Auto-detect file paths ──
    # Search common locations relative to the script and project root
    if movies_path is None or credits_path is None:
        search_dirs = [
            'data',
            '../data',
            'Movie-Success-Prediction-and-Sentiment-Study/data',
            os.path.join(os.path.dirname(__file__), '..', 'data'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data'),
        ]
        for d in search_dirs:
            m = os.path.join(d, 'tmdb_5000_movies.csv')
            c = os.path.join(d, 'tmdb_5000_credits.csv')
            if os.path.exists(m) and os.path.exists(c):
                if movies_path is None:
                    movies_path = m
                if credits_path is None:
                    credits_path = c
                break

    if movies_path is None or credits_path is None:
        raise FileNotFoundError(
            "Could not find TMDB data files. Please provide paths to "
            "tmdb_5000_movies.csv and tmdb_5000_credits.csv"
        )

    print(f"      Loading movies from: {movies_path}")
    print(f"      Loading credits from: {credits_path}")

    # ── EXTRACT: Load CSVs ──
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)

    print(f"      Raw movies:  {len(movies_df):,} rows × {len(movies_df.columns)} columns")
    print(f"      Raw credits: {len(credits_df):,} rows × {len(credits_df.columns)} columns")

    # ── MERGE: Join movies with cast/crew ──
    # Credits uses 'movie_id', movies uses 'id' — rename for consistency
    if 'movie_id' in credits_df.columns:
        credits_df = credits_df.rename(columns={'movie_id': 'id'})

    df = pd.merge(movies_df, credits_df[['id', 'cast', 'crew']], on='id', how='inner')
    print(f"      After merge: {len(df):,} rows")

    # ── TRANSFORM: Parse JSON columns ──
    print("      Parsing JSON columns (genres, cast, crew, companies)...")
    df['genres_list'] = df['genres'].apply(_parse_json_column).apply(_extract_all_genres)
    df['cast_list'] = df['cast'].apply(_parse_json_column)
    df['crew_list'] = df['crew'].apply(_parse_json_column)
    df['companies_list'] = df['production_companies'].apply(_parse_json_column)

    # Extract key features from JSON
    df['director'] = df['crew_list'].apply(_extract_director)
    df['top_cast'] = df['cast_list'].apply(_extract_top_cast)
    df['lead_actor'] = df['top_cast'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    df['studio_tier'] = df['companies_list'].apply(_classify_studio)

    # Primary genre (for backward compatibility with existing code)
    df['Genre'] = df['genres_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')

    # ── FILTER: Remove bad data ──
    initial_count = len(df)
    df = df[df['revenue'] >= min_revenue].copy()
    df = df[df['budget'] >= min_budget].copy()
    filtered_count = initial_count - len(df)
    print(f"      Filtered out {filtered_count:,} movies with zero/minimal revenue or budget")

    # ── CONVERT: Create standard columns ──
    # Convert to millions for consistency with existing codebase
    df['Revenue_Millions'] = df['revenue'] / 1_000_000.0
    df['Budget_Millions'] = df['budget'] / 1_000_000.0

    # Use 'overview' as the text column for sentiment analysis
    if 'overview' in df.columns:
        df['User_Review'] = df['overview'].fillna('')

    print(f"      Final dataset: {len(df):,} movies ready for analysis")
    print(f"      Revenue range: ${df['Revenue_Millions'].min():.1f}M — ${df['Revenue_Millions'].max():.1f}M")
    print(f"      Budget range:  ${df['Budget_Millions'].min():.2f}M — ${df['Budget_Millions'].max():.1f}M")
    print(f"      Genres: {df['Genre'].nunique()} unique | Directors: {df['director'].nunique()} unique")

    # ── CLEANUP: Drop intermediate columns ──
    cols_to_drop = ['cast_list', 'crew_list', 'companies_list', 'cast', 'crew',
                    'production_companies', 'genres']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


# ──────────────────────────────────────────────────────────────────
#  Standalone Test
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TMDB Data Loader — Standalone Test")
    print("=" * 60)

    df = load_tmdb_data()

    print(f"\n--- Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nStudio distribution:")
    print(df['studio_tier'].value_counts().to_string())
    print(f"\nTop 10 genres:")
    print(df['Genre'].value_counts().head(10).to_string())
    print(f"\nSample directors: {df['director'].value_counts().head(5).to_string()}")
    print(f"\nSample row:")
    print(df[['title', 'Budget_Millions', 'Revenue_Millions', 'director', 'Genre', 'studio_tier']].head(3).to_string())
