"""
Main Pipeline: Movie Success Prediction and Sentiment Study

This script runs the complete data science pipeline:
1. Phase 1: Data Acquisition & Setup
2. Phase 2: Sentiment Analysis with VADER
3. Phase 3: Exploratory Data Analysis
4. Phase 4: Predictive Modeling

Usage:
    python src/main.py
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_data_acquisition import (
    check_dependencies, 
    load_data_from_csv, 
    inspect_data,
    prepare_data_directory,
    create_sample_data
)
from phase2_sentiment_analysis import (
    preprocess_text, 
    analyze_sentiment
)
from phase3_eda import perform_eda
from phase4_modeling import (
    prepare_features,
    split_data,
    build_models,
    predict_new_movie
)


def run_pipeline(data_file=None, use_sample_data=False):
    """
    Run the complete movie success prediction pipeline.
    
    Args:
        data_file (str): Path to CSV file with movie data (optional)
        use_sample_data (bool): If True, use sample data for demonstration
    """
    print("="*80)
    print("MOVIE SUCCESS PREDICTION AND SENTIMENT STUDY")
    print("Complete Data Science Pipeline")
    print("="*80)
    
    # Phase 1: Data Acquisition & Setup
    print("\n" + "="*80)
    print("PHASE 1: DATA ACQUISITION & SETUP")
    print("="*80)
    
    if not check_dependencies():
        print("ERROR: Missing dependencies. Please install requirements.txt")
        return
    
    prepare_data_directory()
    
    if use_sample_data or data_file is None or not os.path.exists(data_file):
        print("\nUsing sample data for demonstration...")
        df = create_sample_data()
        if not os.path.exists('data'):
            os.makedirs('data')
        # Try to save sample data (handle permission errors gracefully)
        try:
            df.to_csv('data/sample_movies_dataset.csv', index=False)
        except PermissionError:
            print("[WARNING] Could not save sample data (file may be open). Continuing...")
    else:
        print(f"\nLoading data from {data_file}...")
        df = load_data_from_csv(data_file)
        if df is None:
            print("ERROR: Could not load data. Exiting.")
            return
    
    inspect_data(df)

    # Harmonize column names for external datasets (e.g., Kaggle IMDB, TMDB)
    # so that the rest of the pipeline (EDA + modeling) can run unchanged.
    print("\nNormalizing dataset columns for EDA and modeling...")

    # Create Revenue_Millions from common alternatives (e.g., 'gross', 'revenue')
    if 'Revenue_Millions' not in df.columns:
        revenue_col = None
        if 'gross' in df.columns:
            revenue_col = 'gross'
        elif 'revenue' in df.columns:
            revenue_col = 'revenue'
        elif 'Revenue (Millions)' in df.columns:
            revenue_col = 'Revenue (Millions)'
            
        if revenue_col:
            revenue_values = pd.to_numeric(df[revenue_col], errors='coerce')
            # Check if data is already in millions or needs conversion
            if revenue_values.max() > 1000000:  # Likely in raw dollars
                print(f" - Creating 'Revenue_Millions' from '{revenue_col}' (dividing by 1e6).")
                df['Revenue_Millions'] = revenue_values / 1_000_000.0
            elif revenue_values.max() <= 1.0 and revenue_values.min() >= 0.0:
                print(f" - Creating 'Revenue_Millions' from '{revenue_col}' (data appears normalized, using as-is).")
                df['Revenue_Millions'] = revenue_values
            else:
                print(f" - Creating 'Revenue_Millions' from '{revenue_col}' (assuming already in millions).")
                df['Revenue_Millions'] = revenue_values
        else:
            print(" - WARNING: Could not find a revenue column like 'gross' or 'revenue'.")

    # Create Budget_Millions from 'budget' if not already present
    if 'Budget_Millions' not in df.columns and 'budget' in df.columns:
        budget_values = pd.to_numeric(df['budget'], errors='coerce')
        # Check if data is already normalized (0-1 range) or in raw dollars
        if budget_values.max() <= 1.0 and budget_values.min() >= 0.0:
            print(" - Creating 'Budget_Millions' from 'budget' (data appears normalized, using as-is).")
            df['Budget_Millions'] = budget_values
        elif budget_values.max() > 1000000:  # Likely in raw dollars
            print(" - Creating 'Budget_Millions' from 'budget' (dividing by 1e6).")
            df['Budget_Millions'] = budget_values / 1_000_000.0
        else:
            print(" - Creating 'Budget_Millions' from 'budget' (assuming already in millions).")
            df['Budget_Millions'] = budget_values

    # Create Genre from 'genres' (handle both JSON format and pipe-separated)
    if 'Genre' not in df.columns and 'genres' in df.columns:
        import json
        import ast
        
        def extract_first_genre(genres_str):
            """Extract first genre from JSON array or pipe-separated string."""
            if pd.isna(genres_str) or genres_str == '' or genres_str == '[]':
                return 'Unknown'
            try:
                # Try JSON format first (TMDB style)
                if genres_str.startswith('['):
                    genres_list = json.loads(genres_str.replace("'", '"'))
                    if genres_list and isinstance(genres_list, list):
                        return genres_list[0].get('name', 'Unknown')
                # Try pipe-separated format
                elif '|' in str(genres_str):
                    return str(genres_str).split('|')[0].strip()
                else:
                    return str(genres_str).strip()
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                # Fallback: try simple string split
                return str(genres_str).split('|')[0].split(',')[0].strip()
            return 'Unknown'
        
        print(" - Creating 'Genre' from 'genres' (extracting first genre).")
        df['Genre'] = df['genres'].apply(extract_first_genre)
        print(f"   Found {df['Genre'].nunique()} unique genres: {df['Genre'].value_counts().head(10).index.tolist()}")

    # Phase 2: Sentiment Analysis with VADER
    print("\n" + "="*80)
    print("PHASE 2: SENTIMENT ANALYSIS WITH VADER")
    print("="*80)
    
    # Ensure we have a review / text column
    if 'User_Review' not in df.columns:
        print("WARNING: 'User_Review' column not found. Trying to choose a suitable text column...")
        text_columns = list(df.select_dtypes(include=['object']).columns)

        # Prefer columns that look like plot/keywords/description over generic ones like 'color'
        preferred_order = ['plot_keywords', 'story', 'description', 'overview', 'synopsis', 'review', 'movie_title']
        review_col = None
        for cand in preferred_order:
            if cand in df.columns:
                review_col = cand
                break

        # Fallback: first text column if no preferred one found
        if review_col is None and len(text_columns) > 0:
            review_col = text_columns[0]

        if review_col is not None:
            print(f"Using '{review_col}' as review text column (renamed to 'User_Review').")
            df.rename(columns={review_col: 'User_Review'}, inplace=True)
        else:
            print("ERROR: No text column found for sentiment analysis.")
            return
    
    df = preprocess_text(df)
    df = analyze_sentiment(df)
    
    print("\nSentiment analysis completed!")
    # Robustly pick a title column if present
    title_col = None
    if 'Movie_Title' in df.columns:
        title_col = 'Movie_Title'
    elif 'movie_title' in df.columns:
        title_col = 'movie_title'
    elif 'title' in df.columns:
        title_col = 'title'

    if title_col:
        print(df[[title_col, 'Sentiment_Score']].head(10))
    else:
        print(df[['Sentiment_Score']].head(10))
    
    # Phase 3: Exploratory Data Analysis
    print("\n" + "="*80)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Ensure we have required columns for EDA
    required_cols = ['Sentiment_Score', 'Revenue_Millions', 'Genre']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"WARNING: Missing columns for EDA: {missing_cols}")
        print("Please ensure your dataset has these columns.")
    else:
        perform_eda(df)
    
    # Phase 4: Predictive Modeling
    print("\n" + "="*80)
    print("PHASE 4: PREDICTIVE MODELING")
    print("="*80)
    
    # Check if we have the required columns for modeling
    if 'Revenue_Millions' not in df.columns:
        print("ERROR: 'Revenue_Millions' column required for modeling.")
        return

    # Drop rows with missing target (Revenue_Millions) before modeling
    initial_rows = len(df)
    model_df = df.dropna(subset=['Revenue_Millions']).copy()
    dropped = initial_rows - len(model_df)
    if dropped > 0:
        print(f"\nDropping {dropped} rows with missing Revenue_Millions for modeling ({len(model_df)} rows left).")
    
    # Check data quality - warn if target has very little variance
    revenue_unique = model_df['Revenue_Millions'].nunique()
    if revenue_unique < 10:
        print(f"\nWARNING: Revenue_Millions has only {revenue_unique} unique values.")
        print("This dataset may not be suitable for prediction as there's very little variance in the target variable.")
        print("The model may not learn meaningful patterns. Consider using a dataset with more diverse revenue values.")
    
    # Prepare features
    X, y, df_encoded, scaling_params = prepare_features(model_df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build models
    results = build_models(X_train, X_test, y_train, y_test, scaling_params)
    
    # Make a prediction for a new movie
    print("\n" + "="*80)
    print("PREDICTIONS ON REAL MOVIES FROM DATASET")
    print("="*80)
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    
    # Get predictions for actual movies from the test set
    # Find title column
    title_col = None
    for col in ['title', 'Title', 'Movie_Title', 'movie_title']:
        if col in model_df.columns:
            title_col = col
            break
    
    # Make predictions on test set and compare with actual
    test_indices = y_test.index
    test_predictions = best_model.predict(X_test)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Movie': model_df.loc[test_indices, title_col].values if title_col else [f'Movie_{i}' for i in range(len(test_indices))],
        'Genre': model_df.loc[test_indices, 'Genre'].values,
        'Budget_M': model_df.loc[test_indices, 'Budget_Millions'].values,
        'Sentiment': model_df.loc[test_indices, 'Sentiment_Score'].values,
        'Actual_Revenue_M': y_test.values,
        'Predicted_Revenue_M': test_predictions,
    })
    comparison_df['Error_M'] = abs(comparison_df['Actual_Revenue_M'] - comparison_df['Predicted_Revenue_M'])
    comparison_df['Accuracy_%'] = 100 - (comparison_df['Error_M'] / (comparison_df['Actual_Revenue_M'] + 0.1) * 100)
    comparison_df['Accuracy_%'] = comparison_df['Accuracy_%'].clip(0, 100)
    
    # Sort by actual revenue to show a mix of blockbusters and smaller films
    comparison_df_sorted = comparison_df.sort_values('Actual_Revenue_M', ascending=False)
    
    # Show top 10 highest grossing movies from test set
    print("\n--- Top 10 Blockbusters (Actual vs Predicted Revenue) ---")
    print(f"{'Movie':<40} {'Genre':<15} {'Budget':<10} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
    print("-" * 109)
    
    for _, row in comparison_df_sorted.head(10).iterrows():
        movie_name = str(row['Movie'])[:38]
        print(f"{movie_name:<40} {row['Genre']:<15} ${row['Budget_M']:>6.1f}M   ${row['Actual_Revenue_M']:>8.1f}M   ${row['Predicted_Revenue_M']:>8.1f}M   ${row['Error_M']:>6.1f}M")
    
    # Show 10 random mid-range movies
    mid_range = comparison_df[(comparison_df['Actual_Revenue_M'] > 10) & (comparison_df['Actual_Revenue_M'] < 200)]
    if len(mid_range) >= 10:
        sample_movies = mid_range.sample(n=10, random_state=42)
    else:
        sample_movies = mid_range
    
    print("\n--- 10 Mid-Range Movies (Actual vs Predicted Revenue) ---")
    print(f"{'Movie':<40} {'Genre':<15} {'Budget':<10} {'Actual':<12} {'Predicted':<12} {'Error':<10}")
    print("-" * 109)
    
    for _, row in sample_movies.iterrows():
        movie_name = str(row['Movie'])[:38]
        print(f"{movie_name:<40} {row['Genre']:<15} ${row['Budget_M']:>6.1f}M   ${row['Actual_Revenue_M']:>8.1f}M   ${row['Predicted_Revenue_M']:>8.1f}M   ${row['Error_M']:>6.1f}M")
    
    # Summary statistics
    print("\n--- Prediction Summary Statistics ---")
    print(f"Total movies in test set: {len(comparison_df)}")
    print(f"Average prediction error: ${comparison_df['Error_M'].mean():.2f}M")
    print(f"Median prediction error: ${comparison_df['Error_M'].median():.2f}M")
    print(f"Movies with error < $50M: {(comparison_df['Error_M'] < 50).sum()} ({(comparison_df['Error_M'] < 50).sum() / len(comparison_df) * 100:.1f}%)")
    print(f"Movies with error < $100M: {(comparison_df['Error_M'] < 100).sum()} ({(comparison_df['Error_M'] < 100).sum() / len(comparison_df) * 100:.1f}%)")
    
    # Save predictions to CSV
    comparison_df_sorted.to_csv('results/movie_predictions.csv', index=False)
    print(f"\n[OK] All {len(comparison_df)} movie predictions saved to results/movie_predictions.csv")
    
    # Example predictions for hypothetical new movies
    print("\n" + "="*80)
    print("PREDICTIONS FOR NEW/HYPOTHETICAL MOVIES")
    print("="*80)
    
    examples = [
        {'title': 'Inception 2', 'sentiment': 0.9, 'budget': 180, 'genre': 'Science Fiction', 'description': 'High-budget Sci-Fi with great reviews'},
        {'title': 'Quiet Hearts', 'sentiment': 0.3, 'budget': 30, 'genre': 'Drama', 'description': 'Low-budget Drama with mixed reviews'},
        {'title': 'Fury Road 2', 'sentiment': 0.7, 'budget': 150, 'genre': 'Adventure', 'description': 'Big-budget Adventure with good reviews'}
    ]
    
    for example in examples:
        predicted = predict_new_movie(
            best_model, 
            X.columns, 
            sentiment=example['sentiment'],
            budget=example['budget'], 
            genre=example['genre'],
            scaling_params=scaling_params
        )
        print(f"\n'{example['title']}' - {example['description']}:")
        print(f"  Sentiment Score: {example['sentiment']:.2f}")
        print(f"  Budget: ${example['budget']}M")
        print(f"  Genre: {example['genre']}")
        print(f"  -> Predicted Revenue for '{example['title']}': ${predicted:.2f}M")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save processed data
    output_file = 'data/movies_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    # Avoid non-ASCII symbols for Windows consoles
    print(f"[OK] Processed data saved to {output_file}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nSummary:")
    print(f"  - Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"  - R² Score: {results[best_model_name]['metrics']['r2']:.3f}")
    print(f"  - MAE: ${results[best_model_name]['metrics']['mae']:.2f}M")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Movie Success Prediction and Sentiment Study Pipeline'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to CSV file with movie data'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data for demonstration'
    )
    
    args = parser.parse_args()
    
    run_pipeline(data_file=args.data, use_sample_data=args.sample)
