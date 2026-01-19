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
        df.to_csv('data/sample_movies_dataset.csv', index=False)
    else:
        print(f"\nLoading data from {data_file}...")
        df = load_data_from_csv(data_file)
        if df is None:
            print("ERROR: Could not load data. Exiting.")
            return
    
    inspect_data(df)

    # Harmonize column names for external datasets (e.g., Kaggle IMDB)
    # so that the rest of the pipeline (EDA + modeling) can run unchanged.
    print("\nNormalizing dataset columns for EDA and modeling...")

    # Create Revenue_Millions from common alternatives (e.g., 'gross')
    if 'Revenue_Millions' not in df.columns:
        if 'gross' in df.columns:
            gross_values = pd.to_numeric(df['gross'], errors='coerce')
            # Check if data is already normalized (0-1 range)
            if gross_values.max() <= 1.0 and gross_values.min() >= 0.0:
                print(" - Creating 'Revenue_Millions' from 'gross' (data appears normalized, using as-is).")
                df['Revenue_Millions'] = gross_values
            else:
                print(" - Creating 'Revenue_Millions' from 'gross' (dividing by 1e6).")
                df['Revenue_Millions'] = gross_values / 1_000_000.0
        else:
            print(" - WARNING: Could not find a revenue column like 'gross'.")

    # Create Budget_Millions from 'budget' if not already present
    if 'Budget_Millions' not in df.columns and 'budget' in df.columns:
        budget_values = pd.to_numeric(df['budget'], errors='coerce')
        # Check if data is already normalized (0-1 range)
        if budget_values.max() <= 1.0 and budget_values.min() >= 0.0:
            print(" - Creating 'Budget_Millions' from 'budget' (data appears normalized, using as-is).")
            df['Budget_Millions'] = budget_values
        else:
            print(" - Creating 'Budget_Millions' from 'budget' (dividing by 1e6).")
            df['Budget_Millions'] = budget_values / 1_000_000.0

    # Create Genre from 'genres' (take the first genre in pipe-separated list)
    if 'Genre' not in df.columns and 'genres' in df.columns:
        print(" - Creating 'Genre' from 'genres' (first genre before '|').")
        df['Genre'] = df['genres'].astype(str).str.split('|').str[0]

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
    print("PREDICTION EXAMPLE")
    print("="*80)
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    
    # Example predictions
    examples = [
        {'title': 'Inception 2', 'sentiment': 0.9, 'budget': 180, 'genre': 'Sci-Fi', 'description': 'High-budget Sci-Fi with great reviews'},
        {'title': 'Quiet Hearts', 'sentiment': 0.3, 'budget': 30, 'genre': 'Drama', 'description': 'Low-budget Drama with mixed reviews'},
        {'title': 'Fury Road 2', 'sentiment': 0.7, 'budget': 150, 'genre': 'Action', 'description': 'Big-budget Action with good reviews'}
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
