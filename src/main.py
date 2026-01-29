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
    print("\n" + "="*70)
    print("       MOVIE SUCCESS PREDICTION & SENTIMENT STUDY")
    print("="*70)
    
    # Phase 1: Data Acquisition & Setup
    print("\n[1/4] Loading Data...")
    
    if not check_dependencies():
        print("ERROR: Missing dependencies. Please install requirements.txt")
        return
    
    prepare_data_directory()
    
    if use_sample_data or data_file is None or not os.path.exists(data_file):
        df = create_sample_data()
        if not os.path.exists('data'):
            os.makedirs('data')
        try:
            df.to_csv('data/sample_movies_dataset.csv', index=False)
        except PermissionError:
            pass
        print(f"      Loaded {len(df)} movies | {df['Genre'].nunique()} genres")
    else:
        df = load_data_from_csv(data_file)
        if df is None:
            print("ERROR: Could not load data. Exiting.")
            return
        print(f"      Loaded {len(df)} movies from {data_file}")

    # Harmonize column names silently
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
            if revenue_values.max() > 1000000:
                df['Revenue_Millions'] = revenue_values / 1_000_000.0
            elif revenue_values.max() <= 1.0 and revenue_values.min() >= 0.0:
                df['Revenue_Millions'] = revenue_values
            else:
                df['Revenue_Millions'] = revenue_values

    if 'Budget_Millions' not in df.columns and 'budget' in df.columns:
        budget_values = pd.to_numeric(df['budget'], errors='coerce')
        if budget_values.max() <= 1.0 and budget_values.min() >= 0.0:
            df['Budget_Millions'] = budget_values
        elif budget_values.max() > 1000000:
            df['Budget_Millions'] = budget_values / 1_000_000.0
        else:
            df['Budget_Millions'] = budget_values

    if 'Genre' not in df.columns and 'genres' in df.columns:
        import json
        
        def extract_first_genre(genres_str):
            if pd.isna(genres_str) or genres_str == '' or genres_str == '[]':
                return 'Unknown'
            try:
                if genres_str.startswith('['):
                    genres_list = json.loads(genres_str.replace("'", '"'))
                    if genres_list and isinstance(genres_list, list):
                        return genres_list[0].get('name', 'Unknown')
                elif '|' in str(genres_str):
                    return str(genres_str).split('|')[0].strip()
                else:
                    return str(genres_str).strip()
            except:
                return str(genres_str).split('|')[0].split(',')[0].strip()
            return 'Unknown'
        
        df['Genre'] = df['genres'].apply(extract_first_genre)

    # Phase 2: Sentiment Analysis
    print("\n[2/4] Analyzing Sentiment...")
    
    # Ensure we have a review column
    if 'User_Review' not in df.columns:
        text_columns = list(df.select_dtypes(include=['object']).columns)
        preferred_order = ['plot_keywords', 'story', 'description', 'overview', 'synopsis', 'review', 'movie_title']
        review_col = None
        for cand in preferred_order:
            if cand in df.columns:
                review_col = cand
                break
        if review_col is None and len(text_columns) > 0:
            review_col = text_columns[0]
        if review_col is not None:
            df.rename(columns={review_col: 'User_Review'}, inplace=True)
        else:
            print("ERROR: No text column found for sentiment analysis.")
            return
    
    df = preprocess_text(df)
    df = analyze_sentiment(df)
    
    avg_sentiment = df['Sentiment_Score'].mean()
    print(f"      Average Sentiment: {avg_sentiment:.2f} (-1 to +1 scale)")
    
    # Phase 3: Exploratory Data Analysis
    print("\n[3/4] Creating Visualizations...")
    
    required_cols = ['Sentiment_Score', 'Revenue_Millions', 'Genre']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if not missing_cols:
        perform_eda(df)
        print("      Saved to: results/figures/")
    
    # Phase 4: Predictive Modeling
    print("\n[4/4] Training Prediction Model...")
    
    if 'Revenue_Millions' not in df.columns:
        print("ERROR: 'Revenue_Millions' column required for modeling.")
        return

    initial_rows = len(df)
    model_df = df.dropna(subset=['Revenue_Millions']).copy()
    
    # Prepare features
    X, y, df_encoded, scaling_params = prepare_features(model_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = build_models(X_train, X_test, y_train, y_test, scaling_params)
    
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    best_r2 = results[best_model_name]['metrics']['r2']
    best_mae = results[best_model_name]['metrics']['mae']
    
    print(f"\n      Best Model: {best_model_name.replace('_', ' ').title()}")
    print(f"      Accuracy (R²): {best_r2:.1%}")
    print(f"      Avg Error: ${best_mae:.1f}M")
    
    # Save predictions
    title_col = None
    for col in ['title', 'Title', 'Movie_Title', 'movie_title']:
        if col in model_df.columns:
            title_col = col
            break
    
    test_indices = y_test.index
    test_predictions = best_model.predict(X_test)
    
    comparison_df = pd.DataFrame({
        'Movie': model_df.loc[test_indices, title_col].values if title_col else [f'Movie_{i}' for i in range(len(test_indices))],
        'Genre': model_df.loc[test_indices, 'Genre'].values,
        'Budget_M': model_df.loc[test_indices, 'Budget_Millions'].values,
        'Actual_Revenue_M': y_test.values,
        'Predicted_Revenue_M': test_predictions,
    })
    comparison_df['Error_M'] = abs(comparison_df['Actual_Revenue_M'] - comparison_df['Predicted_Revenue_M'])
    comparison_df = comparison_df.sort_values('Actual_Revenue_M', ascending=False)
    comparison_df.to_csv('results/movie_predictions.csv', index=False)
    
    # Show sample predictions
    print("\n" + "-"*70)
    print("  SAMPLE PREDICTIONS (Top 5 Movies)")
    print("-"*70)
    print(f"  {'Movie':<30} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
    print("  " + "-"*66)
    
    for _, row in comparison_df.head(5).iterrows():
        movie_name = str(row['Movie'])[:28]
        print(f"  {movie_name:<30} ${row['Actual_Revenue_M']:>9.1f}M  ${row['Predicted_Revenue_M']:>9.1f}M  ${row['Error_M']:>7.1f}M")
    
    # New movie predictions
    print("\n" + "-"*70)
    print("  NEW MOVIE PREDICTIONS")
    print("-"*70)
    
    examples = [
        {'title': 'Inception 2', 'sentiment': 0.9, 'budget': 180, 'genre': 'Sci-Fi'},
        {'title': 'Quiet Hearts', 'sentiment': 0.3, 'budget': 30, 'genre': 'Drama'},
        {'title': 'Fury Road 2', 'sentiment': 0.7, 'budget': 150, 'genre': 'Adventure'}
    ]
    
    print(f"  {'Movie':<20} {'Budget':>10} {'Sentiment':>12} {'Predicted':>15}")
    print("  " + "-"*60)
    
    for example in examples:
        predicted = predict_new_movie(
            best_model, X.columns, 
            sentiment=example['sentiment'],
            budget=example['budget'], 
            genre=example['genre'],
            scaling_params=scaling_params
        )
        sent_label = "Positive" if example['sentiment'] > 0.5 else "Mixed" if example['sentiment'] > 0 else "Negative"
        print(f"  {example['title']:<20} ${example['budget']:>7}M   {sent_label:>10}   ${predicted:>12.1f}M")
    
    # Save results
    output_file = 'data/movies_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    
    # Final Summary
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print(f"  Model: {best_model_name.replace('_', ' ').title()} | R²: {best_r2:.1%} | MAE: ${best_mae:.1f}M")
    print(f"  Files: results/movie_predictions.csv, {output_file}")
    print("="*70 + "\n")


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
