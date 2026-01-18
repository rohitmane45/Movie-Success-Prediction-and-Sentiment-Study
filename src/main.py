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
    
    # Phase 2: Sentiment Analysis with VADER
    print("\n" + "="*80)
    print("PHASE 2: SENTIMENT ANALYSIS WITH VADER")
    print("="*80)
    
    # Ensure we have a review column
    if 'User_Review' not in df.columns:
        print("WARNING: 'User_Review' column not found. Using 'Plot' or first text column.")
        text_columns = df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            review_col = text_columns[0]
            df.rename(columns={review_col: 'User_Review'}, inplace=True)
        else:
            print("ERROR: No text column found for sentiment analysis.")
            return
    
    df = preprocess_text(df)
    df = analyze_sentiment(df)
    
    print("\nSentiment analysis completed!")
    print(df[['Movie_Title', 'Sentiment_Score']].head(10))
    
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
    
    # Prepare features
    X, y, df_encoded = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build models
    results = build_models(X_train, X_test, y_train, y_test)
    
    # Make a prediction for a new movie
    print("\n" + "="*80)
    print("PREDICTION EXAMPLE")
    print("="*80)
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    
    # Example predictions
    examples = [
        {'sentiment': 0.9, 'budget': 180, 'genre': 'Sci-Fi', 'description': 'High-budget Sci-Fi with great reviews'},
        {'sentiment': 0.3, 'budget': 30, 'genre': 'Drama', 'description': 'Low-budget Drama with mixed reviews'},
        {'sentiment': 0.7, 'budget': 150, 'genre': 'Action', 'description': 'Big-budget Action with good reviews'}
    ]
    
    for example in examples:
        predicted = predict_new_movie(
            best_model, 
            X.columns, 
            sentiment=example['sentiment'],
            budget=example['budget'], 
            genre=example['genre']
        )
        print(f"\n{example['description']}:")
        print(f"  Sentiment: {example['sentiment']:.2f}")
        print(f"  Budget: ${example['budget']}M")
        print(f"  Genre: {example['genre']}")
        print(f"  Predicted Revenue: ${predicted:.2f}M")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save processed data
    output_file = 'data/movies_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    print(f"✓ Processed data saved to {output_file}")
    
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
