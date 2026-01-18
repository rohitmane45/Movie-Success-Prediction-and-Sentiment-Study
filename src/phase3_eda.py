"""
Phase 3: Exploratory Data Analysis (EDA)

Goal: Understand the data and create the "Sentiment Visuals."
Answer the core question: Is there actually a relationship between how much 
people like a movie (Sentiment) and how much money it makes (Revenue)?

In this phase, we stop looking at raw numbers and start looking for patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def create_scatter_plot(df, sentiment_col='Sentiment_Score', 
                       revenue_col='Revenue_Millions', genre_col='Genre'):
    """
    Create a scatter plot showing relationship between sentiment and revenue.
    
    Why we are doing this: A Scatter Plot is the standard tool in Data Science 
    to compare two continuous variables.
    - X-axis: Sentiment Score (Cause?)
    - Y-axis: Revenue (Effect?)
    
    We look for a trend line. If the dots go "up and to the right," it means 
    better reviews = more money. If they are scattered randomly, there is no relationship.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment, revenue, and genre data
        sentiment_col (str): Column name for sentiment scores
        revenue_col (str): Column name for revenue values
        genre_col (str): Column name for genre (for color coding)
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot with genre color coding
    sns.scatterplot(data=df, x=sentiment_col, y=revenue_col, 
                   hue=genre_col, s=100, palette='viridis')
    
    # Add titles and labels
    plt.title('Movie Success: Sentiment vs. Box Office Revenue', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score (-1 to +1)', fontsize=12)
    plt.ylabel('Revenue (Millions $)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def create_genre_sentiment_bar(df, sentiment_col='Sentiment_Score', 
                               genre_col='Genre'):
    """
    Create a bar chart showing average sentiment by genre.
    
    Why we are doing this: This fulfills the requirement: "Analyze genre-wise 
    sentiment trends." Some genres (like Comedies) might naturally have higher 
    sentiment scores than others (like Horror), regardless of how much money 
    they make. We need to see which genres are the "happiest."
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment and genre data
        sentiment_col (str): Column name for sentiment scores
        genre_col (str): Column name for genre
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate average sentiment per genre
    genre_sentiment = df.groupby(genre_col)[sentiment_col].mean().reset_index()
    genre_sentiment = genre_sentiment.sort_values(sentiment_col, ascending=False)
    
    # Create the bar chart
    sns.barplot(data=genre_sentiment, x=genre_col, y=sentiment_col, 
               palette='viridis')
    
    plt.title('Average Sentiment Score by Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return genre_sentiment


def create_correlation_analysis(df, numerical_cols=None):
    """
    Calculate and visualize correlation matrix.
    
    Why we are doing this: Visuals are great, but "Correlation" gives us a 
    hard number (from -1 to 1) to prove the relationship.
    - 1.0: Perfect positive relationship (Sentiment goes up, Revenue goes up)
    - 0.0: No relationship
    - -1.0: Inverse relationship
    
    This number is crucial for your "Predictive Model Summary" deliverable.
    
    Args:
        df (pd.DataFrame): DataFrame with numerical columns
        numerical_cols (list): List of column names to include in correlation
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    if numerical_cols is None:
        numerical_cols = ['Sentiment_Score', 'Revenue_Millions', 'Budget_Millions']
    
    # Select only numerical columns that exist
    available_cols = [col for col in numerical_cols if col in df.columns]
    numerical_data = df[available_cols]
    
    # Calculate correlation
    correlation = numerical_data.corr()
    
    print("--- Correlation Matrix ---")
    print(correlation)
    print()
    
    # Visualize the matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Sentiment vs. Revenue', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return correlation


def perform_eda(df, sentiment_col='Sentiment_Score', 
               revenue_col='Revenue_Millions', genre_col='Genre'):
    """
    Perform complete exploratory data analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with movie data
        sentiment_col (str): Column name for sentiment scores
        revenue_col (str): Column name for revenue
        genre_col (str): Column name for genre
    """
    print("=" * 60)
    print("Phase 3: Exploratory Data Analysis")
    print("=" * 60)
    
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData summary:")
    print(df.describe())
    
    # Scatter plot
    print("\nCreating scatter plot...")
    create_scatter_plot(df, sentiment_col, revenue_col, genre_col)
    
    # Genre sentiment analysis
    print("\nAnalyzing genre-wise sentiment...")
    genre_sentiment = create_genre_sentiment_bar(df, sentiment_col, genre_col)
    print("\nGenre Sentiment Summary:")
    print(genre_sentiment)
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    correlation = create_correlation_analysis(df)
    
    # Check specific correlation between sentiment and revenue
    if sentiment_col in correlation.columns and revenue_col in correlation.columns:
        sentiment_revenue_corr = correlation.loc[sentiment_col, revenue_col]
        print(f"\n--- Key Finding ---")
        print(f"Correlation between Sentiment and Revenue: {sentiment_revenue_corr:.3f}")
        
        if abs(sentiment_revenue_corr) > 0.3:
            direction = "positive" if sentiment_revenue_corr > 0 else "negative"
            strength = "strong" if abs(sentiment_revenue_corr) > 0.7 else "moderate"
            print(f"This indicates a {strength} {direction} relationship.")
        else:
            print("This indicates a weak or no relationship.")
    
    print("\nPhase 3 completed successfully!")


if __name__ == "__main__":
    # Example usage with dummy data
    # In real usage, this data would come from Phase 2 output
    
    data = {
        'Movie_Title': ['Inception', 'The Room', 'Average Joe', 'Super Hero 5',
                       'Epic Adventure', 'Dark Thriller', 'Romantic Comedy', 'Sci-Fi Epic'],
        'Sentiment_Score': [0.9, -0.8, 0.1, 0.5, 0.85, -0.5, 0.2, 0.95],
        'Revenue_Millions': [800, 0.5, 15, 200, 750, 2, 25, 900],
        'Budget_Millions': [160, 6, 20, 100, 150, 10, 30, 200],
        'Genre': ['Sci-Fi', 'Drama', 'Comedy', 'Action', 'Sci-Fi', 'Drama', 
                 'Comedy', 'Action']
    }
    
    df = pd.DataFrame(data)
    
    perform_eda(df)
