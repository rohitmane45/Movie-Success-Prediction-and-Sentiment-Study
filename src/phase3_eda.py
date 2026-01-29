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
import os

# Create figures directory for saving plots
FIGURES_DIR = 'results/figures'


def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def create_sentiment_distribution(df, sentiment_col='Sentiment_Score', save=True):
    """
    Create histogram showing distribution of sentiment scores.
    
    Why we are doing this: We need to understand how sentiment scores are 
    distributed across our dataset. Are most movies reviewed positively or negatively?
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        sentiment_col (str): Column name for sentiment scores
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with KDE
    sns.histplot(data=df, x=sentiment_col, bins=30, kde=True, 
                color='steelblue', edgecolor='black', alpha=0.7, ax=ax)
    
    # Add mean and median lines
    mean_val = df[sentiment_col].mean()
    median_val = df[sentiment_col].median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Sentiment Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Movie Review Sentiments', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/sentiment_distribution.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {FIGURES_DIR}/sentiment_distribution.png")
    
    plt.show()


def create_scatter_plot(df, sentiment_col='Sentiment_Score', 
                       revenue_col='Revenue_Millions', genre_col='Genre', save=True):
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
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    plt.figure(figsize=(12, 7))
    
    # Create the scatter plot with genre color coding
    scatter = sns.scatterplot(data=df, x=sentiment_col, y=revenue_col, 
                             hue=genre_col, s=100, palette='viridis', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df[sentiment_col], df[revenue_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[sentiment_col].min(), df[sentiment_col].max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend Line')
    
    # Add titles and labels
    plt.title('Movie Success: Sentiment vs. Box Office Revenue', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score (-1 to +1)', fontsize=12)
    plt.ylabel('Revenue (Millions $)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/sentiment_vs_revenue.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {FIGURES_DIR}/sentiment_vs_revenue.png")
    
    plt.show()


def create_genre_sentiment_bar(df, sentiment_col='Sentiment_Score', 
                               genre_col='Genre', save=True):
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
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Calculate average sentiment per genre
    genre_sentiment = df.groupby(genre_col)[sentiment_col].mean().reset_index()
    genre_sentiment = genre_sentiment.sort_values(sentiment_col, ascending=False)
    
    # Create the bar chart
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(genre_sentiment)))
    bars = plt.bar(genre_sentiment[genre_col], genre_sentiment[sentiment_col], 
                   color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, genre_sentiment[sentiment_col]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Average Sentiment Score by Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Average Sentiment Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/genre_sentiment.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {FIGURES_DIR}/genre_sentiment.png")
    
    plt.show()
    
    return genre_sentiment


def create_genre_revenue_boxplot(df, revenue_col='Revenue_Millions', genre_col='Genre', save=True):
    """
    Create boxplot showing revenue distribution by genre.
    
    Why we are doing this: Boxplots show not just the average, but the full 
    distribution including outliers. This helps identify which genres have 
    the most consistent revenue vs high variance.
    
    Args:
        df (pd.DataFrame): DataFrame with revenue and genre data
        revenue_col (str): Column name for revenue
        genre_col (str): Column name for genre
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    plt.figure(figsize=(14, 7))
    
    # Order genres by median revenue
    genre_order = df.groupby(genre_col)[revenue_col].median().sort_values(ascending=False).index
    
    # Create boxplot
    sns.boxplot(data=df, x=genre_col, y=revenue_col, order=genre_order,
               palette='viridis', showfliers=True)
    
    plt.title('Revenue Distribution by Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Revenue (Millions $)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/genre_revenue_boxplot.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {FIGURES_DIR}/genre_revenue_boxplot.png")
    
    plt.show()


def create_correlation_analysis(df, numerical_cols=None, save=True):
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
        save (bool): Whether to save the figure
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    ensure_figures_dir()
    
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
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               square=True, fmt='.3f', cbar_kws={"shrink": 0.8},
               linewidths=0.5, annot_kws={"size": 12, "weight": "bold"})
    
    plt.title('Correlation Heatmap: Key Movie Metrics', 
             fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {FIGURES_DIR}/correlation_heatmap.png")
    
    plt.show()
    
    return correlation


def create_budget_vs_revenue_plot(df, budget_col='Budget_Millions', 
                                  revenue_col='Revenue_Millions', 
                                  sentiment_col='Sentiment_Score', save=True):
    """
    Create scatter plot of budget vs revenue, colored by sentiment.
    
    Why we are doing this: This shows the relationship between investment (budget)
    and return (revenue), with sentiment as a third dimension via color.
    
    Args:
        df (pd.DataFrame): DataFrame with budget, revenue, and sentiment data
        budget_col (str): Column name for budget
        revenue_col (str): Column name for revenue
        sentiment_col (str): Column name for sentiment (for coloring)
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df[budget_col], df[revenue_col], 
                        c=df[sentiment_col], cmap='RdYlGn', 
                        s=100, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sentiment Score', fontsize=12)
    
    # Add break-even line (Revenue = Budget)
    max_val = max(df[budget_col].max(), df[revenue_col].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, 
           label='Break-even (Revenue = Budget)', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df[budget_col], df[revenue_col], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[budget_col].min(), df[budget_col].max(), 100)
    ax.plot(x_line, p(x_line), "b-", linewidth=2, label='Trend Line', alpha=0.7)
    
    ax.set_xlabel('Budget (Millions $)', fontsize=12)
    ax.set_ylabel('Revenue (Millions $)', fontsize=12)
    ax.set_title('Budget vs Revenue (colored by Sentiment)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/budget_vs_revenue.png', dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_genre_sentiment(df, sentiment_col='Sentiment_Score', 
                           revenue_col='Revenue_Millions', genre_col='Genre'):
    """
    Perform comprehensive genre-wise sentiment analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment, revenue, and genre data
        sentiment_col (str): Column name for sentiment scores
        revenue_col (str): Column name for revenue
        genre_col (str): Column name for genre
        
    Returns:
        pd.DataFrame: Genre analysis summary
    """
    genre_analysis = df.groupby(genre_col).agg({
        sentiment_col: ['mean', 'std', 'min', 'max', 'count'],
        revenue_col: ['mean', 'median', 'std']
    }).round(2)
    
    # Flatten column names
    genre_analysis.columns = ['_'.join(col).strip() for col in genre_analysis.columns.values]
    genre_analysis = genre_analysis.reset_index()
    genre_analysis = genre_analysis.sort_values(f'{sentiment_col}_mean', ascending=False)
    
    print("\n=== Genre-wise Sentiment Analysis ===")
    print(genre_analysis.to_string(index=False))
    
    return genre_analysis


def create_movies_by_genre(df, genre_col='Genre', save=True):
    """
    Create a bar chart showing number of movies by genre.
    
    Args:
        df (pd.DataFrame): DataFrame with genre data
        genre_col (str): Column name for genre
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Count movies per genre
    genre_counts = df[genre_col].value_counts().sort_values(ascending=False)
    
    # Create the bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(genre_counts)))
    bars = plt.bar(genre_counts.index, genre_counts.values, 
                   color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, genre_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Number of Movies by Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/movies_by_genre.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return genre_counts


def create_revenue_by_genre(df, revenue_col='Revenue_Millions', genre_col='Genre', save=True):
    """
    Create a bar chart showing total/average revenue by genre.
    
    Args:
        df (pd.DataFrame): DataFrame with revenue and genre data
        revenue_col (str): Column name for revenue
        genre_col (str): Column name for genre
        save (bool): Whether to save the figure
    """
    ensure_figures_dir()
    
    plt.figure(figsize=(12, 6))
    
    # Calculate average revenue per genre
    genre_revenue = df.groupby(genre_col)[revenue_col].mean().sort_values(ascending=False)
    
    # Create the bar chart
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(genre_revenue)))
    bars = plt.bar(genre_revenue.index, genre_revenue.values, 
                   color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, genre_revenue.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'${val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.title('Average Revenue by Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Average Revenue (Millions $)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{FIGURES_DIR}/revenue_by_genre.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return genre_revenue


def perform_eda(df, sentiment_col='Sentiment_Score', 
               revenue_col='Revenue_Millions', genre_col='Genre', save_figures=True):
    """
    Perform complete exploratory data analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with movie data
        sentiment_col (str): Column name for sentiment scores
        revenue_col (str): Column name for revenue
        genre_col (str): Column name for genre
        save_figures (bool): Whether to save all figures
    """
    ensure_figures_dir()
    
    # 1. Number of Movies by Genre
    create_movies_by_genre(df, genre_col, save=save_figures)
    
    # 2. Revenue by Genre
    create_revenue_by_genre(df, revenue_col, genre_col, save=save_figures)
    
    # 3. Budget vs Revenue plot (colored by sentiment)
    if 'Budget_Millions' in df.columns:
        create_budget_vs_revenue_plot(df, 'Budget_Millions', revenue_col, sentiment_col, save=save_figures)


if __name__ == "__main__":
    # Example usage with expanded sample data
    np.random.seed(42)
    n_samples = 50
    
    genres = ['Sci-Fi', 'Drama', 'Comedy', 'Action', 'Horror', 'Romance', 'Thriller', 'Animation']
    
    data = {
        'Movie_Title': [f'Movie_{i}' for i in range(n_samples)],
        'Sentiment_Score': np.random.uniform(-0.8, 0.95, n_samples),
        'Revenue_Millions': [],
        'Budget_Millions': np.random.uniform(10, 250, n_samples),
        'Genre': np.random.choice(genres, n_samples)
    }
    
    # Generate correlated revenue based on budget and sentiment
    for i in range(n_samples):
        budget = data['Budget_Millions'][i]
        sentiment = data['Sentiment_Score'][i]
        sentiment_factor = 0.5 + (sentiment + 1) * 0.5
        noise = np.random.normal(0, budget * 0.3)
        revenue = max(1, budget * 2.5 * sentiment_factor + noise)
        data['Revenue_Millions'].append(round(revenue, 1))
    
    data['Sentiment_Score'] = [round(x, 2) for x in data['Sentiment_Score']]
    data['Budget_Millions'] = [round(x, 1) for x in data['Budget_Millions']]
    
    df = pd.DataFrame(data)
    
    perform_eda(df)
