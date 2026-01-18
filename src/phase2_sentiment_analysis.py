"""
Phase 2: Sentiment Analysis with VADER

Goal: Turn text reviews into numerical scores.
This fulfills the "Use VADER for sentiment on user reviews" requirement.

Step-by-step implementation with explanations:
- Why we are doing this: VADER is specifically designed for sentiment analysis 
  of social media and short texts (like movie reviews) because it understands 
  intensity (e.g., "good" vs "GREAT!!!") and context.
"""

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (run once)
# WHY: VADER relies on a pre-built dictionary of words (lexicon) to score sentiment.
# We must download this lexicon once before using it.
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def get_sentiment_score(text, analyzer):
    """
    Extract sentiment compound score from text.
    
    Why we are doing this: VADER returns a dictionary of scores (Positive, 
    Neutral, Negative, Compound). For this project, we only care about the 
    Compound Score - a single normalized number between -1 (Most Negative) 
    and +1 (Most Positive).
    
    This single number is mathematically easier to feed into Regression Model 
    later than three separate scores.
    
    Args:
        text (str): Input text to analyze
        analyzer: VADER SentimentIntensityAnalyzer instance
        
    Returns:
        float: Compound sentiment score (-1 to 1)
    """
    scores = analyzer.polarity_scores(text)
    return scores['compound']


def analyze_sentiment(df, review_column='User_Review'):
    """
    Apply VADER sentiment analysis to a DataFrame column.
    
    Why we are doing this: We need to run our function on every single row of 
    the User_Review column and save the result in a new column. This transforms 
    qualitative text data (words) into quantitative data (numbers) that your 
    machine learning model can understand.
    
    Args:
        df (pd.DataFrame): DataFrame containing movie data
        review_column (str): Name of the column containing review text
        
    Returns:
        pd.DataFrame: DataFrame with added 'Sentiment_Score' column
    """
    # Initialize the VADER sentiment analyzer
    # WHY: We need to create an "instance" of the VADER analyzer. 
    # Think of this as turning on the machine that will process the text.
    analyzer = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to each review
    df['Sentiment_Score'] = df[review_column].apply(
        lambda x: get_sentiment_score(str(x), analyzer)
    )
    
    return df


def preprocess_text(df, review_column='User_Review'):
    """
    Basic text preprocessing: remove HTML tags and normalize text.
    
    Why we are doing this: Raw text from reviews may contain HTML tags, 
    special characters, or inconsistent formatting that could affect sentiment 
    analysis. Cleaning the text ensures more accurate sentiment scores.
    
    Args:
        df (pd.DataFrame): DataFrame with review column
        review_column (str): Name of column containing text reviews
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text
    """
    import re
    
    # Remove HTML tags
    df[review_column] = df[review_column].apply(
        lambda x: re.sub(r'<[^>]+>', '', str(x))
    )
    
    # Convert to lowercase for consistency
    df[review_column] = df[review_column].str.lower()
    
    return df


if __name__ == "__main__":
    # Example usage with dummy data
    print("=" * 60)
    print("Phase 2: Sentiment Analysis with VADER")
    print("=" * 60)
    
    # Creating a sample dataset to simulate IMDB data
    # WHY: Since we don't have your specific Kaggle CSV loaded right now, 
    # we need to create a small "dummy" dataset to test our logic. 
    # This ensures our code works before we apply it to thousands of real rows.
    data = {
        'Movie_Title': ['Inception', 'The Room', 'Average Joe', 'Super Hero 5'],
        'User_Review': [
            "This movie was an absolute masterpiece! The plot was mind-blowing.",
            "Worst movie ever. Complete waste of time and money.",
            "It was okay. Not great, not terrible, just fine.",
            "I loved the action scenes, but the dialogue was a bit weak."
        ]
    }
    
    df = pd.DataFrame(data)
    print("\n--- Raw Data ---")
    print(df)
    
    # Preprocess text
    df = preprocess_text(df)
    
    # Apply sentiment analysis
    df = analyze_sentiment(df)
    
    # Display results
    print("\n--- Processed Data with Sentiment Scores ---")
    print(df[['Movie_Title', 'User_Review', 'Sentiment_Score']])
    
    print("\n--- Interpretation ---")
    print("Masterpiece review → Score near +0.8 or +0.9")
    print("Worst movie review → Score near -0.6 or -0.8")
    print("Okay review → Score near 0.0")
    
    # Save results (optional)
    # df.to_csv('data/movies_with_sentiment.csv', index=False)
    print("\nPhase 2 completed successfully!")
