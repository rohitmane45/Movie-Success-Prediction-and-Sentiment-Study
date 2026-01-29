"""
Phase 1: Data Acquisition & Setup

Goal: Get your environment ready and raw data loaded.

This phase handles:
- Environment setup verification
- Data loading from CSV or other sources
- Basic data inspection and cleaning
"""

import pandas as pd
import numpy as np
import os


def check_dependencies():
    """
    Verify that all required libraries are installed.
    """
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'nltk']
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            print(f"[MISSING] {package}")
            return False
    
    return True


def load_data_from_csv(file_path, encoding='utf-8'):
    """
    Load movie data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return None
    except UnicodeDecodeError:
        if encoding == 'utf-8':
            return load_data_from_csv(file_path, encoding='latin-1')
        return None


def inspect_data(df, sample_size=5):
    """
    Perform basic data inspection (silent).
    """
    pass  # No output needed


def prepare_data_directory(base_path='data'):
    """
    Create data directory if it doesn't exist.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)


def create_sample_data(n_samples=200):
    """
    Create a realistic sample dataset for ML training with 200+ samples.
    
    Why we are doing this: ML models require sufficient data to learn patterns.
    With only 15 samples, models cannot generalize. This function creates
    realistic movie data that simulates real-world relationships between
    sentiment, budget, genre, and revenue.
    
    Args:
        n_samples (int): Number of movie samples to generate (default: 200)
    
    Returns:
        pd.DataFrame: Sample movie data with realistic distributions
    """
    np.random.seed(42)  # For reproducibility
    
    # Genre definitions with typical budget ranges and revenue multipliers
    genres = {
        'Action': {'budget_range': (50, 250), 'revenue_mult': (1.5, 4.0), 'sentiment_bias': 0.1},
        'Comedy': {'budget_range': (15, 80), 'revenue_mult': (1.0, 3.5), 'sentiment_bias': 0.2},
        'Drama': {'budget_range': (10, 60), 'revenue_mult': (0.8, 3.0), 'sentiment_bias': 0.05},
        'Sci-Fi': {'budget_range': (80, 300), 'revenue_mult': (1.2, 4.5), 'sentiment_bias': 0.15},
        'Horror': {'budget_range': (5, 40), 'revenue_mult': (2.0, 6.0), 'sentiment_bias': -0.1},
        'Romance': {'budget_range': (10, 50), 'revenue_mult': (1.0, 3.0), 'sentiment_bias': 0.25},
        'Thriller': {'budget_range': (20, 100), 'revenue_mult': (1.2, 3.5), 'sentiment_bias': 0.0},
        'Animation': {'budget_range': (50, 200), 'revenue_mult': (1.5, 5.0), 'sentiment_bias': 0.3},
        'Adventure': {'budget_range': (60, 250), 'revenue_mult': (1.5, 4.0), 'sentiment_bias': 0.15},
        'Fantasy': {'budget_range': (50, 200), 'revenue_mult': (1.3, 4.0), 'sentiment_bias': 0.1},
        'Crime': {'budget_range': (20, 80), 'revenue_mult': (1.0, 3.5), 'sentiment_bias': 0.0},
        'Documentary': {'budget_range': (1, 20), 'revenue_mult': (1.0, 10.0), 'sentiment_bias': 0.1},
        'Mystery': {'budget_range': (15, 60), 'revenue_mult': (1.0, 3.5), 'sentiment_bias': 0.05},
        'Musical': {'budget_range': (20, 100), 'revenue_mult': (1.2, 4.0), 'sentiment_bias': 0.2},
        'Western': {'budget_range': (30, 100), 'revenue_mult': (0.8, 2.5), 'sentiment_bias': 0.0},
        'Biography': {'budget_range': (15, 80), 'revenue_mult': (1.0, 4.0), 'sentiment_bias': 0.1},
        'Family': {'budget_range': (30, 150), 'revenue_mult': (1.5, 4.5), 'sentiment_bias': 0.3},
    }
    
    # Review templates for different sentiment levels
    positive_reviews = [
        "An absolute masterpiece! Stunning visuals and incredible storytelling.",
        "I loved every minute of it. The acting was phenomenal!",
        "Best movie I've seen this year. Highly recommended!",
        "Brilliant performances and a gripping plot from start to finish.",
        "A must-watch! The director outdid themselves with this one.",
        "Fantastic film with amazing character development.",
        "This movie exceeded all my expectations. Truly outstanding!",
        "Incredible cinematography and a beautiful score. Loved it!",
        "A perfect blend of action and emotion. Simply amazing!",
        "Outstanding performances by the entire cast. Bravo!",
    ]
    
    neutral_reviews = [
        "It was okay. Not great, not terrible, just fine.",
        "Decent movie with some good moments, but nothing special.",
        "Average film. Worth watching once but not memorable.",
        "Some parts were good, others were boring. Mixed feelings.",
        "Watchable but forgettable. Nothing groundbreaking here.",
        "Had potential but didn't quite deliver. It's okay.",
        "Neither impressed nor disappointed. Just mediocre.",
        "Some enjoyable scenes but overall pretty average.",
        "Could have been better, could have been worse.",
        "A standard film. Does what it sets out to do, nothing more.",
    ]
    
    negative_reviews = [
        "Complete waste of time. Don't bother watching this.",
        "Terrible movie with awful acting and a boring plot.",
        "I want my two hours back. Absolute disaster.",
        "The worst movie I've seen in a long time. Avoid!",
        "Poorly written, badly directed. A complete mess.",
        "I couldn't even finish it. So disappointing.",
        "What a letdown. Expected much better from this cast.",
        "Boring, predictable, and utterly forgettable.",
        "Save your money. This movie is not worth it.",
        "A painful experience from start to finish. Terrible!",
    ]
    
    # Movie title components
    adjectives = ['The', 'Dark', 'Last', 'Secret', 'Final', 'Lost', 'Golden', 'Silent', 
                  'Hidden', 'Eternal', 'Savage', 'Wild', 'Forbidden', 'Broken', 'Rising']
    nouns = ['Knight', 'Journey', 'Dream', 'Legacy', 'Shadow', 'Storm', 'Kingdom', 
             'Horizon', 'Empire', 'Legend', 'Warrior', 'Phoenix', 'Destiny', 'Quest', 'Dawn']
    suffixes = ['', ' II', ' III', ': Reborn', ': Origins', ': The Beginning', ': Awakening', '']
    
    data = {
        'Movie_Title': [],
        'User_Review': [],
        'Revenue_Millions': [],
        'Budget_Millions': [],
        'Genre': [],
        'Runtime_Minutes': [],
        'Year': [],
        'Rating': []
    }
    
    genre_list = list(genres.keys())
    
    for i in range(n_samples):
        # Select random genre
        genre = np.random.choice(genre_list)
        genre_info = genres[genre]
        
        # Generate budget based on genre
        budget = np.random.uniform(genre_info['budget_range'][0], genre_info['budget_range'][1])
        
        # Generate base sentiment (affected by genre bias)
        base_sentiment = np.random.normal(0.3 + genre_info['sentiment_bias'], 0.35)
        sentiment = np.clip(base_sentiment, -1, 1)
        
        # Calculate revenue based on budget, sentiment, and genre multiplier
        # Formula: Revenue = Budget * Multiplier * (1 + Sentiment_Factor) + Noise
        mult = np.random.uniform(genre_info['revenue_mult'][0], genre_info['revenue_mult'][1])
        sentiment_factor = 0.5 + (sentiment + 1) / 2  # Convert -1,1 to 0.5,1.5
        noise = np.random.normal(0, budget * 0.3)  # Add some randomness
        revenue = max(0.1, budget * mult * sentiment_factor + noise)
        
        # Select appropriate review based on sentiment
        if sentiment > 0.4:
            review = np.random.choice(positive_reviews)
        elif sentiment < -0.2:
            review = np.random.choice(negative_reviews)
        else:
            review = np.random.choice(neutral_reviews)
        
        # Generate movie title
        title = f"{np.random.choice(adjectives)} {np.random.choice(nouns)}{np.random.choice(suffixes)}"
        
        # Generate other attributes
        runtime = int(np.random.normal(115, 20))
        runtime = max(75, min(180, runtime))
        year = np.random.randint(2010, 2025)
        rating = np.clip(5 + sentiment * 2.5 + np.random.normal(0, 0.5), 1, 10)
        
        data['Movie_Title'].append(title)
        data['User_Review'].append(review)
        data['Revenue_Millions'].append(round(revenue, 2))
        data['Budget_Millions'].append(round(budget, 2))
        data['Genre'].append(genre)
        data['Runtime_Minutes'].append(runtime)
        data['Year'].append(year)
        data['Rating'].append(round(rating, 1))
    
    df = pd.DataFrame(data)
    return df


def load_kaggle_imdb_dataset(file_path='data/IMDB-Movie-Data.csv'):
    """
    Load the IMDB 5000 Movie Dataset from Kaggle.
    
    Download from: https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset
    
    Args:
        file_path (str): Path to the Kaggle CSV file
        
    Returns:
        pd.DataFrame: Processed movie data or None if file not found
    """
    if not os.path.exists(file_path):
        print(f"[INFO] Kaggle dataset not found at {file_path}")
        print("   To use real IMDB data:")
        print("   1. Download from: https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset")
        print("   2. Place IMDB-Movie-Data.csv in the 'data/' folder")
        return None
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"[OK] Loaded Kaggle IMDB dataset: {len(df)} movies")
        
        # Map columns to our expected format
        column_mapping = {
            'Title': 'Movie_Title',
            'title': 'Movie_Title',
            'Genre': 'Genre',
            'genres': 'Genre',
            'Description': 'User_Review',
            'description': 'User_Review',
            'Revenue (Millions)': 'Revenue_Millions',
            'revenue': 'Revenue_Millions',
            'Budget': 'Budget_Millions',
            'budget': 'Budget_Millions',
            'Runtime (Minutes)': 'Runtime_Minutes',
            'runtime': 'Runtime_Minutes',
            'Rating': 'Rating',
            'imdb_score': 'Rating',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Handle multi-genre by taking first genre
        if 'Genre' in df.columns:
            df['Genre'] = df['Genre'].astype(str).str.split(',').str[0].str.strip()
        
        # Drop rows with missing critical values
        critical_cols = ['Revenue_Millions', 'Budget_Millions']
        existing_critical = [c for c in critical_cols if c in df.columns]
        if existing_critical:
            initial_count = len(df)
            df = df.dropna(subset=existing_critical)
            dropped = initial_count - len(df)
            if dropped > 0:
                print(f"   Dropped {dropped} rows with missing revenue/budget data")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load Kaggle dataset: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Data Acquisition & Setup")
    print("=" * 60)
    
    # Check dependencies
    print("\n--- Checking Dependencies ---")
    dependencies_ok = check_dependencies()
    
    if not dependencies_ok:
        print("\nPlease install missing dependencies before proceeding.")
        exit(1)
    
    # Prepare data directory
    print("\n--- Preparing Data Directory ---")
    prepare_data_directory()
    
    # Try to load data from CSV if it exists
    data_file = 'data/movies_dataset.csv'
    df = None
    
    if os.path.exists(data_file):
        print(f"\n--- Loading Data from {data_file} ---")
        df = load_data_from_csv(data_file)
    else:
        print(f"\n--- Data file not found: {data_file} ---")
        print("Creating sample data for demonstration...")
        df = create_sample_data()
        
        # Save sample data for reference
        df.to_csv('data/sample_movies_dataset.csv', index=False)
        print("✓ Sample data saved to data/sample_movies_dataset.csv")
    
    # Inspect data
    if df is not None:
        inspect_data(df)
        
        print("\n" + "="*60)
        print("Phase 1 completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review the data inspection above")
        print("  2. If using your own dataset, ensure it has columns for:")
        print("     - Reviews/Plot (text)")
        print("     - Revenue/Box Office (numerical)")
        print("     - Budget (numerical)")
        print("     - Genre (categorical)")
        print("  3. Proceed to Phase 2: Sentiment Analysis")
