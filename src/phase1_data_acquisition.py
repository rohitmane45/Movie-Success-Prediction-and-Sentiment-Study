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
    
    Why we are doing this: Before starting any data science project, we need 
    to ensure all necessary libraries are available. This prevents errors 
    later in the pipeline.
    """
    required_packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'matplotlib': 'plt',
        'seaborn': 'sns',
        'sklearn': 'sklearn',
        'nltk': 'nltk'
    }
    
    missing_packages = []
    
    for package, alias in required_packages.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nPlease install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    print("\nAll dependencies are installed!")
    return True


def load_data_from_csv(file_path, encoding='utf-8'):
    """
    Load movie data from a CSV file.
    
    Why we are doing this: Most datasets from Kaggle or other sources come 
    as CSV files. This function provides a standard way to load the data 
    into a pandas DataFrame for analysis.
    
    Args:
        file_path (str): Path to the CSV file
        encoding (str): File encoding (try 'latin-1' or 'ISO-8859-1' if utf-8 fails)
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"✓ Successfully loaded data from {file_path}")
        print(f"  Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print("  Please ensure the file exists or provide the correct path.")
        return None
    except UnicodeDecodeError:
        if encoding == 'utf-8':
            print("✗ UTF-8 encoding failed, trying latin-1...")
            return load_data_from_csv(file_path, encoding='latin-1')
        else:
            print(f"✗ Error: Could not decode file with {encoding} encoding")
            return None


def inspect_data(df, sample_size=5):
    """
    Perform basic data inspection.
    
    Why we are doing this: Before processing data, we need to understand:
    - What columns are available
    - What the data looks like
    - If there are missing values
    - Data types
    
    This helps us plan the rest of the pipeline.
    
    Args:
        df (pd.DataFrame): Data to inspect
        sample_size (int): Number of sample rows to display
    """
    if df is None or df.empty:
        print("No data to inspect.")
        return
    
    print("\n" + "="*60)
    print("Data Inspection")
    print("="*60)
    
    print(f"\nData Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst {sample_size} rows:")
    print(df.head(sample_size))
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values!")
    
    print(f"\nBasic Statistics:")
    print(df.describe())


def prepare_data_directory(base_path='data'):
    """
    Create data directory if it doesn't exist.
    
    Why we are doing this: We need a consistent location to store datasets 
    and processed files. Creating this structure early keeps the project organized.
    
    Args:
        base_path (str): Base path for data directory
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"✓ Created directory: {base_path}")
    else:
        print(f"✓ Directory exists: {base_path}")


def create_sample_data():
    """
    Create a sample dataset for testing purposes.
    
    Why we are doing this: When you don't have the actual dataset yet, or 
    for testing the pipeline, we can create a dummy dataset that mimics the 
    structure we expect from IMDB/Kaggle data.
    
    Returns:
        pd.DataFrame: Sample movie data
    """
    data = {
        'Movie_Title': [
            'Inception', 'The Room', 'Average Joe', 'Super Hero 5',
            'Epic Adventure', 'Dark Thriller', 'Romantic Comedy', 'Sci-Fi Epic',
            'Action Packed', 'Dramatic Tale', 'Funny Movie', 'Horror Flick',
            'Space Odyssey', 'Love Story', 'Crime Drama'
        ],
        'User_Review': [
            "This movie was an absolute masterpiece! The plot was mind-blowing.",
            "Worst movie ever. Complete waste of time and money.",
            "It was okay. Not great, not terrible, just fine.",
            "I loved the action scenes, but the dialogue was a bit weak.",
            "Incredible special effects and a gripping storyline!",
            "Too dark and depressing. Not my cup of tea.",
            "Sweet and funny, perfect date movie!",
            "Groundbreaking sci-fi with amazing visuals!",
            "Non-stop action from start to finish!",
            "Deep and emotional, made me cry.",
            "Hilarious! Laughed throughout the entire film.",
            "Scary but predictable plot.",
            "Thought-provoking and visually stunning!",
            "Beautiful romantic story, loved it!",
            "Intense and well-acted thriller."
        ],
        'Revenue_Millions': [800, 0.5, 15, 200, 750, 2, 25, 900, 300, 8, 
                           45, 12, 650, 30, 5],
        'Budget_Millions': [160, 6, 20, 100, 150, 10, 30, 200, 80, 15, 
                          25, 8, 180, 20, 12],
        'Genre': ['Sci-Fi', 'Drama', 'Comedy', 'Action', 'Sci-Fi', 'Drama', 
                 'Comedy', 'Action', 'Action', 'Drama', 'Comedy', 'Horror',
                 'Sci-Fi', 'Romance', 'Drama'],
        'Runtime_Minutes': [148, 99, 95, 120, 135, 110, 100, 165, 105, 
                          115, 90, 98, 142, 108, 122]
    }
    
    df = pd.DataFrame(data)
    return df


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
