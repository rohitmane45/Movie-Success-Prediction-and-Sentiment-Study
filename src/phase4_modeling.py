"""
Phase 4: Predictive Modeling

This is the core "Data Science" part of the project. We are moving from 
describing what happened (EDA) to predicting what will happen (Regression).

Goal: Build regression model to predict box office success based on 
sentiment, budget, genre, and other features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_features(df, target_col='Revenue_Millions', 
                    categorical_cols=None, numerical_cols=None):
    """
    Prepare features for machine learning by encoding categorical variables.
    
    Why we are doing this:
    - Numerical Input Only: Machine Learning models cannot understand words like 
      "Action" or "Comedy." We must convert these into numbers using One-Hot 
      Encoding (creating a column for each genre with a 1 or 0).
    - Feature Selection: We select relevant numerical and categorical features 
      that we believe influence box office success.
    
    Args:
        df (pd.DataFrame): DataFrame with all movie data
        target_col (str): Name of target variable (what we want to predict)
        categorical_cols (list): List of categorical column names to encode
        numerical_cols (list): List of numerical column names to include
        
    Returns:
        X (pd.DataFrame): Features dataframe
        y (pd.Series): Target variable
    """
    if categorical_cols is None:
        categorical_cols = ['Genre'] if 'Genre' in df.columns else []
    
    if numerical_cols is None:
        numerical_cols = ['Sentiment_Score', 'Budget_Millions']
        # Only include columns that exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # One-Hot Encode categorical variables
    # drop_first=True avoids redundancy (dummy variable trap)
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                   drop_first=True, prefix='Genre')
    
    # Select features (all columns except target)
    feature_cols = numerical_cols + [col for col in df_encoded.columns 
                                    if col.startswith('Genre_')]
    
    # Remove target from features if it exists
    feature_cols = [col for col in feature_cols if col != target_col]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col] if target_col in df_encoded.columns else df[target_col]
    
    print(f"--- Features Prepared ---")
    print(f"Number of features: {len(X.columns)}")
    print(f"Feature names: {list(X.columns)}")
    print(f"Number of samples: {len(X)}")
    
    return X, y, df_encoded


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Why we are doing this: We cannot test the model on the same data we used 
    to teach it. That would be like giving a student the exam questions with 
    the answers attached—they would get 100%, but they wouldn't learn anything.
    - Training Set (80%): The data the model studies
    - Testing Set (20%): The unseen data we use to grade the model's performance
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n--- Data Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Why we are doing this: We are using Linear Regression. Imagine plotting 
    all your data points on a graph. Linear Regression tries to draw the 
    "best fit" straight line through them.
    
    Mathematically, it tries to solve: 
    Revenue = (Weight×Sentiment) + (Weight×Budget) + Bias
    
    model.fit is the command where the math actually happens.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        model: Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nLinear Regression model trained successfully!")
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor model.
    
    Why we are doing this: Random Forest often handles non-linear relationships 
    in movie data better than Linear Regression. It can capture complex 
    interactions between features (e.g., "High sentiment + Action genre" might 
    be worth more than "High sentiment + Drama genre").
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility
        
    Returns:
        model: Trained Random Forest model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                 random_state=random_state,
                                 max_depth=10)
    model.fit(X_train, y_train)
    print("\nRandom Forest model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance on test data.
    
    Why we are doing this: Now we ask the model to predict the revenue for 
    the Test Set (which it has never seen). We compare its guess to the actual revenue.
    
    - MAE (Mean Absolute Error): On average, how many millions of dollars 
      was the model off by?
    - RMSE (Root Mean Square Error): Penalizes larger errors more heavily
    - R² Score: A score from 0 to 1. How well does our model explain the 
      variance? (Closer to 1 is better)
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Actual Values: {list(y_test.values[:5])}")  # Show first 5
    print(f"Predicted Values: {[round(p, 2) for p in predictions[:5]]}")
    print(f"Mean Absolute Error: ${mae:.2f} Million")
    print(f"Root Mean Square Error: ${rmse:.2f} Million")
    print(f"R² Score: {r2:.3f}")
    
    # Interpretation
    if r2 > 0.7:
        print("Excellent model fit!")
    elif r2 > 0.5:
        print("Good model fit.")
    elif r2 > 0.3:
        print("Moderate model fit.")
    else:
        print("Poor model fit - consider feature engineering.")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions
    }


def plot_predictions(y_test, predictions, model_name="Model"):
    """
    Visualize predicted vs actual values.
    
    Args:
        y_test (pd.Series): Actual values
        predictions (np.array): Predicted values
        model_name (str): Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, predictions, alpha=0.6)
    
    # Perfect prediction line (y=x)
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
            label='Perfect Prediction')
    
    plt.xlabel('Actual Revenue (Millions $)', fontsize=12)
    plt.ylabel('Predicted Revenue (Millions $)', fontsize=12)
    plt.title(f'{model_name}: Predicted vs Actual Revenue', 
             fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def predict_new_movie(model, X_train_columns, sentiment, budget, genre=None):
    """
    Make a prediction for a hypothetical new movie.
    
    Why we are doing this: This is how you demonstrate value in a portfolio. 
    You can say, "If we make a Sci-Fi movie with a $120M budget and get 
    Great reviews (0.8 sentiment), here is what we will earn."
    
    Args:
        model: Trained model
        X_train_columns (list): List of feature column names from training data
        sentiment (float): Sentiment score (-1 to 1)
        budget (float): Budget in millions
        genre (str): Genre name (must match encoding from training)
        
    Returns:
        float: Predicted revenue in millions
    """
    # Create a row matching the training data structure
    new_movie = pd.DataFrame(0, index=[0], columns=X_train_columns)
    
    # Set numerical features
    if 'Sentiment_Score' in new_movie.columns:
        new_movie['Sentiment_Score'] = sentiment
    if 'Budget_Millions' in new_movie.columns:
        new_movie['Budget_Millions'] = budget
    
    # Set genre encoding (set the appropriate genre column to 1)
    if genre:
        genre_col = f'Genre_{genre}'
        if genre_col in new_movie.columns:
            new_movie[genre_col] = 1
    
    predicted_revenue = model.predict(new_movie)[0]
    
    return predicted_revenue


def build_models(X_train, X_test, y_train, y_test):
    """
    Build and compare multiple models.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        
    Returns:
        dict: Dictionary containing trained models and their evaluations
    """
    results = {}
    
    # Linear Regression
    print("\n" + "="*60)
    print("Training Linear Regression Model...")
    print("="*60)
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    plot_predictions(y_test, lr_results['predictions'], "Linear Regression")
    results['linear_regression'] = {'model': lr_model, 'metrics': lr_results}
    
    # Random Forest
    print("\n" + "="*60)
    print("Training Random Forest Model...")
    print("="*60)
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    plot_predictions(y_test, rf_results['predictions'], "Random Forest")
    results['random_forest'] = {'model': rf_model, 'metrics': rf_results}
    
    # Compare models
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"Linear Regression R²: {lr_results['r2']:.3f}")
    print(f"Random Forest R²: {rf_results['r2']:.3f}")
    
    if rf_results['r2'] > lr_results['r2']:
        print("\nRandom Forest performs better!")
        best_model_name = 'random_forest'
    else:
        print("\nLinear Regression performs better!")
        best_model_name = 'linear_regression'
    
    results['best_model'] = best_model_name
    
    return results


if __name__ == "__main__":
    # Example usage with dummy data
    print("=" * 60)
    print("Phase 4: Predictive Modeling")
    print("=" * 60)
    
    # Expanded dummy data for ML (need more samples)
    data = {
        'Sentiment_Score': [0.9, -0.8, 0.1, 0.5, 0.85, -0.5, 0.2, 0.95, 
                           -0.2, 0.4, 0.75, -0.3, 0.6, 0.3, -0.1],
        'Budget_Millions': [160, 6, 20, 100, 150, 10, 30, 200, 15, 50, 
                           120, 8, 90, 25, 12],
        'Genre': ['Sci-Fi', 'Drama', 'Comedy', 'Action', 'Sci-Fi', 'Drama', 
                 'Comedy', 'Action', 'Drama', 'Action', 'Sci-Fi', 'Drama',
                 'Action', 'Comedy', 'Drama'],
        'Revenue_Millions': [800, 0.5, 15, 200, 750, 2, 25, 900, 5, 120, 
                           600, 3, 180, 18, 8]
    }
    
    df = pd.DataFrame(data)
    
    # Prepare features
    X, y, df_encoded = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build models
    results = build_models(X_train, X_test, y_train, y_test)
    
    # Make a prediction for a new movie
    print("\n" + "="*60)
    print("Making Prediction for New Movie")
    print("="*60)
    best_model = results[results['best_model']]['model']
    predicted = predict_new_movie(best_model, X.columns, 
                                  sentiment=0.9, budget=180, genre='Sci-Fi')
    print(f"Predicted Box Office for new Sci-Fi hit: ${predicted:.2f} Million")
    
    print("\nPhase 4 completed successfully!")
