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
        scaling_params (dict): Dictionary with min/max values for scaling
    """
    # Check dataset size
    if len(df) < 30:
        print("[WARNING] You have very few samples. ML models need at least 50-100 samples.")
        print(f"   Current samples: {len(df)}")
        print("   Predictions may be unreliable!\n")
    
    if categorical_cols is None:
        categorical_cols = ['Genre'] if 'Genre' in df.columns else []
    
    if numerical_cols is None:
        numerical_cols = ['Sentiment_Score', 'Budget_Millions']
        # Only include columns that exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    
    # Store scaling parameters for budget (to normalize predictions)
    scaling_params = {}
    if 'Budget_Millions' in df_encoded.columns:
        budget_vals = df_encoded['Budget_Millions'].dropna()
        if len(budget_vals) > 0:
            # Check if data is normalized (0-1 range)
            if budget_vals.max() <= 1.0 and budget_vals.min() >= 0.0:
                # Data is normalized, estimate reasonable min/max for real-world values
                # Typical movie budgets: $1M to $300M
                scaling_params['budget_min'] = 1.0  # $1M
                scaling_params['budget_max'] = 300.0  # $300M
                scaling_params['budget_normalized'] = True
            else:
                # Data is in real values
                scaling_params['budget_min'] = budget_vals.min()
                scaling_params['budget_max'] = budget_vals.max()
                scaling_params['budget_normalized'] = False
    
    if target_col in df_encoded.columns:
        revenue_vals = df_encoded[target_col].dropna()
        if len(revenue_vals) > 0:
            # Check if data is normalized (0-1 range)
            if revenue_vals.max() <= 1.0 and revenue_vals.min() >= 0.0:
                # Data is normalized, estimate reasonable min/max for real-world values
                # Typical movie revenues: $0.1M to $1000M
                scaling_params['revenue_min'] = 0.1  # $0.1M
                scaling_params['revenue_max'] = 1000.0  # $1000M
                scaling_params['revenue_normalized'] = True
            else:
                # Data is in real values
                scaling_params['revenue_min'] = revenue_vals.min()
                scaling_params['revenue_max'] = revenue_vals.max()
                scaling_params['revenue_normalized'] = False
    
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
    print(f"Target range: ${y.min():.2f}M - ${y.max():.2f}M")
    print(f"Target mean: ${y.mean():.2f}M")
    if scaling_params.get('budget_normalized'):
        print(f"Note: Budget data is normalized. Using estimated scale: ${scaling_params['budget_min']:.1f}M - ${scaling_params['budget_max']:.1f}M")
    if scaling_params.get('revenue_normalized'):
        print(f"Note: Revenue data is normalized. Using estimated scale: ${scaling_params['revenue_min']:.1f}M - ${scaling_params['revenue_max']:.1f}M")
    print()
    
    return X, y, df_encoded, scaling_params


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
    
    print(f"--- Data Split ---")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {len(X_train.columns)}")
    
    if len(X_train) < 20:
        print("[WARNING] Very few training samples! Model will struggle to learn.")
    print()
    
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
    
    print("[OK] Linear Regression model trained successfully!")
    
    print("\nLearned Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print()
    
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
    # Adaptive hyperparameters for small datasets
    if len(X_train) < 30:
        n_estimators = 20
        max_depth = 3
        print(f"[WARNING] Small dataset detected. Adjusting RF parameters:")
        print(f"   n_estimators: {n_estimators}, max_depth: {max_depth}")
    else:
        max_depth = 10
    
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    print("[OK] Random Forest model trained successfully!")
    
    print("\nFeature Importance:")
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    print()
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model", scaling_params=None):
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
        scaling_params (dict): Dictionary with scaling parameters for denormalization
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Denormalize if needed for display
    if scaling_params and scaling_params.get('revenue_normalized'):
        revenue_min = scaling_params['revenue_min']
        revenue_max = scaling_params['revenue_max']
        y_test_display = revenue_min + y_test.values * (revenue_max - revenue_min)
        predictions_display = revenue_min + predictions * (revenue_max - revenue_min)
    else:
        y_test_display = y_test.values
        predictions_display = predictions
    
    # Calculate metrics (always on normalized scale for accuracy)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Calculate denormalized MAE for display if needed
    if scaling_params and scaling_params.get('revenue_normalized'):
        mae_display = mean_absolute_error(y_test_display, predictions_display)
        rmse_display = np.sqrt(mean_squared_error(y_test_display, predictions_display))
    else:
        mae_display = mae
        rmse_display = rmse
    
    print(f"--- {model_name} Results ---")
    print(f"Number of test samples: {len(y_test)}")
    print(f"\nFirst 5 predictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test_display[i]
        pred = predictions_display[i]
        diff = abs(actual - pred)
        print(f"  Sample {i+1}: Actual=${actual:.2f}M, Predicted=${pred:.2f}M, Error=${diff:.2f}M")
    
    print(f"\nPerformance Metrics:")
    print(f"  Mean Absolute Error (MAE): ${mae_display:.2f} Million")
    print(f"  Root Mean Square Error (RMSE): ${rmse_display:.2f} Million")
    print(f"  R² Score: {r2:.3f}")
    
    print(f"\nInterpretation:")
    if r2 > 0.7:
        print("  [OK] Excellent model fit! (R² > 0.7)")
    elif r2 > 0.5:
        print("  [OK] Good model fit. (R² > 0.5)")
    elif r2 > 0.3:
        print("  [WARNING] Moderate model fit. (R² > 0.3)")
    elif r2 > 0:
        print("  [WARNING] Poor model fit. Consider:")
        print("     - Adding more training data")
        print("     - Feature engineering (add more features)")
        print("     - Using a simpler model")
    else:
        print("  [ERROR] Very poor model fit - model performs worse than baseline.")
    print()
    
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
    
    plt.scatter(y_test, predictions, alpha=0.6, s=100, edgecolors='black')
    
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


def predict_new_movie(model, X_train_columns, sentiment, budget, genre=None, scaling_params=None):
    """
    Make a prediction for a hypothetical new movie.
    
    Why we are doing this: This is how you demonstrate value in a portfolio. 
    You can say, "If we make a Sci-Fi movie with a $120M budget and get 
    Great reviews (0.8 sentiment), here is what we will earn."
    
    Args:
        model: Trained model
        X_train_columns (list): List of feature column names from training data
        sentiment (float): Sentiment score (-1 to 1)
        budget (float): Budget in millions (real-world value)
        genre (str): Genre name (must match encoding from training)
        scaling_params (dict): Dictionary with min/max values for scaling
        
    Returns:
        float: Predicted revenue in millions (real-world value)
    """
    # Create a row matching the training data structure
    new_movie = pd.DataFrame(0, index=[0], columns=X_train_columns)
    
    # Set numerical features
    if 'Sentiment_Score' in new_movie.columns:
        new_movie['Sentiment_Score'] = sentiment
    
    if 'Budget_Millions' in new_movie.columns:
        # Normalize budget if needed
        if scaling_params and scaling_params.get('budget_normalized'):
            # Normalize budget to 0-1 range
            budget_min = scaling_params['budget_min']
            budget_max = scaling_params['budget_max']
            normalized_budget = (budget - budget_min) / (budget_max - budget_min)
            normalized_budget = max(0.0, min(1.0, normalized_budget))  # Clip to [0, 1]
            new_movie['Budget_Millions'] = normalized_budget
        else:
            new_movie['Budget_Millions'] = budget
    
    # Set genre encoding (set the appropriate genre column to 1)
    if genre:
        genre_col = f'Genre_{genre}'
        if genre_col in new_movie.columns:
            new_movie[genre_col] = 1
        else:
            print(f"[WARNING] Genre '{genre}' not found in training data.")
            available_genres = [col.replace('Genre_', '') for col in new_movie.columns if col.startswith('Genre_')]
            if available_genres:
                print(f"   Available genres: {available_genres}")
    
    print(f"\nNew Movie Input Features:")
    print(f"  Sentiment Score: {sentiment}")
    print(f"  Budget: ${budget}M")
    print(f"  Genre: {genre if genre else 'Not specified'}")
    
    predicted_revenue_normalized = model.predict(new_movie)[0]
    
    # Denormalize revenue if needed
    if scaling_params and scaling_params.get('revenue_normalized'):
        revenue_min = scaling_params['revenue_min']
        revenue_max = scaling_params['revenue_max']
        # Denormalize from 0-1 range to real values
        predicted_revenue = revenue_min + predicted_revenue_normalized * (revenue_max - revenue_min)
    else:
        predicted_revenue = predicted_revenue_normalized
    
    return max(0.0, predicted_revenue)  # Ensure non-negative


def build_models(X_train, X_test, y_train, y_test, scaling_params=None):
    """
    Build and compare multiple models.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        scaling_params (dict): Dictionary with scaling parameters
        
    Returns:
        dict: Dictionary containing trained models and their evaluations
    """
    results = {}
    
    # Linear Regression
    print("\n" + "="*70)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*70)
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression", scaling_params)
    plot_predictions(y_test, lr_results['predictions'], "Linear Regression")
    results['linear_regression'] = {'model': lr_model, 'metrics': lr_results}
    
    # Random Forest
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*70)
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", scaling_params)
    plot_predictions(y_test, rf_results['predictions'], "Random Forest")
    results['random_forest'] = {'model': rf_model, 'metrics': rf_results}
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"Linear Regression R²: {lr_results['r2']:.3f}")
    print(f"Random Forest R²: {rf_results['r2']:.3f}")
    
    if rf_results['r2'] > lr_results['r2']:
        print("\n[OK] Random Forest performs better!")
        best_model_name = 'random_forest'
    else:
        print("\n[OK] Linear Regression performs better!")
        best_model_name = 'linear_regression'
    
    results['best_model'] = best_model_name
    
    return results


if __name__ == "__main__":
    # Example usage with dummy data
    print("=" * 70)
    print("Phase 4: Predictive Modeling")
    print("=" * 70)
    
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
    X, y, df_encoded, scaling_params = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build models
    results = build_models(X_train, X_test, y_train, y_test, scaling_params)
    
    # Make a prediction for a new movie
    print("\n" + "="*70)
    print("MAKING PREDICTION FOR NEW MOVIE")
    print("="*70)
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    
    predicted = predict_new_movie(best_model, X.columns, 
                                  sentiment=0.9, budget=180, genre='Sci-Fi',
                                  scaling_params=scaling_params)
    print(f"\nPrediction Result:")
    print(f"   Predicted Box Office for new Sci-Fi hit: ${predicted:.2f} Million")
    
    print("\nPhase 4 completed successfully!")
