"""
Phase 4: Predictive Modeling

This is the core "Data Science" part of the project. We are moving from 
describing what happened (EDA) to predicting what will happen (Regression).

Goal: Build regression model to predict box office success based on 
sentiment, budget, genre, and other features.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    if categorical_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                   drop_first=True, prefix='Genre')
    
    # Select features (all columns except target)
    feature_cols = numerical_cols + [col for col in df_encoded.columns 
                                    if col.startswith('Genre_')]
    feature_cols = [col for col in feature_cols if col != target_col]
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col] if target_col in df_encoded.columns else df[target_col]
    
    return X, y, df_encoded, scaling_params


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
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
    
    return {
        'mae': mae_display,
        'rmse': rmse_display,
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
    """
    # Create a row matching the training data structure
    new_movie = pd.DataFrame(0, index=[0], columns=X_train_columns)
    
    # Set numerical features
    if 'Sentiment_Score' in new_movie.columns:
        new_movie['Sentiment_Score'] = sentiment
    
    if 'Budget_Millions' in new_movie.columns:
        if scaling_params and scaling_params.get('budget_normalized'):
            budget_min = scaling_params['budget_min']
            budget_max = scaling_params['budget_max']
            normalized_budget = (budget - budget_min) / (budget_max - budget_min)
            normalized_budget = max(0.0, min(1.0, normalized_budget))
            new_movie['Budget_Millions'] = normalized_budget
        else:
            new_movie['Budget_Millions'] = budget
    
    # Set genre encoding
    if genre:
        genre_col = f'Genre_{genre}'
        if genre_col in new_movie.columns:
            new_movie[genre_col] = 1
    
    predicted_revenue_normalized = model.predict(new_movie)[0]
    
    # Denormalize revenue if needed
    if scaling_params and scaling_params.get('revenue_normalized'):
        revenue_min = scaling_params['revenue_min']
        revenue_max = scaling_params['revenue_max']
        predicted_revenue = revenue_min + predicted_revenue_normalized * (revenue_max - revenue_min)
    else:
        predicted_revenue = predicted_revenue_normalized
    
    return max(0.0, predicted_revenue)


def build_models(X_train, X_test, y_train, y_test, scaling_params=None, save_models=True):
    """
    Build and compare multiple models.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        scaling_params (dict): Dictionary with scaling parameters
        save_models (bool): Whether to save trained models to disk
        
    Returns:
        dict: Dictionary containing trained models and their evaluations
    """
    results = {}
    
    # Create models directory if saving
    if save_models:
        os.makedirs('models', exist_ok=True)
    
    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression", scaling_params)
    results['linear_regression'] = {'model': lr_model, 'metrics': lr_results}
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", scaling_params)
    results['random_forest'] = {'model': rf_model, 'metrics': rf_results}
    
    # Determine best model
    if rf_results['r2'] > lr_results['r2']:
        best_model_name = 'random_forest'
    else:
        best_model_name = 'linear_regression'
    
    results['best_model'] = best_model_name
    
    # Save models silently
    if save_models:
        best_model = results[best_model_name]['model']
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(lr_model, 'models/linear_regression.pkl')
        joblib.dump(rf_model, 'models/random_forest.pkl')
        
        os.makedirs('results', exist_ok=True)
        comparison_df = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'],
            'MAE': [lr_results['mae'], rf_results['mae']],
            'RMSE': [lr_results['rmse'], rf_results['rmse']],
            'R2': [lr_results['r2'], rf_results['r2']]
        })
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Create summary report
        summary = f"""=== Movie Success Prediction Model Summary ===

Dataset Statistics:
- Training Samples: {len(X_train)}
- Testing Samples: {len(X_test)}
- Features: {len(X_train.columns)}

Model Performance:
- Linear Regression R²: {lr_results['r2']:.3f}
- Random Forest R²: {rf_results['r2']:.3f}

Best Model: {best_model_name.replace('_', ' ').title()}
- MAE: ${results[best_model_name]['metrics']['mae']:.2f} Million
- R² Score: {results[best_model_name]['metrics']['r2']:.3f}
"""
        with open('results/model_summary.txt', 'w') as f:
            f.write(summary)
    
    return results


if __name__ == "__main__":
    # Example usage with robust synthetic data
    np.random.seed(42)
    n_samples = 100
    
    genres = ['Sci-Fi', 'Drama', 'Comedy', 'Action', 'Horror', 'Romance', 'Thriller', 'Animation']
    genre_multipliers = {'Sci-Fi': 3.5, 'Drama': 1.8, 'Comedy': 2.5, 'Action': 3.0, 
                        'Horror': 4.0, 'Romance': 2.0, 'Thriller': 2.5, 'Animation': 3.5}
    
    data = {
        'Sentiment_Score': [],
        'Budget_Millions': [],
        'Genre': [],
        'Revenue_Millions': []
    }
    
    for _ in range(n_samples):
        genre = np.random.choice(genres)
        budget = np.random.uniform(10, 250)
        sentiment = np.random.uniform(-0.8, 0.95)
        
        # Revenue formula: Budget * Genre_Multiplier * Sentiment_Factor + Noise
        sentiment_factor = 0.6 + (sentiment + 1) * 0.4  # Maps -1,1 to 0.2,1.4
        noise = np.random.normal(0, budget * 0.25)
        revenue = max(0.5, budget * genre_multipliers[genre] * sentiment_factor + noise)
        
        data['Sentiment_Score'].append(round(sentiment, 2))
        data['Budget_Millions'].append(round(budget, 1))
        data['Genre'].append(genre)
        data['Revenue_Millions'].append(round(revenue, 1))
    
    df = pd.DataFrame(data)
    print(f"\nGenerated {n_samples} synthetic movie samples for demonstration")
    print(f"Revenue range: ${df['Revenue_Millions'].min():.1f}M - ${df['Revenue_Millions'].max():.1f}M")
    
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
