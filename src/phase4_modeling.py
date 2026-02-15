"""
Phase 4: Predictive Modeling (Enhanced with Advanced Models)

This is the core "Data Science" part of the project. We are moving from 
describing what happened (EDA) to predicting what will happen (Regression).

Goal: Build regression models to predict box office success based on 
sentiment, budget, genre, director track record, star power, and more.

Models Available:
    1. Linear Regression     — simple baseline, fast, interpretable
    2. Random Forest          — ensemble of decision trees, handles non-linearity
    3. Gradient Boosting      — sequential tree boosting, strong on tabular data
    4. XGBoost                — optimized gradient boosting with regularization

Evaluation:
    - 5-fold Cross-Validation for reliable performance estimates
    - Feature importance visualization to explain predictions
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

# XGBoost — try to import, fall back gracefully if not installed
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[INFO] XGBoost not installed. Install with: pip install xgboost")
    print("       Gradient Boosting (sklearn) will be used as fallback.")


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
        # Extended feature set: include all engineered features if they exist
        numerical_cols = [
            'Sentiment_Score', 'Budget_Millions',
            # Temporal features
            'release_month', 'is_summer_release', 'is_holiday_release',
            # Personnel features
            'director_avg_revenue', 'director_movie_count',
            'lead_actor_avg_revenue',
            # Quality & popularity signals
            'vote_average', 'vote_count', 'popularity',
            # Content features
            'runtime', 'num_genres',
            # Financial derivations
            'log_budget', 'budget_per_vote',
        ]
        # Only include columns that actually exist in the DataFrame
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
    
    # One-Hot Encode categorical variables (only if Genre_* columns don't already exist)
    existing_genre_cols = [c for c in df_encoded.columns if c.startswith('Genre_')]
    if categorical_cols and not existing_genre_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, 
                                   drop_first=True, prefix='Genre')
    
    # Also include Studio_* columns if present (from feature engineering)
    studio_cols = [col for col in df_encoded.columns if col.startswith('Studio_')]
    
    # Select features (all columns except target)
    feature_cols = numerical_cols + [col for col in df_encoded.columns 
                                    if col.startswith('Genre_')] + studio_cols
    # Deduplicate while preserving order
    seen = set()
    unique_feature_cols = []
    for col in feature_cols:
        if col not in seen and col != target_col:
            seen.add(col)
            unique_feature_cols.append(col)
    feature_cols = unique_feature_cols
    
    X = df_encoded[feature_cols]
    y = df_encoded[target_col] if target_col in df_encoded.columns else df[target_col]
    
    # Drop any rows with NaN in features or target
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
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


def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.1, random_state=42):
    """
    Train a Gradient Boosting Regressor model.
    
    Why we are doing this: Gradient Boosting builds trees SEQUENTIALLY —
    each new tree focuses on correcting the errors of the previous ones.
    Think of it as a team where each member specializes in fixing the
    mistakes the others made.
    
    How it differs from Random Forest:
        - Random Forest: builds trees INDEPENDENTLY, then averages (parallel)
        - Gradient Boosting: builds trees SEQUENTIALLY, each improving on the last (serial)
    
    The 'learning_rate' controls how aggressively each tree corrects errors.
    Lower = more trees needed but often better generalization (less overfitting).
    
    Parameters explained:
        - n_estimators=200: Number of sequential trees to build
        - learning_rate=0.1: Step size for each tree's correction
        - max_depth=5: Each tree is shallow (prevents overfitting)
        - subsample=0.8: Uses 80% of data per tree (adds randomness)
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of boosting stages
        learning_rate (float): Shrinkage rate
        random_state (int): Random seed
        
    Returns:
        model: Trained Gradient Boosting model
    """
    if len(X_train) < 100:
        n_estimators = 50
        max_depth = 3
    else:
        max_depth = 5
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=300, learning_rate=0.1, random_state=42):
    """
    Train an XGBoost Regressor model.
    
    Why we are doing this: XGBoost (eXtreme Gradient Boosting) is the go-to
    algorithm for tabular data in competitions. It consistently outperforms
    other algorithms because:
    
    1. Built-in regularization (L1 + L2) — prevents overfitting
    2. Efficient handling of missing values
    3. Parallelized tree building — much faster than sklearn's GBM
    4. Column subsampling — randomly selects feature subsets per tree
    
    Parameters explained:
        - n_estimators=300: More trees than GBM because XGBoost regularizes better
        - colsample_bytree=0.8: Each tree sees 80% of features
        - reg_alpha=0.1: L1 regularization (sparsity)
        - reg_lambda=1.0: L2 regularization (shrinkage)
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of boosting rounds
        learning_rate (float): Step size shrinkage
        random_state (int): Random seed
        
    Returns:
        model: Trained XGBoost model, or None if XGBoost not installed
    """
    if not HAS_XGBOOST:
        print("      [SKIP] XGBoost not installed. Using Gradient Boosting instead.")
        return None
    
    if len(X_train) < 100:
        n_estimators = 50
        max_depth = 3
    else:
        max_depth = 6
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X, y, cv=5, model_name="Model"):
    """
    Perform k-fold cross-validation for reliable performance estimates.
    
    Why: A single train/test split can be misleading. Cross-validation
    tests the model on EVERY sample by rotating the test set:
    
    Fold 1: [TEST] [train] [train] [train] [train]  → R² = 0.72
    Fold 2: [train] [TEST] [train] [train] [train]  → R² = 0.68
    ...
    Average CV R² = 0.716 ± 0.025
    
    Low standard deviation = stable model. High std = unreliable.
    
    Method: cross_val_score(cv=5) splits data into 5 equal folds,
    trains on 4, tests on 1, rotates 5 times.
    
    Args:
        model: An unfitted model instance
        X (pd.DataFrame): All features
        y (pd.Series): All targets
        cv (int): Number of folds
        model_name (str): For display
        
    Returns:
        dict: {'mean_r2': float, 'std_r2': float, 'scores': array}
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    return {
        'mean_r2': scores.mean(),
        'std_r2': scores.std(),
        'scores': scores
    }


def plot_feature_importance(model, feature_names, model_name="Model", top_n=15, save=True):
    """
    Visualize which features drive predictions the most.
    
    Why: Feature importance answers the BUSINESS question: "What factors
    matter most for a movie's success?" This is often MORE valuable than
    the prediction itself.
    
    Method: model.feature_importances_ — tree-based models track how much
    each feature reduces prediction error across all trees.
    
    Args:
        model: Trained tree-based model with feature_importances_
        feature_names: List of feature column names
        model_name (str): Name for chart title
        top_n (int): Number of top features to show
        save (bool): Whether to save to disk
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"      [SKIP] {model_name} doesn't support feature importance.")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        range(len(indices)),
        importances[indices][::-1],
        align='center',
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    )
    plt.yticks(
        range(len(indices)),
        [feature_names[i] for i in indices[::-1]],
        fontsize=10
    )
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'{model_name} — Top {top_n} Feature Importances',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save:
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/feature_importance.png', dpi=150, bbox_inches='tight')
        print(f"      Saved: results/figures/feature_importance.png")
    
    plt.close()


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


def build_models(X_train, X_test, y_train, y_test, scaling_params=None, save_models=True,
                 X_full=None, y_full=None):
    """
    Build, compare, and cross-validate multiple models.
    
    Enhanced to include 4 algorithms and cross-validation for reliable
    performance estimates. Also generates feature importance charts.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        scaling_params (dict): Dictionary with scaling parameters
        save_models (bool): Whether to save trained models to disk
        X_full (pd.DataFrame): Full feature set for cross-validation (optional)
        y_full (pd.Series): Full target for cross-validation (optional)
        
    Returns:
        dict: Dictionary containing trained models and their evaluations
    """
    results = {}
    model_names = []
    model_metrics = []
    
    # Create models directory if saving
    if save_models:
        os.makedirs('models', exist_ok=True)
    
    # ── 1. Linear Regression ──
    print("      Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression", scaling_params)
    results['linear_regression'] = {'model': lr_model, 'metrics': lr_results}
    model_names.append('Linear Regression')
    
    # ── 2. Random Forest ──
    print("      Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", scaling_params)
    results['random_forest'] = {'model': rf_model, 'metrics': rf_results}
    model_names.append('Random Forest')
    
    # ── 3. Gradient Boosting ──
    print("      Training Gradient Boosting...")
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", scaling_params)
    results['gradient_boosting'] = {'model': gb_model, 'metrics': gb_results}
    model_names.append('Gradient Boosting')
    
    # ── 4. XGBoost (if available) ──
    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        print("      Training XGBoost...")
        xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost", scaling_params)
        results['xgboost'] = {'model': xgb_model, 'metrics': xgb_results}
        model_names.append('XGBoost')
    
    # ── Cross-Validation (if full dataset provided) ──
    if X_full is not None and y_full is not None:
        print("      Running 5-fold Cross-Validation...")
        for key, name in [('linear_regression', 'Linear Regression'),
                          ('random_forest', 'Random Forest'),
                          ('gradient_boosting', 'Gradient Boosting')]:
            if key == 'linear_regression':
                cv_model = LinearRegression()
            elif key == 'random_forest':
                cv_model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                                 random_state=42)
            else:
                cv_model = GradientBoostingRegressor(n_estimators=200,
                                                      learning_rate=0.1,
                                                      max_depth=5, random_state=42)
            cv_results = cross_validate_model(cv_model, X_full, y_full, cv=5, model_name=name)
            results[key]['cv'] = cv_results
            print(f"        {name}: CV R² = {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}")
        
        if HAS_XGBOOST and 'xgboost' in results:
            cv_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1,
                                         max_depth=6, random_state=42, verbosity=0)
            cv_results = cross_validate_model(cv_model, X_full, y_full, cv=5, model_name='XGBoost')
            results['xgboost']['cv'] = cv_results
            print(f"        XGBoost: CV R² = {cv_results['mean_r2']:.3f} ± {cv_results['std_r2']:.3f}")
    
    # ── Determine best model (by test R², or CV R² if available) ──
    best_r2 = -999
    best_model_name = 'linear_regression'
    for key in results:
        if key == 'best_model':
            continue
        if not isinstance(results[key], dict) or 'metrics' not in results[key]:
            continue
        # Prefer CV score if available, otherwise use test score
        if 'cv' in results[key]:
            r2 = results[key]['cv']['mean_r2']
        else:
            r2 = results[key]['metrics']['r2']
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = key
    
    results['best_model'] = best_model_name
    
    # ── Feature Importance (from best tree-based model) ──
    best_model_obj = results[best_model_name]['model']
    if hasattr(best_model_obj, 'feature_importances_'):
        plot_feature_importance(
            best_model_obj,
            list(X_train.columns),
            model_name=best_model_name.replace('_', ' ').title(),
            save=save_models
        )
    
    # ── Save models and results ──
    if save_models:
        best_model = results[best_model_name]['model']
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(lr_model, 'models/linear_regression.pkl')
        joblib.dump(rf_model, 'models/random_forest.pkl')
        joblib.dump(gb_model, 'models/gradient_boosting.pkl')
        if xgb_model is not None:
            joblib.dump(xgb_model, 'models/xgboost.pkl')
        
        os.makedirs('results', exist_ok=True)
        
        # Build comparison table
        comparison_rows = []
        for key in ['linear_regression', 'random_forest', 'gradient_boosting', 'xgboost']:
            if key not in results or not isinstance(results[key], dict):
                continue
            row = {
                'Model': key.replace('_', ' ').title(),
                'MAE': results[key]['metrics']['mae'],
                'RMSE': results[key]['metrics']['rmse'],
                'R2_Test': results[key]['metrics']['r2'],
            }
            if 'cv' in results[key]:
                row['R2_CV_Mean'] = results[key]['cv']['mean_r2']
                row['R2_CV_Std'] = results[key]['cv']['std_r2']
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv('results/model_comparison.csv', index=False)
        
        # Create summary report
        summary_lines = [
            "=== Movie Success Prediction Model Summary ===",
            "",
            "Dataset Statistics:",
            f"- Training Samples: {len(X_train)}",
            f"- Testing Samples: {len(X_test)}",
            f"- Features: {len(X_train.columns)}",
            f"- Feature Names: {list(X_train.columns)}",
            "",
            "Model Performance (Test Set):",
        ]
        for key in ['linear_regression', 'random_forest', 'gradient_boosting', 'xgboost']:
            if key in results and isinstance(results[key], dict) and 'metrics' in results[key]:
                name = key.replace('_', ' ').title()
                m = results[key]['metrics']
                line = f"- {name}: R²={m['r2']:.3f} | MAE=${m['mae']:.2f}M | RMSE=${m['rmse']:.2f}M"
                if 'cv' in results[key]:
                    cv = results[key]['cv']
                    line += f" | CV R²={cv['mean_r2']:.3f}±{cv['std_r2']:.3f}"
                summary_lines.append(line)
        
        summary_lines.extend([
            "",
            f"Best Model: {best_model_name.replace('_', ' ').title()}",
            f"- MAE: ${results[best_model_name]['metrics']['mae']:.2f} Million",
            f"- R² Score: {results[best_model_name]['metrics']['r2']:.3f}",
        ])
        if 'cv' in results[best_model_name]:
            cv = results[best_model_name]['cv']
            summary_lines.append(f"- CV R²: {cv['mean_r2']:.3f} ± {cv['std_r2']:.3f}")
        
        with open('results/model_summary.txt', 'w') as f:
            f.write('\n'.join(summary_lines))
    
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
