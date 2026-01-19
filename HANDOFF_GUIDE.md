# Project Handoff Guide: Movie Success Prediction & Sentiment Study

This guide is written for someone who did **not** build this project.
It explains what the project does, what to install (and why), how to run it on the dataset, and what outputs you should expect.

---

## 1) What this project does (high-level)

This is an end-to-end **data science pipeline** that:

1. Loads a movies dataset (CSV)
2. Extracts/normalizes key columns (budget, revenue, genre)
3. Runs **VADER sentiment analysis** on text (movie overview/review)
4. Generates **EDA visualizations** (plots + correlation analysis)
5. Trains regression models to predict **box office revenue**
6. Prints **real movie predictions** from the dataset (not just hypothetical)
7. Saves outputs (processed data, plots, models, prediction CSV)

**Main question:** can we predict revenue using **budget + genre + sentiment**?

---

## 2) What dataset to use (recommended)

### Recommended dataset: TMDB 5000 Movies (CSV)
This repo is currently set up to run cleanly on a TMDB-style dataset with columns like:

- `title` (movie name)
- `overview` (text used for sentiment)
- `budget` (raw dollars)
- `revenue` (raw dollars)
- `genres` (JSON list of objects, e.g. `[{"id": 28, "name": "Action"}, ...]`)

In this workspace, the dataset already exists at:
- [data/tmdb_5000_movies.csv](data/tmdb_5000_movies.csv)

### If you use another dataset
The pipeline tries to adapt using common column names.

**For sentiment text:**
- If `User_Review` is missing, it auto-picks a reasonable text column (preferably `overview`).

**For revenue:**
- If `Revenue_Millions` is missing, it tries `gross`, then `revenue`, then `Revenue (Millions)`.

**For budget:**
- If `Budget_Millions` is missing, it tries `budget`.

**For genre:**
- If `Genre` is missing, it tries `genres`.
  - If `genres` is JSON (TMDB), it extracts the first genre name.
  - If `genres` is pipe-separated (e.g. `Action|Drama`), it takes the first part.

---

## 3) What to install (and why)

All dependencies are in:
- [requirements.txt](requirements.txt)

### Packages (what they do)
- **pandas**: load CSVs, clean columns, create feature tables
- **numpy**: numeric operations and sampling
- **matplotlib**: plotting base library
- **seaborn**: nicer statistical plots (heatmaps, boxplots)
- **scikit-learn**: train/test split, Linear Regression, Random Forest, metrics (MAE/RMSE/R²)
- **joblib**: save trained models to disk (`models/*.pkl`)
- **nltk**: VADER sentiment analyzer + lexicon
- **jupyter + notebook**: run the interactive notebook version of the analysis

### NLTK VADER lexicon
VADER needs a dictionary file called `vader_lexicon`.
The code in [src/phase2_sentiment_analysis.py](src/phase2_sentiment_analysis.py) downloads it automatically if missing.

---

## 4) How the pipeline works (Phase-by-phase)

### Phase 1: Data acquisition + inspection
Code:
- [src/phase1_data_acquisition.py](src/phase1_data_acquisition.py)

What happens:
- Checks dependencies
- Loads a CSV OR generates sample data
- Prints shape, columns, missing values, stats
- Normalizes columns in the main pipeline so later phases work consistently

Outputs:
- If using sample mode, writes [data/sample_movies_dataset.csv](data/sample_movies_dataset.csv) (if possible)


### Phase 2: Sentiment analysis (VADER)
Code:
- [src/phase2_sentiment_analysis.py](src/phase2_sentiment_analysis.py)

What happens:
- Finds a text column (`User_Review` preferred; otherwise `overview` for TMDB)
- Preprocesses text (lowercase + basic cleaning)
- VADER computes `Sentiment_Score` in range [-1, +1]
- Adds `Sentiment_Category` bins (Very Negative … Very Positive)

Outputs (columns added):
- `Sentiment_Score`
- `Sentiment_Category`


### Phase 3: EDA + visualizations
Code:
- [src/phase3_eda.py](src/phase3_eda.py)

What happens:
- Plots distributions and relationships
- Computes correlations (especially sentiment vs revenue, budget vs revenue)

Outputs:
- [results/figures/sentiment_distribution.png](results/figures/sentiment_distribution.png)
- [results/figures/sentiment_vs_revenue.png](results/figures/sentiment_vs_revenue.png)
- [results/figures/genre_sentiment.png](results/figures/genre_sentiment.png)
- [results/figures/genre_revenue_boxplot.png](results/figures/genre_revenue_boxplot.png)
- [results/figures/budget_vs_revenue.png](results/figures/budget_vs_revenue.png)
- [results/figures/correlation_heatmap.png](results/figures/correlation_heatmap.png)
- [results/genre_analysis.csv](results/genre_analysis.csv)


### Phase 4: Modeling (prediction)
Code:
- [src/phase4_modeling.py](src/phase4_modeling.py)

What happens:
- Builds features:
  - numeric: `Budget_Millions`, `Sentiment_Score`
  - categorical: one-hot encoded `Genre_*`
- Splits into train/test (80/20)
- Trains models:
  - Linear Regression
  - Random Forest
- Evaluates with:
  - MAE (mean absolute error)
  - RMSE
  - R²
- Selects the best model by R²
- Saves trained models and comparison reports

Outputs:
- [models/best_model.pkl](models/best_model.pkl)
- [models/linear_regression.pkl](models/linear_regression.pkl)
- [models/random_forest.pkl](models/random_forest.pkl)
- [results/model_comparison.csv](results/model_comparison.csv)
- [results/model_summary.txt](results/model_summary.txt)


### “Real movie predictions” output
Code:
- [src/main.py](src/main.py)

What happens:
- Uses the best model on the **test set**
- Prints **10 real blockbusters** + **10 mid-range movies** with:
  - Movie title
  - Genre
  - Budget
  - Actual vs Predicted revenue
  - Error
- Saves all test-set predictions to:
  - [results/movie_predictions.csv](results/movie_predictions.csv)


### Final processed dataset
At the end, the pipeline writes:
- [data/movies_with_sentiment.csv](data/movies_with_sentiment.csv)

This file contains the original dataset plus engineered columns like:
- `Revenue_Millions`, `Budget_Millions`, `Genre`, `Sentiment_Score`, `Sentiment_Category`

---

## 5) How to run (Windows-friendly)

### Option A: Run the complete pipeline on TMDB dataset
From the repo root:

```bash
python src/main.py --data data/tmdb_5000_movies.csv
```

You should see:
- Phase 1 data inspection
- Phase 2 VADER sentiment summary
- Phase 3 plots saved into `results/figures/`
- Phase 4 model training metrics (MAE/RMSE/R²)
- “PREDICTIONS ON REAL MOVIES FROM DATASET” section


### Option B: Run the complete pipeline with generated sample data

```bash
python src/main.py --sample
```

Use this to quickly verify the environment works even without a real dataset.


### Option C: Run phases individually

```bash
python src/phase1_data_acquisition.py
python src/phase2_sentiment_analysis.py
python src/phase3_eda.py
python src/phase4_modeling.py
```

(Important: running individual phases may require you to pass/prepare data between phases; the recommended path is running [src/main.py](src/main.py).)

---

## 6) What outcomes to expect (interpreting results)

### Model metrics
- **MAE**: average absolute error (in millions). Example: MAE $55M means predictions are off by ~$55M on average.
- **RMSE**: like MAE but penalizes big mistakes more.
- **R²**: how much variance the model explains. Closer to 1 is better.

### Typical insights from this project
- **Budget** usually correlates strongly with revenue.
- **Sentiment** may be weakly correlated with revenue (dataset dependent), but it can still help as a feature.
- **Genre** shifts the baseline expectation (some genres earn more on average).

### Where to see the outcomes
- Visual insights: [results/figures/](results/figures/)
- Quantitative model comparison: [results/model_comparison.csv](results/model_comparison.csv)
- Detailed prediction list (real movies): [results/movie_predictions.csv](results/movie_predictions.csv)

---

## 7) Notebook (interactive workflow)

Notebook:
- [notebooks/main_analysis.ipynb](notebooks/main_analysis.ipynb)

Run:
```bash
jupyter notebook
```
Open the notebook and run cells top-to-bottom.

---

## 8) Troubleshooting

### “Downloading VADER lexicon…” every time
NLTK will download once and cache it, but in locked-down environments it may re-download.
Fix by ensuring your user has write access to NLTK’s data directory.

### Unicode / Windows console issues
The code avoids non-ASCII symbols in prints.
If you still see encoding issues, run PowerShell with UTF-8:
```powershell
chcp 65001
```

### Dataset column mismatch
If your CSV doesn’t have `overview/title/budget/revenue/genres`, update the mapping section in:
- [src/main.py](src/main.py)

---

## 9) Advanced: how to extend this project

Good next improvements (typical “advanced” upgrades):
- Add more predictors: `runtime`, `vote_average`, `vote_count`, `popularity`, release year
- Add log-transform: train on `log1p(revenue)` to reduce blockbuster skew
- Try stronger models: Gradient Boosting, XGBoost/LightGBM, CatBoost
- Add proper cross-validation + hyperparameter search
- Evaluate with train/validation/test split

Code starting points:
- Feature engineering: [src/phase4_modeling.py](src/phase4_modeling.py)
- Dataset normalization: [src/main.py](src/main.py)

---

## 10) “Where are the main things?” (cheat sheet)

- Main entrypoint: [src/main.py](src/main.py)
- Sentiment (VADER): [src/phase2_sentiment_analysis.py](src/phase2_sentiment_analysis.py)
- Plots/EDA: [src/phase3_eda.py](src/phase3_eda.py)
- Modeling + saved models: [src/phase4_modeling.py](src/phase4_modeling.py)
- Outputs:
  - Processed data: [data/movies_with_sentiment.csv](data/movies_with_sentiment.csv)
  - Predictions CSV: [results/movie_predictions.csv](results/movie_predictions.csv)
  - Figures: [results/figures/](results/figures/)
  - Models: [models/](models/)
