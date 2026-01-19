<!--
Slide deck in Markdown.
Tip: In VS Code you can present this using a Markdown slideshow extension (e.g., Marp) or copy/paste into PPT/Google Slides.
-->

# Movie Success Prediction & Sentiment Study
### End-to-end NLP + ML pipeline (TMDB 5000)

**Team/Author:** _<Your Name>_  
**Date:** Jan 19, 2026  

<!-- Speaker notes:
One-liner: We built an end-to-end pipeline that turns movie text + metadata into revenue predictions.
-->

---

## Agenda
- Problem statement + motivation
- Dataset + feature mapping
- System workflow (Phases 1–4)
- EDA insights
- Modeling results + real-movie predictions
- Deliverables
- Limitations
- Future scope

<!-- Speaker notes:
Keep this to ~20–30 seconds.
-->

---

## Problem Statement
**Goal:** Predict a movie’s box-office revenue (in millions) using:
- Budget (production cost)
- Genre (categorical)
- Text sentiment (from overview/review-like text)

**Why it matters:**
- Early-stage planning: budget allocation & risk estimation
- Marketing: sentiment signals for positioning
- Analytics: quantify which factors correlate with revenue

<!-- Speaker notes:
Frame it as a regression problem and explain why revenue is a useful proxy for “success”.
-->

---

## What We Built (High Level)
An **end-to-end data science pipeline** that:
- Loads a movie dataset (CSV)
- Normalizes columns into a consistent schema
- Runs VADER sentiment analysis on text
- Performs EDA + saves plots
- Trains regression models + evaluates (MAE/RMSE/R²)
- Produces **real-movie predictions** from the dataset test split
- Saves artifacts (processed data, figures, models, predictions)

<!-- Speaker notes:
Emphasize: not just a model — a reproducible pipeline with outputs.
-->

---

## Dataset (TMDB 5000 Movies)
**Dataset:** TMDB 5000 Movies CSV (~4803 movies)

**Key fields (raw → project schema):**
- `title` → `Movie_Title`
- `overview` → `User_Review` (sentiment input)
- `budget` (USD) → `Budget_Millions`
- `revenue` (USD) → `Revenue_Millions` (target)
- `genres` (JSON) → `Genre` (first genre extracted)

<!-- Speaker notes:
Explain why we convert dollars to millions and why we take “first genre” for a baseline model.
-->

---

## Tools & Libraries Used
- **Python** (pipeline + notebooks)
- **pandas / numpy**: data handling
- **NLTK VADER**: sentiment scoring (`SentimentIntensityAnalyzer`)
- **scikit-learn**: training + evaluation
  - Linear Regression, Random Forest
  - metrics: MAE, RMSE, R²
- **matplotlib / seaborn**: EDA plots
- **joblib**: model persistence (`models/*.pkl`)

<!-- Speaker notes:
If asked “why VADER”: fast, interpretable, good for short text like overviews/reviews.
-->

---

## Workflow (Phase-by-Phase)
**Phase 1 — Data Acquisition & Setup**
- Load CSV / sample data
- Inspect columns + missing values
- Normalize schema to support multiple datasets

**Phase 2 — Sentiment Analysis**
- Clean text
- Compute sentiment score in [-1, +1]

**Phase 3 — EDA**
- Visualize distributions + relationships
- Save plots + basic correlation analysis

**Phase 4 — Modeling**
- Feature engineering (numeric + one-hot genre)
- Train/test split
- Train models, compare metrics, select best

<!-- Speaker notes:
Mention that `src/main.py` runs the full pipeline end-to-end.
-->

---

## EDA Highlights (From TMDB Run)
**Correlations (TMDB 5000):**
- Sentiment vs Revenue: **-0.016** (essentially no linear relationship)
- Budget vs Revenue: **0.731** (strong positive relationship)

**Interpretation:**
- Budget is a major driver of revenue.
- Sentiment from overviews alone is weak for predicting revenue (needs richer text/features).

<!-- Speaker notes:
This is an important “learning”: sentiment is useful, but not sufficient by itself.
-->

---

## Modeling Results (From TMDB Run)
We compared regression models using the same feature set.

**Best model selected:** Linear Regression
- Linear Regression: **R² = 0.629**, **MAE ≈ $55.95M**
- Random Forest: **R² = 0.592**, **MAE ≈ $56.71M**

**What this means:**
- The model explains ~63% of the variance in revenue using budget + sentiment + genre.

<!-- Speaker notes:
Explain R² and MAE quickly (variance explained and average error in millions).
-->

---

## Real-Movie Predictions (What We Output)
Instead of hypothetical examples only, the pipeline:
- Predicts revenue for **real movies in the test set**
- Prints **10 blockbusters + ~10 mid-range movies** with:
  - Movie title, genre, budget, actual vs predicted revenue, error
- Exports all test-set predictions to a CSV in `results/`

<!-- Speaker notes:
This directly satisfies the requirement: “10–20 movies prediction of revenue of the dataset used”.
-->

---

## Deliverables (Artifacts Generated)
When you run:
- `python src/main.py --data data/tmdb_5000_movies.csv`

You get:
- Processed dataset: `data/movies_with_sentiment.csv`
- EDA plots: `results/figures/*.png`
- EDA tables: `results/genre_analysis.csv`
- Model reports: `results/model_comparison.csv`, `results/model_summary.txt`
- Predictions: `results/movie_predictions.csv`
- Saved models: `models/best_model.pkl` (+ individual models)

<!-- Speaker notes:
Highlight reproducibility: rerun produces the same artifacts.
-->

---

## What We Are Doing Now
- Keeping the **CLI pipeline** and **notebook workflow** aligned to the same TMDB schema
- Improving reliability and usability:
  - consistent dataset loading paths
  - consistent column normalization
  - clearer documentation for new users

**Documentation included:**
- `README.md`, `QUICK_START.md`, `PROJECT_SUMMARY.md`, `HANDOFF_GUIDE.md`

<!-- Speaker notes:
If someone new joins, they can use QUICK_START and HANDOFF to run everything.
-->

---

## Limitations
- Revenue is influenced by many factors not included here:
  - marketing spend, release window, competition
  - cast/director popularity (“star power”)
  - distribution reach, franchise effect
- Sentiment source: using `overview` text (not actual user reviews) limits predictive power
- First-genre simplification: multi-genre effects are not captured fully

<!-- Speaker notes:
Limitations show maturity; they also set up the future scope.
-->

---

## Future Scope (Next Iterations)
**Feature engineering**
- Add release date/seasonality, runtime
- Add cast/director popularity metrics
- Use full multi-genre encoding (not just first genre)

**NLP upgrades**
- TF-IDF / embeddings on overview + keywords
- Use real user reviews or social media sentiment

**Modeling upgrades**
- Regularized linear models (Ridge/Lasso)
- Gradient boosting (XGBoost/LightGBM)
- Predict log-revenue to handle outliers, then invert

**Productization**
- Streamlit dashboard for interactive predictions
- Automated reporting and experiment tracking

<!-- Speaker notes:
Pick 2–3 that best fit your audience/time.
-->

---

# Thank You
Questions?

**Demo command:**
- `python src/main.py --data data/tmdb_5000_movies.csv`

<!-- Speaker notes:
Offer to show the results folder and the predictions CSV.
-->