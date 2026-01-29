# Movie Success Prediction & Sentiment Study

A **machine learning pipeline** that predicts movie box office revenue using sentiment analysis, budget, and genre data.

---

## What This Project Does

| Input | Method | Output |
|-------|--------|--------|
| Movie Reviews | VADER Sentiment Analysis | Sentiment Score (-1 to +1) |
| Budget | Feature Engineering | Normalized Value |
| Genre | One-Hot Encoding | Category Features |
| **Combined** | **Machine Learning** | **Predicted Revenue ($M)** |

### Core Question
> **Can we predict how much money a movie will make based on its budget, genre, and audience sentiment?**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with sample data
python src/main.py --sample

# Run with your data
python src/main.py --data data/your_file.csv
```

**Output:**
```
======================================================================
       MOVIE SUCCESS PREDICTION & SENTIMENT STUDY
======================================================================

[1/4] Loading Data...           ✓ 200 movies
[2/4] Analyzing Sentiment...    ✓ Avg: 0.33
[3/4] Creating Visualizations... ✓ Saved
[4/4] Training Model...         ✓ R²: 70.9%

PIPELINE COMPLETE
Model: Linear Regression | R²: 70.9% | MAE: $75.6M
======================================================================
```

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Best Model** | Linear Regression |
| **R² Score** | 70.9% |
| **MAE** | $75.6M |

### Feature Importance

| Feature | Importance |
|---------|------------|
| Budget | 91% |
| Genre | 4% |
| Sentiment | 5% |

### Key Insights

- **Budget is the #1 predictor** - Investment drives revenue
- **Sentiment helps** - Positive reviews add value
- **Sci-Fi & Animation** - Highest earning genres
- **Drama & Western** - Lower revenue potential

---

## Prediction Examples

| Movie | Genre | Budget | Sentiment | Predicted Revenue |
|-------|-------|--------|-----------|-------------------|
| Inception 2 | Sci-Fi | $180M | Positive | **$765.8M** |
| Quiet Hearts | Drama | $30M | Mixed | **$70.0M** |
| Fury Road 2 | Adventure | $150M | Positive | **$485.7M** |

---

## Project Structure

```
Movie-Success-Prediction/
├── src/
│   ├── main.py                      # Run this
│   ├── phase1_data_acquisition.py   # Data loading
│   ├── phase2_sentiment_analysis.py # VADER sentiment
│   ├── phase3_eda.py                # Visualizations
│   └── phase4_modeling.py           # ML models
├── data/                            # Input datasets
├── models/                          # Saved models
├── results/
│   ├── figures/                     # Charts
│   └── *.csv                        # Results
├── notebooks/                       # Jupyter analysis
└── requirements.txt                 # Dependencies
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.x |
| NLP | NLTK + VADER |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |

---

## Output Files

| File | Description |
|------|-------------|
| `results/model_comparison.csv` | Model performance metrics |
| `results/movie_predictions.csv` | Revenue predictions |
| `results/figures/` | Visualization charts |
| `models/*.pkl` | Trained model files |

### Visualizations Generated

1. **Movies by Genre** - Distribution across genres
2. **Revenue by Genre** - Average revenue per genre
3. **Budget vs Revenue** - Scatter plot colored by sentiment

---

## How It Works

```
PHASE 1: LOAD DATA
  → 200 movies, 17 genres, budget & revenue
         ↓
PHASE 2: SENTIMENT ANALYSIS
  → "Amazing movie!" → VADER → Score: +0.85
         ↓
PHASE 3: VISUALIZATIONS
  → 3 charts saved to results/figures/
         ↓
PHASE 4: TRAIN & PREDICT
  → Linear Regression wins → Predict new movies
```

---

## Limitations

| Limitation | Impact |
|------------|--------|
| Sample Data | Uses synthetic 200-movie dataset |
| Limited Features | Only budget, sentiment, genre |
| No Time Factor | Ignores release date |
| Basic Models | Linear regression only |

---

## Future Enhancements

- [ ] Real IMDB/TMDB datasets (50,000+ movies)
- [ ] Add cast, director features
- [ ] Implement XGBoost, neural networks
- [ ] Build web interface (Streamlit)
- [ ] Deploy to cloud (AWS/Heroku)

---

## Business Value

| Stakeholder | Benefit |
|-------------|---------|
| Movie Studios | Estimate ROI before production |
| Marketing Teams | Target campaigns effectively |
| Investors | Data-driven funding decisions |
| Distributors | Optimize release strategies |

---

## Project Info

**Project**: Movie Success Prediction & Sentiment Study  
**Technology**: Python + VADER + Scikit-learn  
**Date**: January 2026

---

*Built with Python | Powered by VADER Sentiment + Linear Regression*

