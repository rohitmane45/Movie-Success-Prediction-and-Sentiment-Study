# 🎬 Movie Success Prediction & Sentiment Study

---

## 📌 PROJECT OVERVIEW

### What This Project Does

A **complete Data Science pipeline** that predicts movie box office revenue by combining:

| Input | Method | Output |
|-------|--------|--------|
| Movie Reviews | VADER Sentiment Analysis | Sentiment Score (-1 to +1) |
| Budget | Feature Engineering | Normalized Value |
| Genre | One-Hot Encoding | Category Features |
| **Combined** | **Machine Learning** | **Predicted Revenue ($M)** |

### The Core Question
> **Can we predict how much money a movie will make based on its budget, genre, and audience sentiment?**

---

## 🎯 PROJECT OBJECTIVES

1. **Analyze** movie review sentiment using NLP (VADER)
2. **Visualize** relationships between sentiment, budget, genre, and revenue
3. **Predict** box office success using regression models
4. **Build** a production-ready data science workflow

---

## 🔬 HOW IT WORKS

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: LOAD DATA                                         │
│  → 200 movies, 17 genres, budget & revenue data            │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: SENTIMENT ANALYSIS                                │
│  → "Amazing movie!" → VADER → Score: +0.85                 │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: VISUALIZATIONS                                    │
│  → Movies by Genre | Revenue by Genre | Budget vs Revenue  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: TRAIN MODELS & PREDICT                           │
│  → Linear Regression vs Random Forest → Best Model Wins    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 PROJECT OUTCOMES

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Best Model** | Linear Regression | Outperformed Random Forest |
| **R² Score** | **70.9%** | Model explains 71% of revenue variance |
| **MAE** | **$75.6M** | Average prediction error |
| **Accuracy** | Good | Suitable for estimations |

### Key Insights Discovered

| Finding | Insight |
|---------|---------|
| 🏆 **Budget = 91% importance** | Investment is the #1 revenue predictor |
| 💬 **Sentiment = 5% importance** | Positive reviews help, but budget dominates |
| 🎬 **Sci-Fi & Animation** | Highest earning genres |
| 📉 **Drama & Western** | Lower revenue potential |

---

## 🎥 PREDICTION EXAMPLES

### New Movie Revenue Predictions

| Movie | Genre | Budget | Sentiment | Predicted Revenue |
|-------|-------|--------|-----------|-------------------|
| **Inception 2** | Sci-Fi | $180M | Positive | **$765.8M** |
| **Quiet Hearts** | Drama | $30M | Mixed | **$70.0M** |
| **Fury Road 2** | Adventure | $150M | Positive | **$485.7M** |

### How Prediction Works
```
Revenue = f(Budget, Sentiment, Genre)

Example: Inception 2
  Budget: $180M (high investment)
  Sentiment: 0.9 (positive reviews expected)
  Genre: Sci-Fi (high-earning category)
  → Model predicts: $765.8M revenue
```

---

## 📁 PROJECT DELIVERABLES

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **Genre Distribution** | `results/figures/movies_by_genre.png` | Number of movies per genre |
| **Revenue Analysis** | `results/figures/revenue_by_genre.png` | Average revenue by genre |
| **Budget vs Revenue** | `results/figures/budget_vs_revenue.png` | Scatter plot with sentiment colors |
| **Predictions CSV** | `results/movie_predictions.csv` | All test set predictions |
| **Model Summary** | `results/model_summary.txt` | Performance metrics report |
| **Trained Model** | `models/best_model.pkl` | Reusable prediction model |
| **Processed Data** | `data/movies_with_sentiment.csv` | Data with sentiment scores |

---

## 🛠️ TECHNICAL STACK

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.x | Core development |
| **NLP** | NLTK + VADER | Sentiment analysis |
| **Machine Learning** | Scikit-learn | Regression models |
| **Visualization** | Matplotlib, Seaborn | Charts & plots |
| **Data Processing** | Pandas, NumPy | Data manipulation |

---

## 📂 PROJECT STRUCTURE

```
Movie-Success-Prediction/
├── src/
│   ├── main.py                      # Main pipeline (run this)
│   ├── phase1_data_acquisition.py   # Data loading
│   ├── phase2_sentiment_analysis.py # VADER sentiment
│   ├── phase3_eda.py                # Visualizations
│   └── phase4_modeling.py           # ML models
├── data/                            # Datasets
├── models/                          # Saved models
├── results/
│   ├── figures/                     # Charts
│   └── *.csv                        # Results
├── notebooks/                       # Jupyter analysis
└── requirements.txt                 # Dependencies
```

---

## ⚠️ CURRENT LIMITATIONS

| Limitation | Impact |
|------------|--------|
| **Sample Data** | Uses synthetic 200-movie dataset |
| **Limited Features** | Only budget, sentiment, genre used |
| **No Time Factor** | Ignores release date, competition |
| **Basic Models** | Linear regression only |
| **Single Text Source** | No multi-platform sentiment |

---

## 🚀 FUTURE ENHANCEMENTS

### Phase 1: Data Improvements
- [ ] Integrate real IMDB/TMDB/Kaggle datasets (50,000+ movies)
- [ ] Scrape live reviews from Rotten Tomatoes, Metacritic
- [ ] Add cast, director, production company features

### Phase 2: Model Upgrades
- [ ] Implement Gradient Boosting, XGBoost, LightGBM
- [ ] Try deep learning (LSTM for review sequences)
- [ ] Use BERT/RoBERTa for advanced sentiment analysis
- [ ] Add cross-validation for robust evaluation

### Phase 3: New Features
- [ ] Release date and seasonality effects
- [ ] Marketing budget and social media buzz
- [ ] Sequel/franchise information
- [ ] Competition analysis (same-week releases)
- [ ] Actor star power scoring

### Phase 4: Deployment
- [ ] Build web interface (Streamlit/Flask)
- [ ] Create REST API for predictions
- [ ] Deploy to cloud (AWS/Heroku/Vercel)
- [ ] Real-time sentiment monitoring dashboard

---

## 💼 BUSINESS VALUE

| Stakeholder | How This Helps |
|-------------|----------------|
| **Movie Studios** | Estimate ROI before greenlighting projects |
| **Marketing Teams** | Target campaigns to improve sentiment |
| **Investors** | Make data-driven funding decisions |
| **Distributors** | Optimize release strategies |
| **Analysts** | Understand what drives box office success |

---

## 🏃 HOW TO RUN

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run complete pipeline
python src/main.py

# Output:
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

## 📈 CONCLUSION

### What We Built
✅ End-to-end ML pipeline for movie revenue prediction  
✅ Automated sentiment analysis from text reviews  
✅ Clean visualizations showing data relationships  
✅ Trained model ready for new predictions  

### What We Learned
✅ **Budget is the strongest predictor** (91% importance)  
✅ **Sentiment matters** but less than investment  
✅ **Genre significantly impacts** revenue expectations  
✅ **Simple models can work well** with good features  

### Impact
✅ Data-driven movie investment decisions  
✅ Quantified relationship between reviews and revenue  
✅ Framework extensible to real-world data  

---

## 📞 PROJECT INFO

**Project**: Movie Success Prediction & Sentiment Study  
**Technology**: Python + VADER + Scikit-learn  
**Date**: January 2026  

---

*Built with Python | Powered by VADER Sentiment + Linear Regression*
