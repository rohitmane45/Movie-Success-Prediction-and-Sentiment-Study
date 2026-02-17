# 🎬 Movie Success Prediction & Sentiment Study

> **Predict movie box office revenue using machine learning, NLP sentiment analysis, and real-time TMDB data.**

An end-to-end data science project that combines **natural language processing**, **feature engineering**, and **ensemble ML models** to predict how much money a movie will earn — trained on **4,800+ real TMDB movies** and capable of making **live predictions** on any movie using the TMDB API.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best_Model-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Problem Statement

> **Can we predict how much revenue a movie will generate based on its budget, genre, audience sentiment, cast, studio, and release timing?**

This project answers that question by building a full ML pipeline — from raw data ingestion to a production-ready dashboard and REST API.

---

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **R² Score** | **80.3%** |
| **MAE** | $45.5M |
| **RMSE** | $102.0M |
| **Dataset** | 4,800+ real TMDB movies |
| **Features** | 36 engineered features |

### Model Comparison

| Model | R² (Test) | MAE ($M) | RMSE ($M) | CV R² |
|-------|-----------|----------|-----------|-------|
| **XGBoost** | **0.803** | **45.5** | **102.0** | 0.556 |
| Gradient Boosting | 0.792 | 46.6 | 104.8 | 0.578 |
| Random Forest | 0.756 | 47.7 | 113.3 | 0.524 |
| Linear Regression | 0.735 | 59.1 | 118.1 | 0.241 |

---

## ✨ Features

### 🔬 ML Pipeline
- **Dual Sentiment Analysis** — VADER (rule-based) + optional DistilBERT (transformer-based)
- **36 Engineered Features** — budget, genre, cast/director track records, release timing, studio tier, sentiment scores, and more
- **4 ML Models** — Linear Regression, Random Forest, Gradient Boosting, and XGBoost with cross-validation
- **Real TMDB Data** — Trained on 4,800+ movies from the TMDB 5000 dataset

### 📊 Interactive Dashboard (Streamlit)
- **Overview** — KPI cards, revenue distribution, genre analysis
- **Data Explorer** — Filter and explore the full dataset interactively
- **Sentiment Analysis** — Visualize VADER vs. transformer sentiment comparisons
- **Model Performance** — Compare all 4 models side-by-side with actual vs predicted charts
- **🔴 Live Predictor** — Search any movie → fetch live TMDB data → real-time sentiment analysis → ML revenue prediction
- **💼 Business Intelligence** — ROI calculator, genre trend analysis, competitive benchmarking

### 🌐 REST API (FastAPI)
- `POST /predict` — Predict revenue from budget, genre, and sentiment
- `GET /search/{query}` — Search any movie on TMDB and get instant prediction
- `GET /now-playing` — Currently playing movies with predictions
- `GET /trending` — Trending movies with predictions
- `GET /health` — Health check with model and TMDB connection status
- Auto-generated Swagger docs at `/docs`

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/Movie-Success-Prediction-and-Sentiment-Study.git
cd Movie-Success-Prediction-and-Sentiment-Study

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train on the TMDB 5000 dataset (recommended)
python src/main.py --tmdb

# Or use sample data for a quick demo
python src/main.py --sample
```

**Output:**
```
======================================================================
       MOVIE SUCCESS PREDICTION & SENTIMENT STUDY
======================================================================

[1/4] Loading Data...              ✓ 4803 movies loaded
[2/4] Analyzing Sentiment...       ✓ VADER avg: 0.33
[3/4] Creating Visualizations...   ✓ 9 charts saved
[4/4] Training Models...           ✓ Best: XGBoost (R²: 80.3%)

PIPELINE COMPLETE ✅
======================================================================
```

### 3. Launch the Dashboard

```bash
python src/main.py --dashboard
# Open http://localhost:8501
```

### 4. Launch the REST API

```bash
python src/main.py --api
# Swagger docs at http://localhost:8000/docs
```

### 5. Enable Live Predictions (Optional)

To use the **Live Predictor** and **REST API TMDB endpoints**, you need a free TMDB API key:

```bash
# Get a free key at https://www.themoviedb.org/settings/api
# Create a .env file in the project root:
echo TMDB_API_KEY=your_api_key_here > .env
```

---

## 📁 Project Structure

```
Movie-Success-Prediction-and-Sentiment-Study/
│
├── src/
│   ├── main.py                         # Entry point — CLI with --tmdb, --dashboard, --api flags
│   │
│   ├── phase1_data_acquisition.py      # Data loading & sample data generation
│   ├── phase1_tmdb_loader.py           # TMDB 5000 dataset loader & preprocessing
│   ├── phase1_feature_engineering.py   # 36-feature engineering pipeline
│   ├── phase1_review_loader.py         # Real review data loader
│   ├── phase1_tmdb_api.py              # Live TMDB API client (search, details, reviews)
│   │
│   ├── phase2_sentiment_analysis.py    # VADER sentiment analysis
│   ├── phase2_transformer_sentiment.py # DistilBERT transformer sentiment
│   │
│   ├── phase3_eda.py                   # EDA & visualization generation
│   ├── phase4_modeling.py              # ML model training & evaluation
│   │
│   ├── dashboard.py                    # Streamlit dashboard (7 pages)
│   └── api.py                          # FastAPI REST API
│
├── data/                               # TMDB datasets (auto-downloaded)
├── models/                             # Trained model files (.pkl)
│   ├── best_model.pkl                  # Best performing model (XGBoost)
│   ├── xgboost.pkl
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   └── linear_regression.pkl
│
├── results/
│   ├── figures/                        # 9 visualization charts
│   ├── model_comparison.csv            # Model performance metrics
│   ├── movie_predictions.csv           # Revenue predictions for all movies
│   └── genre_analysis.csv              # Genre-level statistics
│
├── notebooks/                          # Jupyter notebook for exploration
├── requirements.txt                    # Python dependencies
├── .env.example                        # Template for TMDB API key
└── README.md
```

---

## 🔧 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│  TMDB 5000 dataset → 4,803 movies with budget, revenue,    │
│  genres, cast, crew, reviews, and metadata                  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│  36 features: budget, log_budget, genre one-hot,            │
│  director/actor track records, studio tier,                 │
│  release timing, sentiment scores, vote metrics             │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 SENTIMENT ANALYSIS                          │
│  VADER (rule-based)  ──→  Sentiment Score [-1, +1]          │
│  DistilBERT (optional) ─→  Transformer Confidence           │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               MODEL TRAINING & EVALUATION                   │
│  4 models trained with 5-fold cross-validation              │
│  XGBoost wins → saved as best_model.pkl                     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                               │
│  Streamlit Dashboard (7 pages) + FastAPI REST API           │
│  Live Predictor: Search any movie → TMDB → Prediction       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Visualizations Generated

The pipeline automatically generates 9 publication-ready charts:

| Chart | Description |
|-------|-------------|
| `budget_vs_revenue.png` | Scatter plot of budget vs revenue (colored by sentiment) |
| `sentiment_vs_revenue.png` | Sentiment score impact on revenue |
| `sentiment_distribution.png` | Distribution of VADER sentiment scores |
| `movies_by_genre.png` | Movie count per genre |
| `revenue_by_genre.png` | Average revenue per genre |
| `genre_revenue_boxplot.png` | Revenue spread within each genre |
| `genre_sentiment.png` | Average sentiment by genre |
| `correlation_heatmap.png` | Feature correlation matrix |
| `feature_importance.png` | Top feature importances from ensemble models |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ |
| **NLP** | NLTK (VADER) + HuggingFace Transformers (DistilBERT) |
| **ML Models** | Scikit-learn, XGBoost |
| **Feature Engineering** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **REST API** | FastAPI + Uvicorn |
| **Data Source** | TMDB API + TMDB 5000 Dataset |

---

## 📈 Feature Importance (Top 10)

The model uses **36 engineered features**. The most influential ones:

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `Budget_Millions` | Production budget in millions |
| 2 | `vote_count` | Number of TMDB votes (popularity proxy) |
| 3 | `popularity` | TMDB popularity score |
| 4 | `director_avg_revenue` | Director's historical average revenue |
| 5 | `lead_actor_avg_revenue` | Lead actor's historical average revenue |
| 6 | `vote_average` | TMDB audience rating (1-10) |
| 7 | `log_budget` | Log-transformed budget (reduces skew) |
| 8 | `runtime` | Movie length in minutes |
| 9 | `Sentiment_Score` | VADER sentiment from reviews |
| 10 | `is_summer_release` | Released in May/June/July (blockbuster season) |

---

## 💡 Key Insights

- **Budget is king** — Production investment is the strongest single predictor of revenue
- **Star power matters** — Directors and actors with high historical revenue drive predictions up
- **Timing is strategic** — Summer and holiday releases earn significantly more
- **Studio backing helps** — Major studios (Disney, Warner Bros, Universal) correlate with higher revenue
- **Sentiment adds signal** — Positive audience sentiment contributes measurably to revenue prediction
- **Genre shapes expectations** — Sci-Fi, Animation, and Adventure have the highest earning potential

---

## 💼 Business Value

| Stakeholder | How This Helps |
|-------------|---------------|
| **Movie Studios** | Estimate ROI before greenlighting production |
| **Marketing Teams** | Budget campaigns based on predicted performance |
| **Investors** | Data-driven funding decisions with confidence scores |
| **Distributors** | Optimize release window and regional strategy |
| **Analysts** | Genre trend analysis and competitive benchmarking |

---

## 🧪 CLI Options

```bash
python src/main.py [OPTIONS]

Options:
  --sample        Use synthetic sample data for demo
  --tmdb          Use real TMDB 5000 dataset (recommended)
  --data FILE     Use custom CSV file
  --transformer   Enable DistilBERT sentiment (requires torch)
  --dashboard     Launch Streamlit dashboard
  --api           Launch FastAPI REST API server
```

---

## 📝 API Examples

### Predict Revenue
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"budget_millions": 150, "genre": "Action", "sentiment": 0.7}'
```

### Search & Predict Live
```bash
curl http://localhost:8000/search/Inception
```

### Response:
```json
{
  "movie": { "title": "Inception", "director": "Christopher Nolan", ... },
  "financials": {
    "budget_millions": 160.0,
    "actual_revenue_millions": 839.0,
    "predicted_revenue_millions": 285.4,
    "roi_percent": 78.4
  },
  "sentiment": { "vader_score": 0.634, "label": "Positive", "review_count": 15 }
}
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TMDB_API_KEY` | For live features | Free API key from [themoviedb.org](https://www.themoviedb.org/settings/api) |

### Optional Dependencies

```bash
# Enable DistilBERT transformer sentiment (downloads ~2GB model)
pip install torch transformers
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Built with ❤️ using Python • VADER & DistilBERT NLP • XGBoost • Streamlit • FastAPI**



# Train models
.venv\Scripts\python.exe src/main.py --tmdb

# Launch dashboard
.venv\Scripts\python.exe src/main.py --dashboard

# Launch API (in another terminal)
.venv\Scripts\python.exe src/main.py --api