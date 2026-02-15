"""
FastAPI REST API — Movie Success Prediction & Sentiment Study

Endpoints:
    POST /predict       — Predict revenue from budget, genre, sentiment
    GET  /search/{q}    — Search TMDB & predict revenue for any movie
    GET  /now-playing    — Currently playing movies with predictions
    GET  /trending       — Trending movies with predictions
    GET  /health         — Health check

Launch:
    python src/main.py --api
    # or directly:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup ──────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ── App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Success Prediction API",
    description=(
        "Predict movie box office revenue using ML models trained on 5000+ TMDB movies. "
        "Supports live TMDB search, sentiment analysis, and revenue prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Request body for movie revenue prediction."""
    budget_millions: float = Field(..., gt=0, description="Production budget in millions of dollars")
    genre: str = Field(..., description="Primary genre (e.g., Action, Comedy, Drama)")
    sentiment: float = Field(0.0, ge=-1, le=1, description="Audience sentiment score (-1 to +1)")
    vote_average: Optional[float] = Field(None, ge=0, le=10, description="TMDB vote average (0-10)")
    vote_count: Optional[int] = Field(None, ge=0, description="Number of votes on TMDB")
    runtime: Optional[int] = Field(None, gt=0, description="Runtime in minutes")
    popularity: Optional[float] = Field(None, ge=0, description="TMDB popularity score")


class PredictionResponse(BaseModel):
    """Response for movie revenue prediction."""
    predicted_revenue_millions: float
    roi_percent: float
    model_name: str
    confidence: str
    input_summary: dict


class MovieSearchResult(BaseModel):
    """Movie search result with prediction."""
    title: str
    release_date: str
    budget_millions: float
    actual_revenue_millions: float
    predicted_revenue_millions: float
    genres: List[str]
    director: Optional[str]
    vote_average: float
    sentiment_score: float
    sentiment_label: str
    review_count: int
    poster_url: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    tmdb_connected: bool


# ── Load model at startup ──────────────────────────────────────────
_model = None
_model_name = None


def _load_model():
    """Load the best available model."""
    global _model, _model_name
    
    # Try best_model first
    best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if os.path.exists(best_path):
        _model = joblib.load(best_path)
        _model_name = 'best_model'
        return
    
    # Try named models in order of preference
    for name in ['xgboost', 'gradient_boosting', 'random_forest', 'linear_regression']:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            _model = joblib.load(path)
            _model_name = name
            return


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    _load_model()
    if _model:
        print(f"✅ Model loaded: {_model_name}")
    else:
        print("⚠️ No trained model found. Run 'python src/main.py --tmdb' first.")


def _predict(budget: float, genre: str, sentiment: float,
             vote_average=None, vote_count=None, runtime=None, popularity=None,
             release_date=None, genres=None, studio=None):
    """Make a prediction using the loaded model, populating all engineered features."""
    import math
    
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Train a model first.")
    
    feature_cols = _model.feature_names_in_ if hasattr(_model, 'feature_names_in_') else []
    feature_dict = {col: 0.0 for col in feature_cols}
    
    # Core features
    if 'Budget_Millions' in feature_dict:
        feature_dict['Budget_Millions'] = budget
    if 'Sentiment_Score' in feature_dict:
        feature_dict['Sentiment_Score'] = sentiment
    
    # Genre one-hot
    genre_col = f'Genre_{genre}'
    if genre_col in feature_dict:
        feature_dict[genre_col] = 1
    if genres:
        for g in genres:
            gc = f'Genre_{g}'
            if gc in feature_dict:
                feature_dict[gc] = 1
        if 'num_genres' in feature_dict:
            feature_dict['num_genres'] = len(genres)
    elif 'num_genres' in feature_dict:
        feature_dict['num_genres'] = 1
    
    # TMDB metadata
    if vote_average is not None and 'vote_average' in feature_dict:
        feature_dict['vote_average'] = vote_average
    if vote_count is not None and 'vote_count' in feature_dict:
        feature_dict['vote_count'] = vote_count
    if runtime is not None and 'runtime' in feature_dict:
        feature_dict['runtime'] = runtime or 120
    elif 'runtime' in feature_dict:
        feature_dict['runtime'] = 120
    if popularity is not None and 'popularity' in feature_dict:
        feature_dict['popularity'] = popularity or 10
    elif 'popularity' in feature_dict:
        feature_dict['popularity'] = 10
    
    # Derived financial features
    if 'log_budget' in feature_dict:
        feature_dict['log_budget'] = math.log1p(budget) if budget > 0 else 0
    if 'budget_per_vote' in feature_dict and vote_count and vote_count > 0:
        feature_dict['budget_per_vote'] = budget / vote_count
    
    # Temporal features
    if release_date and len(release_date) >= 7:
        try:
            month = int(release_date.split('-')[1])
            if 'release_month' in feature_dict:
                feature_dict['release_month'] = month
            if 'is_summer_release' in feature_dict:
                feature_dict['is_summer_release'] = 1 if month in [5, 6, 7] else 0
            if 'is_holiday_release' in feature_dict:
                feature_dict['is_holiday_release'] = 1 if month in [11, 12] else 0
        except (ValueError, IndexError):
            pass
    
    # Studio classification
    major_studios = ['Warner Bros', 'Universal', 'Paramount', 'Walt Disney',
                     'Columbia', 'Sony', '20th Century', 'Lionsgate', 'Marvel',
                     'New Line', 'DreamWorks', 'Metro-Goldwyn']
    mid_studios = ['A24', 'Focus Features', 'Miramax', 'Relativity',
                   'Summit', 'STX', 'Blumhouse', 'Legendary']
    if studio:
        if 'Studio_Major' in feature_dict:
            feature_dict['Studio_Major'] = 1 if any(s.lower() in studio.lower() for s in major_studios) else 0
        if 'Studio_Mid' in feature_dict:
            feature_dict['Studio_Mid'] = 1 if any(s.lower() in studio.lower() for s in mid_studios) else 0
    
    # Director & actor track record (use reasonable defaults)
    if 'director_avg_revenue' in feature_dict:
        feature_dict['director_avg_revenue'] = 100  # dataset median ~100M
    if 'director_movie_count' in feature_dict:
        feature_dict['director_movie_count'] = 3
    if 'lead_actor_avg_revenue' in feature_dict:
        feature_dict['lead_actor_avg_revenue'] = 100
    
    df = pd.DataFrame([feature_dict])
    predicted = max(0, float(_model.predict(df)[0]))
    return predicted


# ════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API health, model status, and TMDB connection."""
    tmdb_ok = False
    try:
        from phase1_tmdb_api import TMDBClient
        client = TMDBClient()
        tmdb_ok, _ = client.test_connection()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_name=_model_name,
        tmdb_connected=tmdb_ok,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(req: PredictionRequest):
    """
    Predict box office revenue for a movie.
    
    Provide budget, genre, and sentiment to get a revenue prediction.
    """
    predicted = _predict(
        budget=req.budget_millions,
        genre=req.genre,
        sentiment=req.sentiment,
        vote_average=req.vote_average,
        vote_count=req.vote_count,
        runtime=req.runtime,
        popularity=req.popularity,
    )
    
    roi = ((predicted - req.budget_millions) / req.budget_millions) * 100
    
    return PredictionResponse(
        predicted_revenue_millions=round(predicted, 2),
        roi_percent=round(roi, 1),
        model_name=_model_name or "none",
        confidence="high" if req.budget_millions > 10 else "medium",
        input_summary={
            "budget": f"${req.budget_millions}M",
            "genre": req.genre,
            "sentiment": f"{req.sentiment:+.2f}",
        }
    )


@app.get("/search/{query}", tags=["TMDB Live"])
async def search_movie(
    query: str,
    year: Optional[int] = Query(None, description="Release year filter")
):
    """
    Search for a movie on TMDB, fetch reviews, run sentiment analysis,
    and predict revenue — all in one call.
    """
    try:
        from phase1_tmdb_api import TMDBClient, analyze_live_reviews
        client = TMDBClient()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ImportError:
        raise HTTPException(status_code=503, detail="TMDB API module not available")
    
    results = client.search_movie(query, year=year)
    if not results:
        raise HTTPException(status_code=404, detail=f"No movies found for '{query}'")
    
    # Get details for top result
    movie = results[0]
    details = client.get_movie_details(movie['id'])
    reviews = client.get_movie_reviews(movie['id'])
    sentiment = analyze_live_reviews(reviews)
    
    if not details:
        raise HTTPException(status_code=404, detail="Could not fetch movie details")
    
    # Predict revenue
    budget = details['budget_millions'] if details['budget_millions'] > 0 else 50  # default
    try:
        predicted = _predict(
            budget=budget,
            genre=details['primary_genre'],
            sentiment=sentiment['vader_avg'],
            vote_average=details['vote_average'],
            vote_count=details['vote_count'],
            runtime=details['runtime'],
            popularity=details['popularity'],
            release_date=details.get('release_date'),
            genres=details.get('genres'),
            studio=details.get('primary_studio'),
        )
    except Exception:
        predicted = 0.0
    
    return {
        "movie": {
            "title": details['title'],
            "release_date": details['release_date'],
            "status": details['status'],
            "genres": details['genres'],
            "director": details['director'],
            "lead_actor": details['lead_actor'],
            "studio": details['primary_studio'],
            "runtime": details['runtime'],
            "vote_average": details['vote_average'],
            "vote_count": details['vote_count'],
            "poster_url": details['poster_url'],
        },
        "financials": {
            "budget_millions": details['budget_millions'],
            "actual_revenue_millions": details['revenue_millions'],
            "predicted_revenue_millions": round(predicted, 2),
            "roi_percent": round(((predicted - budget) / budget) * 100, 1) if budget > 0 else 0,
        },
        "sentiment": {
            "vader_score": round(sentiment['vader_avg'], 3),
            "label": sentiment['sentiment_label'],
            "review_count": sentiment['review_count'],
            "confidence": round(sentiment['confidence'], 2),
        },
        "other_results": [
            {"title": r['title'], "id": r['id'], "release_date": r['release_date']}
            for r in results[1:5]
        ]
    }


@app.get("/now-playing", tags=["TMDB Live"])
async def now_playing(region: str = Query("US", description="Country code (US, IN, GB, etc.)")):
    """Get movies currently playing in theatres with revenue predictions."""
    try:
        from phase1_tmdb_api import TMDBClient
        client = TMDBClient()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    movies = client.get_now_playing(region=region)
    genre_map = client.get_genre_map()
    
    results = []
    for m in movies[:10]:
        genres = [genre_map.get(gid, 'Unknown') for gid in m.get('genre_ids', [])]
        primary_genre = genres[0] if genres else 'Unknown'
        
        try:
            predicted = _predict(budget=50, genre=primary_genre, sentiment=0.0,
                                 vote_average=m['vote_average'])
        except Exception:
            predicted = 0.0
        
        results.append({
            "title": m['title'],
            "release_date": m['release_date'],
            "vote_average": m['vote_average'],
            "genres": genres,
            "predicted_revenue_millions": round(predicted, 2),
            "poster_url": m.get('poster_url'),
        })
    
    return {"region": region, "count": len(results), "movies": results}


@app.get("/trending", tags=["TMDB Live"])
async def trending(window: str = Query("week", description="Time window: 'day' or 'week'")):
    """Get trending movies this week/today with predictions."""
    try:
        from phase1_tmdb_api import TMDBClient
        client = TMDBClient()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    movies = client.get_trending(time_window=window)
    genre_map = client.get_genre_map()
    
    results = []
    for m in movies[:10]:
        genres = [genre_map.get(gid, 'Unknown') for gid in m.get('genre_ids', [])]
        primary_genre = genres[0] if genres else 'Unknown'
        
        try:
            predicted = _predict(budget=50, genre=primary_genre, sentiment=0.0,
                                 vote_average=m['vote_average'])
        except Exception:
            predicted = 0.0
        
        results.append({
            "title": m['title'],
            "release_date": m.get('release_date', ''),
            "vote_average": m['vote_average'],
            "genres": genres,
            "predicted_revenue_millions": round(predicted, 2),
            "poster_url": m.get('poster_url'),
        })
    
    return {"window": window, "count": len(results), "movies": results}
