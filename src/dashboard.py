"""
Interactive Streamlit Dashboard — Movie Success Prediction & Sentiment Study

Launch with:
    streamlit run src/dashboard.py

A premium 5-page analytics dashboard with dark glassmorphism theme,
interactive Plotly charts, and real-time revenue predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# ── Path setup ──────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium Dark Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* === GLOBAL DARK THEME === */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 1.05rem;
        margin-top: 4px;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #10b981;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #a5b4fc;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }

    /* Data tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading (cached) ──────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the best available dataset."""
    # Priority 1: sentiment-enriched data
    sentiment_path = os.path.join(DATA_DIR, 'movies_with_sentiment.csv')
    if os.path.exists(sentiment_path):
        df = pd.read_csv(sentiment_path)
        if len(df) > 50:
            return df, "movies_with_sentiment.csv"

    # Priority 2: TMDB data (load and process)
    tmdb_path = os.path.join(DATA_DIR, 'tmdb_5000_movies.csv')
    if os.path.exists(tmdb_path):
        try:
            from phase1_tmdb_loader import load_tmdb_data
            df = load_tmdb_data(
                movies_path=tmdb_path,
                credits_path=os.path.join(DATA_DIR, 'tmdb_5000_credits.csv')
            )
            return df, "TMDB 5000 (real)"
        except Exception:
            df = pd.read_csv(tmdb_path)
            return df, "tmdb_5000_movies.csv (raw)"

    # Priority 3: sample data
    sample_path = os.path.join(DATA_DIR, 'sample_movies_dataset.csv')
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path), "sample_movies_dataset.csv"

    # Generate minimal sample
    return pd.DataFrame({
        'title': ['Sample Movie'],
        'Genre': ['Action'],
        'Budget_Millions': [100],
        'Revenue_Millions': [300],
        'Sentiment_Score': [0.5]
    }), "Generated Sample"


@st.cache_data
def load_model_results():
    """Load model comparison results."""
    path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(path):
        results = pd.read_csv(path)
        # Normalize column names — the modeling module saves 'R2_Test'
        # but the dashboard uses 'R2' for simplicity
        if 'R2_Test' in results.columns and 'R2' not in results.columns:
            results = results.rename(columns={'R2_Test': 'R2'})
        return results
    return None


@st.cache_resource
def load_best_model():
    """Load the best trained model."""
    best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    if os.path.exists(best_path):
        return joblib.load(best_path), 'best_model'

    # Try other models
    for name in ['xgboost', 'gradient_boosting', 'random_forest', 'linear_regression']:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            return joblib.load(path), name

    return None, None


@st.cache_data
def load_predictions():
    """Load prediction results."""
    path = os.path.join(RESULTS_DIR, 'movie_predictions.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ── Plotly Theme ────────────────────────────────────────────────────
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': 'rgba(255,255,255,0.8)', 'family': 'Inter, sans-serif'},
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.06)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.06)',
            'zerolinecolor': 'rgba(255,255,255,0.1)'
        },
        'colorway': ['#6366f1', '#a855f7', '#ec4899', '#10b981',
                      '#f59e0b', '#3b82f6', '#ef4444', '#14b8a6'],
    }
}

COLOR_PALETTE = ['#6366f1', '#a855f7', '#ec4899', '#10b981',
                  '#f59e0b', '#3b82f6', '#ef4444', '#14b8a6',
                  '#8b5cf6', '#06b6d4', '#f97316', '#84cc16']


def apply_plotly_theme(fig):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.8)', family='Inter, sans-serif'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                   zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                   zerolinecolor='rgba(255,255,255,0.1)'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


# ── Sidebar Navigation ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span style="font-size: 3rem;">🎬</span>
        <h2 style="background: linear-gradient(135deg, #6366f1, #a855f7);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 10px 0 5px 0;">Movie Predictor</h2>
        <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">
            ML-Powered Revenue Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🎯 Revenue Predictor", "🔴 Live Predictor",
         "📊 Sentiment Trends", "🔍 Movie Comparison",
         "🤖 Model Performance", "💼 Business Intelligence"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: center;">
        Powered by Scikit-learn & VADER<br>
        Live Data via TMDB API<br>
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Load Data ───────────────────────────────────────────────────────
df, data_source = load_dataset()
model, model_name = load_best_model()
model_results = load_model_results()
predictions_df = load_predictions()


# ================================================================
#  PAGE 1: OVERVIEW
# ================================================================
if page == "🏠 Overview":
    st.markdown('<h1 class="hero-title">Movie Success Prediction & Sentiment Study</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Predicting box office revenue using ML, NLP sentiment analysis, and real movie data</p>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Movies</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        n_genres = df['Genre'].nunique() if 'Genre' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Genres</div>
            <div class="metric-value">{n_genres}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if model_results is not None and 'R2' in model_results.columns:
            best_r2 = model_results['R2'].max()
        else:
            best_r2 = 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best R² Score</div>
            <div class="metric-value">{best_r2:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if 'Revenue_Millions' in df.columns:
            avg_rev = df['Revenue_Millions'].mean()
        else:
            avg_rev = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Revenue</div>
            <div class="metric-value">${avg_rev:.0f}M</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">📈 Revenue Distribution</div>',
                    unsafe_allow_html=True)
        if 'Revenue_Millions' in df.columns:
            fig = px.histogram(
                df, x='Revenue_Millions',
                nbins=40,
                color_discrete_sequence=['#6366f1'],
                labels={'Revenue_Millions': 'Revenue ($M)', 'count': 'Movies'}
            )
            fig.update_layout(showlegend=False)
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">🎭 Genre Distribution</div>',
                    unsafe_allow_html=True)
        if 'Genre' in df.columns:
            genre_counts = df['Genre'].value_counts().head(12)
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                color=genre_counts.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Number of Movies', 'y': 'Genre'}
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis=dict(autorange='reversed'))
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    # Data source info
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <span style="color: rgba(255,255,255,0.5);">Data Source:</span>
        <strong style="color: #a5b4fc;"> {data_source}</strong>
        &nbsp;•&nbsp;
        <span style="color: rgba(255,255,255,0.5);">Pipeline:</span>
        <strong style="color: #a5b4fc;"> VADER + {'DistilBERT' if 'Transformer_Sentiment' in df.columns else 'VADER only'}</strong>
    </div>
    """, unsafe_allow_html=True)


# ================================================================
#  PAGE 2: REVENUE PREDICTOR
# ================================================================
elif page == "🎯 Revenue Predictor":
    st.markdown('<h1 class="hero-title">🎯 Revenue Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Enter movie details to predict box office revenue</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ No trained model found. Run `python src/main.py --tmdb` first to train a model.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="section-header">🎬 Movie Details</div>',
                        unsafe_allow_html=True)

            # Budget input
            budget = st.slider(
                "💰 Budget ($M)",
                min_value=1, max_value=400, value=100, step=5,
                help="Production budget in millions of dollars"
            )

            # Genre input
            available_genres = sorted(df['Genre'].unique().tolist()) if 'Genre' in df.columns else [
                'Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Animation'
            ]
            genre = st.selectbox("🎭 Genre", available_genres, index=0)

            # Sentiment input
            sentiment = st.slider(
                "💬 Audience Sentiment",
                min_value=-1.0, max_value=1.0, value=0.5, step=0.05,
                help="Expected audience sentiment (-1 = very negative, +1 = very positive)"
            )

            # Sentiment label
            if sentiment > 0.5:
                sent_emoji, sent_label = "😍", "Very Positive"
            elif sentiment > 0.05:
                sent_emoji, sent_label = "😊", "Positive"
            elif sentiment > -0.05:
                sent_emoji, sent_label = "😐", "Neutral"
            elif sentiment > -0.5:
                sent_emoji, sent_label = "😕", "Negative"
            else:
                sent_emoji, sent_label = "😡", "Very Negative"

            st.markdown(f"<p style='color: rgba(255,255,255,0.5);'>"
                        f"{sent_emoji} Sentiment: <strong>{sent_label}</strong></p>",
                        unsafe_allow_html=True)

            predict_btn = st.button("🚀 Predict Revenue", type="primary",
                                     use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">📊 Prediction Result</div>',
                        unsafe_allow_html=True)

            if predict_btn:
                try:
                    from phase4_modeling import predict_new_movie

                    # Load feature columns from training data
                    if predictions_df is not None:
                        # Use the model directly
                        feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                    else:
                        feature_cols = None

                    # Build feature vector manually
                    if feature_cols is not None:
                        feature_dict = {col: 0.0 for col in feature_cols}
                        if 'Budget_Millions' in feature_dict:
                            feature_dict['Budget_Millions'] = budget
                        if 'Sentiment_Score' in feature_dict:
                            feature_dict['Sentiment_Score'] = sentiment
                        # Set genre one-hot
                        genre_col = f'Genre_{genre}'
                        if genre_col in feature_dict:
                            feature_dict[genre_col] = 1
                        feature_df = pd.DataFrame([feature_dict])
                        predicted = float(model.predict(feature_df)[0])
                    else:
                        predicted = predict_new_movie(
                            model, getattr(model, 'feature_names_in_', []),
                            sentiment=sentiment, budget=budget, genre=genre
                        )

                    predicted = max(0, predicted)  # No negative revenue

                    # ROI calculation
                    roi = ((predicted - budget) / budget) * 100 if budget > 0 else 0

                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                            PREDICTED BOX OFFICE REVENUE
                        </div>
                        <div class="prediction-value">${predicted:,.1f}M</div>
                        <div style="color: rgba(255,255,255,0.5); margin-top: 8px;">
                            ROI: <strong style="color: {'#10b981' if roi > 0 else '#ef4444'};">
                            {roi:+.0f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Comparison gauge
                    if 'Revenue_Millions' in df.columns:
                        avg_rev = df['Revenue_Millions'].mean()
                        median_rev = df['Revenue_Millions'].median()

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=predicted,
                            delta={'reference': avg_rev, 'suffix': 'M'},
                            number={'prefix': '$', 'suffix': 'M'},
                            gauge={
                                'axis': {'range': [0, df['Revenue_Millions'].quantile(0.95)]},
                                'bar': {'color': '#6366f1'},
                                'bgcolor': 'rgba(255,255,255,0.05)',
                                'steps': [
                                    {'range': [0, median_rev], 'color': 'rgba(239,68,68,0.1)'},
                                    {'range': [median_rev, avg_rev * 1.5], 'color': 'rgba(245,158,11,0.1)'},
                                    {'range': [avg_rev * 1.5, df['Revenue_Millions'].quantile(0.95)],
                                     'color': 'rgba(16,185,129,0.1)'}
                                ],
                                'threshold': {
                                    'line': {'color': '#a855f7', 'width': 2},
                                    'thickness': 0.8,
                                    'value': avg_rev
                                }
                            },
                            title={'text': 'vs. Average Revenue',
                                   'font': {'color': 'rgba(255,255,255,0.6)'}}
                        ))
                        apply_plotly_theme(fig)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("Make sure you've run the pipeline first: `python src/main.py --tmdb`")
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 60px 20px;">
                    <span style="font-size: 4rem;">🎬</span>
                    <p style="color: rgba(255,255,255,0.4); margin-top: 15px;">
                        Adjust the parameters on the left and click<br>
                        <strong style="color: #6366f1;">Predict Revenue</strong> to see results
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
#  PAGE 3: SENTIMENT TRENDS
# ================================================================
elif page == "📊 Sentiment Trends":
    st.markdown('<h1 class="hero-title">📊 Sentiment Trends</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore how audience sentiment varies across genres and relates to revenue</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    has_sentiment = 'Sentiment_Score' in df.columns
    has_transformer = 'Transformer_Sentiment' in df.columns

    if not has_sentiment:
        st.warning("⚠️ No sentiment data found. Run the pipeline first: `python src/main.py --tmdb`")
    else:
        tab1, tab2, tab3 = st.tabs(["Genre Trends", "Sentiment vs Revenue", "Distribution"])

        with tab1:
            st.markdown('<div class="section-header">🎭 Average Sentiment by Genre</div>',
                        unsafe_allow_html=True)
            if 'Genre' in df.columns:
                genre_sent = df.groupby('Genre')['Sentiment_Score'].agg(['mean', 'std', 'count']).reset_index()
                genre_sent = genre_sent[genre_sent['count'] >= 5].sort_values('mean', ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=genre_sent['Genre'],
                    x=genre_sent['mean'],
                    orientation='h',
                    marker=dict(
                        color=genre_sent['mean'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='Score')
                    ),
                    error_x=dict(type='data', array=genre_sent['std'], visible=True,
                                 color='rgba(255,255,255,0.3)'),
                    hovertemplate="<b>%{y}</b><br>Avg Sentiment: %{x:.3f}<br>Movies: %{customdata}<extra></extra>",
                    customdata=genre_sent['count']
                ))
                fig.update_layout(
                    title='Genre-wise Sentiment (with standard deviation)',
                    xaxis_title='Average Sentiment Score',
                    height=500
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # If transformer scores exist, show comparison
                if has_transformer:
                    st.markdown('<div class="section-header">🤖 VADER vs Transformer by Genre</div>',
                                unsafe_allow_html=True)
                    genre_comp = df.groupby('Genre').agg({
                        'Sentiment_Score': 'mean',
                        'Transformer_Sentiment': 'mean'
                    }).reset_index()
                    genre_comp = genre_comp.sort_values('Sentiment_Score')

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='VADER',
                        y=genre_comp['Genre'],
                        x=genre_comp['Sentiment_Score'],
                        orientation='h',
                        marker_color='#6366f1'
                    ))
                    fig.add_trace(go.Bar(
                        name='DistilBERT',
                        y=genre_comp['Genre'],
                        x=genre_comp['Transformer_Sentiment'],
                        orientation='h',
                        marker_color='#a855f7'
                    ))
                    fig.update_layout(
                        barmode='group',
                        title='Sentiment Engine Comparison by Genre',
                        xaxis_title='Average Sentiment',
                        height=500
                    )
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown('<div class="section-header">💰 Sentiment vs Revenue</div>',
                        unsafe_allow_html=True)
            if 'Revenue_Millions' in df.columns:
                fig = px.scatter(
                    df,
                    x='Sentiment_Score',
                    y='Revenue_Millions',
                    color='Genre' if 'Genre' in df.columns else None,
                    size='Budget_Millions' if 'Budget_Millions' in df.columns else None,
                    hover_data=['title'] if 'title' in df.columns else None,
                    color_discrete_sequence=COLOR_PALETTE,
                    labels={
                        'Sentiment_Score': 'Sentiment Score',
                        'Revenue_Millions': 'Revenue ($M)',
                        'Budget_Millions': 'Budget ($M)'
                    },
                    opacity=0.7
                )
                fig.update_layout(
                    title='How Sentiment Relates to Box Office Revenue',
                    height=600
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown('<div class="section-header">📊 Sentiment Score Distribution</div>',
                        unsafe_allow_html=True)

            cols_to_plot = ['Sentiment_Score']
            names = ['VADER']
            colors = ['#6366f1']
            if has_transformer:
                cols_to_plot.append('Transformer_Sentiment')
                names.append('DistilBERT')
                colors.append('#a855f7')

            fig = go.Figure()
            for col, name, color in zip(cols_to_plot, names, colors):
                fig.add_trace(go.Histogram(
                    x=df[col],
                    name=name,
                    marker_color=color,
                    opacity=0.7,
                    nbinsx=50
                ))
            fig.update_layout(
                barmode='overlay',
                title='Sentiment Score Distribution',
                xaxis_title='Sentiment Score',
                yaxis_title='Count',
                height=500
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Stats cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Sentiment</div>
                    <div class="metric-value">{df['Sentiment_Score'].mean():+.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                positive_pct = (df['Sentiment_Score'] > 0.05).mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">% Positive</div>
                    <div class="metric-value">{positive_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                negative_pct = (df['Sentiment_Score'] < -0.05).mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">% Negative</div>
                    <div class="metric-value">{negative_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
#  PAGE 4: MOVIE COMPARISON
# ================================================================
elif page == "🔍 Movie Comparison":
    st.markdown('<h1 class="hero-title">🔍 Movie Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Compare movies side-by-side across all dimensions</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    title_col = None
    for col in ['title', 'Title', 'Movie_Title', 'movie_title']:
        if col in df.columns:
            title_col = col
            break

    if title_col is None:
        st.warning("No movie title column found in the dataset.")
    else:
        all_titles = sorted(df[title_col].dropna().unique().tolist())

        # Movie selection
        selected_movies = st.multiselect(
            "🎬 Select movies to compare (2-5)",
            all_titles,
            default=all_titles[:3] if len(all_titles) >= 3 else all_titles,
            max_selections=5
        )

        if len(selected_movies) >= 2:
            comp_df = df[df[title_col].isin(selected_movies)].copy()

            # Comparison table
            st.markdown('<div class="section-header">📋 Side-by-Side Comparison</div>',
                        unsafe_allow_html=True)

            display_cols = [title_col]
            for col in ['Genre', 'Budget_Millions', 'Revenue_Millions', 'Sentiment_Score',
                         'Transformer_Sentiment', 'vote_average', 'director', 'lead_actor']:
                if col in comp_df.columns:
                    display_cols.append(col)

            st.dataframe(
                comp_df[display_cols].set_index(title_col).T,
                use_container_width=True
            )

            # Radar chart
            st.markdown('<div class="section-header">🕸️ Radar Comparison</div>',
                        unsafe_allow_html=True)

            numeric_cols = []
            for col in ['Budget_Millions', 'Revenue_Millions', 'Sentiment_Score', 'vote_average']:
                if col in comp_df.columns:
                    numeric_cols.append(col)

            if len(numeric_cols) >= 3:
                fig = go.Figure()
                for _, row in comp_df.iterrows():
                    # Normalize values to 0-1 for radar
                    values = []
                    for col in numeric_cols:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        if col_max != col_min:
                            normalized = (row[col] - col_min) / (col_max - col_min)
                        else:
                            normalized = 0.5
                        values.append(normalized)
                    values.append(values[0])  # Close the polygon

                    labels = [c.replace('_', ' ').replace('Millions', '$M') for c in numeric_cols]
                    labels.append(labels[0])

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=labels,
                        fill='toself',
                        name=str(row[title_col]),
                        opacity=0.6
                    ))

                fig.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            color='rgba(255,255,255,0.5)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            color='rgba(255,255,255,0.7)'
                        )
                    ),
                    height=500,
                    title='Normalized Feature Comparison'
                )
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            # Bar comparison
            st.markdown('<div class="section-header">📊 Revenue & Budget Comparison</div>',
                        unsafe_allow_html=True)
            if 'Revenue_Millions' in comp_df.columns and 'Budget_Millions' in comp_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=comp_df[title_col],
                    y=comp_df['Budget_Millions'],
                    name='Budget ($M)',
                    marker_color='#6366f1'
                ))
                fig.add_trace(go.Bar(
                    x=comp_df[title_col],
                    y=comp_df['Revenue_Millions'],
                    name='Revenue ($M)',
                    marker_color='#10b981'
                ))
                fig.update_layout(barmode='group', height=400)
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

        elif len(selected_movies) == 1:
            st.info("Select at least 2 movies to compare.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <span style="font-size: 3rem;">🔍</span>
                <p style="color: rgba(255,255,255,0.4); margin-top: 10px;">
                    Select 2-5 movies from the dropdown above to compare them
                </p>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
#  PAGE 5: MODEL PERFORMANCE
# ================================================================
elif page == "🤖 Model Performance":
    st.markdown('<h1 class="hero-title">🤖 Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Detailed evaluation of ML models and feature importance</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Predictions", "Feature Importance"])

    with tab1:
        if model_results is not None:
            st.markdown('<div class="section-header">📊 Model Comparison</div>',
                        unsafe_allow_html=True)

            # Metrics cards
            cols = st.columns(len(model_results))
            for i, (_, row) in enumerate(model_results.iterrows()):
                with cols[i]:
                    is_best = row['R2'] == model_results['R2'].max()
                    border_color = '#10b981' if is_best else 'rgba(255,255,255,0.1)'
                    badge = '<span style="color: #10b981; font-size: 0.8rem;">★ BEST</span>' if is_best else ''

                    st.markdown(f"""
                    <div class="glass-card" style="border-color: {border_color}; text-align: center;">
                        <strong style="color: #a5b4fc;">{row['Model']}</strong> {badge}
                        <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">R² Score</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #6366f1;">{row['R2']:.1%}</div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 8px;">MAE</div>
                        <div style="font-size: 1.2rem; color: #a855f7;">${row['MAE']:.1f}M</div>
                        <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 8px;">RMSE</div>
                        <div style="font-size: 1.2rem; color: #ec4899;">${row['RMSE']:.1f}M</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Bar chart comparison
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=model_results['Model'],
                y=model_results['R2'],
                name='R² Score',
                marker_color='#6366f1',
                text=[f"{v:.1%}" for v in model_results['R2']],
                textposition='outside'
            ))
            fig.update_layout(
                title='R² Score by Model (higher is better)',
                yaxis_title='R² Score',
                height=400,
                yaxis_range=[0, 1]
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model results found. Run the pipeline first.")

    with tab2:
        st.markdown('<div class="section-header">🎯 Actual vs Predicted Revenue</div>',
                    unsafe_allow_html=True)

        if predictions_df is not None:
            fig = go.Figure()

            # Perfect prediction line
            max_val = max(predictions_df['Actual_Revenue_M'].max(),
                          predictions_df['Predicted_Revenue_M'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='rgba(255,255,255,0.3)')
            ))

            # Actual points
            fig.add_trace(go.Scatter(
                x=predictions_df['Actual_Revenue_M'],
                y=predictions_df['Predicted_Revenue_M'],
                mode='markers',
                name='Predictions',
                marker=dict(
                    color=predictions_df['Error_M'] if 'Error_M' in predictions_df.columns else '#6366f1',
                    colorscale='Viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(title='Error ($M)')
                ),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Actual: $%{x:.1f}M<br>"
                    "Predicted: $%{y:.1f}M<extra></extra>"
                ),
                customdata=predictions_df['Movie'] if 'Movie' in predictions_df.columns else None
            ))

            fig.update_layout(
                xaxis_title='Actual Revenue ($M)',
                yaxis_title='Predicted Revenue ($M)',
                height=600
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction table
            st.markdown('<div class="section-header">📋 Detailed Predictions</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                predictions_df.head(20).style.format({
                    'Actual_Revenue_M': '${:.1f}M',
                    'Predicted_Revenue_M': '${:.1f}M',
                    'Error_M': '${:.1f}M',
                    'Budget_M': '${:.1f}M'
                }),
                use_container_width=True
            )
        else:
            st.info("No predictions available yet. Run the pipeline first.")

    with tab3:
        st.markdown('<div class="section-header">🔑 Feature Importance</div>',
                    unsafe_allow_html=True)

        if model is not None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
                f'Feature_{i}' for i in range(len(importances))
            ]

            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(20)

            fig = go.Figure(go.Bar(
                y=imp_df['Feature'],
                x=imp_df['Importance'],
                orientation='h',
                marker=dict(
                    color=imp_df['Importance'],
                    colorscale='Viridis',
                    showscale=False
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            ))
            fig.update_layout(
                title='Top 20 Most Important Features',
                xaxis_title='Importance Score',
                height=600
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        elif model is not None and hasattr(model, 'coef_'):
            # Linear model — show coefficients
            coefs = model.coef_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
                f'Feature_{i}' for i in range(len(coefs))
            ]

            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': np.abs(coefs)
            }).sort_values('Coefficient', ascending=True).tail(20)

            fig = go.Figure(go.Bar(
                y=coef_df['Feature'],
                x=coef_df['Coefficient'],
                orientation='h',
                marker=dict(color='#6366f1')
            ))
            fig.update_layout(
                title='Top 20 Feature Coefficients (absolute)',
                xaxis_title='|Coefficient|',
                height=600
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for the current model type.")


# ================================================================
#  PAGE 6: LIVE PREDICTOR (TMDB API)
# ================================================================
elif page == "🔴 Live Predictor":
    st.markdown('<h1 class="hero-title">🔴 Live Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Search any movie — get live TMDB data, real-time sentiment & revenue prediction</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Try to initialize TMDB client
    tmdb_client = None
    try:
        from phase1_tmdb_api import TMDBClient, analyze_live_reviews
        tmdb_client = TMDBClient()
        api_ok = True
    except ValueError:
        api_ok = False
        st.warning("⚠️ TMDB API key not found. Create a `.env` file in the project root with:\n\n`TMDB_API_KEY=your_key_here`\n\nGet a free key at [themoviedb.org](https://www.themoviedb.org/settings/api)")
    except ImportError:
        api_ok = False
        st.error("❌ `phase1_tmdb_api.py` module not found.")

    if api_ok and tmdb_client:
        # Three tabs: Search, Now Playing, Trending
        tab_search, tab_playing, tab_trending = st.tabs(
            ["🔍 Search Movie", "🎬 Now Playing", "🔥 Trending"])

        with tab_playing:
            st.markdown('<div class="section-header">🎬 Currently in Theatres</div>',
                        unsafe_allow_html=True)
            now_movies = tmdb_client.get_now_playing()
            if now_movies:
                cols = st.columns(4)
                for i, m in enumerate(now_movies[:8]):
                    with cols[i % 4]:
                        if m.get('poster_url'):
                            st.image(m['poster_url'], use_container_width=True)
                        st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 15px;">
                            <strong style="color: #a5b4fc;">{m['title']}</strong><br>
                            <span style="color: rgba(255,255,255,0.5);">⭐ {m['vote_average']}/10</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Could not fetch now playing movies.")

        with tab_trending:
            st.markdown('<div class="section-header">🔥 Trending This Week</div>',
                        unsafe_allow_html=True)
            trending = tmdb_client.get_trending()
            if trending:
                cols = st.columns(4)
                for i, m in enumerate(trending[:8]):
                    with cols[i % 4]:
                        if m.get('poster_url'):
                            st.image(m['poster_url'], use_container_width=True)
                        st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 15px;">
                            <strong style="color: #a5b4fc;">{m['title']}</strong><br>
                            <span style="color: rgba(255,255,255,0.5);">⭐ {m['vote_average']}/10</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Could not fetch trending movies.")

        with tab_search:
            st.markdown('<div class="section-header">🔍 Search Any Movie</div>',
                        unsafe_allow_html=True)

            search_query = st.text_input(
                "Enter movie title",
                placeholder="e.g. Inception, Pushpa, Avengers...",
                key="tmdb_search"
            )

            if search_query:
                with st.spinner("Searching TMDB..."):
                    results = tmdb_client.search_movie(search_query)

                if not results:
                    st.warning(f"No movies found for '{search_query}'")
                else:
                    # Let user pick from results
                    movie_options = {
                        f"{m['title']} ({m['release_date'][:4] if m['release_date'] else '?'}) — ⭐ {m['vote_average']}": m
                        for m in results
                    }
                    selected_label = st.selectbox("Select a match:", list(movie_options.keys()))
                    selected_movie = movie_options[selected_label]

                    # Fetch full details
                    with st.spinner("Fetching movie details & reviews..."):
                        details = tmdb_client.get_movie_details(selected_movie['id'])
                        reviews = tmdb_client.get_movie_reviews(selected_movie['id'])
                        sentiment = analyze_live_reviews(reviews)

                    if details:
                        # ── Movie Info Card ──
                        info_col, poster_col = st.columns([3, 1])

                        with poster_col:
                            if details.get('poster_url'):
                                st.image(details['poster_url'], use_container_width=True)

                        with info_col:
                            status_color = '#10b981' if details['status'] == 'Released' else '#f59e0b'
                            st.markdown(f"""
                            <div class="glass-card">
                                <h2 style="color: #e2e8f0; margin: 0;">{details['title']}</h2>
                                <p style="color: #a5b4fc; font-style: italic;">{details.get('tagline', '')}</p>
                                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin: 15px 0;">
                                    <span style="color: rgba(255,255,255,0.6);">📅 {details['release_date']}</span>
                                    <span style="color: rgba(255,255,255,0.6);">⏱️ {details['runtime']} min</span>
                                    <span style="color: {status_color}; font-weight: 600;">● {details['status']}</span>
                                    <span style="color: rgba(255,255,255,0.6);">⭐ {details['vote_average']}/10 ({details['vote_count']:,} votes)</span>
                                </div>
                                <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0;">
                                    {''.join(f'<span style="background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.4); border-radius: 20px; padding: 4px 12px; font-size: 0.8rem; color: #a5b4fc;">{g}</span>' for g in details['genres'])}
                                </div>
                                <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-top: 10px;">
                                    🎬 Director: <strong style="color: #e2e8f0;">{details['director'] or 'N/A'}</strong> &nbsp;•&nbsp;
                                    ⭐ Lead: <strong style="color: #e2e8f0;">{details['lead_actor']}</strong> &nbsp;•&nbsp;
                                    🏢 Studio: <strong style="color: #e2e8f0;">{details['primary_studio']}</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # ── Metrics Row ──
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            budget_display = f"${details['budget_millions']:.0f}M" if details['budget_millions'] > 0 else "N/A"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Budget</div>
                                <div class="metric-value">{budget_display}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m2:
                            rev_display = f"${details['revenue_millions']:.0f}M" if details['revenue_millions'] > 0 else "TBD"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Box Office</div>
                                <div class="metric-value">{rev_display}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m3:
                            sent_color = '#10b981' if sentiment['vader_avg'] > 0.1 else '#ef4444' if sentiment['vader_avg'] < -0.1 else '#f59e0b'
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Live Sentiment</div>
                                <div class="metric-value" style="background: none; -webkit-text-fill-color: {sent_color};">{sentiment['vader_avg']:+.2f}</div>
                                <div style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">{sentiment['sentiment_label']} ({sentiment['review_count']} reviews)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m4:
                            conf_color = '#10b981' if sentiment['confidence'] >= 0.6 else '#f59e0b' if sentiment['confidence'] >= 0.3 else '#ef4444'
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value" style="background: none; -webkit-text-fill-color: {conf_color};">{sentiment['confidence']:.0%}</div>
                                <div style="color: rgba(255,255,255,0.4); font-size: 0.75rem;">Based on review count</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Revenue Prediction ──
                        st.markdown('<div class="section-header">🎯 Revenue Prediction</div>',
                                    unsafe_allow_html=True)

                        # Allow manual budget input if not available
                        pred_budget = details['budget_millions']
                        if pred_budget <= 0:
                            pred_budget = st.slider(
                                "Budget not available on TMDB — enter estimated budget ($M):",
                                min_value=1, max_value=400, value=50, step=5)

                        if model is not None:
                            try:
                                import math
                                feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
                                feature_dict = {col: 0.0 for col in feature_cols}

                                # ── Core features ──
                                if 'Budget_Millions' in feature_dict:
                                    feature_dict['Budget_Millions'] = pred_budget
                                if 'Sentiment_Score' in feature_dict:
                                    feature_dict['Sentiment_Score'] = sentiment['vader_avg']

                                # ── Genre one-hot ──
                                for g in details['genres']:
                                    genre_col = f'Genre_{g}'
                                    if genre_col in feature_dict:
                                        feature_dict[genre_col] = 1
                                if 'num_genres' in feature_dict:
                                    feature_dict['num_genres'] = len(details['genres'])

                                # ── TMDB metadata ──
                                if 'vote_average' in feature_dict:
                                    feature_dict['vote_average'] = details['vote_average']
                                if 'vote_count' in feature_dict:
                                    feature_dict['vote_count'] = details['vote_count']
                                if 'runtime' in feature_dict:
                                    feature_dict['runtime'] = details.get('runtime', 120) or 120
                                if 'popularity' in feature_dict:
                                    feature_dict['popularity'] = details.get('popularity', 10) or 10

                                # ── Derived financial features ──
                                if 'log_budget' in feature_dict:
                                    feature_dict['log_budget'] = math.log1p(pred_budget) if pred_budget > 0 else 0
                                if 'budget_per_vote' in feature_dict and details['vote_count'] > 0:
                                    feature_dict['budget_per_vote'] = pred_budget / details['vote_count']

                                # ── Temporal features from release date ──
                                release_date = details.get('release_date', '')
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

                                # ── Studio classification ──
                                major_studios = ['Warner Bros', 'Universal', 'Paramount', 'Walt Disney',
                                                 'Columbia', 'Sony', '20th Century', 'Lionsgate', 'Marvel',
                                                 'New Line', 'DreamWorks', 'Metro-Goldwyn']
                                mid_studios = ['A24', 'Focus Features', 'Miramax', 'Relativity',
                                               'Summit', 'STX', 'Blumhouse', 'Legendary']
                                studio = details.get('primary_studio', '')
                                if 'Studio_Major' in feature_dict:
                                    feature_dict['Studio_Major'] = 1 if any(s.lower() in studio.lower() for s in major_studios) else 0
                                if 'Studio_Mid' in feature_dict:
                                    feature_dict['Studio_Mid'] = 1 if any(s.lower() in studio.lower() for s in mid_studios) else 0

                                # ── Director & actor track record (use dataset medians as reasonable defaults) ──
                                if 'Budget_Millions' in df.columns and 'Revenue_Millions' in df.columns:
                                    valid = df[df['Revenue_Millions'] > 0]
                                    median_rev = valid['Revenue_Millions'].median() if len(valid) > 0 else 100
                                else:
                                    median_rev = 100
                                if 'director_avg_revenue' in feature_dict:
                                    feature_dict['director_avg_revenue'] = median_rev
                                if 'director_movie_count' in feature_dict:
                                    feature_dict['director_movie_count'] = 3
                                if 'lead_actor_avg_revenue' in feature_dict:
                                    feature_dict['lead_actor_avg_revenue'] = median_rev

                                feature_df = pd.DataFrame([feature_dict])
                                predicted = max(0, float(model.predict(feature_df)[0]))

                                roi = ((predicted - pred_budget) / pred_budget) * 100 if pred_budget > 0 else 0

                                pred_col, gauge_col = st.columns([1, 1])
                                with pred_col:
                                    st.markdown(f"""
                                    <div class="prediction-box">
                                        <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
                                            ML-PREDICTED REVENUE
                                        </div>
                                        <div class="prediction-value">${predicted:,.1f}M</div>
                                        <div style="color: rgba(255,255,255,0.5); margin-top: 8px;">
                                            ROI: <strong style="color: {'#10b981' if roi > 0 else '#ef4444'};">
                                            {roi:+.0f}%</strong>
                                        </div>
                                        <div style="color: rgba(255,255,255,0.3); font-size: 0.8rem; margin-top: 12px;">
                                            {'✅ Based on real TMDB budget' if details['budget_millions'] > 0 else '⚠️ Using estimated budget'}
                                            &nbsp;•&nbsp; Sentiment from {sentiment['review_count']} live reviews
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    if details['revenue_millions'] > 0:
                                        actual = details['revenue_millions']
                                        error = abs(predicted - actual)
                                        accuracy = max(0, (1 - error / actual)) * 100
                                        st.markdown(f"""
                                        <div class="glass-card" style="text-align: center;">
                                            <div style="color: rgba(255,255,255,0.5);">Actual Revenue (TMDB)</div>
                                            <div style="font-size: 1.5rem; font-weight: 700; color: #a855f7;">${actual:,.1f}M</div>
                                            <div style="color: rgba(255,255,255,0.4); margin-top: 5px;">
                                                Error: ${error:,.1f}M &nbsp;•&nbsp; Accuracy: {accuracy:.0f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                with gauge_col:
                                    if 'Revenue_Millions' in df.columns:
                                        avg_rev = df['Revenue_Millions'].mean()
                                        fig = go.Figure(go.Indicator(
                                            mode="gauge+number+delta",
                                            value=predicted,
                                            delta={'reference': avg_rev, 'suffix': 'M'},
                                            number={'prefix': '$', 'suffix': 'M'},
                                            gauge={
                                                'axis': {'range': [0, df['Revenue_Millions'].quantile(0.95)]},
                                                'bar': {'color': '#6366f1'},
                                                'bgcolor': 'rgba(255,255,255,0.05)',
                                                'steps': [
                                                    {'range': [0, df['Revenue_Millions'].median()],
                                                     'color': 'rgba(239,68,68,0.1)'},
                                                    {'range': [df['Revenue_Millions'].median(), avg_rev * 1.5],
                                                     'color': 'rgba(245,158,11,0.1)'},
                                                    {'range': [avg_rev * 1.5, df['Revenue_Millions'].quantile(0.95)],
                                                     'color': 'rgba(16,185,129,0.1)'}
                                                ],
                                                'threshold': {
                                                    'line': {'color': '#a855f7', 'width': 2},
                                                    'thickness': 0.8, 'value': avg_rev
                                                }
                                            },
                                            title={'text': 'vs. Dataset Average',
                                                   'font': {'color': 'rgba(255,255,255,0.6)'}}
                                        ))
                                        apply_plotly_theme(fig)
                                        fig.update_layout(height=350)
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                        else:
                            st.warning("⚠️ No trained model found. Run `python src/main.py --tmdb` first.")

                        # ── Review Sentiment Details ──
                        if reviews:
                            st.markdown('<div class="section-header">💬 Review Sentiment Breakdown</div>',
                                        unsafe_allow_html=True)
                            rev_col1, rev_col2 = st.columns([1, 1])
                            with rev_col1:
                                if sentiment['vader_scores']:
                                    fig = go.Figure()
                                    fig.add_trace(go.Histogram(
                                        x=sentiment['vader_scores'],
                                        nbinsx=20,
                                        marker_color='#6366f1',
                                        name='VADER'
                                    ))
                                    if sentiment['transformer_scores']:
                                        fig.add_trace(go.Histogram(
                                            x=sentiment['transformer_scores'],
                                            nbinsx=20,
                                            marker_color='#a855f7',
                                            opacity=0.7,
                                            name='DistilBERT'
                                        ))
                                    fig.update_layout(
                                        title='Sentiment Score Distribution',
                                        xaxis_title='Score', yaxis_title='Count',
                                        barmode='overlay', height=300
                                    )
                                    apply_plotly_theme(fig)
                                    st.plotly_chart(fig, use_container_width=True)

                            with rev_col2:
                                st.markdown("**Sample Reviews:**")
                                for r in reviews[:3]:
                                    snippet = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                                    rating_badge = f"⭐ {r['rating']}/10" if r['rating'] else ""
                                    st.markdown(f"""
                                    <div class="glass-card" style="padding: 12px; margin: 5px 0;">
                                        <strong style="color: #a5b4fc;">{r['author']}</strong>
                                        <span style="color: rgba(255,255,255,0.4); float: right;">{rating_badge}</span>
                                        <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 8px;">{snippet}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="text-align: center; padding: 60px 20px;">
                    <span style="font-size: 4rem;">🔍</span>
                    <p style="color: rgba(255,255,255,0.4); margin-top: 15px;">
                        Type a movie name above to search TMDB<br>
                        <strong style="color: #6366f1;">Live data • Real reviews • ML prediction</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ================================================================
#  PAGE 7: BUSINESS INTELLIGENCE
# ================================================================
elif page == "💼 Business Intelligence":
    st.markdown('<h1 class="hero-title">💼 Business Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">ROI analysis, genre trends, and strategic market insights</p>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_roi, tab_trends, tab_compete = st.tabs(
        ["💰 ROI Calculator", "📈 Genre Trends", "🏆 Competitive Analysis"])

    # ── ROI Calculator ──
    with tab_roi:
        st.markdown('<div class="section-header">💰 ROI Probability Calculator</div>',
                    unsafe_allow_html=True)

        if 'Revenue_Millions' in df.columns and 'Budget_Millions' in df.columns:
            roi_col1, roi_col2 = st.columns([1, 1])

            with roi_col1:
                invest_budget = st.slider(
                    "Investment Budget ($M)", min_value=1, max_value=400,
                    value=100, step=5, key="roi_budget")
                target_multiplier = st.slider(
                    "Target Return Multiplier (X)",
                    min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                    key="roi_target",
                    help="A 2x multiplier means you want $200M return on a $100M investment")
                genre_filter = st.selectbox(
                    "Genre Filter",
                    ["All Genres"] + sorted(df['Genre'].unique().tolist()) if 'Genre' in df.columns else ["All Genres"],
                    key="roi_genre")

            with roi_col2:
                # Calculate ROI statistics from historical data
                analysis_df = df[(df['Budget_Millions'] > 0) & (df['Revenue_Millions'] > 0)].copy()
                analysis_df['ROI'] = (analysis_df['Revenue_Millions'] - analysis_df['Budget_Millions']) / analysis_df['Budget_Millions']

                if genre_filter != "All Genres" and 'Genre' in analysis_df.columns:
                    analysis_df = analysis_df[analysis_df['Genre'] == genre_filter]

                # Budget range: +/- 30% of input budget
                budget_low = invest_budget * 0.5
                budget_high = invest_budget * 1.5
                similar = analysis_df[
                    (analysis_df['Budget_Millions'] >= budget_low) &
                    (analysis_df['Budget_Millions'] <= budget_high)
                ]

                if len(similar) >= 3:
                    success_rate = (similar['ROI'] >= (target_multiplier - 1)).mean() * 100
                    avg_roi = similar['ROI'].mean() * 100
                    median_roi = similar['ROI'].median() * 100
                    best_case = similar['ROI'].quantile(0.9) * 100
                    worst_case = similar['ROI'].quantile(0.1) * 100

                    # Display probability
                    prob_color = '#10b981' if success_rate >= 50 else '#f59e0b' if success_rate >= 25 else '#ef4444'
                    st.markdown(f"""
                    <div class="prediction-box" style="border-color: {prob_color}40; background: linear-gradient(135deg, {prob_color}20 0%, {prob_color}10 100%);">
                        <div style="color: rgba(255,255,255,0.6);">
                            Probability of {target_multiplier}x Return ({genre_filter})
                        </div>
                        <div class="prediction-value" style="color: {prob_color};">{success_rate:.0f}%</div>
                        <div style="color: rgba(255,255,255,0.4); margin-top: 8px;">
                            Based on {len(similar)} similar movies (${budget_low:.0f}M-${budget_high:.0f}M budget)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ROI stats
                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Avg ROI</div>
                        <div class="metric-value">{avg_roi:+.0f}%</div></div>""", unsafe_allow_html=True)
                    with s2:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Median ROI</div>
                        <div class="metric-value">{median_roi:+.0f}%</div></div>""", unsafe_allow_html=True)
                    with s3:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Best Case (90th)</div>
                        <div class="metric-value">{best_case:+.0f}%</div></div>""", unsafe_allow_html=True)
                    with s4:
                        st.markdown(f"""<div class="metric-card"><div class="metric-label">Worst Case (10th)</div>
                        <div class="metric-value">{worst_case:+.0f}%</div></div>""", unsafe_allow_html=True)
                else:
                    st.info(f"Not enough data for {genre_filter} movies in the ${budget_low:.0f}M-${budget_high:.0f}M range. Try a different budget or genre.")
        else:
            st.warning("Budget and revenue data required. Run `python src/main.py --tmdb` first.")

    # ── Genre Trends ──
    with tab_trends:
        st.markdown('<div class="section-header">📈 Genre ROI Trends</div>',
                    unsafe_allow_html=True)

        if 'Revenue_Millions' in df.columns and 'Budget_Millions' in df.columns and 'Genre' in df.columns:
            trend_df = df[(df['Budget_Millions'] > 0) & (df['Revenue_Millions'] > 0)].copy()
            trend_df['ROI_pct'] = ((trend_df['Revenue_Millions'] - trend_df['Budget_Millions']) / trend_df['Budget_Millions']) * 100
            trend_df['Profitable'] = trend_df['Revenue_Millions'] > trend_df['Budget_Millions']

            genre_stats = trend_df.groupby('Genre').agg(
                avg_roi=('ROI_pct', 'mean'),
                median_roi=('ROI_pct', 'median'),
                success_rate=('Profitable', 'mean'),
                avg_budget=('Budget_Millions', 'mean'),
                avg_revenue=('Revenue_Millions', 'mean'),
                count=('ROI_pct', 'count')
            ).reset_index()
            genre_stats = genre_stats[genre_stats['count'] >= 10].sort_values('avg_roi', ascending=True)
            genre_stats['success_rate'] = genre_stats['success_rate'] * 100

            # ROI by genre bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=genre_stats['Genre'],
                x=genre_stats['avg_roi'],
                orientation='h',
                marker=dict(
                    color=genre_stats['avg_roi'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='ROI %')
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Avg ROI: %{x:.0f}%<br>"
                    "Movies: %{customdata[0]}<br>"
                    "Success Rate: %{customdata[1]:.0f}%<extra></extra>"
                ),
                customdata=genre_stats[['count', 'success_rate']].values
            ))
            fig.update_layout(
                title='Average ROI by Genre (higher = more profitable)',
                xaxis_title='Average ROI %',
                height=500
            )
            apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

            # Success rate chart
            genre_stats_sorted = genre_stats.sort_values('success_rate', ascending=True)
            fig2 = go.Figure(go.Bar(
                y=genre_stats_sorted['Genre'],
                x=genre_stats_sorted['success_rate'],
                orientation='h',
                marker=dict(
                    color=genre_stats_sorted['success_rate'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{v:.0f}%" for v in genre_stats_sorted['success_rate']],
                textposition='outside'
            ))
            fig2.update_layout(
                title='Profitability Rate by Genre (% of movies that made a profit)',
                xaxis_title='Success Rate %',
                xaxis_range=[0, 100],
                height=500
            )
            apply_plotly_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

            # Key insights
            best_genre = genre_stats.sort_values('avg_roi', ascending=False).iloc[0]
            safest_genre = genre_stats.sort_values('success_rate', ascending=False).iloc[0]
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="color: #a5b4fc;">💡 Key Insights</h3>
                <ul style="color: rgba(255,255,255,0.7);">
                    <li><strong style="color: #10b981;">{best_genre['Genre']}</strong> has the highest average ROI
                        at <strong>{best_genre['avg_roi']:.0f}%</strong></li>
                    <li><strong style="color: #6366f1;">{safest_genre['Genre']}</strong> is the safest bet with
                        <strong>{safest_genre['success_rate']:.0f}%</strong> profitability rate</li>
                    <li>Average budget for top-ROI genre: <strong>${best_genre['avg_budget']:.0f}M</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Budget and revenue data required for trend analysis.")

    # ── Competitive Analysis ──
    with tab_compete:
        st.markdown('<div class="section-header">🏆 Competitive Analysis</div>',
                    unsafe_allow_html=True)

        if 'Genre' in df.columns and 'Revenue_Millions' in df.columns:
            comp_genre = st.selectbox(
                "Select genre for competitive landscape:",
                sorted(df['Genre'].unique().tolist()),
                key="comp_genre")

            genre_movies = df[df['Genre'] == comp_genre].copy()

            if 'Budget_Millions' in genre_movies.columns:
                comp_budget = st.slider(
                    "Your movie's budget ($M):",
                    min_value=1, max_value=400, value=50, step=5,
                    key="comp_budget")

                # Find similar-budget movies in same genre
                budget_range = comp_budget * 0.4
                competitors = genre_movies[
                    (genre_movies['Budget_Millions'] >= comp_budget - budget_range) &
                    (genre_movies['Budget_Millions'] <= comp_budget + budget_range)
                ].sort_values('Revenue_Millions', ascending=False)

                title_col = None
                for col in ['title', 'Title', 'Movie_Title', 'movie_title']:
                    if col in competitors.columns:
                        title_col = col
                        break

                if len(competitors) >= 1 and title_col:
                    st.markdown(f"**Found {len(competitors)} similar {comp_genre} movies** "
                                f"(budget: ${comp_budget - budget_range:.0f}M - ${comp_budget + budget_range:.0f}M)")

                    # Scatter plot: budget vs revenue for genre
                    fig = px.scatter(
                        genre_movies[genre_movies['Budget_Millions'] > 0],
                        x='Budget_Millions',
                        y='Revenue_Millions',
                        hover_data=[title_col] if title_col else None,
                        color_discrete_sequence=['rgba(99,102,241,0.5)'],
                        labels={'Budget_Millions': 'Budget ($M)', 'Revenue_Millions': 'Revenue ($M)'}
                    )
                    # Highlight competitors
                    fig.add_trace(go.Scatter(
                        x=competitors['Budget_Millions'],
                        y=competitors['Revenue_Millions'],
                        mode='markers+text',
                        marker=dict(color='#10b981', size=12, symbol='star'),
                        text=competitors[title_col].apply(lambda x: x[:15] + '...' if len(str(x)) > 15 else x),
                        textposition='top center',
                        textfont=dict(color='#10b981', size=10),
                        name='Similar Movies',
                        hovertemplate="<b>%{text}</b><br>Budget: $%{x:.0f}M<br>Revenue: $%{y:.0f}M<extra></extra>"
                    ))
                    # Add "your movie" marker
                    fig.add_trace(go.Scatter(
                        x=[comp_budget], y=[competitors['Revenue_Millions'].median()],
                        mode='markers+text',
                        marker=dict(color='#ef4444', size=15, symbol='diamond'),
                        text=['YOUR MOVIE'],
                        textposition='top center',
                        textfont=dict(color='#ef4444', size=11),
                        name='Your Movie'
                    ))
                    fig.update_layout(
                        title=f'{comp_genre} — Budget vs Revenue Landscape',
                        height=500, showlegend=True
                    )
                    apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    # Competitor table
                    display_cols = [title_col]
                    for col in ['Budget_Millions', 'Revenue_Millions', 'Sentiment_Score', 'vote_average']:
                        if col in competitors.columns:
                            display_cols.append(col)
                    st.dataframe(
                        competitors[display_cols].head(15),
                        use_container_width=True
                    )

                    # Stats summary
                    avg_rev = competitors['Revenue_Millions'].mean()
                    median_rev = competitors['Revenue_Millions'].median()
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3 style="color: #a5b4fc;">📊 Competitive Summary</h3>
                        <ul style="color: rgba(255,255,255,0.7);">
                            <li>Avg revenue for similar movies: <strong style="color: #10b981;">${avg_rev:.0f}M</strong></li>
                            <li>Median revenue: <strong>${median_rev:.0f}M</strong></li>
                            <li>Best performer: <strong style="color: #a855f7;">{competitors.iloc[0][title_col]}</strong>
                                (${competitors.iloc[0]['Revenue_Millions']:.0f}M)</li>
                            <li>Expected range: ${competitors['Revenue_Millions'].quantile(0.25):.0f}M - ${competitors['Revenue_Millions'].quantile(0.75):.0f}M</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No {comp_genre} movies found in that budget range. Try adjusting the budget.")
        else:
            st.warning("Genre and revenue data required for competitive analysis.")

