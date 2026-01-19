# Movie Success Prediction and Sentiment Study

A comprehensive Data Science project that combines Natural Language Processing (NLP) with Regression analysis to predict box office success based on movie reviews, budgets, and genres.

## Project Overview

This project demonstrates the full data science pipeline:
- **Sentiment Analysis**: Using VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze movie reviews
- **Exploratory Data Analysis**: Visualizing relationships between sentiment, revenue, and genres
- **Predictive Modeling**: Building regression models to predict box office success

For a full “handoff-style” explanation (what we built, why, how to run, and what outputs to expect), see:
- [HANDOFF_GUIDE.md](HANDOFF_GUIDE.md)

## Project Structure

```
Movie-Success-Prediction-and-Sentiment-Study/
├── data/                      # Data directory (add your datasets here)
├── notebooks/                 # Jupyter notebooks
│   └── main_analysis.ipynb   # Main analysis notebook
├── src/                      # Source code
│   ├── phase1_data_acquisition.py
│   ├── phase2_sentiment_analysis.py
│   ├── phase3_eda.py
│   └── phase4_modeling.py
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. VADER lexicon: this project auto-downloads `vader_lexicon` on first run.

## Data Sources

You can use datasets from:
- Kaggle: "IMDB 5000 Movie Dataset" or "The Movies Dataset"
- TMDb (The Movie Database) API
- IMDB (via scraping, with rate limit considerations)

## Project Phases

### Phase 1: Data Acquisition & Setup
- Environment setup
- Data loading and preprocessing

### Phase 2: Sentiment Analysis with VADER
- Text preprocessing
- Sentiment scoring using VADER
- Feature creation

### Phase 3: Exploratory Data Analysis
- Genre-wise sentiment analysis
- Correlation analysis
- Visualization (scatter plots, bar charts, heatmaps)

### Phase 4: Predictive Modeling
- Feature engineering
- Model building (Linear Regression, Random Forest)
- Model evaluation

### Phase 5: Final Deliverables
- Documentation
- Results summary and conclusions

## Usage

Run the complete pipeline (recommended):

```bash
python src/main.py --data data/tmdb_5000_movies.csv
```

Or run using generated sample data:

```bash
python src/main.py --sample
```

You can still run phases individually if you want, but `src/main.py` is the intended end-to-end path.

Or use the Jupyter notebook for interactive analysis:
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

## License

This project is for educational purposes.
