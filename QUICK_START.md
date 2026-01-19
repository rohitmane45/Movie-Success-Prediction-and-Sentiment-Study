# Quick Start Guide

Get started with the Movie Success Prediction and Sentiment Study project in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data (Required for Phase 2)

Open Python and run:

```python
import nltk
nltk.download('vader_lexicon')
```

Or let the script handle it automatically - it will download if missing.

## Running the Project

### Option 1: Run Complete Pipeline with Sample Data

```bash
python src/main.py --sample
```

This will run all 4 phases using sample data for demonstration.

### Option 2: Run Complete Pipeline with Your Data

```bash
python src/main.py --data path/to/your/movies_dataset.csv
```

Recommended (TMDB 5000 Movies CSV already included in this repo):

```bash
python src/main.py --data data/tmdb_5000_movies.csv
```

### Option 3: Run Phases Individually

```bash
# Phase 1: Data Acquisition
python src/phase1_data_acquisition.py

# Phase 2: Sentiment Analysis
python src/phase2_sentiment_analysis.py

# Phase 3: Exploratory Data Analysis
python src/phase3_eda.py

# Phase 4: Predictive Modeling
python src/phase4_modeling.py
```

## Expected Output

### Phase 1: Data Acquisition & Setup
- ✓ Dependency check
- ✓ Data directory creation
- ✓ Data loading and inspection

### Phase 2: Sentiment Analysis
- ✓ Text preprocessing
- ✓ VADER sentiment scoring
- ✓ Sentiment scores added to dataframe

### Phase 3: Exploratory Data Analysis
- ✓ Scatter plot: Sentiment vs Revenue
- ✓ Bar chart: Average sentiment by genre
- ✓ Correlation heatmap
- ✓ Correlation statistics

Saved outputs:
- `results/figures/*.png`
- `results/genre_analysis.csv`

### Phase 4: Predictive Modeling
- ✓ Feature preparation and encoding
- ✓ Train/test split
- ✓ Linear Regression model training
- ✓ Random Forest model training
- ✓ Model evaluation (MAE, RMSE, R²)
- ✓ Real movie predictions from the dataset (test set)
- ✓ Prediction examples for new/hypothetical movies

Saved outputs:
- `models/best_model.pkl`, `models/linear_regression.pkl`, `models/random_forest.pkl`
- `results/model_comparison.csv`, `results/model_summary.txt`
- `results/movie_predictions.csv` (all test-set movies with actual vs predicted)

## Your Dataset Format

If using your own dataset, ensure it has these columns:

**Required Columns:**
- `User_Review` or text column: Movie reviews or plot descriptions
- `Revenue_Millions`: Box office revenue (target variable)
- `Budget_Millions`: Production budget
- `Genre`: Movie genre (e.g., "Action", "Drama", "Sci-Fi")

**TMDB dataset notes (supported automatically):**
- Text: `overview` is used for sentiment if `User_Review` is missing
- Revenue: `revenue` (raw dollars) is converted to `Revenue_Millions`
- Budget: `budget` (raw dollars) is converted to `Budget_Millions`
- Genre: `genres` (JSON list) is converted to a single `Genre` (first genre)

**Optional Columns:**
- `Movie_Title`: Movie names
- `Runtime_Minutes`: Movie runtime

## Troubleshooting

### Import Errors
If you get import errors, make sure you've installed all requirements:
```bash
pip install -r requirements.txt
```

### NLTK Download Issues
If VADER lexicon download fails, try:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
```

### Missing Data Columns
If you get errors about missing columns, check that your CSV has the required columns listed above. The script will try to adapt, but having the exact column names helps.

### Visualization Not Showing
If plots don't display:
- **Windows**: May need to install `tkinter` (usually included with Python)
- **Linux**: May need `python3-tk` package
- **Alternative**: Modify scripts to save plots instead: `plt.savefig('output.png')` instead of `plt.show()`

## Next Steps

1. **Run with sample data** to verify everything works
2. **Replace with your dataset** from Kaggle/IMDB
3. **Experiment with different features** or models
4. **Review PROJECT_SUMMARY.md** for detailed findings

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test with sample data
python src/main.py --sample

# 3. Use your own data (after downloading from Kaggle)
python src/main.py --data data/my_movies_dataset.csv

# 4. Check outputs in data/ directory
```

## Support

For detailed project information, see:
- `README.md` - Project overview
- `PROJECT_SUMMARY.md` - Findings and conclusions
- `HANDOFF_GUIDE.md` - Full handoff documentation (what/why/how/outputs)
- Individual phase scripts - Detailed code comments explaining each step

Happy analyzing.
