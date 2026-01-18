# Movie Success Prediction and Sentiment Study - Project Summary

## Project Overview

This project demonstrates a complete data science pipeline combining Natural Language Processing (NLP) with Regression analysis to predict box office success based on movie reviews, budgets, and genres.

## Key Findings & Conclusions

### Research Question
**Can we predict a movie's box office success based on sentiment analysis of user reviews, budget, and genre?**

### Methodology

1. **Sentiment Analysis (Phase 2)**
   - Used VADER (Valence Aware Dictionary and sEntiment Reasoner) to convert text reviews into numerical sentiment scores (-1 to +1)
   - **Why VADER?** VADER is specifically designed for social media and short texts (like movie reviews) because it understands intensity (e.g., "good" vs "GREAT!!!") and context

2. **Exploratory Data Analysis (Phase 3)**
   - Visualized relationships between sentiment scores and revenue
   - Analyzed genre-wise sentiment trends
   - Calculated correlation coefficients between sentiment and box office success

3. **Predictive Modeling (Phase 4)**
   - Built Linear Regression and Random Forest models
   - Used features: Sentiment Score, Budget, Genre (one-hot encoded)
   - Evaluated using MAE, RMSE, and R² metrics

### Key Results

#### Correlation Findings
- **Sentiment vs Revenue**: The correlation coefficient indicates the strength of relationship
  - High correlation (>0.7): Strong positive relationship - positive reviews strongly predict higher revenue
  - Moderate correlation (0.3-0.7): Moderate relationship - sentiment is one factor among many
  - Low correlation (<0.3): Weak relationship - sentiment alone doesn't strongly predict success

#### Model Performance
- **Linear Regression**: Provides interpretable coefficients showing how each feature affects revenue
- **Random Forest**: Often captures non-linear relationships better, handling complex interactions between features
- **Best Model Selection**: Choose based on R² score - higher is better (closer to 1.0)

### Insights

1. **Budget Impact**: Typically the strongest predictor of box office success
   - Higher budgets often correlate with higher revenue (though not always profitable)

2. **Sentiment Impact**: 
   - Positive sentiment generally correlates with higher revenue
   - However, sentiment alone may not be as predictive as budget
   - Genre-specific sentiment patterns exist (e.g., comedies may naturally score higher)

3. **Genre Effects**:
   - Different genres show different average sentiment scores
   - Action and Sci-Fi may have different sentiment-revenue relationships than Drama or Comedy

### Business Implications

1. **For Studio Executives**:
   - Budget allocation should consider genre and expected sentiment response
   - Early sentiment analysis from test screenings can inform marketing strategies

2. **For Marketing Teams**:
   - Monitor sentiment trends early in release
   - Genre-specific marketing strategies based on typical sentiment patterns

3. **For Investors**:
   - Use model predictions to assess potential ROI
   - Consider sentiment trends alongside budget and genre

### Limitations

1. **Data Quality**: Results depend heavily on quality of review data
   - Missing reviews, biased samples, or fake reviews can skew results

2. **Feature Completeness**: Additional factors not included:
   - Marketing budget
   - Release timing (holiday seasons, competition)
   - Star power / cast popularity
   - Critical reviews vs user reviews

3. **Model Constraints**:
   - Linear models assume linear relationships
   - May not capture complex market dynamics

### Recommendations for Future Work

1. **Feature Engineering**:
   - Add marketing budget as a feature
   - Include release date/time of year
   - Add cast/director popularity metrics

2. **Advanced Modeling**:
   - Try ensemble methods combining multiple models
   - Use deep learning for more complex patterns
   - Implement time series analysis for revenue prediction over time

3. **Data Collection**:
   - Collect real-time streaming review data
   - Include critical reviews alongside user reviews
   - Add social media sentiment (Twitter, Reddit)

4. **Domain-Specific Analysis**:
   - Analyze by production company
   - Study sequel vs original movie patterns
   - International vs domestic market differences

## Technical Deliverables

### Code Structure
- **Phase 1**: Data acquisition and preprocessing
- **Phase 2**: Sentiment analysis implementation
- **Phase 3**: EDA and visualization
- **Phase 4**: Model building and evaluation
- **Main Pipeline**: Complete end-to-end workflow

### Outputs
- Processed dataset with sentiment scores
- Visualizations (scatter plots, bar charts, heatmaps)
- Trained models with performance metrics
- Predictions for hypothetical new movies

## Conclusion

This project successfully demonstrates:
1. ✅ NLP application using VADER for sentiment analysis
2. ✅ Data visualization and exploratory analysis
3. ✅ Machine learning model development (regression)
4. ✅ End-to-end data science pipeline

**Final Answer**: While sentiment analysis provides valuable insights, **budget and genre are typically stronger predictors** of box office success. However, **sentiment can be a valuable secondary indicator**, especially when combined with other features. The model shows that a holistic approach considering multiple factors yields better predictions than relying on any single metric alone.

---

## How to Use This Project

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run complete pipeline**: `python src/main.py --sample` (for demo) or `python src/main.py --data path/to/your/data.csv`
3. **Or run phases individually**:
   - `python src/phase1_data_acquisition.py`
   - `python src/phase2_sentiment_analysis.py`
   - `python src/phase3_eda.py`
   - `python src/phase4_modeling.py`

## Dataset Requirements

Your dataset should include:
- **User_Review** or **Plot**: Text reviews/descriptions
- **Revenue_Millions**: Box office revenue (target variable)
- **Budget_Millions**: Production budget
- **Genre**: Movie genre (categorical)
- **Movie_Title**: Movie name (optional)

---

*Project completed as part of Data Science Portfolio - Movie Success Prediction and Sentiment Study*
