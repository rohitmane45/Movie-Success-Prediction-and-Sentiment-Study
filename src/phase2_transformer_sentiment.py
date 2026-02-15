"""
Phase 2 (Enhanced): Transformer-Based Sentiment Analysis with DistilBERT

Goal: Augment VADER with a deep-learning sentiment model for higher accuracy.

Why DistilBERT?
    VADER is lexicon-based — it scores individual words and sums them up.
    This fails on negation, sarcasm, and complex sentences:
        "This movie is not bad"  → VADER scores "bad" as negative  → WRONG
        DistilBERT reads the FULL sentence in context              → CORRECT

    DistilBERT-SST2 is pre-trained on 67K movie reviews from Stanford
    Sentiment Treebank, making it perfect for our use case.

Technical Notes:
    - Model: distilbert-base-uncased-finetuned-sst-2-english (~260MB)
    - No fine-tuning needed — works out of the box for movie reviews
    - Batched inference for speed on large datasets
    - Falls back gracefully to VADER if torch/transformers not installed
"""

import pandas as pd
import numpy as np

# ── Check for transformer dependencies ──────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _get_transformer_pipeline():
    """
    Initialize the DistilBERT sentiment pipeline (cached after first call).

    Why we use a function: The model loads ~260MB into memory. We want to
    load it exactly once and reuse it across all calls.

    Returns:
        transformers.Pipeline or None: Sentiment analysis pipeline
    """
    if not HAS_TRANSFORMERS:
        return None

    try:
        classifier = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,           # CPU (use 0 for GPU if available)
            truncation=True,
            max_length=512       # DistilBERT max token length
        )
        return classifier
    except Exception as e:
        print(f"[WARNING] Could not load transformer model: {e}")
        return None


def _score_single_text(text, classifier):
    """
    Score a single text string with the transformer model.

    The model returns:
        [{'label': 'POSITIVE', 'score': 0.9998}]

    We convert this to a -1 to +1 scale to match VADER's compound score:
        POSITIVE with score 0.95 → +0.95
        NEGATIVE with score 0.85 → -0.85

    Args:
        text (str): Review text
        classifier: Hugging Face pipeline

    Returns:
        float: Sentiment score from -1 (negative) to +1 (positive)
    """
    if not text or pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    try:
        # Truncate very long texts to avoid memory issues
        truncated = text[:1500]
        result = classifier(truncated)[0]
        score = result['score']

        if result['label'] == 'NEGATIVE':
            return -score
        else:
            return score
    except Exception:
        return 0.0


def _batch_score(texts, classifier, batch_size=32):
    """
    Score a batch of texts efficiently using the transformer pipeline.

    Why batching: Processing one text at a time is slow because of GPU/CPU
    overhead per call. Batching amortizes this cost. For 3,500 movies:
        - One-by-one: ~7 minutes
        - Batch of 32: ~1 minute

    Args:
        texts (list): List of text strings
        classifier: Hugging Face pipeline
        batch_size (int): Texts per batch

    Returns:
        list: Sentiment scores for each text
    """
    scores = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]

        # Clean batch — replace empty/NaN with placeholder
        cleaned = []
        empty_indices = set()
        for j, t in enumerate(batch):
            if not t or pd.isna(t) or not isinstance(t, str) or len(str(t).strip()) == 0:
                cleaned.append("neutral")  # Placeholder to keep batch aligned
                empty_indices.add(j)
            else:
                cleaned.append(str(t)[:1500])

        try:
            results = classifier(cleaned, batch_size=batch_size)
            for j, res in enumerate(results):
                if j in empty_indices:
                    scores.append(0.0)
                else:
                    s = res['score']
                    scores.append(s if res['label'] == 'POSITIVE' else -s)
        except Exception:
            # Fall back to one-by-one on batch failure
            for j, t in enumerate(cleaned):
                if j in empty_indices:
                    scores.append(0.0)
                else:
                    scores.append(_score_single_text(t, classifier))

        # Progress indicator
        done = min(i + batch_size, total)
        pct = done / total * 100
        print(f"\r      Transformer sentiment: {done}/{total} ({pct:.0f}%)", end='', flush=True)

    print()  # newline after progress
    return scores


def analyze_sentiment_transformer(df, review_column='User_Review', batch_size=32):
    """
    Apply DistilBERT transformer sentiment analysis to a DataFrame column.

    This function mirrors the API of analyze_sentiment() from
    phase2_sentiment_analysis.py so it can be used as a drop-in replacement
    or run alongside it for comparison.

    Args:
        df (pd.DataFrame): DataFrame containing review text
        review_column (str): Column name with review text
        batch_size (int): Batch size for inference

    Returns:
        pd.DataFrame: DataFrame with 'Transformer_Sentiment' column added
        bool: True if transformer was used, False if fell back to VADER
    """
    if not HAS_TRANSFORMERS:
        print("      [INFO] Transformers not installed. Using VADER only.")
        print("             To enable: pip install torch transformers")
        return df, False

    classifier = _get_transformer_pipeline()
    if classifier is None:
        print("      [INFO] Could not load transformer model. Using VADER only.")
        return df, False

    print("      Loading DistilBERT sentiment model...")

    # Get texts
    texts = df[review_column].fillna('').tolist()

    # Batch inference
    scores = _batch_score(texts, classifier, batch_size=batch_size)

    df['Transformer_Sentiment'] = scores

    # Add category labels
    df['Transformer_Category'] = pd.cut(
        df['Transformer_Sentiment'],
        bins=[-1.01, -0.5, -0.05, 0.05, 0.5, 1.01],
        labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    )

    return df, True


def compare_sentiment_engines(df, vader_col='Sentiment_Score',
                               transformer_col='Transformer_Sentiment'):
    """
    Compare VADER vs Transformer sentiment scores.

    Why: Understanding where the two models agree vs disagree helps
    identify which model to trust and which reviews are ambiguous.

    Args:
        df (pd.DataFrame): DataFrame with both sentiment columns
        vader_col (str): VADER sentiment column name
        transformer_col (str): Transformer sentiment column name

    Returns:
        dict: Comparison statistics
    """
    if transformer_col not in df.columns or vader_col not in df.columns:
        return None

    # Calculate agreement
    vader_sign = np.sign(df[vader_col])
    trans_sign = np.sign(df[transformer_col])
    agreement = (vader_sign == trans_sign).mean()

    # Correlation
    correlation = df[vader_col].corr(df[transformer_col])

    # Disagreement analysis
    disagree_mask = vader_sign != trans_sign
    n_disagree = disagree_mask.sum()

    # Score distributions
    stats = {
        'agreement_rate': agreement,
        'correlation': correlation,
        'n_disagreements': int(n_disagree),
        'vader_mean': df[vader_col].mean(),
        'transformer_mean': df[transformer_col].mean(),
        'vader_std': df[vader_col].std(),
        'transformer_std': df[transformer_col].std(),
    }

    return stats


# ── Standalone Test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 (Enhanced): Transformer Sentiment Analysis")
    print("=" * 60)

    if not HAS_TRANSFORMERS:
        print("\n[!] torch and/or transformers not installed.")
        print("    Install with: pip install torch transformers")
        print("    The pipeline will fall back to VADER-only mode.")
    else:
        # Test with sample reviews
        data = {
            'Movie_Title': ['Inception', 'The Room', 'Average Joe', 'Tricky Review'],
            'User_Review': [
                "This movie was an absolute masterpiece! Mind-blowing plot.",
                "Worst movie ever. Complete waste of time and money.",
                "It was okay. Not great, not terrible.",
                "This movie is not bad at all, actually quite good!"  # VADER gets this wrong
            ]
        }
        df = pd.DataFrame(data)

        # Run transformer
        df, used_transformer = analyze_sentiment_transformer(df)

        if used_transformer:
            print("\n--- Transformer Sentiment Scores ---")
            for _, row in df.iterrows():
                print(f"  {row['Movie_Title']:<20} → {row['Transformer_Sentiment']:+.3f} ({row['Transformer_Category']})")

            # Compare with VADER
            from phase2_sentiment_analysis import analyze_sentiment
            df = analyze_sentiment(df)

            stats = compare_sentiment_engines(df)
            if stats:
                print(f"\n--- VADER vs Transformer ---")
                print(f"  Agreement rate: {stats['agreement_rate']:.1%}")
                print(f"  Correlation:    {stats['correlation']:.3f}")
                print(f"  Disagreements:  {stats['n_disagreements']}")

    print("\nPhase 2 (Enhanced) complete!")
