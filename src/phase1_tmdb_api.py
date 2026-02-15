"""
Live TMDB API Client — Real-Time Movie Data & Reviews

Provides functions to query The Movie Database (TMDB) API for:
- Movie search by title
- Full movie details (budget, genres, cast, crew)
- User reviews (for live sentiment analysis)
- Now-playing, upcoming, and trending movie lists

Setup:
    1. Get a free API key at https://www.themoviedb.org/settings/api
    2. Create a .env file in project root: TMDB_API_KEY=your_key_here
    3. Or set environment variable: export TMDB_API_KEY=your_key_here

Usage:
    from phase1_tmdb_api import TMDBClient
    client = TMDBClient()
    results = client.search_movie("Inception")
    details = client.get_movie_details(results[0]['id'])
    reviews = client.get_movie_reviews(results[0]['id'])
"""

import os
import sys
import json
import time
import requests
from functools import lru_cache

# ── Load API key ────────────────────────────────────────────────────
def _load_api_key():
    """
    Load TMDB API key from environment or .env file.
    
    Priority:
        1. TMDB_API_KEY environment variable
        2. .env file in project root
        3. .env file in src/ directory
    """
    # Check environment variable first
    key = os.environ.get('TMDB_API_KEY')
    if key:
        return key.strip()
    
    # Look for .env file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    for env_path in [
        os.path.join(project_dir, '.env'),
        os.path.join(src_dir, '.env'),
    ]:
        if os.path.exists(env_path):
            # Try multiple encodings (PowerShell creates UTF-16/BOM files)
            content = None
            for encoding in ['utf-8-sig', 'utf-16', 'utf-8', 'latin-1']:
                try:
                    with open(env_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content:
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('TMDB_API_KEY='):
                        key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if key and key != 'your_key_here':
                            return key
    
    return None


# ── Constants ───────────────────────────────────────────────────────
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w500"      # w92, w154, w185, w342, w500, w780, original
BACKDROP_SIZE = "w780"

# Rate limiting: TMDB allows 40 requests per 10 seconds
_last_request_time = 0
_MIN_REQUEST_INTERVAL = 0.25  # seconds between requests


def _rate_limit():
    """Simple rate limiter to respect TMDB's 40 req/10 sec limit."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


# ════════════════════════════════════════════════════════════════════
#  TMDB CLIENT CLASS
# ════════════════════════════════════════════════════════════════════

class TMDBClient:
    """
    Client for The Movie Database (TMDB) API.
    
    Handles authentication, rate limiting, caching, and error handling.
    All methods return Python dicts/lists (parsed JSON).
    """
    
    def __init__(self, api_key=None):
        """
        Initialize TMDB client.
        
        Args:
            api_key (str): TMDB API key. If None, loads from env/.env file.
        """
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            raise ValueError(
                "TMDB API key not found!\n"
                "Set it via:\n"
                "  1. Environment variable: TMDB_API_KEY=your_key\n"
                "  2. .env file in project root: TMDB_API_KEY=your_key\n"
                "  3. Pass directly: TMDBClient(api_key='your_key')\n"
                "\nGet a free key at: https://www.themoviedb.org/settings/api"
            )
        self._session = requests.Session()
        self._session.params = {'api_key': self.api_key}
        # Simple in-memory cache
        self._cache = {}
    
    def _get(self, endpoint, params=None, cache_key=None):
        """
        Make a GET request to TMDB API with rate limiting and caching.
        
        Args:
            endpoint (str): API endpoint (e.g., '/movie/550')
            params (dict): Additional query parameters
            cache_key (str): Optional cache key to avoid duplicate requests
            
        Returns:
            dict: Parsed JSON response
        """
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        _rate_limit()
        
        url = f"{BASE_URL}{endpoint}"
        try:
            response = self._session.get(url, params=params or {}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if cache_key:
                self._cache[cache_key] = data
            
            return data
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise ValueError("Invalid TMDB API key. Check your key at themoviedb.org/settings/api")
            elif response.status_code == 404:
                return None
            elif response.status_code == 429:
                # Rate limited — wait and retry
                time.sleep(2)
                return self._get(endpoint, params, cache_key)
            else:
                print(f"[TMDB API Error] {response.status_code}: {e}")
                return None
        except requests.exceptions.ConnectionError:
            print("[TMDB API Error] No internet connection")
            return None
        except requests.exceptions.Timeout:
            print("[TMDB API Error] Request timed out")
            return None
    
    # ── Search ──────────────────────────────────────────────────────
    
    def search_movie(self, query, year=None):
        """
        Search for movies by title.
        
        Args:
            query (str): Movie title to search for
            year (int): Optional release year to narrow results
            
        Returns:
            list[dict]: List of matching movies with id, title, 
                        release_date, overview, poster_path, vote_average
        """
        params = {'query': query, 'language': 'en-US', 'page': 1}
        if year:
            params['year'] = year
        
        data = self._get('/search/movie', params, cache_key=f"search:{query}:{year}")
        if not data or 'results' not in data:
            return []
        
        results = []
        for movie in data['results'][:10]:  # Top 10 results
            results.append({
                'id': movie['id'],
                'title': movie.get('title', 'Unknown'),
                'release_date': movie.get('release_date', ''),
                'overview': movie.get('overview', ''),
                'poster_path': movie.get('poster_path', ''),
                'backdrop_path': movie.get('backdrop_path', ''),
                'vote_average': movie.get('vote_average', 0),
                'vote_count': movie.get('vote_count', 0),
                'popularity': movie.get('popularity', 0),
                'genre_ids': movie.get('genre_ids', []),
            })
        
        return results
    
    # ── Movie Details ───────────────────────────────────────────────
    
    def get_movie_details(self, movie_id):
        """
        Get full movie details including budget, revenue, credits.
        
        Args:
            movie_id (int): TMDB movie ID
            
        Returns:
            dict: Full movie details with budget, revenue, genres,
                  cast, crew, runtime, status, etc.
        """
        # Append credits and keywords in a single request
        data = self._get(
            f'/movie/{movie_id}',
            params={'append_to_response': 'credits,keywords'},
            cache_key=f"details:{movie_id}"
        )
        
        if not data:
            return None
        
        # Extract director
        director = None
        if 'credits' in data and 'crew' in data['credits']:
            for person in data['credits']['crew']:
                if person.get('job') == 'Director':
                    director = person.get('name')
                    break
        
        # Extract top cast (first 5)
        cast = []
        if 'credits' in data and 'cast' in data['credits']:
            for person in data['credits']['cast'][:5]:
                cast.append({
                    'name': person.get('name'),
                    'character': person.get('character'),
                    'profile_path': person.get('profile_path'),
                })
        
        # Extract genres
        genres = [g['name'] for g in data.get('genres', [])]
        
        # Extract production companies
        companies = [c['name'] for c in data.get('production_companies', [])]
        
        # Build clean result
        result = {
            'id': data['id'],
            'title': data.get('title', 'Unknown'),
            'original_title': data.get('original_title', ''),
            'overview': data.get('overview', ''),
            'tagline': data.get('tagline', ''),
            'status': data.get('status', ''),           # Released, Post Production, etc.
            'release_date': data.get('release_date', ''),
            'runtime': data.get('runtime', 0),
            'budget': data.get('budget', 0),             # In dollars
            'revenue': data.get('revenue', 0),           # In dollars
            'budget_millions': data.get('budget', 0) / 1_000_000 if data.get('budget') else 0,
            'revenue_millions': data.get('revenue', 0) / 1_000_000 if data.get('revenue') else 0,
            'genres': genres,
            'primary_genre': genres[0] if genres else 'Unknown',
            'director': director,
            'cast': cast,
            'lead_actor': cast[0]['name'] if cast else 'Unknown',
            'production_companies': companies,
            'primary_studio': companies[0] if companies else 'Unknown',
            'vote_average': data.get('vote_average', 0),
            'vote_count': data.get('vote_count', 0),
            'popularity': data.get('popularity', 0),
            'poster_path': data.get('poster_path', ''),
            'backdrop_path': data.get('backdrop_path', ''),
            'poster_url': f"{IMAGE_BASE}/{POSTER_SIZE}{data['poster_path']}" if data.get('poster_path') else None,
            'backdrop_url': f"{IMAGE_BASE}/{BACKDROP_SIZE}{data['backdrop_path']}" if data.get('backdrop_path') else None,
            'imdb_id': data.get('imdb_id', ''),
            'homepage': data.get('homepage', ''),
        }
        
        return result
    
    # ── Reviews ─────────────────────────────────────────────────────
    
    def get_movie_reviews(self, movie_id, max_pages=3):
        """
        Fetch user reviews for a movie.
        
        Args:
            movie_id (int): TMDB movie ID
            max_pages (int): Maximum number of pages to fetch (20 reviews/page)
            
        Returns:
            list[dict]: Reviews with author, content, rating, created_at
        """
        all_reviews = []
        
        for page in range(1, max_pages + 1):
            data = self._get(
                f'/movie/{movie_id}/reviews',
                params={'language': 'en-US', 'page': page},
                cache_key=f"reviews:{movie_id}:p{page}"
            )
            
            if not data or 'results' not in data:
                break
            
            for review in data['results']:
                rating = None
                if review.get('author_details', {}).get('rating'):
                    rating = review['author_details']['rating']
                
                all_reviews.append({
                    'author': review.get('author', 'Anonymous'),
                    'content': review.get('content', ''),
                    'rating': rating,                    # 1-10 scale (or None)
                    'created_at': review.get('created_at', ''),
                    'url': review.get('url', ''),
                })
            
            # Stop if we've gotten all pages
            if page >= data.get('total_pages', 1):
                break
        
        return all_reviews
    
    # ── Lists: Now Playing, Upcoming, Trending ──────────────────────
    
    def get_now_playing(self, region='US', page=1):
        """Get movies currently in theatres."""
        data = self._get(
            '/movie/now_playing',
            params={'language': 'en-US', 'page': page, 'region': region},
            cache_key=f"now_playing:{region}:{page}"
        )
        if not data or 'results' not in data:
            return []
        return self._parse_movie_list(data['results'][:20])
    
    def get_upcoming(self, region='US', page=1):
        """Get upcoming movie releases."""
        data = self._get(
            '/movie/upcoming',
            params={'language': 'en-US', 'page': page, 'region': region},
            cache_key=f"upcoming:{region}:{page}"
        )
        if not data or 'results' not in data:
            return []
        return self._parse_movie_list(data['results'][:20])
    
    def get_trending(self, time_window='week'):
        """
        Get trending movies.
        
        Args:
            time_window (str): 'day' or 'week'
        """
        data = self._get(
            f'/trending/movie/{time_window}',
            cache_key=f"trending:{time_window}"
        )
        if not data or 'results' not in data:
            return []
        return self._parse_movie_list(data['results'][:20])
    
    def _parse_movie_list(self, movies):
        """Parse a list of movie objects from TMDB API."""
        results = []
        for m in movies:
            results.append({
                'id': m['id'],
                'title': m.get('title', 'Unknown'),
                'release_date': m.get('release_date', ''),
                'overview': m.get('overview', ''),
                'poster_path': m.get('poster_path', ''),
                'poster_url': f"{IMAGE_BASE}/{POSTER_SIZE}{m['poster_path']}" if m.get('poster_path') else None,
                'vote_average': m.get('vote_average', 0),
                'vote_count': m.get('vote_count', 0),
                'popularity': m.get('popularity', 0),
                'genre_ids': m.get('genre_ids', []),
            })
        return results
    
    # ── Genre Map ───────────────────────────────────────────────────
    
    def get_genre_map(self):
        """Get mapping of genre IDs to genre names."""
        data = self._get('/genre/movie/list', cache_key="genre_map")
        if not data or 'genres' not in data:
            # Fallback: common genres
            return {
                28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
                80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
                14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
                9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
                10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
            }
        return {g['id']: g['name'] for g in data['genres']}
    
    def resolve_genre_ids(self, genre_ids):
        """Convert genre ID list to genre name list."""
        genre_map = self.get_genre_map()
        return [genre_map.get(gid, 'Unknown') for gid in genre_ids]
    
    # ── Utility ─────────────────────────────────────────────────────
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
    
    def test_connection(self):
        """
        Test if the API key is valid and connection works.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            data = self._get('/configuration')
            if data and 'images' in data:
                return True, "Connected to TMDB API successfully"
            return False, "Unexpected response from TMDB API"
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Connection failed: {e}"


# ════════════════════════════════════════════════════════════════════
#  LIVE PREDICTION HELPER
# ════════════════════════════════════════════════════════════════════

def analyze_live_reviews(reviews, use_transformer=False):
    """
    Run sentiment analysis on live TMDB reviews.
    
    Args:
        reviews (list[dict]): Reviews from get_movie_reviews()
        use_transformer (bool): If True, also run DistilBERT
        
    Returns:
        dict: Sentiment analysis results
            - vader_scores: list of VADER compound scores
            - vader_avg: mean VADER score
            - transformer_scores: list of transformer scores (if enabled)
            - transformer_avg: mean transformer score (if enabled)
            - review_count: number of reviews analyzed
            - sentiment_label: 'Positive', 'Negative', or 'Mixed'
            - confidence: 0-1 based on number of reviews
    """
    if not reviews:
        return {
            'vader_scores': [],
            'vader_avg': 0.0,
            'transformer_scores': [],
            'transformer_avg': 0.0,
            'review_count': 0,
            'sentiment_label': 'No Reviews',
            'confidence': 0.0,
        }
    
    # VADER sentiment
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        sia = SentimentIntensityAnalyzer()
    except ImportError:
        sia = None
    
    vader_scores = []
    review_texts = []
    
    for review in reviews:
        text = review.get('content', '')
        if not text or len(text) < 10:
            continue
        
        # Truncate very long reviews for efficiency
        text = text[:2000]
        review_texts.append(text)
        
        if sia:
            score = sia.polarity_scores(text)['compound']
            vader_scores.append(score)
    
    # Transformer sentiment (optional)
    transformer_scores = []
    if use_transformer and review_texts:
        try:
            from phase2_transformer_sentiment import _get_transformer_pipeline
            pipe = _get_transformer_pipeline()
            if pipe:
                # Batch process (truncate to 512 chars for transformer)
                short_texts = [t[:512] for t in review_texts]
                results = pipe(short_texts, batch_size=8)
                for r in results:
                    # Convert POSITIVE/NEGATIVE to -1 to +1
                    score = r['score'] if r['label'] == 'POSITIVE' else -r['score']
                    transformer_scores.append(score)
        except Exception:
            pass
    
    # Calculate aggregates
    vader_avg = sum(vader_scores) / len(vader_scores) if vader_scores else 0.0
    trans_avg = sum(transformer_scores) / len(transformer_scores) if transformer_scores else 0.0
    
    # Confidence based on review count (more reviews = higher confidence)
    n_reviews = len(review_texts)
    if n_reviews >= 20:
        confidence = 1.0
    elif n_reviews >= 10:
        confidence = 0.8
    elif n_reviews >= 5:
        confidence = 0.6
    elif n_reviews >= 2:
        confidence = 0.4
    elif n_reviews >= 1:
        confidence = 0.2
    else:
        confidence = 0.0
    
    # Determine overall sentiment label
    primary_score = trans_avg if transformer_scores else vader_avg
    if primary_score > 0.1:
        label = 'Positive'
    elif primary_score < -0.1:
        label = 'Negative'
    else:
        label = 'Mixed'
    
    return {
        'vader_scores': vader_scores,
        'vader_avg': vader_avg,
        'transformer_scores': transformer_scores,
        'transformer_avg': trans_avg,
        'review_count': n_reviews,
        'sentiment_label': label,
        'confidence': confidence,
    }


# ════════════════════════════════════════════════════════════════════
#  CLI TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TMDB API Client — Connection Test")
    print("=" * 60)
    
    try:
        client = TMDBClient()
        success, msg = client.test_connection()
        print(f"\n  Status: {'✅' if success else '❌'} {msg}")
        
        if success:
            # Test search
            print("\n  Testing search: 'Inception'")
            results = client.search_movie("Inception")
            if results:
                movie = results[0]
                print(f"  Found: {movie['title']} ({movie['release_date'][:4]})")
                print(f"  Rating: {movie['vote_average']}/10")
                
                # Test details
                details = client.get_movie_details(movie['id'])
                if details:
                    print(f"  Budget: ${details['budget_millions']:.0f}M")
                    print(f"  Revenue: ${details['revenue_millions']:.0f}M")
                    print(f"  Director: {details['director']}")
                    print(f"  Genre: {details['primary_genre']}")
                
                # Test reviews
                reviews = client.get_movie_reviews(movie['id'])
                print(f"  Reviews: {len(reviews)} found")
                
                if reviews:
                    sentiment = analyze_live_reviews(reviews)
                    print(f"  VADER Sentiment: {sentiment['vader_avg']:+.3f}")
                    print(f"  Label: {sentiment['sentiment_label']}")
                    print(f"  Confidence: {sentiment['confidence']:.0%}")
            
            # Test now playing
            print("\n  Now Playing (top 3):")
            now = client.get_now_playing()
            for m in now[:3]:
                print(f"    • {m['title']} ({m['vote_average']}/10)")
            
            # Test trending
            print("\n  Trending This Week (top 3):")
            trending = client.get_trending()
            for m in trending[:3]:
                print(f"    • {m['title']} ({m['vote_average']}/10)")
    
    except ValueError as e:
        print(f"\n  ❌ {e}")
    
    print("\n" + "=" * 60)
