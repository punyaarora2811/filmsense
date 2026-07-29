import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── FastAPI Setup ──────────────────────────────────────────────────────────────

app = FastAPI(title="FilmSense API")

# Allow cross-origin requests so the frontend can call this API from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health Check / Root Endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "FilmSense API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ── Data Loading ──────────────────────────────────────────────────────────────

print("Loading pre-cleaned dataset...")

# Dynamically find path relative to this script so it works from any directory
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, 'data', '50K_Movies.csv')

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

# Load the pre-processed dataset which already has the 'tags' column built
movies = pd.read_csv(
    dataset_path,
    usecols=['id', 'title', 'poster_path', 'vote_average', 'tags']
)

# Ensure no NaNs in tags
movies['tags'] = movies['tags'].fillna('')

print(f"Loaded {len(movies)} movies.")

# ── Model Training ────────────────────────────────────────────────────────────

# Convert tag strings into a TF-IDF numerical matrix (sparse) for similarity comparison
print("Vectorizing...")
tv = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
)
vectors = tv.fit_transform(movies['tags'])

# Free the tags column — the TF-IDF matrix has replaced it
movies = movies.drop(columns=['tags'])

# Fit a KNN model using cosine distance to find the 10 most similar movies (+ 1 for the query itself)
print("Fitting KNN...")
nn = NearestNeighbors(
    n_neighbors=11,
    metric='cosine',
    algorithm='brute'
)
nn.fit(vectors)
print("Backend ready. Extremely memory-optimized.")

# ── API Endpoint ──────────────────────────────────────────────────────────────

@app.get("/api/recommend")
def recommend(query: str):
    """Accept a movie title and return 10 similar movies based on content similarity."""

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty")

    # Case-insensitive title lookup
    match = movies[movies['title'].str.lower() == query.strip().lower()]
    
    if match.empty:
        # Fallback search for partial match if exact match fails
        match = movies[movies['title'].str.lower().str.contains(query.strip().lower(), regex=False)]
        
    if match.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
        
    movie_index = match.index[0]
    movie_title = match.iloc[0]['title']
    
    # Find the 11 nearest neighbors (first result is the query movie itself, so skip it)
    distances, indices = nn.kneighbors(vectors[movie_index], n_neighbors=11)
    
    # Build the response list with poster URLs and ratings
    recs = []
    for i in indices[0][1:]:
        row = movies.iloc[i]
        poster_url = ""
        poster_path = row.get('poster_path')
        if pd.notna(poster_path) and str(poster_path).strip() and str(poster_path) != "nan":
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            
        try:
            rating = float(row['vote_average'])
        except (ValueError, TypeError):
            rating = 0.0
            
        recs.append({
            "id": int(row['id']),
            "title": str(row['title']),
            "rating": round(rating, 1),
            "posterUrl": poster_url
        })
        
    return {
        "movieTitle": str(movie_title),
        "recs": recs
    }
