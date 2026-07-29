<div align="center">
  <h1>🎬 FilmSense</h1>
  <p><strong>A dynamic, content-based movie recommendation engine built with React and Python.</strong></p>
  <p>🚀 <strong><a href="https://filmsense-jade.vercel.app">Live Demo</a></strong></p>
  
  [![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](#)
  [![Vite](https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E)](#)
  [![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](#)
  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
  [![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)](#)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](#)
</div>

<br />

## 🌟 Overview

FilmSense is a full-stack web application that intelligently suggests movies based on content similarity. Utilizing the comprehensive TMDB dataset, it analyzes movie overviews, genres, keywords, cast, and directors to uncover the 10 closest recommendations to your favorite films via cosine similarity.

## ✨ Features

- **Smart Recommendations:** Leverages TF-IDF vectorization and K-Nearest Neighbors (KNN) to generate accurate, content-based movie recommendations.
- **Beautiful UI:** A responsive, visually striking frontend crafted with React, Vite, and Tailwind CSS.
- **Lightning Fast API:** A robust backend powered by FastAPI that serves predictions rapidly.
- **Rich Data Processing:** Uses scikit-learn for TF-IDF vectorization and KNN-based similarity search.

## 📂 Project Structure

```text
FilmSense/
├── backend/
│   ├── data/
│   │   └── 50K_Movies.csv            # Pre-cleaned, lightweight dataset
│   ├── main.py                       # FastAPI application & ML pipeline
│   └── requirements.txt              # Python dependencies
├── frontend/
│   ├── public/                       # Static assets (favicon)
│   ├── src/
│   │   ├── components/               # Reusable React components
│   │   ├── lib/                      # Helper functions & API utilities
│   │   ├── styles/                   # Global styling (Tailwind CSS v4)
│   │   ├── App.jsx                   # Main application component
│   │   └── main.jsx                  # React application entry point
│   ├── .env.example                  # Environment variables template
│   ├── index.html                    # HTML entry point
│   ├── package.json                  # Frontend dependencies
│   └── vite.config.js                # Vite configuration
├── .gitignore
├── package.json                      # Root monorepo scripts
└── README.md
```

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **TMDB API Key** (Get one for free [here](https://www.themoviedb.org/settings/api))

### 1. Clone the Repository

```bash
git clone https://github.com/punyaarora2811/filmsense.git
cd filmsense
```

### 2. Dataset

The repository already includes the `50K_Movies.csv` in the `backend/data/` directory. It contains the top 50,000 highest-rated movies with pre-computed features for lightning-fast ML modeling. No extra downloads needed!

### 3. Backend Setup (FastAPI & ML)

Install the Python dependencies and start the backend server. The model will build its vectors on startup (takes a few seconds).

```bash
# Install backend dependencies
npm run install:backend

# Start the FastAPI server (Runs on http://127.0.0.1:8000)
npm run dev:backend
```

### 4. Frontend Setup (React UI)

Configure the environment variables and start the frontend application.

```bash
# Set up your environment variables
cp frontend/.env.example frontend/.env

# Edit frontend/.env and add your keys:
# VITE_TMDB_API_KEY=your_api_key_here
# VITE_API_URL=http://127.0.0.1:8000

# Install dependencies and start the dev server
npm run install:frontend
npm run dev:frontend
```

The frontend will be available at `http://localhost:5173` (or similar, check terminal output).

## 🧠 How the ML Pipeline Works

To run lightning-fast on the free tier, the pipeline is split into an offline preprocessing stage and a runtime inference stage:

### 1. Offline Preprocessing (Already done)
1. **Data Cleaning:** The raw 400K TMDB dataset is filtered to remove empty rows and kept only to the top 50,000 movies by rating to save memory.
2. **Feature Extraction:** Genres, keywords, cast, and directors are merged into lists. Multi-word names are collapsed (e.g., `Tom Hanks` → `TomHanks`) to prevent false positive matches.
3. **Weighted Tagging:** Constructs a master "tag" string for each movie where directors and genres are weighted 3×, keywords 2×, and overview/cast 1×.

### 2. Runtime (FastAPI Server)
1. **Vectorization:** Upon startup, the backend instantly converts the pre-computed tags into a numerical matrix using TF-IDF (Term Frequency-Inverse Document Frequency).
2. **Recommendation:** When you search for a movie, the server computes the closest neighbors via Cosine Similarity utilizing the K-Nearest Neighbors (KNN) algorithm and returns the top 10 results in milliseconds.

## 🚀 Deployment

FilmSense is deployed with the frontend on [Vercel](https://vercel.com) and the backend on [Railway](https://railway.app).

### 1. Backend (Railway)

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select **filmsense**
2. Configure under **Settings → Build & Deploy**:
   | Field | Value |
   |---|---|
   | **Root Directory** | `backend` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
3. Go to **Settings → Networking** → **Generate Domain** and copy the URL.

### 2. Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → **Add New** → **Project** → import **filmsense**
2. Set **Framework Preset** to `Vite` and **Root Directory** to `frontend`
3. Add environment variables:
   | Key | Value |
   |---|---|
   | `VITE_API_URL` | Your Railway backend URL |
   | `VITE_TMDB_API_KEY` | Your TMDB API key |
4. Click **Deploy**.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
