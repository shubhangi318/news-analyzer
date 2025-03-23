# News Analyzer Backend

The backend service for the News Analyzer application built with FastAPI. This service provides APIs for news extraction, sentiment analysis, and comprehensive report generation.

## Technologies Used

- **FastAPI**: High-performance web framework for building APIs
- **OpenAI**: Integration for advanced text analysis and translation
- **gTTS (Google Text-to-Speech)**: For generating audio summaries
- **NLP Models**: Hybrid approach with multiple sentiment analysis models

## Features

- 📰 News extraction from multiple sources
- 🔍 Sentiment analysis using hybrid models
- 🏷️ Topic extraction and categorization
- 📊 Comparative analysis across sources
- 🔄 Hindi translation and text-to-speech capabilities

## Directory Structure

```
backend/
├── audio_output/           # Generated audio files
├── model/                  # Pydantic models
├── news_extract_metadata/  # News extraction utilities
├── sentiment_analysis/     # Sentiment analysis models
├── utils/                  # Utility functions
├── main.py                 # FastAPI application
└── sentiment_analysis.log  # Log file
```

## API Endpoints

- `GET /`: Root endpoint (health check)
- `POST /api/extract-news`: Extract news articles for a company
- `POST /api/analyze-sentiment`: Analyze sentiment of news articles
- `POST /api/compare-sentiment`: Compare sentiment across news sources
- `POST /api/final-analysis`: Generate comprehensive analysis report

## Setup Instructions

1. Install required dependencies:

```bash
pip install -r ../requirements.txt
```

2. Set up environment variables (if needed)

3. Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The server will start on http://localhost:8000 by default.

## Development

To extend the functionality:

1. Add new endpoint handlers in `main.py`
2. Implement additional sentiment analysis models in `sentiment_analysis/`
3. Enhance news extraction capabilities in `news_extract_metadata/` 
