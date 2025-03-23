# News Analyzer Frontend

A Streamlit-based web interface for the News Analyzer application that provides an intuitive dashboard for company news sentiment analysis.

## Technologies Used

- **Streamlit**: Python framework for creating interactive data apps
- **Plotly**: Interactive visualization library
- **Matplotlib & Seaborn**: Data visualization
- **PyWaffle**: For creating waffle charts
- **NLTK**: Natural Language Toolkit for text processing

## Features

- 📊 Interactive dashboards for sentiment analysis
- 📈 Visualizations including donut charts and waffle charts
- 🔀 Comparative analysis between different news sources
- 📑 Detailed article display with sentiment highlighting
- 🔍 Keyword-based news filtering
- 🔊 Audio playback of translated summaries

## Directory Structure

```
frontend/
├── app.py                       # Main Streamlit application
└── sentiment_frequency_chart.png # Sample visualization
```

## UI Sections

The application is organized into three main tabs:

1. **News & Sentiment Analysis**: Basic news extraction and sentiment overview
2. **Comparative Analysis**: Compare sentiment across different news sources
3. **Detailed Analysis**: In-depth report with topic distribution and trends

## Setup Instructions

1. Install required dependencies:

```bash
pip install -r ../requirements.txt
```

2. Ensure the backend service is running (default: http://localhost:8000)

3. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will launch in your default web browser at http://localhost:8501.

## Configuration

The frontend connects to the backend API at the URL specified by the `API_URL` constant in `app.py`. Update this value if your backend is running on a different host or port.

## Development

To extend the UI:

1. Add new visualization components to the relevant sections in `app.py`
2. Create additional tabs for new features
3. Implement new API call functions to interact with backend endpoints 
