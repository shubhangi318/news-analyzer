---
title: News Analyzer
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# News Analyzer

![News Analyzer](https://img.shields.io/badge/News-Analyzer-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-red)

A comprehensive platform for analyzing company news sentiment using advanced NLP techniques. Extract, analyze, and visualize sentiment patterns from multiple news sources.

## 🌟 Features

- **Multi-source News Extraction**: Gather news articles from various online sources
- **Advanced Sentiment Analysis**: Hybrid model approach combining multiple sentiment analysis techniques
- **Comparative Analytics**: Compare sentiment across different news sources
- **Topic Extraction**: Identify and categorize key topics in news coverage
- **Multilingual Support**: Translation and audio generation in Hindi
- **Interactive Visualizations**: Rich set of charts and graphs for data exploration
- **Detailed Reporting**: Comprehensive summaries and article-specific insights

## 📁 Project Structure

```
News-Analyzer/
├── backend/                  # FastAPI backend service
│   ├── audio_output/         # Generated audio files
│   ├── model/                # Data models
│   ├── news_extract_metadata/# News extraction utilities
│   ├── sentiment_analysis/   # Sentiment analysis modules
│   ├── utils/                # Utility functions
│   └── main.py               # Main FastAPI application
├── frontend/                 # Streamlit frontend
│   └── app.py                # Main Streamlit application
└── requirements.txt          # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/News-Analyzer.git
   cd News-Analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

4. In a new terminal, start the frontend:
   ```bash
   cd frontend
   streamlit run app.py
   ```

5. Open your browser and navigate to:
   - Frontend: http://localhost:8501
   - Backend API docs: http://localhost:8000/docs

## 🧩 Architecture

The application follows a client-server architecture:

- **Backend**: FastAPI service providing RESTful endpoints for news extraction, sentiment analysis, and report generation
- **Frontend**: Streamlit web application providing an intuitive user interface with interactive visualizations

## 📊 Example Use Cases

- Monitor public sentiment for your company across different news sources
- Compare sentiment between competitors in the same industry
- Identify trending topics related to your business
- Generate comprehensive reports for management or stakeholders

## 🛠️ Tech Stack

- **Backend**: FastAPI, OpenAI, gTTS, NLTK, TextBlob, Azure Text Analytics
- **Frontend**: Streamlit, Plotly, Matplotlib, PyWaffle
- **NLP**: Spacy, NLTK, Hugging Face Transformers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📬 Contact

For any questions or suggestions, please drop an email to shubhangi318@gmail.com. Thanks!
