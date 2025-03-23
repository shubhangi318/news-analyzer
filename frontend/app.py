import streamlit as st
import requests
import pandas as pd
import json
import base64
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pywaffle import Waffle
import plotly.express as px

import nltk
nltk.download('vader_lexicon', quiet=True)

# API configuration
API_URL = "http://localhost:8000/api"

# Type definitions
ArticleType = Dict[str, Any]
SentimentDistribution = Dict[str, int]

# Sentiment color mapping
SENTIMENT_COLORS = {
    "positive": "#32CD32",  # Green
    "negative": "#B90E0A",  # Red
    "neutral": "#b0bce5"    # Blue
}

###################
# API FUNCTIONS
###################

def call_api_endpoint(endpoint: str, payload: Dict[str, Any]) -> Any:
    """
    Make a POST request to the specified API endpoint.
    
    Args:
        endpoint: API endpoint path
        payload: Request payload
        
    Returns:
        API response parsed as JSON
        
    Raises:
        requests.RequestException: If the API call fails
    """
    response = requests.post(f"{API_URL}/{endpoint}", json=payload)
    response.raise_for_status()
    return response.json()


def get_news(company_name: str, num_articles: int = 10, keywords: Optional[List[str]] = None) -> List[ArticleType]:
    """
    Fetch news articles for a company from the API.
    
    Args:
        company_name: Name of the company to get news for
        num_articles: Number of articles to retrieve
        keywords: Optional list of keywords to refine the search
        
    Returns:
        List of article dictionaries
    """
    payload = {
        "company_name": company_name, 
        "num_articles": num_articles
    }
    
    # Include keywords in payload if provided
    if keywords:
        payload["keywords"] = keywords
        
    return call_api_endpoint("extract-news", payload)


def get_sentiment_comparison(company_name: str, num_articles: int = 10, 
                             articles: Optional[List[ArticleType]] = None, 
                             keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get sentiment comparison analysis for articles about a company.
    If articles are provided, they will be sent to the API instead of fetching new ones.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to analyze
        articles: Optional list of pre-fetched articles
        keywords: Optional list of keywords to refine the search
        
    Returns:
        Dictionary with sentiment comparison results
    """
    payload = {
        "company_name": company_name, 
        "num_articles": num_articles
    }
    
    # Include articles in payload if provided
    if articles:
        payload["articles"] = articles
    
    # Include keywords in payload if provided
    if keywords:
        payload["keywords"] = keywords
        
    return call_api_endpoint("compare-sentiment", payload)


def get_final_analysis(company_name: str, num_articles: int = 10, 
                       articles: Optional[List[ArticleType]] = None, 
                       keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate comprehensive sentiment analysis for a company.
    If articles are provided, they will be sent to the API instead of fetching new ones.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to analyze
        articles: Optional list of pre-fetched articles
        keywords: Optional list of keywords to refine the search
        
    Returns:
        Dictionary with the final analysis report
    """
    payload = _build_api_payload(company_name, num_articles, articles, keywords)
    return call_api_endpoint("final-analysis", payload)


def _build_api_payload(company_name: str, num_articles: int, 
                      articles: Optional[List[ArticleType]] = None, 
                      keywords: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build a payload for API requests with common parameters.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to analyze
        articles: Optional list of pre-fetched articles
        keywords: Optional list of keywords to refine the search
        
    Returns:
        Dictionary with payload parameters
    """
    payload = {
        "company_name": company_name, 
        "num_articles": num_articles
    }
    
    if articles:
        payload["articles"] = articles
    
    if keywords:
        payload["keywords"] = keywords
        
    return payload


###################
# DATA PROCESSING FUNCTIONS
###################

def ensure_articles_loaded(company_name: str, num_articles: int, 
                          keywords: Optional[List[str]] = None) -> List[ArticleType]:
    """
    Ensure articles are loaded, either from session state or by fetching them.
    
    Args:
        company_name: Name of the company
        num_articles: Number of articles to analyze
        keywords: Optional list of keywords to refine the search
        
    Returns:
        List of articles
    """
    # If company name is empty, return empty list
    if not company_name:
        return []
    
    # Standardize keywords to None if empty list
    if keywords is None or len(keywords) == 0:
        keywords = None
    
    # Get current state for comparison
    stored_keywords = getattr(st.session_state, 'last_keywords', None)
    if stored_keywords is not None and len(stored_keywords) == 0:
        stored_keywords = None
        
    # Check if we already have these articles in session state
    if (st.session_state.articles is not None and 
        len(st.session_state.articles) >= num_articles and 
        st.session_state.last_company == company_name and
        st.session_state.last_article_count == num_articles and
        stored_keywords == keywords):
        
        articles = st.session_state.articles
        st.info(f"Using {len(articles)} articles from previous extraction")
    else:
        # Fetch fresh articles
        keywords_msg = f" with keywords: {', '.join(keywords)}" if keywords else ""
        st.info(f"Fetching {num_articles} articles for {company_name}{keywords_msg}...")
        articles = get_news(company_name, num_articles, keywords)
        
        # Update session state
        st.session_state.articles = articles
        st.session_state.last_company = company_name
        st.session_state.last_article_count = num_articles
        st.session_state.last_keywords = keywords
        
        # Reset tab-specific data when articles change
        st.session_state.tab1_articles = None 
        st.session_state.tab1_sentiment_counts = None
        st.session_state.tab1_final_report = None
        
        st.success(f"Fetched {len(articles)} new articles")
    
    return articles


def get_sentiment_color(sentiment: str) -> str:
    """
    Get the color code for a sentiment value.
    
    Args:
        sentiment: Sentiment value (positive, negative, neutral)
        
    Returns:
        Hex color code
    """
    return SENTIMENT_COLORS.get(sentiment.lower(), "#b0bce5")


def sort_articles_by_date(articles: List[ArticleType]) -> List[ArticleType]:
    """
    Sort articles by date with unknown dates at the end.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Sorted list of articles
    """
    def get_sort_key(article):
        date = article.get('date', 'Unknown')
        if date == 'Unknown':
            # Return a tuple with high value to sort Unknown dates at the end
            return (1, '0000-00-00')
        else:
            # Return a tuple with 0 first to sort known dates before Unknown
            # Use the date string as second element for sorting
            return (0, date)
    
    return sorted(articles, key=get_sort_key, reverse=True)


def calculate_sentiment_distribution(articles: List[ArticleType]) -> SentimentDistribution:
    """
    Calculate the distribution of sentiments across articles.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with counts for each sentiment
    """
    return {
        "Positive": sum(1 for a in articles if a.get('sentiment', '').lower() == 'positive'),
        "Neutral": sum(1 for a in articles if a.get('sentiment', '').lower() == 'neutral'),
        "Negative": sum(1 for a in articles if a.get('sentiment', '').lower() == 'negative')
    }


###################
# VISUALIZATION FUNCTIONS
###################

def display_audio_player(audio_content: Optional[str]) -> None:
    """
    Display an audio player if audio content is available.
    
    Args:
        audio_content: Base64-encoded audio content
    """
    if not audio_content:
        st.error("Audio not available")
        return
        
    try:
        audio_bytes = base64.b64decode(audio_content)
        st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"Error playing audio: {e}")


def create_sentiment_donut_chart(sentiment_counts: SentimentDistribution) -> px.pie:
    """
    Create a donut chart visualization of sentiment distribution.
    
    Args:
        sentiment_counts: Dictionary with sentiment counts
        
    Returns:
        Plotly pie chart figure
    """
    # Filter out zero-value sentiments
    filtered_counts = {k: v for k, v in sentiment_counts.items() if v > 0}
    
    # Create DataFrame for the chart
    sentiment_df = pd.DataFrame({
        'Sentiment': list(filtered_counts.keys()),
        'Count': list(filtered_counts.values())
    })
    
    # Create expanded DataFrame with repeated rows based on count
    expanded_df = pd.DataFrame({
        'Sentiment': np.repeat(sentiment_df['Sentiment'].values, sentiment_df['Count'].values)
    })
    
    # Create donut chart
    fig = px.pie(
        expanded_df,
        names='Sentiment',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#32CD32',  # Green
            'Neutral': '#b0bce5',   # Blue
            'Negative': '#B90E0A'   # Red
        },
        hole=0.4
    )
    
    # Configure layout
    fig.update_layout(
        legend_orientation="h",
        legend_yanchor="top",
        legend_y=-0.1,
        legend_x=0.5,
        legend_xanchor="center",
        height=350, 
        margin=dict(l=58, r=40, t=5, b=5)
    )
    
    # Configure trace display
    fig.update_traces(
        texttemplate='%{percent:.1f}%',
        hovertemplate='%{label}: %{count} articles (%{percent:.1f}%)<extra></extra>',
        textinfo='percent',
        textposition='inside',
    )
    
    return fig


def create_topic_waffle(topic_df: pd.DataFrame) -> plt.Figure:
    """
    Create a waffle chart visualization of article topics.
    
    Args:
        topic_df: DataFrame with topic counts
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(
        FigureClass=Waffle, 
        rows=3,
        values=topic_df['count'].tolist(),
        labels=topic_df['topic'].tolist(),
        legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'prop': {'size': 5}},
        figsize=(4, 2)
    )
    
    return fig


def display_keywords(keywords: List[str]) -> None:
    """
    Display article keywords with styled badges.
    
    Args:
        keywords: List of keyword strings
    """
    if not keywords:
        return
        
    keyword_html = ""
    for kw in keywords:
        keyword_html += f'<span style="background-color:#E0E0E0; color:#000000; padding:5px; border-radius:10px; margin-right:5px; margin-bottom:5px; display:inline-block;">{kw}</span>'
    
    st.markdown("**All Keywords:**", unsafe_allow_html=True)
    st.markdown(keyword_html, unsafe_allow_html=True)


def display_article_card(article: ArticleType) -> None:
    """
    Display a single article as a styled card.
    
    Args:
        article: Article data dictionary
    """
    with st.container():
        # Add sentiment badge
        sentiment = article.get('sentiment', 'neutral')
        sentiment_color = get_sentiment_color(sentiment)
        st.markdown(f"""
        <div style="display: inline-block; padding: 5px 10px; background-color:{sentiment_color}; 
        color:white; border-radius:20px; font-weight:bold; margin-bottom:10px;">
        {sentiment.upper()}
        </div>
        """, unsafe_allow_html=True)
        
        # Display article details
        st.subheader(article['title'])
        st.write(f"**Date:** {article.get('date', 'Unknown')}")
        st.write(f"**Source:** {article.get('source', 'Unknown')}")
        st.write(f"**Industry:** {article.get('industry', 'Unknown')}")
        st.write(f"**Author:** {article.get('author', 'Unknown')}")
        st.write(f"**Read Time:** {article.get('read_time', 'Unknown')}")
        st.write(f"**Relevance:** {article.get('relevance', 'Unknown')}")
        st.write(f"**Summary:** {article.get('summary', 'Unknown')}")
        st.write(f"**Original Link:** [View Article]({article['url']})")
        
        # Display keywords
        display_keywords(article.get('keywords', []))
        
        st.markdown("---")


###################
# TAB RENDERING FUNCTIONS
###################

def render_news_sentiment_tab() -> None:
    """Render the News & Sentiment tab content."""
    company_name = st.session_state.company_name
    num_articles = st.session_state.num_articles
    keywords = st.session_state.keywords
    
    # Initialize session state variables for tab 1 if not already present
    if 'tab1_articles' not in st.session_state:
        st.session_state.tab1_articles = None
    if 'tab1_sentiment_counts' not in st.session_state:
        st.session_state.tab1_sentiment_counts = None
    if 'tab1_final_report' not in st.session_state:
        st.session_state.tab1_final_report = None
    
    analyze_button = st.button("Extract & Analyze News")
    
    # Check if we already have analysis results in session state or button was pressed
    if analyze_button or st.session_state.tab1_articles is not None:
        
        # Only fetch new data if the button was pressed or no previous data exists
        if analyze_button or st.session_state.tab1_articles is None:
            with st.spinner(f"Extracting and analyzing news for {company_name}..."):
                try:
                    # Use ensure_articles_loaded instead of direct API call
                    articles = ensure_articles_loaded(company_name, num_articles, keywords)
                    
                    if not articles:
                        st.error(f"No articles found for {company_name}")
                        return
                        
                    # Store in tab1-specific session state
                    st.session_state.tab1_articles = articles
                    
                    st.success(f"Extraction completed! Found {len(articles)} articles. Analyzing...")
                    
                    # Calculate sentiment distribution
                    sentiment_counts = calculate_sentiment_distribution(articles)
                    st.session_state.tab1_sentiment_counts = sentiment_counts
                    
                    # Generate final analysis report
                    try:
                        final_report = get_final_analysis(company_name, num_articles, articles, keywords)
                        st.session_state.tab1_final_report = final_report
                    except Exception as e:
                        st.error(f"Error generating comprehensive report: {str(e)}")
                        st.session_state.tab1_final_report = {"Final Sentiment Analysis": "Unable to generate comprehensive analysis."}
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        # Display results if available in session state
        if st.session_state.tab1_articles is not None:
            _display_news_sentiment_results()


def _display_news_sentiment_results() -> None:
    """Display news sentiment analysis results from session state."""
    articles = st.session_state.tab1_articles
    sentiment_counts = st.session_state.tab1_sentiment_counts
    final_report = st.session_state.tab1_final_report
    company_name = st.session_state.company_name
    
    # Create layout for visualization and summary
    left_col, right_col = st.columns([0.65, 0.35])
    
    # Left column: Final Analysis, Audio, and Download button
    with left_col:
        st.info(final_report.get("Final Sentiment Analysis", "Analysis not available"))
        
        # Audio player
        display_audio_player(final_report.get("AudioContent"))
        
        # Download JSON button
        json_str = json.dumps(final_report, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"{company_name}_analysis_report.json",
            mime="application/json",
            key="download_json_tab1",
        )
    
    # Sort articles by date
    sorted_articles = sort_articles_by_date(articles)
    
    # Right column: Donut chart
    with right_col:
        # Create and display donut chart
        fig = create_sentiment_donut_chart(sentiment_counts)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add a divider after the two columns
    st.markdown("---")
    
    # Display article cards
    for article in sorted_articles:
        display_article_card(article)


def render_comparative_analysis_tab() -> None:
    """Render the Comparative Analysis tab content."""
    company_name = st.session_state.company_name
    num_articles = st.session_state.num_articles
    keywords = st.session_state.keywords
    
    st.header("Sentiment analysis across news sources")  
    compare_button = st.button("Generate analysis chart")
    
    if compare_button:
        with st.spinner("Analyzing sentiment across articles..."):
            try:
                # First ensure we have articles loaded, reusing them if possible
                articles = ensure_articles_loaded(company_name, num_articles, keywords)
                if not articles:
                    st.error(f"No articles found for {company_name}")
                    return
                    
                # Pass the already loaded articles to the sentiment comparison API
                comparison_results = get_sentiment_comparison(company_name, num_articles, articles, keywords)
                
                # Display frequency chart if available
                if "charts" in comparison_results and "sentiment_frequency_chart" in comparison_results["charts"]:
                    frequency_chart_path = comparison_results["charts"]["sentiment_frequency_chart"]
                    if os.path.exists(frequency_chart_path):
                        st.image(frequency_chart_path, use_column_width=True)
                        st.info("This visualization compares sentiment trends across different news sources.")
                    else:
                        st.warning("Sentiment frequency chart not found.")
                
            except Exception as e:
                st.error(f"Error generating comparison: {str(e)}")


def render_detailed_analysis_tab() -> None:
    """Render the Final Analysis Report tab with a waffle chart showing topic distribution."""
    company_name = st.session_state.company_name
    num_articles = st.session_state.num_articles
    keywords = st.session_state.keywords
    
    st.header("Visualization for topic distribution")
    report_button = st.button("Generate Chart")
    
    if report_button:
        with st.spinner("Generating detailed analysis..."):
            try:
                # First ensure we have articles loaded, reusing them if possible
                articles = ensure_articles_loaded(company_name, num_articles, keywords)
                if not articles:
                    st.error(f"No articles found for {company_name}")
                    return
                    
                # Get comparison results using the same articles
                comparison_results = get_sentiment_comparison(company_name, num_articles, articles, keywords)
                
                # Extract articles from the comparison results
                articles = comparison_results.get("articles", [])
                
                # Count main topics and prepare data for visualization
                topic_counts = _calculate_topic_distribution(articles)
                
                # Create dataframe for visualizations
                topic_df = pd.DataFrame({
                    'topic': list(topic_counts.keys()),
                    'count': list(topic_counts.values())
                })
                
                # Ensure count is numeric
                topic_df['count'] = pd.to_numeric(topic_df['count'])
                
                # Visualize with waffle chart
                fig = create_topic_waffle(topic_df)
                st.pyplot(fig)

                st.info("The waffle chart above shows the relative distribution of key topics discussed in the articles.")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")


def _calculate_topic_distribution(articles: List[ArticleType]) -> Dict[str, int]:
    """
    Calculate the distribution of topics across articles.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with counts for each topic
    """
    topic_counts = {}
    for article in articles:
        main_topic = article.get('main_topic', 'Uncategorized')
        topic_counts[main_topic] = topic_counts.get(main_topic, 0) + 1
    
    return topic_counts


###################
# STATE MANAGEMENT FUNCTIONS
###################

def init_session_state() -> None:
    """Initialize session state variables."""
    # Define all session state variables with default values
    session_vars = {
        # Main article cache
        'articles': None,
        
        # Last parameters used for fetching
        'last_company': "",
        'last_article_count': 0,
        'last_keywords': [],
        
        # Current parameters (will be set in UI)
        'company_name': "",
        'num_articles': 10,
        'keywords': [],
        
        # Tab 1 specific session state variables
        'tab1_articles': None,
        'tab1_sentiment_counts': None,
        'tab1_final_report': None,
    }
    
    # Initialize all session state variables if not already present
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def process_keywords_input(keywords_input: str) -> List[str]:
    """
    Process raw keywords input string into a list of keywords.
    
    Args:
        keywords_input: Comma-separated string of keywords
        
    Returns:
        List of processed keywords (max 3)
    """
    if not keywords_input:
        return []
        
    # Split by comma, strip whitespace, and limit to 3
    keywords_list = [k.strip() for k in keywords_input.split(',') if k.strip()][:3]
    return keywords_list


def handle_input_changes() -> None:
    """Handle changes to input parameters and reset state if needed."""
    # Get prior values for comparison
    prior_company = st.session_state.last_company
    prior_article_count = st.session_state.last_article_count
    prior_keywords = st.session_state.last_keywords if hasattr(st.session_state, 'last_keywords') else []
    
    # Standardize keywords for comparison
    current_keywords = None if not st.session_state.keywords else st.session_state.keywords
    
    # Handle the case where prior_keywords is None
    prior_keywords_empty = prior_keywords is None or len(prior_keywords) == 0
    if prior_keywords_empty:
        prior_keywords = None
    
    # Reset session state if inputs change
    if (st.session_state.company_name != prior_company or 
        st.session_state.num_articles != prior_article_count or
        current_keywords != prior_keywords):
        
        # Clear all article-related session state
        st.session_state.articles = None
        st.session_state.tab1_articles = None
        st.session_state.tab1_sentiment_counts = None
        st.session_state.tab1_final_report = None


###################
# MAIN APPLICATION
###################

def main() -> None:
    """Main application entry point."""
    # Set environment variable to prevent tokenizers warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    st.set_page_config(
        page_title="NewsPulse",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    st.title("📊 NewsPulse: AI-Driven Sentiment & Voice Reports")
    
    # Common inputs in sidebar
    st.session_state.company_name = st.sidebar.text_input("Enter Company Name", placeholder="Company Name", value="")
    st.session_state.num_articles = st.sidebar.number_input("Number of Articles", 
                                                          min_value=1, 
                                                          max_value=30,
                                                          value=10)
    
    # Add keywords input with a 3 keyword limit
    keywords_help = "Enter up to 3 keywords separated by commas (e.g., financial, merger, profit)"
    keywords_input = st.sidebar.text_input("Search Keywords (Optional)", 
                                          placeholder="keyword1, keyword2, keyword3",
                                          help=keywords_help)
    
    # Process keywords input
    st.session_state.keywords = process_keywords_input(keywords_input)
    
    # If more than 3 keywords were provided, show a message
    if keywords_input and len(keywords_input.split(',')) > 3:
        st.sidebar.warning("Only the first 3 keywords will be used.")
    
    # Handle changes to input parameters
    handle_input_changes()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["News & Sentiment", "Comparative Analysis", "Detailed Analysis"])
    
    # Render tab content
    with tab1:
        render_news_sentiment_tab()
    with tab2:
        render_comparative_analysis_tab()
    with tab3:
        render_detailed_analysis_tab()


if __name__ == "__main__":
    main()