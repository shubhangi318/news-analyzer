"""
Sentiment analysis module for financial news articles.
Provides functionality to analyze sentiment using hybrid approach with VADER and FinBERT.
"""
from typing import Dict, List, Any, Optional, Tuple
import os
import re
import statistics
from urllib.parse import urlparse
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('sentiment_analysis.log')  # Also save to file
    ]
)

# Initialize NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Load FinBERT model once at module import
print("Loading FinBERT model...")
finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)
print("FinBERT model loaded successfully")

# Constants
MAX_TEXT_LENGTH = 512
FINBERT_WEIGHT = 0.6
VADER_WEIGHT = 0.4

# Term lists for contextual analysis
BUSINESS_NEGATIVE_TERMS = [
    'bankruptcy', 'bankrupt', 'arbitration', 'lawsuit', 'legal battle', 
    'financial struggle', 'debt', 'litigation', 'layoffs', 'downsizing',
    'cautious', 'unsustainable', 'decline', 'struggles', 'struggling',
    'failed'
]

REPUTATION_NEGATIVE_TERMS = [
    'criticizes', 'accuses', 'exploiting', 'fraud', 'controversy',
    'backlash', 'outrage', 'complaint', 'protest', 'scandal', 'investigation',
    'attack', 'slam', 'blame', 'condemn', 'dispute', 'anger',
    'skeptical', 'cautious about', 'not viable'
]

REPUTATION_POSITIVE_TERMS = [
    'launches', 'improves', 'enhances', 'expands',
    'success', 'achievement', 'innovation', 'growth', 'award',
    'breakthrough', 'milestone', 'leadership', 'significant growth'
]


def preprocess_article(text: str) -> str:
    """
    Preprocess article by removing irrelevant content like ads, 
    navigation elements, and standardizing whitespace.
    
    Args:
        text: Raw article text
        
    Returns:
        Processed article text
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove extra whitespace, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common article footers
    footers = [
        "This article was produced by", "Follow us on", "Copyright ©",
        "All rights reserved", "Terms of Service", "Privacy Policy"
    ]
    for footer in footers:
        if footer in text:
            text = text.split(footer)[0]
    
    # Remove common ad or metadata phrases
    ad_patterns = [
        r'ADVERTISEMENT', r'SPONSORED CONTENT', r'Read more:',
        r'Click here to subscribe', r'Share this article', r'Read more at', r'Also read'
    ]
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def split_into_paragraphs(text: str, max_length: int = MAX_TEXT_LENGTH) -> List[str]:
    """
    Split article text into meaningful paragraphs for analysis.
    
    Args:
        text: Article text to split
        max_length: Maximum length of each paragraph
        
    Returns:
        List of paragraph strings
    """
    # First try to split by newlines (natural paragraph breaks)
    paragraphs = [p for p in text.split('\n') if p.strip()]
    
    # If no natural paragraphs or very few, split by sentences
    if len(paragraphs) <= 1:
        sentences = nltk.sent_tokenize(text)
        paragraphs = []
        current_paragraph = ""
        
        for sentence in sentences:
            if len(current_paragraph) + len(sentence) > max_length:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = sentence
            else:
                if current_paragraph:
                    current_paragraph += " " + sentence
                else:
                    current_paragraph = sentence
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
    
    # Handle very long paragraphs that exceed model limits
    result_paragraphs = []
    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            # Split by sentence and recombine to keep under limit
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_length:
                    result_paragraphs.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            if current_chunk:
                result_paragraphs.append(current_chunk)
        else:
            result_paragraphs.append(paragraph)
    
    return result_paragraphs


def get_finbert_sentiment(text: str) -> Dict[str, float]:
    """
    Use FinBERT to get financial sentiment scores.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Truncate if needed
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    # Get sentiment using pre-loaded pipeline
    result = finbert_pipeline(text)[0]
    
    # Convert results to the expected format
    sentiment_scores = {
        item['label']: item['score'] for item in result
    }
    
    # Ensure we have all three keys
    for label in ['positive', 'negative', 'neutral']:
        if label not in sentiment_scores:
            sentiment_scores[label] = 0.0
    
    return sentiment_scores


def get_vader_sentiment(text: str) -> Dict[str, float]:
    """
    Get VADER sentiment scores for text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment


def _analyze_title_sentiment(title: str) -> float:
    """
    Analyze sentiment of article title.
    
    Args:
        title: Article title
        
    Returns:
        Title sentiment score
    """
    if not title:
        return 0.0
    
    title_vader = get_vader_sentiment(title)
    title_finbert = get_finbert_sentiment(title)
    
    # Calculate title sentiment score (stronger weight on title)
    title_sentiment_score = (
        (title_finbert.get("positive", 0) - title_finbert.get("negative", 0)) * 0.6 +
        (title_vader['compound']) * 0.4
    )
    
    return title_sentiment_score


def _analyze_paragraphs_with_vader(paragraphs: List[str]) -> Tuple[float, float, float, float]:
    """
    Analyze paragraphs with VADER sentiment analyzer.
    
    Args:
        paragraphs: List of text paragraphs
        
    Returns:
        Tuple of (compound, positive, negative, neutral) scores
    """
    # Analyze each paragraph with VADER
    vader_paragraph_scores = []
    for paragraph in paragraphs:
        if len(paragraph.strip()) > 20:  # Only analyze substantial paragraphs
            vader_score = get_vader_sentiment(paragraph)
            vader_paragraph_scores.append(vader_score)
    
    # Calculate aggregate VADER scores
    if vader_paragraph_scores:
        vader_compound = sum(score['compound'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_pos = sum(score['pos'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_neg = sum(score['neg'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
        vader_neu = sum(score['neu'] for score in vader_paragraph_scores) / len(vader_paragraph_scores)
    else:
        # Fallback if no substantial paragraphs
        vader_full = get_vader_sentiment(' '.join(paragraphs))
        vader_compound = vader_full['compound']
        vader_pos = vader_full['pos']
        vader_neg = vader_full['neg']
        vader_neu = vader_full['neu']
    
    return vader_compound, vader_pos, vader_neg, vader_neu


def _detect_sentiment_terms(text: str, title: Optional[str] = None) -> Tuple[bool, bool]:
    """
    Detect presence of specific sentiment terms in text.
    
    Args:
        text: Text to analyze
        title: Optional article title
        
    Returns:
        Tuple of (negative_term_found, positive_term_found)
    """
    # Check for negative business terms
    negative_term_found = False
    for term in BUSINESS_NEGATIVE_TERMS:
        if term in text.lower():
            negative_term_found = True
            print(f"Negative financial term found: {term}")
            break
    
    # Check reputation negative terms
    if not negative_term_found:
        for term in REPUTATION_NEGATIVE_TERMS:
            if term in text.lower() or (title and term in title.lower()):
                negative_term_found = True
                print(f"Negative reputation term found: {term}")
                break
    
    # Check for positive terms
    positive_term_found = False
    for term in REPUTATION_POSITIVE_TERMS:
        # Give more weight if in title
        if title and term in title.lower():
            positive_term_found = True
            print(f"Positive term found in title: {term}")
            break
        elif term in text.lower():
            # Count occurrences in text
            occurrences = len(re.findall(fr'\b{term}\b', text.lower()))
            if occurrences >= 2:  # Multiple occurrences strengthen positive signal
                positive_term_found = True
                print(f"Multiple positive terms found: {term} ({occurrences} times)")
                break
    
    return negative_term_found, positive_term_found


def analyze_sentiment(text: str, title: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive sentiment analysis with hybrid approach.
    
    Args:
        text: Article text to analyze
        title: Optional article title
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Preprocess the article
    processed_text = preprocess_article(text)
    
    # Analyze title if provided
    title_sentiment_score = _analyze_title_sentiment(title) if title else 0
    
    # Analyze with FinBERT (financial domain model)
    finbert_scores = get_finbert_sentiment(processed_text)
    
    # Get the predominant FinBERT sentiment
    finbert_sentiment = max(finbert_scores.items(), key=lambda x: x[1])[0]
    finbert_confidence = max(finbert_scores.values())
    
    # Break into paragraphs
    paragraphs = split_into_paragraphs(processed_text)
    
    # Analyze paragraphs with VADER
    vader_compound, vader_pos, vader_neg, vader_neu = _analyze_paragraphs_with_vader(paragraphs)
    
    # Calculate weighted sentiment scores for each category
    weighted_scores = {
        "positive": (finbert_scores.get("positive", 0) * FINBERT_WEIGHT) + 
                   (vader_pos * VADER_WEIGHT),
        "negative": (finbert_scores.get("negative", 0) * FINBERT_WEIGHT) + 
                   (vader_neg * VADER_WEIGHT),
        "neutral": (finbert_scores.get("neutral", 0) * FINBERT_WEIGHT) + 
                  (vader_neu * VADER_WEIGHT)
    }
    
    # Determine initial sentiment
    final_sentiment = max(weighted_scores.items(), key=lambda x: x[1])[0]
    
    # Detect sentiment terms
    negative_term_found, positive_term_found = _detect_sentiment_terms(processed_text, title)
    
    # Calculate initial sentiment score
    finbert_factor = (finbert_scores.get("positive", 0) - finbert_scores.get("negative", 0))
    vader_factor = vader_compound
    sentiment_score = (finbert_factor * FINBERT_WEIGHT) + (vader_factor * VADER_WEIGHT)
    
    # Apply sentiment adjustments based on detected terms
    if negative_term_found and not positive_term_found:
        if weighted_scores["positive"] < 0.6:
            final_sentiment = "negative"
            # Ensure polarity is negative
            sentiment_score = -abs(sentiment_score)
    elif positive_term_found and weighted_scores["negative"] < 0.6:
        final_sentiment = "positive"
        # Ensure polarity is positive
        sentiment_score = abs(sentiment_score)
    
    # Apply title influence if significantly different from body
    if title and abs(title_sentiment_score) > 0.5:
        # If title has strong sentiment different from current analysis
        if (title_sentiment_score > 0.5 and final_sentiment != "positive"):
            if weighted_scores["negative"] < 0.7:  # Not overwhelmingly negative
                final_sentiment = "positive"
                print(f"Sentiment adjusted to positive based on strong positive title")
        elif (title_sentiment_score < -0.5 and final_sentiment != "negative"):
            if weighted_scores["positive"] < 0.7:  # Not overwhelmingly positive
                final_sentiment = "negative"
                print(f"Sentiment adjusted to negative based on strong negative title")
    
    # Scale down polarity for neutral sentiment
    if final_sentiment == "neutral":
        sentiment_score = sentiment_score * 0.25  # Dampen the polarity

    # Return comprehensive results
    return {
        "sentiment": final_sentiment,
        "polarity": sentiment_score,
        "vader_compound": vader_compound,
        "vader_pos": vader_pos,
        "vader_neg": vader_neg,
        "vader_neu": vader_neu,
        "finbert_sentiment": finbert_sentiment,
    }


def compare_sentiments(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform comparative analysis across multiple articles.
    
    Args:   
        articles: List of article dictionaries with sentiment data
        
    Returns:
        Dictionary with comparative analysis results
    """
    if not articles:
        return {
            "average_sentiment": "N/A",
            "charts": {}
        }
    
    # Extract sentiment data
    polarities = [article.get('polarity', 0) for article in articles]
    sentiments = [article.get('sentiment', 'neutral') for article in articles]
    
    # Calculate statistics
    avg_polarity = statistics.mean(polarities) if polarities else 0
    
    # Determine overall sentiment
    if avg_polarity > 0.1:
        overall_sentiment = "positive"
    elif avg_polarity < -0.1:
        overall_sentiment = "negative"
    else:
        overall_sentiment = "neutral"
    
    # Count sentiment distribution
    sentiment_count = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative")
    }
    
    # Generate charts
    charts = {}
    frequency_chart_path = generate_sentiment_frequency_chart(articles)
    if frequency_chart_path:
        charts["sentiment_frequency_chart"] = frequency_chart_path
    
    return {
        "average_sentiment": overall_sentiment,
        "average_polarity": avg_polarity,
        "sentiment_distribution": sentiment_count,
        "charts": charts
    }


def _process_domain(domain: str) -> str:
    """
    Process domain name for cleaner display.
    
    Args:
        domain: Domain name to process
        
    Returns:
        Processed domain name
    """
    parts = domain.split('.')
    return parts[1] if len(parts) > 2 else parts[0]


def _extract_source_from_article(article: Dict[str, Any]) -> str:
    """
    Extract source name from article data.
    
    Args:
        article: Article data dictionary
        
    Returns:
        Source name
    """
    source = article.get('source', 'Unknown')
    
    # Extract the domain if full URL is provided
    if source.startswith('http'):
        source = urlparse(source).netloc
    
    # Use domain from URL if source not available
    if source == 'Unknown' and 'url' in article:
        source = urlparse(article['url']).netloc
        
    return source


def generate_sentiment_frequency_chart(articles: List[Dict[str, Any]], max_sources: int = 15) -> Optional[str]:
    """
    Create a diverging chart showing the frequency of positive, negative, and neutral
    sentiments for each news source.
    
    Args:
        articles: List of article dictionaries
        max_sources: Maximum number of sources to display
        
    Returns:
        Path to the generated chart image or None if chart couldn't be generated
    """
    if not articles:
        print("No articles provided for frequency chart.")
        return None
    
    # Extract source and sentiment data
    data = []
    for article in articles:
        source = _extract_source_from_article(article)
        sentiment = article.get('sentiment', 'neutral')
        data.append({
            'source': source,
            'sentiment': sentiment
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Group by source and sentiment, count occurrences
    sentiment_counts = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
    
    # Ensure all sentiment columns exist
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0
    
    # Sort by total number of articles per source (descending)
    sentiment_counts['total'] = sentiment_counts.sum(axis=1)
    sentiment_counts = sentiment_counts.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Limit to top sources
    if len(sentiment_counts) > max_sources:
        sentiment_counts = sentiment_counts.iloc[:max_sources]
    
    # Process domain names for cleaner display
    processed_domains = [_process_domain(domain) for domain in sentiment_counts.index]
    
    return _create_frequency_chart(sentiment_counts, processed_domains)


def _create_frequency_chart(sentiment_counts: pd.DataFrame, processed_domains: List[str]) -> Optional[str]:
    """
    Create the actual frequency chart visualization.
    
    Args:
        sentiment_counts: DataFrame with sentiment counts by source
        processed_domains: List of processed domain names
        
    Returns:
        Path to the generated chart image or None if chart couldn't be generated
    """
    # Create figure with appropriate size
    fig_height = max(5, len(sentiment_counts) * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Calculate maximum count for any sentiment type
    max_count = sentiment_counts.values.max()
    
    # Parameters for chart layout
    scale_factor = 3.0  # Compress the x-axis scale
    midpoint = max_count / scale_factor
    
    # Set up positions for y-axis
    y_pos = np.arange(len(sentiment_counts.index))
    
    # Create empty bars for the legend
    ax.barh(-1, 0, color='#32CD32', label='Positive')
    ax.barh(-1, 0, color='#b0bce5', label='Neutral')
    ax.barh(-1, 0, color='#B90E0A', label='Negative')
    
    # For each source, plot the sentiments
    for i, source in enumerate(sentiment_counts.index):
        pos_count = sentiment_counts.loc[source, 'positive']
        neg_count = sentiment_counts.loc[source, 'negative']
        neu_count = sentiment_counts.loc[source, 'neutral']
        
        # Calculate starting positions
        neutral_start = midpoint - (neu_count / (scale_factor * 2))
        negative_start = neutral_start - (neg_count / scale_factor)
        positive_start = neutral_start + (neu_count / scale_factor)
        
        # Plot the bars with labels
        if neg_count > 0:
            ax.barh(i, neg_count/scale_factor, left=negative_start, color='#B90E0A', alpha=0.8)
            ax.text(negative_start + (neg_count/(scale_factor*2)), i, str(int(neg_count)), 
                   ha='center', va='center', color='white', fontweight='bold')
        
        if neu_count > 0:
            ax.barh(i, neu_count/scale_factor, left=neutral_start, color='#b0bce5', alpha=0.8)
            ax.text(neutral_start + (neu_count/(scale_factor*2)), i, str(int(neu_count)), 
                   ha='center', va='center', color='black', fontweight='bold')
        
        if pos_count > 0:
            ax.barh(i, pos_count/scale_factor, left=positive_start, color='#32CD32', alpha=0.8)
            ax.text(positive_start + (pos_count/(scale_factor*2)), i, str(int(pos_count)), 
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Add the midpoint line
    ax.axvline(midpoint, color='black', alpha=0.2, ls='-', zorder=10, linewidth=1.0)
    
    # Set axes limits
    max_range = max(sentiment_counts['positive'].max(), 
                    sentiment_counts['negative'].max(), 
                    sentiment_counts['neutral'].max()) / scale_factor
    
    x_min = max(0, midpoint - max_range * 1.3)
    x_max = midpoint + max_range * 1.3
    ax.set_xlim(x_min, x_max)
    
    # Set up y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(processed_domains)
    
    # Remove x-axis labels and configure appearance
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_title('Number of Positive, Negative, and Neutral Articles by Source', fontsize=12, pad=10)
    ax.legend(loc='upper right')
    
    # Remove excess elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Reduce whitespace
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.05)
    
    # Save figure
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))# )up from sentiment_analysis
    
    frontend_dir = os.path.join(project_root, "frontend")
    output_path = os.path.join(frontend_dir, "sentiment_frequency_chart.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Frequency chart saved to {output_path}")
    return output_path
