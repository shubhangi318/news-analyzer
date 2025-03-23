"""
FastAPI backend for company news analysis service.
Provides endpoints for news extraction, sentiment analysis, and report generation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from gtts import gTTS
import time
import logging
from io import BytesIO
import base64
import json
import re

from news_extract_metadata.news_extractor import extract_company_news
from sentiment_analysis.sentiment import analyze_sentiment, compare_sentiments
from sentiment_analysis.keyword_embeddings import find_common_topics, get_article_specific_topics
from openai import OpenAI
from model.pydantic_model import CompanyRequest
from utils.extraction_utils import rate_limited_openai_call

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()

# Initialize FastAPI app
app = FastAPI(title="Company News Analysis API")


# Utility functions
def generate_hindi_audio_content(text: str) -> Optional[str]:
    """
    Generate Hindi audio and return base64 encoded content.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Base64 encoded audio content or None if generation fails
    """
    try:
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return base64.b64encode(audio_bytes.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating audio content: {e}")
        return None

def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi using OpenAI.
    
    Args:
        text: English text to translate
        
    Returns:
        Hindi translated text or original text if translation fails
    """
    try:
        # Use rate limited OpenAI call
        response = rate_limited_openai_call(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the following text from English to Hindi."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error translating to Hindi: {e}")
        return text

def generate_article_comparisons(articles: List[Dict[str, Any]], company_name: str) -> List[Dict[str, Any]]:
    """
    Generate in-depth comparisons between articles using OpenAI.
    
    Args:
        articles: List of article dictionaries
        company_name: Name of the company for context
    
    Returns:
        List of comparison dictionaries with "Comparison" and "Impact" keys
    """
    if len(articles) < 2:
        return []

    try:
        # Prepare article summaries for the prompt
        article_summaries = []
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            summary = article.get('summary', 'No summary available')
            sentiment = article.get('sentiment', 'unknown')
            article_summaries.append(f"Article {i}: '{title}' - {summary} (Sentiment: {sentiment})")

        all_summaries = "\n\n".join(article_summaries)

        prompt = f"""
        You are an experienced financial journalist specializing in sentiment analysis of news articles. Compare the following article summaries about {company_name}, 
        focusing specifically on differences in **tone, sentiment, and overall narrative**. Ensure comparisons highlight how sentiment varies across articles 
        (e.g., positive vs. negative framing, optimistic vs. skeptical outlook, risk-emphasizing vs. opportunity-driven perspectives). You can go beyond these examples as well.

        {all_summaries}
        ### **Output Guidelines**  

        - **Generate 2-3 comparison statements** that contrast sentiment differences between articles.  
        - Ensure **each comparison references article numbers** (e.g., "Article 1..., while Article 2...").  
        - Use **varied sentence structures** to avoid repetitive phrasing.  

        - **For each comparison, provide an impact analysis** explaining how these sentiment differences may influence:  
        - **Investor sentiment** (e.g., confidence, risk perception).  
        - **Stakeholder decisions** (e.g., business partnerships, customer trust).  
        - **Market perception** (e.g., brand reputation, competitive positioning).  
        - Ensure **each impact analysis is nuanced** and not generic.  

        ### **Structured JSON Output (Schema Enforced)**  

        Use the following **JSON format** to structure the response:  

        ```json
        {{
        "type": "array",
        "items": {{
            "type": "object",
            "properties": {{
            "Comparison": {{
                "type": "string",
                "description": "A concise comparison of sentiment differences between two or more articles."
            }},
            "Impact": {{
                "type": "string",
                "description": "Analysis of how the sentiment differences might influence investor sentiment, stakeholder decisions, or market perception."
            }}
            }},
            "required": ["Comparison", "Impact"]
        }}
        }}
        """

        # Use rate limited OpenAI call
        response = rate_limited_openai_call(
            client.chat.completions.create,
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing article comparisons."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content.strip()
        json_match = re.search(r'(\[.*\])', result, re.DOTALL)
        
        if json_match:
            return json.loads(json_match.group(1))
        else:
            logger.warning("Failed to parse OpenAI comparison response as JSON")
            return []

    except Exception as e:
        logger.error(f"Error generating article comparisons with OpenAI: {e}")
        return []

def generate_comprehensive_summary(company_name: str, articles: List[Dict[str, Any]], 
                                 sentiment_counts: Dict[str, int]) -> str:
    """
    Generate a comprehensive summary of all articles using OpenAI.
    
    Args:
        company_name: Name of the company
        articles: List of article dictionaries
        sentiment_counts: Dictionary with sentiment distribution counts
        
    Returns:
        String containing the comprehensive summary
    """
    try:
        # Prepare article data for the prompt
        article_data = []
        for i, article in enumerate(articles, 1):
            article_data.append(
                f"Article {i}:\n"
                f"Title: {article.get('title', 'No title')}\n"
                f"Sentiment: {article.get('sentiment', 'neutral')}\n"
                f"Keywords: {', '.join(article.get('keywords', []))}\n"
                f"Summary: {article.get('summary', 'No summary available')}"
            )

        combined_data = "\n\n".join(article_data)

        # Calculate overall sentiment distribution percentages
        total_articles = sum(sentiment_counts.values())
        sentiment_percentages = {
            key: (count / total_articles) * 100 if total_articles > 0 else 0 
            for key, count in sentiment_counts.items()
        }

        prompt = f"""
        As a financial analyst and news summarizer, create a comprehensive summary paragraph about {company_name} based on these {len(articles)} news articles:
        
        {combined_data}
        
        Overall Sentiment Distribution:
        - Positive: {sentiment_percentages['Positive']:.1f}% ({sentiment_counts['Positive']} articles)
        - Negative: {sentiment_percentages['Negative']:.1f}% ({sentiment_counts['Negative']} articles)
        - Neutral: {sentiment_percentages['Neutral']:.1f}% ({sentiment_counts['Neutral']} articles)
        
        Create a single comprehensive paragraph (roughly 150-200 words) that:
        1. Summarizes the key news about {company_name} from all articles
        2. Mentions major themes, developments, or events
        3. Integrates the overall sentiment landscape 
        4. Notes potential impacts or implications for the company
        5. Provides a holistic view that covers both positive and negative aspects if present
        
        The summary should be factual, balanced, and reader-friendly, suitable for an investor or general audience. 
        Do not use bullet points or numbered lists - create a flowing narrative paragraph.
        """

        # Use rate limited OpenAI call
        response = rate_limited_openai_call(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a skilled financial journalist creating comprehensive news summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating comprehensive summary with OpenAI: {e}")
        return f"Summary generation failed: {str(e)}"

def create_final_report(company_name: str, articles: List[Dict[str, Any]], 
                       comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a structured final report with comparative analysis.
    
    Args:
        company_name: Name of the company
        articles: List of article dictionaries
        comparison_results: Results from sentiment comparison
        
    Returns:
        Dictionary containing the final structured report
    """
    # Format articles for the report
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            "Title": article.get('title', 'No title available'),
            "Summary": article.get('summary', 'No summary available'),
            "Sentiment": article.get('sentiment', 'neutral').capitalize(),
            "Keywords": article.get('keywords', []),
            "Main_Topic": article.get('main_topic', 'Uncategorized'),
            "Industry": article.get('industry', 'Unknown'),
            "URL": article.get('url', '#')
        })

    # Count sentiment distribution
    sentiment_counts = {
        "Positive": sum(1 for a in articles if a.get('sentiment', '').lower() == 'positive'),
        "Negative": sum(1 for a in articles if a.get('sentiment', '').lower() == 'negative'),
        "Neutral": sum(1 for a in articles if a.get('sentiment', '').lower() == 'neutral')
    }

    # Generate coverage differences (pairwise comparisons of articles)
    coverage_differences = generate_article_comparisons(articles, company_name)

    # Process article keywords and topics
    article_keywords = [article.get('keywords', []) for article in articles]
    common_topics = find_common_topics(article_keywords)
    article_specific_topics = get_article_specific_topics(article_keywords, common_topics)

    # Create a dictionary of unique topics per article
    unique_topics = {}
    for i, topics in enumerate(article_specific_topics):
        unique_topic_list = []
        for rep, similar in topics.items():
            unique_topic_list.append(rep)
            unique_topic_list.extend(similar)

        if unique_topic_list:
            unique_topics[f"Unique Topics in Article {i+1}"] = unique_topic_list

    # Convert common topics to a flat list
    common_topics_list = []
    for rep, similar in common_topics.items():
        common_topics_list.append(rep)
        common_topics_list.extend(similar)

    # Generate comprehensive summary
    comprehensive_summary = generate_comprehensive_summary(company_name, articles, sentiment_counts)

    # Create the final report structure
    return {
        "Company": company_name,
        "Articles": formatted_articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_counts,
            "Coverage Differences": coverage_differences,
            "Keyword Overlap": {
                "Common Keywords": common_topics_list,
                **unique_topics
            }
        },
        "Final Sentiment Analysis": comprehensive_summary,
        "Audio": "Not yet generated",
        "Report_Metadata": {
            "Generated_At": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Total_Articles_Analyzed": len(articles),
            "Dominant_Sentiment": max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else "Neutral"
        }
    }

# API endpoints
@app.get("/")
async def root():
    """API root endpoint."""
    return {"message": "Welcome to the Company News Analysis API"}

@app.post("/api/extract-news", response_model=List[Dict[str, Any]])
async def get_news(request: CompanyRequest):
    """
    Extract and analyze news articles for a specified company.
    
    Args:
        request: CompanyRequest object containing company name and options
        
    Returns:
        List of processed article dictionaries with sentiment analysis
    """
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        keywords = request.keywords

        # Log search parameters including keywords if provided
        keywords_info = f" with keywords: {', '.join(keywords)}" if keywords else ""
        logger.info(f"Extracting news for: {company_name}{keywords_info}, articles: {num_articles}")
        
        # Pass keywords to extract_company_news
        articles = extract_company_news(company_name, num_articles, keywords)

        if not articles:
            logger.warning(f"No articles found for {company_name}")
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Add sentiment analysis for each article
        logger.info(f"Analyzing sentiment for {len(articles)} articles...")
        for i, article in enumerate(articles):
            if 'raw_content' in article:
                logger.info(f"Analyzing sentiment for article {i+1}/{len(articles)}")
                sentiment_result = analyze_sentiment(
                    article['raw_content'], 
                    article.get('title', '')
                )
                article['sentiment'] = sentiment_result['sentiment']

        return articles

    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error extracting news: {str(e)}")

@app.post("/api/analyze-sentiment")
async def analyze_news_sentiment(request: CompanyRequest):
    """
    Extract news articles and prepare for sentiment analysis.
    
    Args:
        request: CompanyRequest object containing company name and options
        
    Returns:
        Dictionary containing processed articles
    """
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        keywords = request.keywords

        # Pass keywords to extract_company_news
        articles = extract_company_news(company_name, num_articles, keywords)
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Ensure each article has a main_topic
        for article in articles:
            if 'main_topic' not in article:
                article['main_topic'] = "Uncategorized"

        return {"articles": articles}
    except Exception as e:
        logger.error(f"Error analyzing articles: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing articles: {str(e)}")

@app.post("/api/compare-sentiment")
async def compare_sentiment(request: CompanyRequest):
    """
    Compare sentiment across articles for a specified company.
    
    Args:
        request: CompanyRequest object containing company name and options
        
    Returns:
        Dictionary containing articles and comparison results
    """
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        keywords = request.keywords

        # Use provided articles if available, otherwise fetch new ones
        if request.articles:
            logger.info(f"Using {len(request.articles)} pre-fetched articles for comparison")
            articles = request.articles
        else:
            # Pass keywords to extract_company_news
            keywords_info = f" with keywords: {', '.join(keywords)}" if keywords else ""
            logger.info(f"Fetching new articles for {company_name}{keywords_info}")
            articles = extract_company_news(company_name, num_articles, keywords)
            
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Analyze sentiment for each article
        for article in articles:
            if 'raw_content' in article:
                logger.info(f"Analyzing article: {article.get('title', 'Unknown')}")
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)

                # Ensure we have a main_topic
                if 'main_topic' not in article:
                    article['main_topic'] = "Uncategorized"
            else:
                logger.warning(f"Missing raw_content for article: {article.get('title', 'Unknown')}")
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'vader_compound': 0,
                    'main_topic': 'Uncategorized'
                })

        # Generate comparative analysis
        comparison_results = compare_sentiments(articles)

        # Return both the articles and the comparison results
        return {
            "articles": articles,
            "charts": comparison_results.get("charts", {}),
            "average_sentiment": comparison_results.get("average_sentiment", "neutral"),
            "sentiment_distribution": comparison_results.get("sentiment_distribution", {})
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/api/final-analysis")
async def generate_final_analysis(request: CompanyRequest):
    """
    Generate a comprehensive final analysis for a company including Hindi audio summary.
    
    Args:
        request: CompanyRequest object containing company name and options
        
    Returns:
        Dictionary containing the final report with audio content
    """
    try:
        company_name = request.company_name
        num_articles = request.num_articles
        keywords = request.keywords

        # Use provided articles if available, otherwise fetch new ones
        if request.articles:
            logger.info(f"Using {len(request.articles)} pre-fetched articles for final analysis")
            articles = request.articles
        else:
            # Pass keywords to extract_company_news
            keywords_info = f" with keywords: {', '.join(keywords)}" if keywords else ""
            logger.info(f"Fetching new articles for {company_name}{keywords_info}")
            articles = extract_company_news(company_name, num_articles, keywords)
            
        if not articles:
            raise HTTPException(status_code=404, detail="No news articles found for this company")

        # Analyze sentiment for each article
        analyzed_articles = []
        for article in articles:
            if 'raw_content' in article:
                sentiment_result = analyze_sentiment(article['raw_content'])
                article.update(sentiment_result)
            else:
                article.update({
                    'sentiment': 'Unknown',
                    'polarity': 0,
                    'vader_compound': 0,
                    'speculation_score': 0,
                })
            analyzed_articles.append(article)

        # Generate comparative analysis
        comparison_results = compare_sentiments(analyzed_articles)

        # Create the final report structure
        final_report = create_final_report(company_name, analyzed_articles, comparison_results)

        # Generate Hindi TTS for the final sentiment analysis
        try:
            hindi_summary = translate_to_hindi(final_report["Final Sentiment Analysis"])
            audio_content = generate_hindi_audio_content(hindi_summary)
            final_report["AudioContent"] = audio_content
            final_report["Audio"] = "Generated" if audio_content else "Failed"
        except Exception as e:
            logger.error(f"Error generating Hindi TTS: {e}")
            final_report["Audio"] = "Audio generation failed"
            final_report["AudioContent"] = None

        return final_report

    except Exception as e:
        logger.error(f"Error generating final analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating final analysis: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
