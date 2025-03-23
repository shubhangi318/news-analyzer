"""
A module for extracting news articles about companies from various search engines.
"""
from typing import List, Dict, Any, Set, Tuple, Callable, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from .search_engines import search_google_news, search_bing_news, search_yahoo_news


def create_session_with_retry() -> requests.Session:
    """
    Creates a requests session with retry logic configured.
    
    Returns:
        A requests session with retry strategy configured
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def get_default_headers() -> Dict[str, str]:
    """
    Returns default headers for making HTTP requests.
    
    Returns:
        Dictionary of HTTP headers
    """
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }


def search_and_process(
    search_function: Callable,
    search_name: str,
    args: Tuple,
    articles_lock: Lock,
    remaining_count_lock: Lock,
    news_articles: List[Dict[str, Any]],
    remaining_count: int,
    num_articles: int
) -> int:
    """
    Execute a search function and process the results.
    
    Args:
        search_function: The search engine function to call
        search_name: Name of the search engine for logging
        args: Arguments to pass to the search function
        articles_lock: Lock for thread-safe access to news_articles
        remaining_count_lock: Lock for thread-safe access to remaining_count
        news_articles: List to store found articles
        remaining_count: Counter for remaining articles needed
        num_articles: Maximum number of articles to collect
        
    Returns:
        Number of articles added from this search
    """
    company_name = args[0] if args else "unknown company"
    
    try:
        print(f"Starting {search_name} search for '{company_name}'")
        
        with remaining_count_lock:
            if remaining_count <= 0:
                print(f"Already have enough articles, skipping {search_name}")
                return 0

            target_count = min(remaining_count + 2, num_articles)  # +2 as buffer
        
        results = search_function(*args)
        
        if not results:
            print(f"No results found from {search_name} for '{company_name}'")
            return 0
            
        # Thread-safe update of news_articles and remaining count
        articles_to_add = []
        with articles_lock:
            # Add articles that aren't duplicates
            for article in results:
                url = article.get('url', '')
                # Skip if URL already exists in news_articles
                if url and url not in [a.get('url', '') for a in news_articles]:
                    articles_to_add.append(article)
                    with remaining_count_lock:
                        remaining_count -= 1
            
            news_articles.extend(articles_to_add)
            print(f"Added {len(articles_to_add)} articles from {search_name}. Total: {len(news_articles)}/{num_articles}")
            
        return len(articles_to_add)
    except Exception as e:
        import traceback
        print(f"Error during {search_name} search for '{company_name}': {e}")
        print(f"Details: {traceback.format_exc()}")
        return 0


def extract_company_news(company_name: str, num_articles: int = 10, keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Extract news articles about a company from multiple search engines in parallel.
    
    Args:
        company_name: The name of the company to search for
        num_articles: Maximum number of articles to return
        keywords: Optional list of keywords to refine the search
        
    Returns:
        List of extracted news article dictionaries
    """
    # Setup
    news_articles = []
    processed_urls = set()
    session = create_session_with_retry()
    headers = get_default_headers()
    
    articles_lock = Lock()
    remaining_count_lock = Lock()
    remaining_count = num_articles
    
    # Define search tasks to run in parallel - with more Google pages
    search_tasks = [
        (search_google_news, "Google News page 1", (company_name, num_articles, headers, processed_urls, 1, session, keywords)),
        (search_google_news, "Google News page 2", (company_name, num_articles, headers, processed_urls, 2, session, keywords)),
        (search_google_news, "Google News page 3", (company_name, num_articles, headers, processed_urls, 3, session, keywords)),
        (search_google_news, "Google News page 4", (company_name, num_articles, headers, processed_urls, 4, session, keywords)),
        (search_google_news, "Google News page 5", (company_name, num_articles, headers, processed_urls, 5, session, keywords)),
        (search_google_news, "Google News page 6", (company_name, num_articles, headers, processed_urls, 6, session, keywords)),
        (search_bing_news, "Bing News", (company_name, num_articles, headers, processed_urls, session, keywords)),
        (search_yahoo_news, "Yahoo News", (company_name, num_articles, headers, processed_urls, session, keywords))
    ]
    
    # Process all search engines in parallel
    keyword_info = f" with keywords: {', '.join(keywords)}" if keywords else ""
    print(f"Starting parallel searches for news about '{company_name}'{keyword_info}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all search tasks
        future_to_search = {
            executor.submit(
                search_and_process, 
                func, name, args, 
                articles_lock, remaining_count_lock, 
                news_articles, remaining_count, num_articles
            ): name 
            for func, name, args in search_tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_search):
            search_name = future_to_search[future]
            try:
                articles_found = future.result()
                print(f"{search_name} search completed, found {articles_found} articles")
                
                # Check if we have enough articles already
                if len(news_articles) >= num_articles * 1.5:
                    # Cancel any remaining tasks
                    for f in future_to_search:
                        if not f.done():
                            f.cancel()
                    print(f"Found {len(news_articles)} articles, stopping remaining searches")
                    break
                    
            except Exception as e:
                print(f"Error in {search_name} search: {e}")
    
    # Process and deduplicate articles
    processed_articles = process_articles(news_articles, num_articles)
    
    print(f"Total unique articles found: {len(processed_articles)}/{num_articles} requested")
    return processed_articles


def process_articles(articles: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    """
    Process the list of articles to remove duplicates and limit to requested number.
    
    Args:
        articles: List of article dictionaries
        max_count: Maximum number of articles to return
        
    Returns:
        Processed list of unique articles limited to max_count
    """
    # Remove duplicates and limit to requested number
    unique_articles = []
    seen_urls = set()
    
    for article in articles:
        url = article.get('url', '')
        if url and url not in seen_urls and len(unique_articles) < max_count:
            seen_urls.add(url)
            unique_articles.append(article)
    
    # Combine and format article info
    combined_articles = []
    for article in unique_articles:
        # Extract data from the article
        combined_info = {
            'url': article.get('url', 'No url available'),
            'title': article.get('title', 'No title available'),
            'source': article.get('source', 'Unknown'),
            'date': article.get('date', 'Unknown'),
            'summary': article.get('summary', 'No summary available'),
            'keywords': article.get('keywords', []),
            'relevance': article.get('relevance', 'Unknown'),
            'raw_content': article.get('raw_content', 'No content available'),
            'author': article.get('author', 'Unknown'),
            'read_time': article.get('read_time', 'Unknown'),
            'industry': article.get('industry', 'Unknown'),
            'main_topic': article.get('main_topic', 'Uncategorized')
        }
        combined_articles.append(combined_info)

    return combined_articles


