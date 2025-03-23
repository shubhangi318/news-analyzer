import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any, Set, Optional, Tuple, Union
import time
import random

from .article_processor import process_urls_in_parallel

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]
REQUEST_TIMEOUT = 15
MAX_PAGES = 3
RESULTS_PER_PAGE = 10
MAX_URL_FACTOR = 2  # Collect up to num_articles * MAX_URL_FACTOR URLs before processing

# Search engine configurations
SEARCH_ENGINE_CONFIG = {
    'google': {
        'base_url': 'https://www.google.com/search',
        'query_param': 'q',
        'page_param': 'start',
        'additional_params': {'tbm': 'nws'},
        'page_calc': lambda page: 0 if page == 1 else (page - 1) * RESULTS_PER_PAGE,
        'delay_range': lambda page: (min(5 + (page * 0.5), 15), min(5 + (page * 0.5), 15)) if page > 1 else (0, 0),
        'error_delay_range': (10, 20),
        'query_suffix': 'bistro news'
    },
    'bing': {
        'base_url': 'https://www.bing.com/news/search',
        'query_param': 'q',
        'page_param': 'first',
        'additional_params': {},
        'page_calc': lambda page: 1 + ((page - 1) * RESULTS_PER_PAGE) if page > 1 else 1,
        'delay_range': lambda page: (3, 7) if page > 1 else (0, 0),
        'error_delay_range': (10, 20),
        'query_suffix': 'bistro news'
    },
    'yahoo': {
        'base_url': 'https://news.search.yahoo.com/search',
        'query_param': 'p',
        'page_param': 'b',
        'additional_params': {},
        'page_calc': lambda page: (page - 1) * RESULTS_PER_PAGE + 1 if page > 1 else 1,
        'delay_range': lambda page: (4, 8) if page > 1 else (0, 0),
        'error_delay_range': (10, 20),
        'query_suffix': 'bistronews'
    }
}

def rotate_user_agent(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of headers with a randomly selected User-Agent."""
    headers_copy = headers.copy()
    headers_copy['User-Agent'] = random.choice(USER_AGENTS)
    return headers_copy

def add_delay(min_seconds: float, max_seconds: float, page: int, engine: str) -> None:
    """Add a delay between requests with appropriate logging."""
    if min_seconds <= 0 and max_seconds <= 0:
        return
        
    delay = random.uniform(min_seconds, max_seconds)
    print(f"Waiting {delay:.1f}s before requesting {engine.capitalize()} page {page}")
    time.sleep(delay)

def handle_request_error(status_code: int, page: int, engine: str, delay_range: Tuple[int, int]) -> None:
    """Handle error responses with logging and appropriate delay."""
    print(f"{engine.capitalize()} returned status code {status_code} for page {page}")
    time.sleep(random.uniform(delay_range[0], delay_range[1]))

def build_search_url(engine_config: Dict[str, Any], company_name: str, page: int, keywords: Optional[List[str]] = None) -> str:
    """Build a search URL for the given engine, company, keywords and page."""
    # Create base query with company name
    base_query = company_name
    
    # Add keywords to the query if provided (up to 3)
    if keywords and len(keywords) > 0:
        # Ensure we use at most 3 keywords
        limited_keywords = keywords[:3]
        # Add each keyword to the query
        base_query = f"{base_query} {' '.join(limited_keywords)}"
    
    # Add the engine-specific suffix
    query = f"{base_query} {engine_config['query_suffix']}"
    encoded_query = query.replace(' ', '+')
    
    # Calculate the page parameter value
    page_value = engine_config['page_calc'](page)
    
    # Construct URL with query parameters
    params = {
        engine_config['query_param']: encoded_query,
        engine_config['page_param']: page_value
    }
    params.update(engine_config['additional_params'])
    
    # Convert params to URL query string
    param_strings = [f"{k}={v}" for k, v in params.items()]
    return f"{engine_config['base_url']}?{'&'.join(param_strings)}"

def extract_urls_from_elements(elements: List, engine: str, processed_urls: Set[str]) -> List[str]:
    """Extract valid URLs from HTML elements that haven't been processed yet."""
    page_urls = []
    
    for element in elements:
        try:
            # Extract link
            link_element = element.find('a')
            url = link_element['href'] if link_element and 'href' in link_element.attrs else ""
            
            # Special case for Google redirects
            if engine == 'google' and not url.startswith('http') and url.startswith('/url?'):
                url_match = re.search(r'url\?q=([^&]+)', url)
                if url_match:
                    url = url_match.group(1)
            
            # Skip if invalid or already processed
            if not url or not url.startswith('http') or url in processed_urls:
                continue
            
            page_urls.append(url)
            processed_urls.add(url)
            
        except Exception as e:
            print(f"Error extracting URL from {engine.capitalize()}: {e}")
            continue
            
    return page_urls

def extract_google_elements(soup: BeautifulSoup, page: int) -> List[Any]:
    """Extract news article elements from Google search results."""
    article_elements = soup.find('div', id='center_col')
    
    if not article_elements:
        print(f"No article container found on Google page {page}")
        return []

    divs = article_elements.find_all('div', class_='SoaBEf')
    if not divs:
        divs = article_elements.find_all('g-card')
    if not divs:
        divs = article_elements.find_all('div', class_=['dbsr', 'WlydOe'])
        
    print(f"Found {len(divs)} article elements on Google page {page}")
    return divs

def extract_bing_elements(soup: BeautifulSoup, page: int) -> List[Any]:
    """Extract news article elements from Bing search results."""
    news_cards = soup.find_all('div', class_='news-card')
    if not news_cards:
        news_cards = soup.find_all('div', class_='card-with-cluster')
    if not news_cards:
        news_cards = soup.find_all('div', class_=['newsitem', 'cardcommon'])
        
    print(f"Found {len(news_cards)} article elements on Bing page {page}")
    return news_cards

def extract_yahoo_elements(soup: BeautifulSoup, page: int) -> List[Any]:
    """Extract news article elements from Yahoo search results."""
    news_items = soup.find('div', id='web')
    
    if not news_items:
        print(f"No article container found on Yahoo page {page}")
        return []
            
    ol_element = news_items.find('ol')
    if not ol_element:
        print(f"No list element found on Yahoo page {page}")
        return []
            
    # Get all list items
    list_items = ol_element.find_all('li')
    print(f"Found {len(list_items)} article elements on Yahoo page {page}")
    return list_items

def check_for_blocking(response_text: str, engine: str, page: int) -> bool:
    """Check if the response indicates we're being blocked or rate-limited."""
    if engine == 'google' and ("unusual traffic" in response_text.lower() or "captcha" in response_text.lower()):
        print(f"Google has detected unusual traffic on page {page}, backing off")
        time.sleep(random.uniform(30, 60))  # Very long pause if detected
        return True
    return False

def fetch_page(session: requests.Session, url: str, headers: Dict[str, str], 
               engine: str, page: int, config: Dict[str, Any]) -> Optional[BeautifulSoup]:
    """Fetch a search page and return the parsed BeautifulSoup object if successful."""
    try:
        # Add appropriate delay for pagination
        delay_range = config['delay_range'](page)
        add_delay(delay_range[0], delay_range[1], page, engine)
        
        # Use a fresh copy of headers with rotated user agent for each request
        current_headers = rotate_user_agent(headers) if page > 1 else headers
        
        # Send the request
        response = session.get(url, headers=current_headers, timeout=REQUEST_TIMEOUT)
        
        # Handle non-200 responses
        if response.status_code != 200:
            handle_request_error(response.status_code, page, engine, config['error_delay_range'])
            return None
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for blocking/rate limiting
        if check_for_blocking(response.text, engine, page):
            return None
            
        return soup
        
    except Exception as e:
        print(f"Error during {engine.capitalize()} page fetch (page {page}): {e}")
        time.sleep(random.uniform(5, 10))
        return None

def get_element_extractor(engine: str) -> callable:
    """Return the appropriate element extractor function for the given engine."""
    extractors = {
        'google': extract_google_elements,
        'bing': extract_bing_elements,
        'yahoo': extract_yahoo_elements
    }
    return extractors.get(engine)

def search_news_engine(engine: str, company_name: str, num_articles: int, 
                      headers: Dict[str, str], processed_urls: Set[str], 
                      max_pages: int = MAX_PAGES, session: Optional[requests.Session] = None,
                      keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Generic function to search for news articles on a specified search engine.
    
    Args:
        engine: The search engine to use ('google', 'bing', or 'yahoo')
        company_name: The company name to search for
        num_articles: Maximum number of articles to process
        headers: HTTP headers to use for requests
        processed_urls: Set of already processed URLs to avoid duplicates
        max_pages: Maximum number of pages to search (default 3)
        session: Optional requests.Session object
        keywords: Optional list of keywords to refine the search
    
    Returns:
        List of processed news articles
    """
    news_articles = []
    all_article_urls = []
    
    # Get engine configuration
    if engine not in SEARCH_ENGINE_CONFIG:
        print(f"Unknown search engine: {engine}")
        return []
        
    config = SEARCH_ENGINE_CONFIG[engine]
    element_extractor = get_element_extractor(engine)
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # For Google, we only process a single page at a time (pagination handled by parallel calls)
    page_range = range(1, 2) if engine == 'google' else range(1, max_pages + 1)
    
    for page in page_range:
        # Stop if we have enough URLs
        if len(all_article_urls) >= num_articles * MAX_URL_FACTOR and engine != 'google':
            break
            
        # Build the search URL
        url = build_search_url(config, company_name, page, keywords)
        print(f"Searching {engine.capitalize()} News page {page} for '{company_name}'")
        
        # Fetch and parse the page
        soup = fetch_page(session, url, headers, engine, page, config)
        if not soup:
            continue
            
        # Extract elements containing news articles
        elements = element_extractor(soup, page)
        
        # Extract URLs from elements
        page_urls = extract_urls_from_elements(elements, engine, processed_urls)
        all_article_urls.extend(page_urls)
        
        # If no URLs found on this page and it's not page 1, stop
        if len(page_urls) == 0 and page > 1 and engine != 'google':
            break
    
    # Process the gathered URLs
    if all_article_urls:
        # Add small delays between article processing
        modified_headers = headers.copy()
        modified_headers['X-Add-Delay'] = 'true'
        
        # For non-Google engines, limit to num_articles*2 URLs
        urls_to_process = all_article_urls
        if engine != 'google':
            urls_to_process = all_article_urls[:num_articles * MAX_URL_FACTOR]
            
        print(f"Processing {len(urls_to_process)} URLs from {engine.capitalize()}")
        news_articles.extend(process_urls_in_parallel(urls_to_process, modified_headers, company_name, num_articles))
    
    print(f"Extracted {len(news_articles)} articles from {engine.capitalize()} News")
    return news_articles

def search_google_news(company_name: str, num_articles: int, headers: Dict[str, str], 
                      processed_urls: Set[str], page: int = 1, 
                      session: Optional[requests.Session] = None,
                      keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Search for news articles on Google News.
    
    Args:
        company_name: The company name to search for
        num_articles: Maximum number of articles to process
        headers: HTTP headers to use for requests
        processed_urls: Set of already processed URLs to avoid duplicates
        page: The page number to search (for Google we only handle one page per call)
        session: Optional requests.Session object
        keywords: Optional list of keywords to refine the search
    
    Returns:
        List of processed news articles
    """
    # For Google, we need to handle the page parameter specifically
    config = SEARCH_ENGINE_CONFIG['google'].copy()
    all_article_urls = []
    news_articles = []
    
    # Create a session if not provided
    if session is None:
        session = requests.Session()
    
    # Build the search URL for the specific page
    url = build_search_url(config, company_name, page, keywords)
    print(f"Searching Google News page {page} for '{company_name}'")
    
    try:
        # Add appropriate delay for pagination
        delay_range = config['delay_range'](page)
        add_delay(delay_range[0], delay_range[1], page, 'google')
        
        # Use a fresh copy of headers with rotated user agent for each request
        current_headers = rotate_user_agent(headers) if page > 1 else headers
        
        # Send the request
        response = session.get(url, headers=current_headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code != 200:
            handle_request_error(response.status_code, page, 'google', config['error_delay_range'])
            return []
    
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for CAPTCHA or blocking indicators
        if check_for_blocking(response.text, 'google', page):
            return []
        
        # Extract elements containing news articles
        elements = extract_google_elements(soup, page)
        if not elements:
            return []
    
        # Extract URLs from elements
        page_urls = extract_urls_from_elements(elements, 'google', processed_urls)
        all_article_urls.extend(page_urls)
        
        # Use parallel processing to extract content from URLs
        if page_urls:
            print(f"Processing {len(page_urls)} URLs from Google page {page}")
            # Add small delays between article processing
            modified_headers = current_headers.copy()
            modified_headers['X-Add-Delay'] = 'true'
            news_articles.extend(process_urls_in_parallel(page_urls, modified_headers, company_name, num_articles))
            
    except Exception as e:
        print(f"Error during Google News extraction (page {page}): {e}")
        time.sleep(random.uniform(5, 10))  # Pause on error
    
    print(f"Extracted {len(news_articles)} articles from Google News page {page}")
    return news_articles

def search_bing_news(company_name: str, num_articles: int, headers: Dict[str, str], 
                    processed_urls: Set[str], session: Optional[requests.Session] = None,
                    keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Search for news articles on Bing News.
    
    Args:
        company_name: The company name to search for
        num_articles: Maximum number of articles to process
        headers: HTTP headers to use for requests
        processed_urls: Set of already processed URLs to avoid duplicates
        session: Optional requests.Session object
        keywords: Optional list of keywords to refine the search
    
    Returns:
        List of processed news articles
    """
    return search_news_engine('bing', company_name, num_articles, headers, processed_urls, session=session, keywords=keywords)

def search_yahoo_news(company_name: str, num_articles: int, headers: Dict[str, str], 
                     processed_urls: Set[str], session: Optional[requests.Session] = None,
                     keywords: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Search for news articles on Yahoo News.
    
    Args:
        company_name: The company name to search for
        num_articles: Maximum number of articles to process
        headers: HTTP headers to use for requests
        processed_urls: Set of already processed URLs to avoid duplicates
        session: Optional requests.Session object
        keywords: Optional list of keywords to refine the search
    
    Returns:
        List of processed news articles
    """
    return search_news_engine('yahoo', company_name, num_articles, headers, processed_urls, session=session, keywords=keywords)
