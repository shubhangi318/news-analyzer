import re
import json
from lxml import etree
from dateutil import parser
from openai import OpenAI

client = OpenAI()
import requests
import tempfile

import os 


from llama_index.readers.llama_parse import LlamaParse  # ✅ Another possible location
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


from openai import OpenAI

client = OpenAI()
openai_api_key = os.getenv('OPENAI_API_KEY')

import time
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

# Type variable for generic functions
T = TypeVar('T')

# Rate limiting for OpenAI API calls
_api_rate_limit_lock = Lock()
_last_api_call_time = 0
_MIN_TIME_BETWEEN_CALLS = 0.6  # Seconds between calls (adjust based on your OpenAI tier)


def calculate_reading_time(content: str) -> str:
    """
    Calculate reading time based on content length at 200 words per minute.
    
    Args:
        content: The article content as string
        
    Returns:
        Formatted reading time string (e.g., "5 min read")
    """
    if not content:
        return "Unknown"
        
    # Count words by splitting on whitespace
    word_count = len(content.split())
    
    # Calculate minutes based on 200 words per minute
    minutes = max(1, round(word_count / 200))
    
    return f"{minutes} min read"

def rate_limited_openai_call(call_function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Execute an OpenAI API call with rate limiting to prevent hitting rate limits.
    
    Args:
        call_function: The OpenAI API function to call
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the API call
    """
    global _last_api_call_time
    
    with _api_rate_limit_lock:
        # Calculate time since last API call
        current_time = time.time()
        time_since_last_call = current_time - _last_api_call_time
        
        # If needed, sleep to maintain minimum time between calls
        if time_since_last_call < _MIN_TIME_BETWEEN_CALLS:
            sleep_time = _MIN_TIME_BETWEEN_CALLS - time_since_last_call
            time.sleep(sleep_time)
        
        # Update last call time
        _last_api_call_time = time.time()
    
    # Make the API call
    return call_function(*args, **kwargs)

def extract_author_with_openai(article_text: str) -> str:
    """
    Extract author name from article text using OpenAI.
    
    Args:
        article_text: The article text
        
    Returns:
        Extracted author name or empty string if not found
    """
    try:
        # Truncate article text if too long
        truncated_text = article_text[:1500] if len(article_text) > 1500 else article_text
        
        # Use rate limited OpenAI call
        response = rate_limited_openai_call(
            client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract the author name from the article. Return only the name without any explanation."},
                {"role": "user", "content": truncated_text}
            ],
            temperature=0.1,
            max_tokens=20
        )
        author = response.choices[0].message.content.strip()
        
        # Clean up the response
        if "unknown" in author.lower() or "not mentioned" in author.lower() or "n/a" in author.lower():
            return ""
        
        return author
    except Exception as e:
        print(f"Error extracting author with OpenAI: {e}")
        return ""


def _setup_llamaparse_environment(article_text: str) -> tuple[str, str]:
    """
    Set up environment for LlamaParse by creating a temporary file and getting API key.
    
    Args:
        article_text: The article text to process
        
    Returns:
        Tuple of (temp_file_path, llama_api_key)
    """
    # Create a temporary file to store the article
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(article_text)
    
    # Get API key from environment
    llama_api_key = os.getenv('LLAMA_CLOUD_API_KEY', '')
    
    return temp_file_path, llama_api_key


def _create_llama_index(temp_file_path: str, llama_api_key: str) -> VectorStoreIndex:
    """
    Create a LlamaIndex from a file using LlamaParse.
    
    Args:
        temp_file_path: Path to the temporary file
        llama_api_key: LlamaParse API key
        
    Returns:
        Vector store index for querying
    """
    # Initialize LlamaParse and parse the file
    parser_instance = LlamaParse(api_key=llama_api_key, result_type="markdown")
    file_extractor = {".txt": parser_instance}
    documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()
    
    # Create an index for querying
    return VectorStoreIndex.from_documents(documents)


def extract_author_with_llamaparse(article_text: str) -> str:
    """
    Extracts the author from article text using LlamaParse and LlamaIndex.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing author name or empty string if not found
    """
    temp_file_path = ""
    try:
        # Set up environment
        temp_file_path, llama_api_key = _setup_llamaparse_environment(article_text)
        
        # Create index and query engine
        index = _create_llama_index(temp_file_path, llama_api_key)
        query_engine = index.as_query_engine()

        # Query the document for author extraction
        query = (
            "Identify the author's name from this news article. "
            "The author's name is typically found in a byline, following phrases like 'By', 'Written by', 'Published by', or 'Author:'."
            "It may also appear at the beginning of the article, near the title, or at the end. "
            "If no clear author is mentioned, return exactly 'Unknown'. "
            "Provide only the author's name, nothing else.")    
        response = query_engine.query(query)

        author = str(response).strip()

        # Clean up the extracted author name
        author = re.sub(r'^(by|author|written by|published by)\s+', '', author, flags=re.IGNORECASE)
        author = author.strip()

        # Validate author (not too long, not Unknown)
        if author and len(author.split()) < 5 and author.lower() != "unknown":
            return author

        # If LlamaParse didn't find an author, try OpenAI
        if author.lower() == "unknown":
            print("LlamaParse returned Unknown, trying OpenAI fallback...")
            openai_author = extract_author_with_openai(article_text)
            return openai_author  # Return whatever OpenAI found

        return author

    except Exception as e:
        print(f"LlamaParse author extraction failed: {e}")

        # Try OpenAI as fallback when LlamaParse fails completely
        print("LlamaParse failed, trying OpenAI fallback...")
        return extract_author_with_openai(article_text)

    finally:
        # Clean up the temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass  # Ignore cleanup errors

def extract_author(soup: Any, article_content: str) -> str:
    """
    Extract author information from an article using multiple methods.
    
    Args:
        soup: BeautifulSoup object of the article HTML
        article_content: Plain text content of the article
        
    Returns:
        Author name or "Unknown" if not found
    """
    author = "Unknown"

    # Check common author elements
    author_patterns = [
        soup.find(['a', 'span', 'div', 'p'], attrs={'class': lambda c: c and any(author_term in c.lower()
                                                                                for author_term in ['author', 'byline', 'writer', 'creator'])}),
        soup.find('meta', attrs={'property': 'article:author'}),
        soup.find('meta', attrs={'name': 'author'})
    ]

    for pattern in author_patterns:
        if pattern:
            text = pattern.get('content', '').strip() if pattern.name == 'meta' else pattern.text.strip()
            # Filter out "Written by" or unwanted phrases
            text = re.sub(r'(?i)written by\s+', '', text).strip()
            if text and len(text.split()) < 5:  # Avoid long bios
                author = text
                break

    # Filter out publication names mistakenly detected as authors
    bad_authors = {"business standard", "latest entertainment", "editorial team"}
    if isinstance(author, str) and author.lower() in bad_authors:
        author = "Unknown"

    # Try AI-based extraction if conventional methods fail
    if author == "Unknown" and article_content:
        try:
            llama_author = extract_author_with_llamaparse(article_content)
            if llama_author and llama_author != "Unknown":
                author = llama_author
        except Exception as e:
            print(f"LlamaParse author extraction failed: {e}")

    return author

def extract_date_with_openai(article_text: str) -> str:
    """
    Extract date from article text using OpenAI.
    
    Args:
        article_text: The article text
        
    Returns:
        Extracted date in YYYY-MM-DD format or "Unknown"
    """
    try:
        # Truncate article text if too long
        truncated_text = article_text[:1500] if len(article_text) > 1500 else article_text
        
        # Use rate limited OpenAI call
        response = rate_limited_openai_call(
            client.chat.completions.create,
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract the publication date from the article. Return only the date in YYYY-MM-DD format if possible. If only month and year are available, use YYYY-MM format. If only year is available, use YYYY format. If no date is mentioned, return 'Unknown'."},
                {"role": "user", "content": truncated_text}
            ],
            temperature=0.1,
            max_tokens=20
        )
        date = response.choices[0].message.content.strip()
        
        # Clean up the response
        if "unknown" in date.lower() or "not mentioned" in date.lower() or "n/a" in date.lower():
            return "Unknown"
        
        return date
    except Exception as e:
        print(f"Error extracting date with OpenAI: {e}")
        return "Unknown"


def extract_date_with_llamaparse(article_text: str) -> str:
    """
    Extracts the publication date from article text using LlamaParse and LlamaIndex.
    
    Args:
        article_text: Full article text
        
    Returns:
        String containing date in YYYY-MM-DD format or "Unknown"
    """
    temp_file_path = ""
    try:
        # Set up environment
        temp_file_path, llama_api_key = _setup_llamaparse_environment(article_text)
        
        # Create index and query engine
        index = _create_llama_index(temp_file_path, llama_api_key)
        query_engine = index.as_query_engine()

        # Query the document for date extraction
        query = (
            "Identify the publication date of this news article. "
            "The date is typically found near the title or at the beginning of the article, "
            "often preceded by 'Published on', 'Posted on', or similar phrases. "
            "Return the date in YYYY-MM-DD format only. "
            "If no clear date is mentioned, return exactly 'Unknown'."
        )
        response = query_engine.query(query)

        date = str(response).strip()

        # Validate date format or parse if possible
        if date.lower() != "unknown":
            try:
                # Try to parse and standardize the date format
                date = parser.parse(date).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"Couldn't parse date: {date}")
                # If LlamaParse returns a date but it can't be parsed, try OpenAI
                openai_date = extract_date_with_openai(article_text)
                return openai_date

        # If LlamaParse didn't find a date, try OpenAI
        if date.lower() == "unknown":
            print("LlamaParse returned Unknown for date, trying OpenAI fallback...")
            openai_date = extract_date_with_openai(article_text)
            return openai_date

        return date

    except Exception as e:
        print(f"LlamaParse date extraction failed: {e}")

        # Try OpenAI as fallback when LlamaParse fails completely
        print("LlamaParse failed for date extraction, trying OpenAI fallback...")
        return extract_date_with_openai(article_text)

    finally:
        # Clean up the temporary file
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception:
            pass  # Ignore cleanup errors

def _check_meta_tags(soup: Any) -> str:
    """
    Check meta tags for publication date.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Date string or "Unknown"
    """
    meta_tags = [
        soup.find("meta", attrs={"property": "article:published_time"}),
        soup.find("meta", attrs={"property": "og:published_time"}),
        soup.find("meta", attrs={"name": "date"}),
        soup.find("meta", attrs={"name": "dc.date"}),
        soup.find("meta", attrs={"name": "dc.date.issued"}),
        soup.find("meta", attrs={"itemprop": "datePublished"}),
    ]
    
    for tag in meta_tags:
        if tag and tag.get("content"):
            return tag["content"].strip()
    
    return "Unknown"

def _check_json_ld(soup: Any) -> str:
    """
    Check JSON-LD structured data for publication date.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Date string or "Unknown"
    """
    json_ld = soup.find("script", type="application/ld+json")
    if json_ld:
        try:
            data = json.loads(json_ld.string)
            if isinstance(data, list):  # Some sites use an array of JSON-LD objects
                data = data[0]

            if "datePublished" in data:
                return data["datePublished"]
            elif "dateCreated" in data:
                return data["dateCreated"]
        except json.JSONDecodeError:
            pass  # Ignore parsing errors
    
    return "Unknown"

def _normalize_date(date_str: str) -> str:
    """
    Normalize date format to YYYY-MM-DD.
    
    Args:
        date_str: Date string to normalize
        
    Returns:
        Normalized date string or "Unknown" if parsing fails
    """
    if date_str and date_str != "Unknown":
        try:
            return parser.parse(date_str).strftime('%Y-%m-%d')  # Convert to YYYY-MM-DD format
        except (ValueError, TypeError):
            return "Unknown"  # If parsing fails, reset to Unknown
    
    return date_str

def extract_date(soup: Any, article_content: str) -> str:
    """
    Extracts the publication date from a news article using multiple methods.
    
    Args:
        soup: BeautifulSoup object of the article HTML
        article_content: Plain text content of the article
        
    Returns:
        Date in YYYY-MM-DD format or "Unknown" if not found
    """
    date = "Unknown"

    # Check <time> Elements
    time_tag = soup.find("time")
    if time_tag and time_tag.text:
        date = time_tag.text.strip()

    # Check meta tags
    if date == "Unknown":
        date = _check_meta_tags(soup)

    # Check HTML Elements with Date-related Classes
    if date == "Unknown":
        date_classes = ['date', 'time', 'published', 'datetime', 'post-date', 'entry-date']
        date_element = soup.find(['span', 'div', 'p'], attrs={'class': lambda c: c and any(d in c.lower() for d in date_classes)})
        if date_element:
            date = date_element.text.strip()

    # Check JSON-LD Structured Data
    if date == "Unknown":
        date = _check_json_ld(soup)

    # Use XPath for Hidden Elements
    if date == "Unknown":
        tree = etree.HTML(str(soup))
        date_xpath = tree.xpath("//*[contains(@class, 'date') or contains(@class, 'time')]//text()")
        if date_xpath:
            date = date_xpath[0].strip()

    # Backup Regex for Dates in Article Text
    if date == "Unknown":
        text = soup.get_text()
        match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})|(\d{4}-\d{2}-\d{2})', text)  # Supports formats like "12 March 2024" or "2024-03-12"
        if match:
            date = match.group(0)

    # Try AI-based extraction if all else fails
    if date == "Unknown" and article_content:
        try:
            llama_date = extract_date_with_llamaparse(article_content)
            if llama_date and llama_date != "Unknown":
                date = llama_date
        except Exception as e:
            print(f"LlamaParse date extraction failed: {e}")

    # Normalize Date Format
    return _normalize_date(date)