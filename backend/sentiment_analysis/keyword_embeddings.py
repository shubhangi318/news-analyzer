"""
Topic utilities for clustering and analyzing keywords in articles.
Provides functionality for keyword preprocessing, normalization, and clustering.
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from typing import List, Dict, Set, Tuple, Optional, Union

# Constants
DEFAULT_CLUSTERING_THRESHOLD = 0.3
DEFAULT_MIN_ARTICLES = 2
DEFAULT_MODEL_NAME = 'all-mpnet-base-v2'  # Better for business domain
COMMON_PHRASES = ["food delivery", "quick commerce"]

# Lazy loaded model - will be initialized on first use
_model = None

def _get_model() -> SentenceTransformer:
    """
    Lazily initialize and return the sentence transformer model.
    
    Returns:
        SentenceTransformer: Initialized model for text embeddings
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_MODEL_NAME)
    return _model

def preprocess_keyword(keyword: str) -> str:
    """
    Clean and normalize a keyword to handle format variations.
    
    Args:
        keyword: Raw keyword string to preprocess
        
    Returns:
        Normalized and cleaned keyword string
    """
    # Convert to lowercase
    keyword = keyword.lower()
    
    # Replace hyphens and similar punctuation with spaces
    keyword = re.sub(r'[-–—]', ' ', keyword)  # Handle different types of hyphens
    
    # Remove other punctuation that might cause variations
    keyword = re.sub(r'[^\w\s]', '', keyword)
    
    # Normalize whitespace (including multiple spaces)
    keyword = ' '.join(keyword.split())
    
    # Process multi-word keywords
    if len(keyword.split()) > 1:
        keyword = _sort_keyword_words(keyword)
    
    return keyword.strip()

def _sort_keyword_words(keyword: str) -> str:
    """
    Sort words in a keyword while preserving common phrases.
    
    Args:
        keyword: Multi-word keyword string
        
    Returns:
        Keyword with sorted words but preserved phrases
    """
    words = keyword.split()
    
    # Try to keep common phrases intact
    for phrase in COMMON_PHRASES:
        if phrase in keyword:
            # Remove phrase words from sorting
            for word in phrase.split():
                if word in words:
                    words.remove(word)
            # Add phrase at the beginning
            return ' '.join(phrase.split() + sorted(words))
    
    # If no common phrase found, sort all words
    return ' '.join(sorted(words))

def get_normalized_to_original_mapping(keywords: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Create mapping between normalized and original forms of keywords.
    
    Args:
        keywords: List of raw keywords to process
        
    Returns:
        Tuple containing:
        - List of normalized keywords
        - Dictionary mapping normalized forms to original forms
    """
    normalized_keywords = []
    normalized_to_original = {}
    
    for keyword in keywords:
        normalized = preprocess_keyword(keyword)
        normalized_keywords.append(normalized)
        
        # Keep the original form (preferring shortest if multiple originals map to same normalized)
        if normalized not in normalized_to_original or len(keyword) < len(normalized_to_original[normalized]):
            normalized_to_original[normalized] = keyword
    
    return normalized_keywords, normalized_to_original

def _group_exact_duplicates(keywords: List[str], normalized_keywords: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Group keywords that are exact duplicates after normalization.
    
    Args:
        keywords: List of original keywords
        normalized_keywords: List of normalized keywords
        
    Returns:
        Tuple containing:
        - List of unique normalized keywords
        - Dictionary mapping unique keywords to their duplicates
    """
    unique_normalized = []
    unique_to_duplicates = {}
    
    seen = set()
    for i, norm_kw in enumerate(normalized_keywords):
        if norm_kw not in seen:
            seen.add(norm_kw)
            unique_normalized.append(norm_kw)
            unique_to_duplicates[norm_kw] = [keywords[i]]
        else:
            # Add to the duplicates list if we've seen this normalized form before
            for unique_kw in unique_normalized:
                if preprocess_keyword(unique_kw) == norm_kw:
                    unique_to_duplicates[unique_kw].append(keywords[i])
                    break
    
    return unique_normalized, unique_to_duplicates

def _handle_single_cluster_case(unique_normalized: List[str], 
                               normalized_to_original: Dict[str, str],
                               unique_to_duplicates: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Handle the special case when there's only one cluster.
    
    Args:
        unique_normalized: List of unique normalized keywords
        normalized_to_original: Mapping from normalized to original keywords
        unique_to_duplicates: Mapping from unique keywords to their duplicates
        
    Returns:
        Dictionary mapping representative keyword to its duplicates
    """
    if not unique_normalized:
        return {}
        
    representative = normalized_to_original[unique_normalized[0]]
    duplicates = [k for k in unique_to_duplicates[unique_normalized[0]] if k != representative]
    return {representative: duplicates}

def _process_clusters(clusters: Dict[int, List[int]], 
                     unique_normalized: List[str],
                     normalized_to_original: Dict[str, str],
                     unique_to_duplicates: Dict[str, List[str]],
                     embeddings: np.ndarray) -> Dict[str, List[str]]:
    """
    Process clusters to create the final output with representative keywords.
    
    Args:
        clusters: Dictionary mapping cluster labels to lists of indices
        unique_normalized: List of unique normalized keywords
        normalized_to_original: Mapping from normalized to original keywords
        unique_to_duplicates: Mapping from unique keywords to their duplicates
        embeddings: Numpy array of keyword embeddings
        
    Returns:
        Dictionary mapping representative keywords to lists of similar keywords
    """
    result = {}
    
    for label, indices in clusters.items():
        if len(indices) == 1:
            # Single keyword cluster
            norm_keyword = unique_normalized[indices[0]]
            representative = normalized_to_original[norm_keyword]
            
            # Include any duplicates from the normalization step
            duplicates = [k for k in unique_to_duplicates[norm_keyword] if k != representative]
            result[representative] = duplicates
        else:
            # Multi-keyword cluster
            result.update(_process_multi_keyword_cluster(indices, unique_normalized, 
                                                      normalized_to_original, 
                                                      unique_to_duplicates, embeddings))
    
    return result

def _process_multi_keyword_cluster(indices: List[int],
                                  unique_normalized: List[str],
                                  normalized_to_original: Dict[str, str],
                                  unique_to_duplicates: Dict[str, List[str]],
                                  embeddings: np.ndarray) -> Dict[str, List[str]]:
    """
    Process a multi-keyword cluster to find representative and similar keywords.
    
    Args:
        indices: List of indices for keywords in this cluster
        unique_normalized: List of unique normalized keywords
        normalized_to_original: Mapping from normalized to original keywords
        unique_to_duplicates: Mapping from unique keywords to their duplicates
        embeddings: Numpy array of keyword embeddings
        
    Returns:
        Dictionary with one entry mapping representative keyword to similar keywords
    """
    # Find the most central keyword to use as representative
    centroid = np.mean([embeddings[idx] for idx in indices], axis=0)
    distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in indices]
    representative_idx = indices[np.argmin(distances)]
    representative_norm = unique_normalized[representative_idx]
    representative = normalized_to_original[representative_norm]
    
    # Collect all keywords in this cluster, including duplicates
    all_similar = []
    
    # Add similar keywords from other indices
    for idx in indices:
        if idx != representative_idx:
            norm_kw = unique_normalized[idx]
            all_similar.extend(unique_to_duplicates[norm_kw])
    
    # Also add duplicates of the representative itself
    for dup in unique_to_duplicates[representative_norm]:
        if dup != representative:
            all_similar.append(dup)
    
    return {representative: all_similar}

def cluster_keywords(keywords: List[str], threshold: float = DEFAULT_CLUSTERING_THRESHOLD) -> Dict[str, List[str]]:
    """
    Cluster similar keywords using embeddings and DBSCAN clustering.
    
    Args:
        keywords: List of keywords to cluster
        threshold: Similarity threshold for clustering (smaller = more clusters)
        
    Returns:
        Dictionary mapping representative keywords to lists of similar keywords
    """
    # Handle edge cases
    if not keywords or len(keywords) < 2:
        return {k: [] for k in keywords} if keywords else {}
    
    # Normalize keywords and find duplicates
    normalized_keywords, normalized_to_original = get_normalized_to_original_mapping(keywords)
    unique_normalized, unique_to_duplicates = _group_exact_duplicates(keywords, normalized_keywords)
    
    # If only exact duplicates or no keywords, return early
    if len(unique_normalized) <= 1:
        return _handle_single_cluster_case(unique_normalized, normalized_to_original, unique_to_duplicates)
    
    # Generate embeddings
    model = _get_model()
    embeddings = model.encode(unique_normalized)
    
    # Cluster with DBSCAN (using cosine distance for better semantic matching)
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='cosine').fit(embeddings)
    
    # Create clusters dictionary
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Process clusters to get the final result
    return _process_clusters(clusters, unique_normalized, normalized_to_original, 
                            unique_to_duplicates, embeddings)

def _create_keyword_to_articles_mapping(article_keywords: List[List[str]]) -> Dict[str, Set[int]]:
    """
    Create a mapping from keywords to the articles they appear in.
    
    Args:
        article_keywords: List of keyword lists, one per article
        
    Returns:
        Dictionary mapping keywords to sets of article indices
    """
    keyword_to_articles = {}
    for i, keywords in enumerate(article_keywords):
        for kw in keywords:
            if kw not in keyword_to_articles:
                keyword_to_articles[kw] = set()
            keyword_to_articles[kw].add(i)
    
    return keyword_to_articles

def _get_unique_keywords(article_keywords: List[List[str]]) -> List[str]:
    """
    Get a list of unique keywords from all articles.
    
    Args:
        article_keywords: List of keyword lists, one per article
        
    Returns:
        List of unique keywords
    """
    # Flatten all keywords
    all_keywords = []
    for keywords in article_keywords:
        all_keywords.extend(keywords)
    
    # Remove duplicates while preserving order
    unique_keywords = []
    seen = set()
    for kw in all_keywords:
        if kw.lower() not in seen:
            unique_keywords.append(kw)
            seen.add(kw.lower())
    
    return unique_keywords

def find_common_topics(article_keywords: List[List[str]], 
                      min_articles: int = DEFAULT_MIN_ARTICLES) -> Dict[str, List[str]]:
    """
    Find common topics across multiple articles using embeddings.
    
    Args:
        article_keywords: List of keyword lists, one per article
        min_articles: Minimum number of articles a topic must appear in
        
    Returns:
        Dictionary mapping representative keywords to lists of similar keywords
    """
    # Get unique keywords and create mapping to articles
    unique_keywords = _get_unique_keywords(article_keywords)
    keyword_to_articles = _create_keyword_to_articles_mapping(article_keywords)
    
    # Cluster similar keywords
    clusters = cluster_keywords(unique_keywords)
    
    # Find common topics (clusters that appear in multiple articles)
    common_topics = {}
    for representative, similar_keywords in clusters.items():
        # Get all articles that contain any keyword in this cluster
        articles_with_topic = keyword_to_articles.get(representative, set())
        for similar in similar_keywords:
            articles_with_topic.update(keyword_to_articles.get(similar, set()))
        
        # If topic appears in minimum number of articles, add to common topics
        if len(articles_with_topic) >= min_articles:
            common_topics[representative] = similar_keywords
    
    return common_topics

def _get_all_common_keywords(common_topics: Dict[str, List[str]]) -> Set[str]:
    """
    Create a set of all common topic keywords (lowercased).
    
    Args:
        common_topics: Dictionary mapping representative keywords to similar keywords
        
    Returns:
        Set of all keywords in common topics, lowercased
    """
    all_common_keywords = set()
    for rep, similar in common_topics.items():
        all_common_keywords.add(rep.lower())
        all_common_keywords.update(kw.lower() for kw in similar)
    
    return all_common_keywords

def get_article_specific_topics(article_keywords: List[List[str]], 
                               common_topics: Dict[str, List[str]]) -> List[Dict[str, List[str]]]:
    """
    Find topics unique to each article.
    
    Args:
        article_keywords: List of keyword lists, one per article
        common_topics: Dictionary of common topics across articles
        
    Returns:
        List of dictionaries, each mapping article-specific keywords to similar keywords
    """
    # Get all common keywords
    all_common_keywords = _get_all_common_keywords(common_topics)
    
    # Find unique topics for each article
    article_specific_topics = []
    
    for keywords in article_keywords:
        # Cluster the article's keywords
        article_clusters = cluster_keywords(keywords)
        
        # Keep only clusters that don't overlap with common topics
        unique_clusters = {}
        for rep, similar in article_clusters.items():
            if rep.lower() not in all_common_keywords and not any(s.lower() in all_common_keywords for s in similar):
                unique_clusters[rep] = similar
        
        article_specific_topics.append(unique_clusters)
    
    return article_specific_topics