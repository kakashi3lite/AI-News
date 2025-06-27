#!/usr/bin/env python3
"""
RSE Fetch Specialist Agent V1
Handles authenticated retrieval of RSE news items through the AI-News `news` API endpoints.

Author: RSE Fetch Specialist V1
Created: 2024-12-27
Version: 1.0.0

Capabilities:
- API authentication and rate-limit handling
- RSS/XML & JSON feed parsing
- Error retry logic
- Logging and metrics for fetch operations
"""

import os
import sys
import json
import time
import logging
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rse_fetch_specialist.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RSEFetchSpecialist')

@dataclass
class RSENewsItem:
    """Data structure for RSE news items."""
    id: str
    title: str
    content: str
    url: str
    source: str
    category: str
    published_at: datetime
    author: Optional[str] = None
    tags: List[str] = None
    summary: Optional[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat()
        return data

@dataclass
class FetchMetrics:
    """Metrics for fetch operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    total_items_fetched: int = 0
    processing_time: float = 0.0
    last_fetch_time: Optional[datetime] = None
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

class RSEFetchSpecialist:
    """RSE Fetch Specialist for authenticated news retrieval."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RSE Fetch Specialist.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.api_key = self._load_api_key()
        self.base_url = self.config.get('base_url', 'http://localhost:3000/api')
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)
        
        # Initialize metrics
        self.metrics = FetchMetrics()
        
        # RSE-specific sources and endpoints
        self.rse_sources = {
            'rse_news': {
                'endpoint': '/news',
                'method': 'GET',
                'params': {'category': 'research-software-engineering'}
            },
            'academic_feeds': {
                'rss_urls': [
                    'https://www.software.ac.uk/feed',
                    'https://urssi.us/feed/',
                    'https://rse.ac.uk/feed/',
                    'https://society-rse.org/feed/'
                ]
            },
            'github_rse': {
                'endpoint': '/news',
                'method': 'GET', 
                'params': {'tag': 'research-software', 'source': 'github'}
            }
        }
        
        # Cache for storing fetched items
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        logger.info(f"RSE Fetch Specialist initialized with base URL: {self.base_url}")
    
    def _load_api_key(self) -> str:
        """Load NEWS_API_KEY from environment."""
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            logger.warning("NEWS_API_KEY not found in environment variables")
            # Try alternative environment variable names
            api_key = os.getenv('NEXT_PUBLIC_NEWS_API_KEY') or os.getenv('API_KEY')
        
        if api_key:
            logger.info("API key loaded successfully")
        else:
            logger.error("No API key found. Set NEWS_API_KEY environment variable.")
        
        return api_key
    
    async def _make_authenticated_request(
        self, 
        session: aiohttp.ClientSession,
        url: str, 
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Make authenticated HTTP request with retry logic.
        
        Args:
            session: aiohttp session
            url: Request URL
            method: HTTP method
            params: Query parameters
            headers: Additional headers
            
        Returns:
            Response data or None if failed
        """
        # Prepare headers with authentication
        request_headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RSE-Fetch-Specialist/1.0'
        }
        
        if self.api_key:
            request_headers['X-API-Key'] = self.api_key
            request_headers['Authorization'] = f'Bearer {self.api_key}'
        
        if headers:
            request_headers.update(headers)
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                self.metrics.total_requests += 1
                
                async with session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    # Handle rate limiting
                    if response.status == 429:
                        self.metrics.rate_limited_requests += 1
                        retry_after = int(response.headers.get('Retry-After', self.rate_limit_delay))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Validate HTTP status codes
                    if response.status == 200:
                        self.metrics.successful_requests += 1
                        data = await response.json()
                        return data
                    elif response.status == 401:
                        logger.error("Authentication failed. Check API key.")
                        return None
                    elif response.status == 403:
                        logger.error("Access forbidden. Check permissions.")
                        return None
                    else:
                        logger.warning(f"HTTP {response.status}: {await response.text()}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
        
        self.metrics.failed_requests += 1
        logger.error(f"Failed to fetch from {url} after {self.max_retries} attempts")
        return None
    
    async def fetch_rse_news_api(self, session: aiohttp.ClientSession) -> List[RSENewsItem]:
        """Query RSE feed endpoints under `/api/news`.
        
        Args:
            session: aiohttp session
            
        Returns:
            List of RSE news items
        """
        items = []
        
        for source_name, source_config in self.rse_sources.items():
            if 'endpoint' not in source_config:
                continue
                
            url = urljoin(self.base_url, source_config['endpoint'])
            params = source_config.get('params', {})
            
            logger.info(f"Fetching from {source_name}: {url}")
            
            data = await self._make_authenticated_request(
                session, url, 
                method=source_config.get('method', 'GET'),
                params=params
            )
            
            if data and 'articles' in data:
                for article in data['articles']:
                    try:
                        item = self._parse_api_article(article, source_name)
                        if item:
                            items.append(item)
                            self.metrics.total_items_fetched += 1
                    except Exception as e:
                        logger.error(f"Error parsing article from {source_name}: {e}")
            
            # Handle pagination if present
            if data and 'pagination' in data:
                await self._handle_pagination(session, url, params, data['pagination'], source_name, items)
            
            # Rate limiting between requests
            await asyncio.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched {len(items)} items from API endpoints")
        return items
    
    async def _handle_pagination(
        self, 
        session: aiohttp.ClientSession,
        base_url: str, 
        base_params: Dict[str, Any],
        pagination: Dict[str, Any], 
        source_name: str,
        items: List[RSENewsItem]
    ) -> None:
        """Handle pagination for API responses.
        
        Args:
            session: aiohttp session
            base_url: Base URL for requests
            base_params: Base parameters
            pagination: Pagination information
            source_name: Source identifier
            items: List to append items to
        """
        current_page = pagination.get('current_page', 1)
        total_pages = pagination.get('total_pages', 1)
        
        # Fetch remaining pages (limit to prevent excessive requests)
        max_pages = min(total_pages, 5)  # Limit to 5 pages
        
        for page in range(current_page + 1, max_pages + 1):
            params = base_params.copy()
            params['page'] = page
            
            logger.info(f"Fetching page {page}/{total_pages} from {source_name}")
            
            data = await self._make_authenticated_request(session, base_url, params=params)
            
            if data and 'articles' in data:
                for article in data['articles']:
                    try:
                        item = self._parse_api_article(article, source_name)
                        if item:
                            items.append(item)
                            self.metrics.total_items_fetched += 1
                    except Exception as e:
                        logger.error(f"Error parsing article from {source_name} page {page}: {e}")
            
            await asyncio.sleep(self.rate_limit_delay)
    
    def _parse_api_article(self, article: Dict[str, Any], source: str) -> Optional[RSENewsItem]:
        """Parse article from API response.
        
        Args:
            article: Article data from API
            source: Source identifier
            
        Returns:
            RSENewsItem or None if parsing failed
        """
        try:
            # Generate unique ID
            article_id = article.get('id') or f"{source}_{hash(article.get('url', ''))}"
            
            # Parse published date
            published_str = article.get('publishedAt') or article.get('published') or article.get('date')
            published_at = datetime.now()
            if published_str:
                try:
                    published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except:
                    logger.warning(f"Could not parse date: {published_str}")
            
            # Extract tags
            tags = article.get('tags', [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            # Add RSE-specific tags
            rse_tags = ['research-software-engineering', 'rse']
            tags.extend(rse_tags)
            
            return RSENewsItem(
                id=article_id,
                title=article.get('title', ''),
                content=article.get('content') or article.get('description', ''),
                url=article.get('url', ''),
                source=source,
                category=article.get('category', 'research-software-engineering'),
                published_at=published_at,
                author=article.get('author'),
                tags=list(set(tags)),  # Remove duplicates
                summary=article.get('summary'),
                confidence_score=article.get('confidence', 0.8)
            )
            
        except Exception as e:
            logger.error(f"Error parsing article: {e}")
            return None
    
    async def fetch_rse_rss_feeds(self, session: aiohttp.ClientSession) -> List[RSENewsItem]:
        """Fetch and parse RSS/XML feeds for RSE content.
        
        Args:
            session: aiohttp session
            
        Returns:
            List of RSE news items from RSS feeds
        """
        items = []
        
        rss_urls = self.rse_sources.get('academic_feeds', {}).get('rss_urls', [])
        
        for rss_url in rss_urls:
            logger.info(f"Fetching RSS feed: {rss_url}")
            
            try:
                # Fetch RSS content
                async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        rss_content = await response.text()
                        
                        # Parse RSS feed
                        feed = feedparser.parse(rss_content)
                        
                        for entry in feed.entries:
                            try:
                                item = self._parse_rss_entry(entry, rss_url)
                                if item:
                                    items.append(item)
                                    self.metrics.total_items_fetched += 1
                            except Exception as e:
                                logger.error(f"Error parsing RSS entry from {rss_url}: {e}")
                    else:
                        logger.warning(f"Failed to fetch RSS feed {rss_url}: HTTP {response.status}")
                        
            except Exception as e:
                logger.error(f"Error fetching RSS feed {rss_url}: {e}")
            
            await asyncio.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched {len(items)} items from RSS feeds")
        return items
    
    def _parse_rss_entry(self, entry: Any, source_url: str) -> Optional[RSENewsItem]:
        """Parse RSS feed entry.
        
        Args:
            entry: RSS entry object
            source_url: Source RSS URL
            
        Returns:
            RSENewsItem or None if parsing failed
        """
        try:
            # Generate unique ID
            entry_id = getattr(entry, 'id', None) or f"rss_{hash(getattr(entry, 'link', ''))}"
            
            # Parse published date
            published_at = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6])
            
            # Extract content
            content = ''
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value if isinstance(entry.content, list) else entry.content
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            # Extract tags
            tags = ['research-software-engineering', 'rse', 'academic']
            if hasattr(entry, 'tags'):
                for tag in entry.tags:
                    if hasattr(tag, 'term'):
                        tags.append(tag.term)
            
            return RSENewsItem(
                id=entry_id,
                title=getattr(entry, 'title', ''),
                content=content,
                url=getattr(entry, 'link', ''),
                source=f"rss_{urlparse(source_url).netloc}",
                category='research-software-engineering',
                published_at=published_at,
                author=getattr(entry, 'author', None),
                tags=tags,
                summary=getattr(entry, 'summary', None),
                confidence_score=0.7
            )
            
        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cached and valid, False otherwise
        """
        if cache_key not in self.cache:
            return False
        
        cached_time, _ = self.cache[cache_key]
        return (datetime.now() - cached_time).total_seconds() < self.cache_ttl
    
    def _get_cached(self, cache_key: str) -> Optional[List[RSENewsItem]]:
        """Get cached data.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None
        """
        if self._is_cached(cache_key):
            _, data = self.cache[cache_key]
            return data
        return None
    
    def _set_cache(self, cache_key: str, data: List[RSENewsItem]) -> None:
        """Set cache data.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        self.cache[cache_key] = (datetime.now(), data)
    
    async def fetch_all_rse_content(self) -> Dict[str, Any]:
        """Fetch RSE content from all sources with caching and error handling.
        
        Returns:
            Dictionary containing fetched items, metrics, and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = 'all_rse_content'
        cached_data = self._get_cached(cache_key)
        if cached_data:
            logger.info("Returning cached RSE content")
            return {
                'items': [item.to_dict() for item in cached_data],
                'total_count': len(cached_data),
                'source': 'cache',
                'metrics': asdict(self.metrics),
                'fetched_at': datetime.now().isoformat()
            }
        
        logger.info("Starting RSE content fetch from all sources...")
        
        all_items = []
        
        async with aiohttp.ClientSession() as session:
            try:
                # Fetch from API endpoints
                api_items = await self.fetch_rse_news_api(session)
                all_items.extend(api_items)
                
                # Fetch from RSS feeds
                rss_items = await self.fetch_rse_rss_feeds(session)
                all_items.extend(rss_items)
                
            except Exception as e:
                logger.error(f"Error during fetch operation: {e}")
        
        # Remove duplicates based on URL
        unique_items = {}
        for item in all_items:
            if item.url and item.url not in unique_items:
                unique_items[item.url] = item
        
        final_items = list(unique_items.values())
        
        # Update metrics
        self.metrics.processing_time = time.time() - start_time
        self.metrics.last_fetch_time = datetime.now()
        
        # Cache results
        self._set_cache(cache_key, final_items)
        
        logger.info(f"Fetch completed. Retrieved {len(final_items)} unique RSE items in {self.metrics.processing_time:.2f}s")
        
        return {
            'items': [item.to_dict() for item in final_items],
            'total_count': len(final_items),
            'source': 'live_fetch',
            'metrics': asdict(self.metrics),
            'fetched_at': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current fetch metrics.
        
        Returns:
            Dictionary containing current metrics
        """
        metrics_dict = asdict(self.metrics)
        metrics_dict['success_rate'] = self.metrics.success_rate()
        if self.metrics.last_fetch_time:
            metrics_dict['last_fetch_time'] = self.metrics.last_fetch_time.isoformat()
        return metrics_dict
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.metrics = FetchMetrics()
        logger.info("Metrics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on RSE fetch capabilities.
        
        Returns:
            Health check results
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'api_key_configured': bool(self.api_key),
            'base_url': self.base_url,
            'cache_size': len(self.cache),
            'metrics': self.get_metrics()
        }
        
        # Test API connectivity
        try:
            async with aiohttp.ClientSession() as session:
                test_url = urljoin(self.base_url, '/news')
                response = await self._make_authenticated_request(
                    session, test_url, params={'limit': 1}
                )
                health_status['api_connectivity'] = response is not None
        except Exception as e:
            health_status['api_connectivity'] = False
            health_status['api_error'] = str(e)
            health_status['status'] = 'degraded'
        
        return health_status

# CLI Interface
async def main():
    """Main CLI interface for RSE Fetch Specialist."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RSE Fetch Specialist - Authenticated RSE news retrieval')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--health-check', action='store_true', help='Perform health check')
    parser.add_argument('--metrics', action='store_true', help='Show current metrics')
    parser.add_argument('--reset-metrics', action='store_true', help='Reset metrics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize RSE Fetch Specialist
    rse_fetcher = RSEFetchSpecialist(config)
    
    if args.health_check:
        health = await rse_fetcher.health_check()
        print(json.dumps(health, indent=2))
        return
    
    if args.metrics:
        metrics = rse_fetcher.get_metrics()
        print(json.dumps(metrics, indent=2))
        return
    
    if args.reset_metrics:
        rse_fetcher.reset_metrics()
        print("Metrics reset successfully")
        return
    
    # Fetch RSE content
    print("🔍 Starting RSE content fetch...")
    results = await rse_fetcher.fetch_all_rse_content()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))
    
    print(f"\n📊 Fetch Summary:")
    print(f"   Total items: {results['total_count']}")
    print(f"   Source: {results['source']}")
    print(f"   Success rate: {rse_fetcher.metrics.success_rate():.1f}%")
    print(f"   Processing time: {rse_fetcher.metrics.processing_time:.2f}s")

if __name__ == '__main__':
    asyncio.run(main())