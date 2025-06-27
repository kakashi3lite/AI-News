#!/usr/bin/env python3
"""
Test Suite for RSE Fetch Specialist Agent

Author: RSE Fetch Specialist V1
Created: 2024-12-27
Version: 1.0.0

Tests:
- Authentication and API key handling
- RSS/XML & JSON feed parsing
- Error retry logic
- Rate limiting and pagination
- Caching mechanisms
- Health checks and metrics
"""

import os
import sys
import json
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the RSE Fetch Specialist
from agents.rse_fetch_specialist import (
    RSEFetchSpecialist,
    RSENewsItem,
    FetchMetrics
)

class TestRSENewsItem(unittest.TestCase):
    """Test RSENewsItem data structure."""
    
    def test_news_item_creation(self):
        """Test creating a news item."""
        item = RSENewsItem(
            id="test_123",
            title="Test RSE Article",
            content="This is test content about research software engineering.",
            url="https://example.com/article",
            source="test_source",
            category="research-software-engineering",
            published_at=datetime.now(),
            author="Test Author",
            tags=["rse", "testing"],
            summary="Test summary",
            confidence_score=0.9
        )
        
        self.assertEqual(item.id, "test_123")
        self.assertEqual(item.title, "Test RSE Article")
        self.assertEqual(item.category, "research-software-engineering")
        self.assertEqual(len(item.tags), 2)
        self.assertEqual(item.confidence_score, 0.9)
    
    def test_news_item_to_dict(self):
        """Test converting news item to dictionary."""
        now = datetime.now()
        item = RSENewsItem(
            id="test_123",
            title="Test Article",
            content="Test content",
            url="https://example.com",
            source="test",
            category="rse",
            published_at=now
        )
        
        item_dict = item.to_dict()
        
        self.assertIsInstance(item_dict, dict)
        self.assertEqual(item_dict['id'], "test_123")
        self.assertEqual(item_dict['published_at'], now.isoformat())
        self.assertIn('tags', item_dict)

class TestFetchMetrics(unittest.TestCase):
    """Test FetchMetrics functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = FetchMetrics()
        
        self.assertEqual(metrics.total_requests, 0)
        self.assertEqual(metrics.successful_requests, 0)
        self.assertEqual(metrics.failed_requests, 0)
        self.assertEqual(metrics.success_rate(), 0.0)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = FetchMetrics(
            total_requests=10,
            successful_requests=8,
            failed_requests=2
        )
        
        self.assertEqual(metrics.success_rate(), 80.0)
    
    def test_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        metrics = FetchMetrics()
        self.assertEqual(metrics.success_rate(), 0.0)

class TestRSEFetchSpecialist(unittest.IsolatedAsyncioTestCase):
    """Test RSEFetchSpecialist main functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'base_url': 'http://localhost:3000/api',
            'rate_limit_delay': 0.1,  # Faster for testing
            'max_retries': 2,
            'timeout': 5,
            'cache_ttl': 60
        }
        
    @patch.dict(os.environ, {'NEWS_API_KEY': 'test_api_key'})
    def test_initialization(self):
        """Test RSE Fetch Specialist initialization."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        self.assertEqual(fetcher.api_key, 'test_api_key')
        self.assertEqual(fetcher.base_url, 'http://localhost:3000/api')
        self.assertEqual(fetcher.rate_limit_delay, 0.1)
        self.assertEqual(fetcher.max_retries, 2)
        self.assertIsInstance(fetcher.metrics, FetchMetrics)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key(self):
        """Test initialization without API key."""
        fetcher = RSEFetchSpecialist(self.test_config)
        self.assertIsNone(fetcher.api_key)
    
    @patch.dict(os.environ, {'NEXT_PUBLIC_NEWS_API_KEY': 'fallback_key'})
    def test_fallback_api_key(self):
        """Test fallback API key loading."""
        fetcher = RSEFetchSpecialist(self.test_config)
        self.assertEqual(fetcher.api_key, 'fallback_key')
    
    async def test_authenticated_request_success(self):
        """Test successful authenticated request."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock aiohttp session and response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {'articles': []}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await fetcher._make_authenticated_request(
            mock_session, 
            'http://test.com/api/news'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result, {'articles': []})
        self.assertEqual(fetcher.metrics.successful_requests, 1)
    
    async def test_authenticated_request_rate_limit(self):
        """Test rate limiting handling."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock rate limited response followed by success
        mock_session = AsyncMock()
        mock_response_429 = AsyncMock()
        mock_response_429.status = 429
        mock_response_429.headers = {'Retry-After': '1'}
        
        mock_response_200 = AsyncMock()
        mock_response_200.status = 200
        mock_response_200.json.return_value = {'success': True}
        
        mock_session.request.return_value.__aenter__.side_effect = [
            mock_response_429,
            mock_response_200
        ]
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await fetcher._make_authenticated_request(
                mock_session,
                'http://test.com/api/news'
            )
        
        self.assertIsNotNone(result)
        self.assertEqual(fetcher.metrics.rate_limited_requests, 1)
        mock_sleep.assert_called_once_with(1)
    
    async def test_authenticated_request_auth_failure(self):
        """Test authentication failure handling."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'invalid_key'
        
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await fetcher._make_authenticated_request(
            mock_session,
            'http://test.com/api/news'
        )
        
        self.assertIsNone(result)
    
    def test_parse_api_article(self):
        """Test parsing API article response."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        api_article = {
            'id': 'api_123',
            'title': 'RSE Best Practices',
            'content': 'Content about research software engineering best practices.',
            'url': 'https://example.com/rse-practices',
            'publishedAt': '2024-01-15T10:30:00Z',
            'author': 'Dr. RSE Expert',
            'category': 'research-software-engineering',
            'tags': 'rse,best-practices,software'
        }
        
        item = fetcher._parse_api_article(api_article, 'test_api')
        
        self.assertIsNotNone(item)
        self.assertEqual(item.id, 'api_123')
        self.assertEqual(item.title, 'RSE Best Practices')
        self.assertEqual(item.source, 'test_api')
        self.assertIn('rse', item.tags)
        self.assertIn('research-software-engineering', item.tags)
    
    def test_parse_rss_entry(self):
        """Test parsing RSS entry."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        # Mock RSS entry
        mock_entry = Mock()
        mock_entry.id = 'rss_123'
        mock_entry.title = 'Software Sustainability'
        mock_entry.link = 'https://software.ac.uk/article'
        mock_entry.summary = 'Article about software sustainability in research.'
        mock_entry.author = 'SSI Team'
        mock_entry.published_parsed = (2024, 1, 15, 10, 30, 0)
        
        item = fetcher._parse_rss_entry(mock_entry, 'https://software.ac.uk/feed')
        
        self.assertIsNotNone(item)
        self.assertEqual(item.title, 'Software Sustainability')
        self.assertEqual(item.source, 'rss_software.ac.uk')
        self.assertIn('research-software-engineering', item.tags)
        self.assertIn('academic', item.tags)
    
    def test_caching_functionality(self):
        """Test caching mechanisms."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        # Test cache miss
        self.assertFalse(fetcher._is_cached('test_key'))
        self.assertIsNone(fetcher._get_cached('test_key'))
        
        # Test cache set and hit
        test_data = [RSENewsItem(
            id='test',
            title='Test',
            content='Test content',
            url='https://test.com',
            source='test',
            category='test',
            published_at=datetime.now()
        )]
        
        fetcher._set_cache('test_key', test_data)
        self.assertTrue(fetcher._is_cached('test_key'))
        
        cached_data = fetcher._get_cached('test_key')
        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), 1)
        self.assertEqual(cached_data[0].id, 'test')
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        # Use very short TTL for testing
        config = self.test_config.copy()
        config['cache_ttl'] = 0.1  # 0.1 seconds
        
        fetcher = RSEFetchSpecialist(config)
        
        test_data = [RSENewsItem(
            id='test',
            title='Test',
            content='Test content',
            url='https://test.com',
            source='test',
            category='test',
            published_at=datetime.now()
        )]
        
        fetcher._set_cache('test_key', test_data)
        self.assertTrue(fetcher._is_cached('test_key'))
        
        # Wait for cache to expire
        import time
        time.sleep(0.2)
        
        self.assertFalse(fetcher._is_cached('test_key'))
    
    async def test_health_check(self):
        """Test health check functionality."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock successful API connectivity test
        with patch.object(fetcher, '_make_authenticated_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {'status': 'ok'}
            
            health = await fetcher.health_check()
            
            self.assertEqual(health['status'], 'healthy')
            self.assertTrue(health['api_key_configured'])
            self.assertTrue(health['api_connectivity'])
            self.assertIn('timestamp', health)
            self.assertIn('metrics', health)
    
    async def test_health_check_api_failure(self):
        """Test health check with API failure."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock API connectivity failure
        with patch.object(fetcher, '_make_authenticated_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None
            
            health = await fetcher.health_check()
            
            self.assertEqual(health['status'], 'degraded')
            self.assertFalse(health['api_connectivity'])
    
    def test_metrics_operations(self):
        """Test metrics operations."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        # Initial metrics
        metrics = fetcher.get_metrics()
        self.assertEqual(metrics['total_requests'], 0)
        self.assertEqual(metrics['success_rate'], 0.0)
        
        # Update metrics
        fetcher.metrics.total_requests = 10
        fetcher.metrics.successful_requests = 8
        fetcher.metrics.failed_requests = 2
        
        metrics = fetcher.get_metrics()
        self.assertEqual(metrics['total_requests'], 10)
        self.assertEqual(metrics['success_rate'], 80.0)
        
        # Reset metrics
        fetcher.reset_metrics()
        metrics = fetcher.get_metrics()
        self.assertEqual(metrics['total_requests'], 0)

class TestRSEFetchIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for RSE Fetch Specialist."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_config = {
            'base_url': 'http://localhost:3000/api',
            'rate_limit_delay': 0.1,
            'max_retries': 1,
            'timeout': 5
        }
    
    @patch('aiohttp.ClientSession')
    async def test_fetch_rse_news_api_integration(self, mock_session_class):
        """Test integration with news API endpoints."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock API response
        mock_response_data = {
            'articles': [
                {
                    'id': 'rse_001',
                    'title': 'RSE Community Update',
                    'content': 'Latest updates from the RSE community.',
                    'url': 'https://rse.ac.uk/update',
                    'publishedAt': '2024-01-15T10:30:00Z',
                    'category': 'research-software-engineering',
                    'author': 'RSE Team'
                }
            ],
            'totalResults': 1
        }
        
        # Mock session and response
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        with patch.object(fetcher, '_make_authenticated_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response_data
            
            items = await fetcher.fetch_rse_news_api(mock_session)
            
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].id, 'rse_001')
            self.assertEqual(items[0].title, 'RSE Community Update')
            self.assertIn('research-software-engineering', items[0].tags)
    
    @patch('feedparser.parse')
    @patch('aiohttp.ClientSession')
    async def test_fetch_rss_feeds_integration(self, mock_session_class, mock_feedparser):
        """Test integration with RSS feeds."""
        fetcher = RSEFetchSpecialist(self.test_config)
        
        # Mock RSS feed data
        mock_feed = Mock()
        mock_entry = Mock()
        mock_entry.id = 'rss_001'
        mock_entry.title = 'Software Sustainability Guide'
        mock_entry.link = 'https://software.ac.uk/guide'
        mock_entry.summary = 'A comprehensive guide to software sustainability.'
        mock_entry.published_parsed = (2024, 1, 15, 10, 30, 0)
        mock_feed.entries = [mock_entry]
        
        mock_feedparser.return_value = mock_feed
        
        # Mock session and response
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = '<rss>mock rss content</rss>'
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        items = await fetcher.fetch_rse_rss_feeds(mock_session)
        
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, 'Software Sustainability Guide')
        self.assertIn('research-software-engineering', items[0].tags)
    
    @patch('aiohttp.ClientSession')
    async def test_fetch_all_rse_content_integration(self, mock_session_class):
        """Test full integration of fetching all RSE content."""
        fetcher = RSEFetchSpecialist(self.test_config)
        fetcher.api_key = 'test_key'
        
        # Mock both API and RSS responses
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        with patch.object(fetcher, 'fetch_rse_news_api', new_callable=AsyncMock) as mock_api:
            with patch.object(fetcher, 'fetch_rse_rss_feeds', new_callable=AsyncMock) as mock_rss:
                
                # Mock return data
                api_item = RSENewsItem(
                    id='api_001',
                    title='API Article',
                    content='Content from API',
                    url='https://api.example.com/article',
                    source='api',
                    category='rse',
                    published_at=datetime.now()
                )
                
                rss_item = RSENewsItem(
                    id='rss_001',
                    title='RSS Article',
                    content='Content from RSS',
                    url='https://rss.example.com/article',
                    source='rss',
                    category='rse',
                    published_at=datetime.now()
                )
                
                mock_api.return_value = [api_item]
                mock_rss.return_value = [rss_item]
                
                result = await fetcher.fetch_all_rse_content()
                
                self.assertEqual(result['total_count'], 2)
                self.assertEqual(result['source'], 'live_fetch')
                self.assertIn('items', result)
                self.assertIn('metrics', result)
                self.assertIn('fetched_at', result)

class TestRSEFetchCLI(unittest.TestCase):
    """Test CLI functionality."""
    
    @patch('sys.argv', ['rse_fetch_specialist.py', '--health-check'])
    @patch('agents.rse_fetch_specialist.RSEFetchSpecialist')
    async def test_cli_health_check(self, mock_fetcher_class):
        """Test CLI health check command."""
        mock_fetcher = AsyncMock()
        mock_fetcher.health_check.return_value = {
            'status': 'healthy',
            'api_connectivity': True
        }
        mock_fetcher_class.return_value = mock_fetcher
        
        # Import and run main function
        from agents.rse_fetch_specialist import main
        
        with patch('builtins.print') as mock_print:
            await main()
            mock_print.assert_called()
    
    def test_config_loading(self):
        """Test configuration file loading."""
        config_path = Path(__file__).parent.parent / 'config' / 'rse_fetch_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.assertIn('id', config)
            self.assertIn('name', config)
            self.assertIn('capabilities', config)
            self.assertEqual(config['id'], 'rse-fetch-bot-v1')
            self.assertEqual(config['name'], 'RSE Fetch Specialist')

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)