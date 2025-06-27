#!/usr/bin/env python3
"""
Validation Script for RSE Fetch Specialist

This script validates that the RSE Fetch Specialist is properly implemented
and can be imported and used without issues.

Author: RSE Fetch Specialist V1
Created: 2024-12-27
Version: 1.0.0
"""

import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from agents.rse_fetch_specialist import (
        RSEFetchSpecialist,
        RSENewsItem,
        FetchMetrics
    )
    print("✅ Successfully imported RSE Fetch Specialist components")
except ImportError as e:
    print(f"❌ Failed to import RSE Fetch Specialist: {e}")
    sys.exit(1)

def test_data_models():
    """Test the data models."""
    print("\n🧪 Testing Data Models...")
    
    # Test RSENewsItem
    try:
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
        
        # Test to_dict method
        item_dict = item.to_dict()
        assert isinstance(item_dict, dict)
        assert item_dict['id'] == "test_123"
        assert 'published_at' in item_dict
        
        print("  ✅ RSENewsItem creation and serialization")
    except Exception as e:
        print(f"  ❌ RSENewsItem test failed: {e}")
        return False
    
    # Test FetchMetrics
    try:
        metrics = FetchMetrics()
        assert metrics.total_requests == 0
        assert metrics.success_rate() == 0.0
        
        metrics.total_requests = 10
        metrics.successful_requests = 8
        metrics.failed_requests = 2
        assert metrics.success_rate() == 80.0
        
        print("  ✅ FetchMetrics calculation")
    except Exception as e:
        print(f"  ❌ FetchMetrics test failed: {e}")
        return False
    
    return True

def test_rse_fetch_specialist():
    """Test RSE Fetch Specialist initialization."""
    print("\n🧪 Testing RSE Fetch Specialist...")
    
    try:
        config = {
            'base_url': 'http://localhost:3000/api',
            'rate_limit_delay': 0.1,
            'max_retries': 2,
            'timeout': 5,
            'cache_ttl': 60
        }
        
        fetcher = RSEFetchSpecialist(config)
        
        # Test initialization
        assert fetcher.base_url == 'http://localhost:3000/api'
        assert fetcher.rate_limit_delay == 0.1
        assert fetcher.max_retries == 2
        assert isinstance(fetcher.metrics, FetchMetrics)
        
        print("  ✅ RSEFetchSpecialist initialization")
        
        # Test metrics operations
        metrics = fetcher.get_metrics()
        assert isinstance(metrics, dict)
        assert 'total_requests' in metrics
        assert 'success_rate' in metrics
        
        print("  ✅ Metrics operations")
        
        # Test caching functionality
        test_data = [RSENewsItem(
            id='test',
            title='Test',
            content='Test content',
            url='https://test.com',
            source='test',
            category='test',
            published_at=datetime.now()
        )]
        
        # Test cache operations
        assert not fetcher._is_cached('test_key')
        fetcher._set_cache('test_key', test_data)
        assert fetcher._is_cached('test_key')
        
        cached_data = fetcher._get_cached('test_key')
        assert cached_data is not None
        assert len(cached_data) == 1
        assert cached_data[0].id == 'test'
        
        print("  ✅ Caching functionality")
        
    except Exception as e:
        print(f"  ❌ RSEFetchSpecialist test failed: {e}")
        return False
    
    return True

def test_parsing_methods():
    """Test parsing methods."""
    print("\n🧪 Testing Parsing Methods...")
    
    try:
        fetcher = RSEFetchSpecialist({})
        
        # Test API article parsing
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
        assert item is not None
        assert item.id == 'api_123'
        assert item.title == 'RSE Best Practices'
        assert item.source == 'test_api'
        assert 'rse' in item.tags
        
        print("  ✅ API article parsing")
        
    except Exception as e:
        print(f"  ❌ Parsing methods test failed: {e}")
        return False
    
    return True

async def test_health_check():
    """Test health check functionality."""
    print("\n🧪 Testing Health Check...")
    
    try:
        fetcher = RSEFetchSpecialist({})
        
        # Test health check (will show API connectivity as false since no server is running)
        health = await fetcher.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'timestamp' in health
        assert 'api_key_configured' in health
        assert 'metrics' in health
        
        print("  ✅ Health check functionality")
        print(f"  📊 Health Status: {health['status']}")
        print(f"  🔑 API Key Configured: {health['api_key_configured']}")
        print(f"  🌐 API Connectivity: {health.get('api_connectivity', False)}")
        
    except Exception as e:
        print(f"  ❌ Health check test failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    try:
        config_path = Path(__file__).parent / 'config' / 'rse_fetch_config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            assert 'id' in config
            assert 'name' in config
            assert 'capabilities' in config
            assert config['id'] == 'rse-fetch-bot-v1'
            assert config['name'] == 'RSE Fetch Specialist'
            
            print("  ✅ Configuration file loading")
            print(f"  📋 Agent ID: {config['id']}")
            print(f"  📋 Agent Name: {config['name']}")
            print(f"  🎯 Capabilities: {len(config['capabilities'])} items")
        else:
            print("  ⚠️  Configuration file not found (optional)")
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False
    
    return True

async def main():
    """Run all validation tests."""
    print("🚀 RSE Fetch Specialist Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Run synchronous tests
    if test_data_models():
        tests_passed += 1
    
    if test_rse_fetch_specialist():
        tests_passed += 1
    
    if test_parsing_methods():
        tests_passed += 1
    
    if test_configuration():
        tests_passed += 1
    
    # Run async tests
    if await test_health_check():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Validation Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! RSE Fetch Specialist is ready to use.")
        print("\n📚 Next Steps:")
        print("  1. Set NEWS_API_KEY environment variable")
        print("  2. Start the AI-News dashboard server")
        print("  3. Run: python agents/rse_fetch_specialist.py --health-check")
        print("  4. Run: python agents/rse_fetch_specialist.py --fetch-all")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Please check the implementation.")
        return False

if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)