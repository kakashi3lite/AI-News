# RSE Fetch Specialist Agent

## Overview

The **RSE Fetch Specialist** is an autonomous agent designed to handle authenticated retrieval of Research Software Engineering (RSE) news items through the AI-News dashboard's API endpoints. This agent specializes in fetching, parsing, and processing RSE-related content from multiple sources including APIs and RSS feeds.

## Features

### Core Capabilities
- **API Authentication & Rate Limiting**: Handles NEWS_API_KEY authentication with intelligent rate limiting
- **Multi-Source Content Fetching**: Supports both JSON API endpoints and RSS/XML feeds
- **Error Retry Logic**: Implements exponential backoff and retry mechanisms
- **Caching System**: Built-in caching to reduce API calls and improve performance
- **Health Monitoring**: Comprehensive health checks and metrics collection
- **RSE Content Classification**: Automatic categorization and tagging of RSE-related content

### Technical Features
- Asynchronous processing with `aiohttp`
- Robust error handling and logging
- Configurable rate limits and timeouts
- Pagination support for large datasets
- Content deduplication
- Structured logging with metrics

## Installation

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Environment Setup

Set up your environment variables:

```bash
# Primary API key
export NEWS_API_KEY="your_api_key_here"

# Alternative API key (fallback)
export NEXT_PUBLIC_NEWS_API_KEY="your_fallback_key"
```

## Configuration

The agent uses a JSON configuration file located at `config/rse_fetch_config.json`. Key configuration options:

```json
{
  "base_url": "http://localhost:3000/api",
  "rate_limit_delay": 1.0,
  "max_retries": 3,
  "timeout": 30,
  "cache_ttl": 300,
  "batch_size": 50
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|----------|
| `base_url` | Base URL for API endpoints | `http://localhost:3000/api` |
| `rate_limit_delay` | Delay between requests (seconds) | `1.0` |
| `max_retries` | Maximum retry attempts | `3` |
| `timeout` | Request timeout (seconds) | `30` |
| `cache_ttl` | Cache time-to-live (seconds) | `300` |
| `batch_size` | Items per batch | `50` |

## Usage

### Command Line Interface

```bash
# Fetch all RSE content
python agents/rse_fetch_specialist.py --fetch-all

# Fetch only from API endpoints
python agents/rse_fetch_specialist.py --fetch-api

# Fetch only from RSS feeds
python agents/rse_fetch_specialist.py --fetch-rss

# Run health check
python agents/rse_fetch_specialist.py --health-check

# Get current metrics
python agents/rse_fetch_specialist.py --metrics

# Reset metrics
python agents/rse_fetch_specialist.py --reset-metrics

# Enable verbose logging
python agents/rse_fetch_specialist.py --fetch-all --verbose
```

### Programmatic Usage

```python
import asyncio
from agents.rse_fetch_specialist import RSEFetchSpecialist

async def main():
    # Initialize the specialist
    config = {
        'base_url': 'http://localhost:3000/api',
        'rate_limit_delay': 1.0,
        'max_retries': 3
    }
    
    fetcher = RSEFetchSpecialist(config)
    
    # Fetch all RSE content
    result = await fetcher.fetch_all_rse_content()
    
    print(f"Fetched {result['total_count']} items")
    print(f"Success rate: {result['metrics']['success_rate']:.1f}%")
    
    # Process items
    for item in result['items']:
        print(f"- {item.title} ({item.source})")

if __name__ == '__main__':
    asyncio.run(main())
```

## API Endpoints

The RSE Fetch Specialist integrates with the following AI-News API endpoints:

### News API Endpoints
- `GET /api/news` - Fetch general news articles
- `GET /api/news?category=research-software-engineering` - RSE-specific articles
- `GET /api/news?search=rse` - Search for RSE content
- `GET /api/news-explorer` - Advanced news exploration

### RSS Feed Sources
- **Software Sustainability Institute**: `https://www.software.ac.uk/news/rss.xml`
- **Research Software Alliance**: `https://www.researchsoft.org/feed/`
- **US-RSE**: `https://us-rse.org/feed.xml`
- **Society of RSE**: `https://society-rse.org/feed/`
- **Better Scientific Software**: `https://bssw.io/blog/feed`

## Data Models

### RSENewsItem

```python
@dataclass
class RSENewsItem:
    id: str
    title: str
    content: str
    url: str
    source: str
    category: str
    published_at: datetime
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

### FetchMetrics

```python
@dataclass
class FetchMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    cached_responses: int = 0
    total_items_fetched: int = 0
    last_fetch_time: Optional[datetime] = None
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
```

## Error Handling

The agent implements comprehensive error handling:

### HTTP Errors
- **401 Unauthorized**: Invalid API key
- **403 Forbidden**: Insufficient permissions
- **429 Rate Limited**: Automatic retry with backoff
- **500+ Server Errors**: Retry with exponential backoff

### Network Errors
- Connection timeouts
- DNS resolution failures
- SSL/TLS errors

### Data Errors
- Invalid JSON responses
- Missing required fields
- Malformed RSS feeds

## Logging

The agent uses structured logging with different levels:

```python
# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('rse_fetch_specialist')
```

### Log Levels
- **DEBUG**: Detailed request/response information
- **INFO**: General operation status
- **WARNING**: Non-critical issues (rate limits, retries)
- **ERROR**: Failed requests or processing errors
- **CRITICAL**: System-level failures

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest agents/tests/test_rse_fetch_specialist.py -v

# Run with coverage
python -m pytest agents/tests/test_rse_fetch_specialist.py --cov=agents.rse_fetch_specialist

# Run specific test categories
python -m pytest agents/tests/test_rse_fetch_specialist.py::TestRSENewsItem -v
python -m pytest agents/tests/test_rse_fetch_specialist.py::TestRSEFetchSpecialist -v
```

### Test Coverage

The test suite covers:
- ✅ Data model validation
- ✅ Authentication handling
- ✅ Rate limiting and retries
- ✅ API response parsing
- ✅ RSS feed processing
- ✅ Caching mechanisms
- ✅ Error handling
- ✅ Health checks
- ✅ Metrics collection
- ✅ CLI functionality

## Integration with AI-News Dashboard

### Workflow Integration

1. **Scheduled Fetching**: Integrate with the scheduler for regular content updates
2. **Real-time Processing**: Connect to the ML pipeline for immediate analysis
3. **Data Storage**: Store fetched items in the dashboard's database
4. **UI Integration**: Display RSE content in the dashboard interface

### Example Integration

```python
# In scheduler/tasks.py
from agents.rse_fetch_specialist import RSEFetchSpecialist

async def scheduled_rse_fetch():
    """Scheduled task to fetch RSE content."""
    fetcher = RSEFetchSpecialist()
    result = await fetcher.fetch_all_rse_content()
    
    # Process and store results
    for item in result['items']:
        await store_news_item(item)
    
    logger.info(f"Fetched {result['total_count']} RSE items")
```

## Performance Considerations

### Optimization Strategies

1. **Caching**: Implement intelligent caching to reduce API calls
2. **Batch Processing**: Process items in configurable batches
3. **Async Operations**: Use async/await for concurrent processing
4. **Rate Limiting**: Respect API rate limits to avoid throttling
5. **Connection Pooling**: Reuse HTTP connections for efficiency

### Monitoring

```python
# Monitor performance metrics
metrics = fetcher.get_metrics()
print(f"Success Rate: {metrics['success_rate']:.1f}%")
print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
```

## Security

### API Key Management
- Store API keys in environment variables
- Never commit API keys to version control
- Use different keys for development and production
- Implement key rotation procedures

### Data Privacy
- Respect robots.txt for RSS feeds
- Implement proper user-agent headers
- Follow data retention policies
- Ensure GDPR compliance for EU data

## Troubleshooting

### Common Issues

#### Authentication Errors
```bash
# Check API key configuration
echo $NEWS_API_KEY

# Test API connectivity
python agents/rse_fetch_specialist.py --health-check
```

#### Rate Limiting
```python
# Increase rate limit delay
config['rate_limit_delay'] = 2.0  # 2 seconds between requests
```

#### Memory Issues
```python
# Reduce batch size
config['batch_size'] = 25  # Process 25 items at a time
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python agents/rse_fetch_specialist.py --fetch-all --verbose
```

## Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd ai-news-dashboard/AI-News
pip install -r agents/requirements.txt

# Run tests
python -m pytest agents/tests/test_rse_fetch_specialist.py

# Code formatting
black agents/rse_fetch_specialist.py
isort agents/rse_fetch_specialist.py
```

### Adding New Sources

1. Update `config/rse_fetch_config.json` with new RSS feeds
2. Add parsing logic for new API endpoints
3. Update tests to cover new sources
4. Document the new sources in this README

## License

This agent is part of the AI-News Dashboard project and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test suite for examples
3. Check the logs for detailed error information
4. Create an issue in the project repository

---

**RSE Fetch Specialist v1.0.0** - Autonomous RSE Content Retrieval Agent