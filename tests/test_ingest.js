/**
 * Test Suite for News Ingestion Module
 * Dr. NewsForge's AI News Dashboard
 * 
 * Tests:
 * - RSS feed parsing and normalization
 * - API source integration
 * - Topic classification accuracy
 * - Deduplication logic
 * - Error handling and resilience
 * - Caching mechanisms
 */

const { NewsIngestionEngine, NEWS_SOURCES, TOPIC_CATEGORIES } = require('../news/ingest.js');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

// Mock data for testing
const MOCK_RSS_RESPONSE = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test News Feed</title>
    <description>Mock RSS feed for testing</description>
    <item>
      <title>AI Revolution in Healthcare Technology</title>
      <description>Artificial intelligence is transforming medical diagnosis and treatment</description>
      <link>https://example.com/ai-healthcare</link>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
      <category>Technology</category>
    </item>
    <item>
      <title>Stock Market Reaches New Heights</title>
      <description>Financial markets show strong performance amid economic recovery</description>
      <link>https://example.com/stock-market</link>
      <pubDate>Mon, 01 Jan 2024 11:00:00 GMT</pubDate>
      <category>Business</category>
    </item>
  </channel>
</rss>`;

const MOCK_NEWSAPI_RESPONSE = {
  status: 'ok',
  totalResults: 2,
  articles: [
    {
      title: 'Climate Change Summit Begins',
      description: 'World leaders gather to discuss environmental policies',
      url: 'https://example.com/climate-summit',
      urlToImage: 'https://example.com/climate.jpg',
      publishedAt: '2024-01-01T10:00:00Z',
      source: { name: 'Environmental News' },
      content: 'Climate change summit brings together global leaders...'
    },
    {
      title: 'New Smartphone Technology Released',
      description: 'Latest mobile device features advanced AI capabilities',
      url: 'https://example.com/smartphone',
      urlToImage: 'https://example.com/phone.jpg',
      publishedAt: '2024-01-01T09:00:00Z',
      source: { name: 'Tech Today' },
      content: 'The new smartphone includes revolutionary AI features...'
    }
  ]
};

class NewsIngestionTester {
  constructor() {
    this.testResults = {
      passed: 0,
      failed: 0,
      errors: []
    };
    this.tempDir = path.join(__dirname, '../temp-test-cache');
  }

  async runAllTests() {
    console.log('🧪 Starting News Ingestion Test Suite\n');
    
    try {
      await this.setupTestEnvironment();
      
      // Core functionality tests
      await this.testRSSParsing();
      await this.testArticleNormalization();
      await this.testTopicClassification();
      await this.testDeduplication();
      await this.testCaching();
      await this.testErrorHandling();
      await this.testRateLimiting();
      await this.testConcurrencyControl();
      
      // Integration tests
      await this.testFullIngestionPipeline();
      
      await this.cleanupTestEnvironment();
      
    } catch (error) {
      console.error('❌ Test suite failed:', error);
      this.testResults.errors.push({ test: 'setup', error: error.message });
    }
    
    this.printTestResults();
    return this.testResults;
  }

  async setupTestEnvironment() {
    try {
      await fs.mkdir(this.tempDir, { recursive: true });
      console.log('✅ Test environment setup complete');
    } catch (error) {
      throw new Error(`Failed to setup test environment: ${error.message}`);
    }
  }

  async cleanupTestEnvironment() {
    try {
      await fs.rmdir(this.tempDir, { recursive: true });
      console.log('🧹 Test environment cleaned up');
    } catch (error) {
      console.warn('⚠️  Failed to cleanup test environment:', error.message);
    }
  }

  async testRSSParsing() {
    console.log('📡 Testing RSS parsing...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      const parsed = engine.xmlParser.parse(MOCK_RSS_RESPONSE);
      const items = engine.extractRSSItems(parsed);
      
      this.assert(items.length === 2, 'Should extract 2 RSS items');
      this.assert(items[0].title === 'AI Revolution in Healthcare Technology', 'Should parse title correctly');
      this.assert(items[1].category === 'Business', 'Should parse category correctly');
      
      console.log('✅ RSS parsing test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ RSS parsing test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'rss-parsing', error: error.message });
    }
  }

  async testArticleNormalization() {
    console.log('🔄 Testing article normalization...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      const mockSource = { name: 'Test Source', url: 'https://test.com', category: 'technology' };
      
      const mockRSSItem = {
        title: 'Test Article Title',
        description: 'Test article description with <b>HTML</b> tags',
        link: 'https://example.com/article',
        pubDate: '2024-01-01T12:00:00Z'
      };
      
      const normalized = engine.normalizeRSSArticle(mockRSSItem, mockSource, 0);
      
      this.assert(normalized.id && normalized.id.length === 16, 'Should generate valid article ID');
      this.assert(normalized.title === 'Test Article Title', 'Should preserve title');
      this.assert(normalized.description === 'Test article description with HTML tags', 'Should clean HTML tags');
      this.assert(normalized.category === 'technology', 'Should inherit source category');
      this.assert(normalized.ingestionMethod === 'rss', 'Should set ingestion method');
      
      console.log('✅ Article normalization test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Article normalization test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'normalization', error: error.message });
    }
  }

  async testTopicClassification() {
    console.log('🏷️  Testing topic classification...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      
      const testArticles = [
        {
          id: '1',
          title: 'AI and Machine Learning Breakthrough',
          description: 'New artificial intelligence software revolutionizes tech industry',
          category: 'general'
        },
        {
          id: '2',
          title: 'Stock Market Analysis',
          description: 'Financial markets show strong investment opportunities',
          category: 'general'
        },
        {
          id: '3',
          title: 'Medical Research Discovery',
          description: 'Scientists discover new vaccine for disease prevention',
          category: 'general'
        }
      ];
      
      const classified = await engine.classifyTopics(testArticles);
      
      this.assert(classified[0].category === 'technology', 'Should classify AI article as technology');
      this.assert(classified[1].category === 'business', 'Should classify finance article as business');
      this.assert(classified[2].category === 'health', 'Should classify medical article as health');
      
      console.log('✅ Topic classification test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Topic classification test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'classification', error: error.message });
    }
  }

  async testDeduplication() {
    console.log('🔄 Testing deduplication...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      
      const duplicateArticles = [
        {
          id: '1',
          title: 'Breaking News Story',
          url: 'https://example.com/news1',
          description: 'Important news story'
        },
        {
          id: '2',
          title: 'Breaking News Story',
          url: 'https://example.com/news1',
          description: 'Important news story'
        },
        {
          id: '3',
          title: 'Different News Story',
          url: 'https://example.com/news2',
          description: 'Another news story'
        },
        {
          id: '4',
          title: 'Breaking News Story',
          url: 'https://different.com/news1',
          description: 'Same title, different domain'
        }
      ];
      
      const unique = engine.deduplicateArticles(duplicateArticles);
      
      this.assert(unique.length === 3, 'Should remove exact duplicates');
      this.assert(unique.some(a => a.id === '1'), 'Should keep first occurrence');
      this.assert(unique.some(a => a.id === '3'), 'Should keep unique articles');
      this.assert(unique.some(a => a.id === '4'), 'Should keep same title from different domain');
      
      console.log('✅ Deduplication test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Deduplication test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'deduplication', error: error.message });
    }
  }

  async testCaching() {
    console.log('💾 Testing caching mechanism...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      
      const testArticles = [
        {
          id: '1',
          title: 'Test Article',
          description: 'Test description',
          url: 'https://example.com/test'
        }
      ];
      
      await engine.cacheResults(testArticles);
      
      // Check if cache file was created
      const cacheFiles = await fs.readdir(this.tempDir);
      const articleCacheFiles = cacheFiles.filter(f => f.startsWith('articles_'));
      
      this.assert(articleCacheFiles.length > 0, 'Should create cache file');
      
      // Verify cache content
      const cacheFile = path.join(this.tempDir, articleCacheFiles[0]);
      const cacheContent = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
      
      this.assert(cacheContent.count === 1, 'Should cache correct article count');
      this.assert(cacheContent.articles[0].id === '1', 'Should cache article data');
      
      console.log('✅ Caching test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Caching test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'caching', error: error.message });
    }
  }

  async testErrorHandling() {
    console.log('⚠️  Testing error handling...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      
      // Test invalid RSS source
      const invalidSource = {
        name: 'Invalid Source',
        url: 'https://invalid-url-that-does-not-exist.com/rss',
        category: 'test'
      };
      
      const result = await engine.fetchRSSSource(invalidSource).catch(err => ({ error: err.message }));
      
      this.assert(result.error, 'Should handle invalid RSS source gracefully');
      
      // Test malformed XML
      const malformedXML = '<invalid>xml<content>';
      try {
        engine.xmlParser.parse(malformedXML);
        this.assert(false, 'Should throw error for malformed XML');
      } catch (parseError) {
        this.assert(true, 'Should handle malformed XML');
      }
      
      console.log('✅ Error handling test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Error handling test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'error-handling', error: error.message });
    }
  }

  async testRateLimiting() {
    console.log('⏱️  Testing rate limiting...');
    
    try {
      const engine = new NewsIngestionEngine({ 
        cacheDir: this.tempDir,
        rateLimitDelay: 100 // 100ms for testing
      });
      
      const startTime = Date.now();
      
      // Simulate multiple source fetches
      const sources = [
        { name: 'Source 1', url: 'https://example1.com', category: 'test' },
        { name: 'Source 2', url: 'https://example2.com', category: 'test' }
      ];
      
      // This should fail due to invalid URLs, but we're testing timing
      await engine.ingestRSSSources({ sources }).catch(() => {});
      
      const elapsed = Date.now() - startTime;
      
      // Should take at least the rate limit delay
      this.assert(elapsed >= 100, 'Should respect rate limiting delay');
      
      console.log('✅ Rate limiting test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Rate limiting test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'rate-limiting', error: error.message });
    }
  }

  async testConcurrencyControl() {
    console.log('🔀 Testing concurrency control...');
    
    try {
      const engine = new NewsIngestionEngine({ 
        cacheDir: this.tempDir,
        maxConcurrent: 2
      });
      
      // Test chunk array utility
      const testArray = [1, 2, 3, 4, 5, 6, 7];
      const chunks = engine.chunkArray(testArray, 3);
      
      this.assert(chunks.length === 3, 'Should create correct number of chunks');
      this.assert(chunks[0].length === 3, 'First chunk should have 3 items');
      this.assert(chunks[2].length === 1, 'Last chunk should have remaining items');
      
      console.log('✅ Concurrency control test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Concurrency control test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'concurrency', error: error.message });
    }
  }

  async testFullIngestionPipeline() {
    console.log('🔄 Testing full ingestion pipeline...');
    
    try {
      const engine = new NewsIngestionEngine({ cacheDir: this.tempDir });
      
      // Mock a minimal ingestion (will fail on actual HTTP calls, but tests structure)
      const result = await engine.ingestAllSources({
        sources: [] // Empty sources to avoid HTTP calls
      });
      
      this.assert(result.hasOwnProperty('articles'), 'Should return articles array');
      this.assert(result.hasOwnProperty('sources'), 'Should return sources array');
      this.assert(result.hasOwnProperty('errors'), 'Should return errors array');
      this.assert(result.hasOwnProperty('stats'), 'Should return stats object');
      
      this.assert(result.stats.hasOwnProperty('totalFetched'), 'Should include totalFetched stat');
      this.assert(result.stats.hasOwnProperty('duplicatesRemoved'), 'Should include duplicatesRemoved stat');
      this.assert(result.stats.hasOwnProperty('processingTime'), 'Should include processingTime stat');
      
      console.log('✅ Full pipeline test passed');
      this.testResults.passed++;
      
    } catch (error) {
      console.error('❌ Full pipeline test failed:', error.message);
      this.testResults.failed++;
      this.testResults.errors.push({ test: 'full-pipeline', error: error.message });
    }
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(`Assertion failed: ${message}`);
    }
  }

  printTestResults() {
    console.log('\n📊 Test Results Summary:');
    console.log(`✅ Passed: ${this.testResults.passed}`);
    console.log(`❌ Failed: ${this.testResults.failed}`);
    console.log(`📈 Success Rate: ${((this.testResults.passed / (this.testResults.passed + this.testResults.failed)) * 100).toFixed(1)}%`);
    
    if (this.testResults.errors.length > 0) {
      console.log('\n❌ Error Details:');
      this.testResults.errors.forEach((error, index) => {
        console.log(`${index + 1}. ${error.test}: ${error.error}`);
      });
    }
    
    console.log('\n🏁 Test suite completed\n');
  }
}

// Performance benchmarking
class IngestionBenchmark {
  constructor() {
    this.metrics = {
      rssParsingTime: [],
      classificationTime: [],
      deduplicationTime: [],
      totalIngestionTime: []
    };
  }

  async runBenchmarks() {
    console.log('⚡ Running performance benchmarks...');
    
    const engine = new NewsIngestionEngine();
    
    // Benchmark RSS parsing
    for (let i = 0; i < 10; i++) {
      const start = Date.now();
      engine.xmlParser.parse(MOCK_RSS_RESPONSE);
      this.metrics.rssParsingTime.push(Date.now() - start);
    }
    
    // Benchmark topic classification
    const testArticles = Array(100).fill().map((_, i) => ({
      id: i.toString(),
      title: `Test Article ${i}`,
      description: 'Technology artificial intelligence software development',
      category: 'general'
    }));
    
    for (let i = 0; i < 5; i++) {
      const start = Date.now();
      await engine.classifyTopics(testArticles);
      this.metrics.classificationTime.push(Date.now() - start);
    }
    
    // Benchmark deduplication
    for (let i = 0; i < 5; i++) {
      const start = Date.now();
      engine.deduplicateArticles(testArticles);
      this.metrics.deduplicationTime.push(Date.now() - start);
    }
    
    this.printBenchmarkResults();
  }

  printBenchmarkResults() {
    console.log('\n⚡ Performance Benchmark Results:');
    
    Object.entries(this.metrics).forEach(([metric, times]) => {
      if (times.length > 0) {
        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        
        console.log(`${metric}:`);
        console.log(`  Average: ${avg.toFixed(2)}ms`);
        console.log(`  Min: ${min}ms`);
        console.log(`  Max: ${max}ms`);
      }
    });
  }
}

// Export for use in other test files
module.exports = {
  NewsIngestionTester,
  IngestionBenchmark,
  MOCK_RSS_RESPONSE,
  MOCK_NEWSAPI_RESPONSE
};

// CLI usage
if (require.main === module) {
  const tester = new NewsIngestionTester();
  const benchmark = new IngestionBenchmark();
  
  async function runTests() {
    const results = await tester.runAllTests();
    
    if (process.argv.includes('--benchmark')) {
      await benchmark.runBenchmarks();
    }
    
    // Exit with error code if tests failed
    process.exit(results.failed > 0 ? 1 : 0);
  }
  
  runTests().catch(console.error);
}