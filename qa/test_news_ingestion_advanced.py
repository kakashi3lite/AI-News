#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - Advanced News Ingestion Test Suite

Comprehensive testing for AI News Dashboard ingestion pipeline with:
- Persona-based validation
- AI inference accuracy testing
- Chaos engineering simulation
- Performance benchmarking
- Security vulnerability assessment

Author: Dr. Orion "TestMaster" Vanguard
Version: 1.0.0
License: MIT
"""

import pytest
import asyncio
import json
import time
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import redis
import openai
from anthropic import Anthropic

# Test Configuration
TEST_CONFIG = {
    "dashboard_url": "http://localhost:3000",
    "api_base_url": "http://localhost:3000/api",
    "redis_host": "localhost",
    "redis_port": 6379,
    "test_timeout": 30,
    "performance_thresholds": {
        "page_load_ms": 3000,
        "api_response_ms": 2000,
        "cache_hit_rate": 0.80
    }
}

@dataclass
class PersonaProfile:
    """Persona configuration for testing"""
    name: str
    description: str
    behavior_patterns: List[str]
    device_type: str
    performance_expectations: Dict[str, int]
    accessibility_requirements: List[str]

@dataclass
class TestResult:
    """Standardized test result structure"""
    test_name: str
    persona: Optional[str]
    success: bool
    score: float
    duration_ms: int
    metrics: Dict[str, Any]
    errors: List[str]
    recommendations: List[str]

class SuperhumanNewsIngestionTester:
    """Advanced news ingestion testing with superhuman capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.test_results: List[TestResult] = []
        self.personas = self._initialize_personas()
        self.ai_models = self._initialize_ai_models()
        
    def _initialize_personas(self) -> Dict[str, PersonaProfile]:
        """Initialize testing personas"""
        return {
            "casual_reader": PersonaProfile(
                name="Casual Reader",
                description="Regular user browsing news casually",
                behavior_patterns=["scroll_headlines", "click_articles", "casual_search"],
                device_type="desktop",
                performance_expectations={"page_load_ms": 3000, "interaction_ms": 500},
                accessibility_requirements=["keyboard_navigation", "screen_reader"]
            ),
            "breaking_news_alerter": PersonaProfile(
                name="Breaking News Alerter",
                description="Power user seeking real-time updates",
                behavior_patterns=["frequent_refresh", "filter_categories", "quick_scan"],
                device_type="mobile",
                performance_expectations={"page_load_ms": 1500, "interaction_ms": 200},
                accessibility_requirements=["touch_targets", "voice_commands"]
            ),
            "financial_analyst": PersonaProfile(
                name="Financial Analyst",
                description="Professional analyzing market news",
                behavior_patterns=["search_companies", "filter_financial", "deep_analysis"],
                device_type="desktop",
                performance_expectations={"page_load_ms": 2000, "interaction_ms": 300},
                accessibility_requirements=["high_contrast", "data_tables"]
            )
        }
    
    def _initialize_ai_models(self) -> Dict[str, Any]:
        """Initialize AI model clients"""
        return {
            "openai": openai.OpenAI(api_key="test-key"),
            "anthropic": Anthropic(api_key="test-key")
        }
    
    def setup_redis_connection(self):
        """Setup Redis connection for cache testing"""
        try:
            self.redis_client = redis.Redis(
                host=self.config["redis_host"],
                port=self.config["redis_port"],
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis_client = None

class TestNewsIngestionPersonas:
    """Persona-based testing for news ingestion"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tester = SuperhumanNewsIngestionTester(TEST_CONFIG)
        self.tester.setup_redis_connection()
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.quit()
    
    def _create_persona_driver(self, persona: PersonaProfile) -> webdriver.Chrome:
        """Create Selenium driver configured for specific persona"""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Device-specific configurations
        if persona.device_type == "mobile":
            mobile_emulation = {"deviceName": "iPhone 12"}
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        elif persona.device_type == "tablet":
            mobile_emulation = {"deviceName": "iPad"}
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Accessibility configurations
        if "high_contrast" in persona.accessibility_requirements:
            options.add_argument("--force-prefers-color-scheme=dark")
        
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        return driver
    
    @pytest.mark.parametrize("persona_name", ["casual_reader", "breaking_news_alerter", "financial_analyst"])
    def test_persona_homepage_load(self, persona_name: str):
        """Test homepage loading for different personas"""
        persona = self.tester.personas[persona_name]
        self.driver = self._create_persona_driver(persona)
        
        start_time = time.time()
        
        try:
            # Navigate to homepage
            self.driver.get(TEST_CONFIG["dashboard_url"])
            
            # Wait for critical elements
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
            
            load_time_ms = (time.time() - start_time) * 1000
            
            # Validate performance expectations
            expected_load_time = persona.performance_expectations["page_load_ms"]
            performance_score = max(0, 100 - (load_time_ms / expected_load_time) * 100)
            
            # Check for critical elements
            critical_elements = [
                (By.CLASS_NAME, "news-grid"),
                (By.CLASS_NAME, "search-bar"),
                (By.CLASS_NAME, "navigation")
            ]
            
            elements_found = 0
            for selector_type, selector_value in critical_elements:
                try:
                    self.driver.find_element(selector_type, selector_value)
                    elements_found += 1
                except:
                    pass
            
            element_score = (elements_found / len(critical_elements)) * 100
            overall_score = (performance_score + element_score) / 2
            
            # Record test result
            result = TestResult(
                test_name=f"homepage_load_{persona_name}",
                persona=persona_name,
                success=load_time_ms <= expected_load_time,
                score=overall_score,
                duration_ms=int(load_time_ms),
                metrics={
                    "load_time_ms": load_time_ms,
                    "elements_found": elements_found,
                    "performance_score": performance_score,
                    "element_score": element_score
                },
                errors=[],
                recommendations=[
                    "Optimize image loading" if load_time_ms > expected_load_time else "Performance acceptable",
                    "Add missing critical elements" if elements_found < len(critical_elements) else "All elements present"
                ]
            )
            
            self.tester.test_results.append(result)
            
            assert load_time_ms <= expected_load_time, f"Load time {load_time_ms}ms exceeds {expected_load_time}ms for {persona_name}"
            assert elements_found >= len(critical_elements) * 0.8, f"Missing critical elements for {persona_name}"
            
        except Exception as e:
            pytest.fail(f"Homepage load test failed for {persona_name}: {str(e)}")
    
    @pytest.mark.parametrize("persona_name", ["casual_reader", "financial_analyst"])
    def test_persona_search_functionality(self, persona_name: str):
        """Test search functionality for different personas"""
        persona = self.tester.personas[persona_name]
        self.driver = self._create_persona_driver(persona)
        
        try:
            self.driver.get(TEST_CONFIG["dashboard_url"])
            
            # Find search input
            search_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'], input[placeholder*='search']"))
            )
            
            # Persona-specific search terms
            search_terms = {
                "casual_reader": "technology news",
                "financial_analyst": "stock market trends",
                "breaking_news_alerter": "breaking news"
            }
            
            search_term = search_terms.get(persona_name, "general news")
            
            start_time = time.time()
            
            # Perform search
            search_input.clear()
            search_input.send_keys(search_term)
            search_input.submit()
            
            # Wait for results
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "search-results"))
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Validate search results
            results = self.driver.find_elements(By.CLASS_NAME, "news-item")
            results_count = len(results)
            
            # Performance validation
            expected_response_time = persona.performance_expectations["interaction_ms"]
            performance_score = max(0, 100 - (search_time_ms / expected_response_time) * 100)
            
            # Results quality validation
            quality_score = min(100, (results_count / 10) * 100)  # Expect at least 10 results
            
            overall_score = (performance_score + quality_score) / 2
            
            result = TestResult(
                test_name=f"search_functionality_{persona_name}",
                persona=persona_name,
                success=search_time_ms <= expected_response_time and results_count >= 5,
                score=overall_score,
                duration_ms=int(search_time_ms),
                metrics={
                    "search_time_ms": search_time_ms,
                    "results_count": results_count,
                    "search_term": search_term,
                    "performance_score": performance_score,
                    "quality_score": quality_score
                },
                errors=[],
                recommendations=[
                    "Optimize search indexing" if search_time_ms > expected_response_time else "Search performance good",
                    "Improve search relevance" if results_count < 10 else "Good search results"
                ]
            )
            
            self.tester.test_results.append(result)
            
            assert search_time_ms <= expected_response_time, f"Search too slow for {persona_name}"
            assert results_count >= 5, f"Insufficient search results for {persona_name}"
            
        except Exception as e:
            pytest.fail(f"Search functionality test failed for {persona_name}: {str(e)}")

class TestAIInferenceValidation:
    """AI inference accuracy and validation testing"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tester = SuperhumanNewsIngestionTester(TEST_CONFIG)
        self.api_base = TEST_CONFIG["api_base_url"]
    
    def test_summarization_accuracy(self):
        """Test AI summarization accuracy"""
        test_articles = [
            {
                "title": "Breaking: Major Tech Company Announces AI Breakthrough",
                "content": "A leading technology company today announced a revolutionary artificial intelligence system that can predict market trends with 95% accuracy. The new system, developed over three years, uses advanced machine learning algorithms and processes millions of data points in real-time. Industry experts believe this breakthrough could transform financial markets and investment strategies worldwide.",
                "expected_summary": "Tech company unveils AI system with 95% market prediction accuracy, potentially transforming financial markets."
            },
            {
                "title": "Climate Change Report Reveals Alarming Trends",
                "content": "The latest climate change report from international scientists reveals unprecedented warming trends across the globe. Temperature increases of 1.5 degrees Celsius are now inevitable within the next decade, according to the comprehensive study. The report emphasizes urgent need for immediate action to prevent catastrophic environmental consequences affecting billions of people worldwide.",
                "expected_summary": "New climate report shows inevitable 1.5°C warming within decade, urgent action needed to prevent catastrophic consequences."
            }
        ]
        
        total_accuracy = 0
        test_count = 0
        
        for article in test_articles:
            try:
                # Test OpenAI summarization
                response = requests.post(
                    f"{self.api_base}/summarize-openai",
                    json={
                        "text": article["content"],
                        "max_length": 100
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    summary_data = response.json()
                    generated_summary = summary_data.get("summary", "")
                    
                    # Calculate similarity score (simplified)
                    accuracy = self._calculate_summary_similarity(
                        generated_summary,
                        article["expected_summary"]
                    )
                    
                    total_accuracy += accuracy
                    test_count += 1
                    
                    # Check for hallucinations
                    hallucination_score = self._detect_hallucinations(
                        article["content"],
                        generated_summary
                    )
                    
                    assert accuracy >= 0.7, f"Summary accuracy too low: {accuracy}"
                    assert hallucination_score < 0.1, f"Hallucination detected: {hallucination_score}"
                
            except Exception as e:
                pytest.fail(f"Summarization test failed: {str(e)}")
        
        average_accuracy = total_accuracy / test_count if test_count > 0 else 0
        assert average_accuracy >= 0.75, f"Average summarization accuracy too low: {average_accuracy}"
    
    def test_rag_inference_accuracy(self):
        """Test RAG (Retrieval-Augmented Generation) inference accuracy"""
        test_queries = [
            {
                "query": "What are the latest developments in artificial intelligence?",
                "expected_topics": ["machine learning", "AI", "technology", "innovation"],
                "min_relevance": 0.8
            },
            {
                "query": "How is climate change affecting the economy?",
                "expected_topics": ["climate", "economy", "environmental", "impact"],
                "min_relevance": 0.8
            }
        ]
        
        for query_data in test_queries:
            try:
                response = requests.post(
                    f"{self.api_base}/news-explorer",
                    json={"query": query_data["query"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Validate response structure
                    assert "summary" in result_data, "Missing summary in response"
                    assert "articles" in result_data, "Missing articles in response"
                    
                    # Calculate topic relevance
                    relevance_score = self._calculate_topic_relevance(
                        result_data["summary"],
                        query_data["expected_topics"]
                    )
                    
                    assert relevance_score >= query_data["min_relevance"], f"Topic relevance too low: {relevance_score}"
                    
                    # Validate article count
                    articles_count = len(result_data.get("articles", []))
                    assert articles_count >= 3, f"Insufficient articles returned: {articles_count}"
                
            except Exception as e:
                pytest.fail(f"RAG inference test failed: {str(e)}")
    
    def _calculate_summary_similarity(self, generated: str, expected: str) -> float:
        """Calculate similarity between generated and expected summaries"""
        # Simplified similarity calculation
        generated_words = set(generated.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        intersection = generated_words.intersection(expected_words)
        return len(intersection) / len(expected_words)
    
    def _detect_hallucinations(self, original_text: str, summary: str) -> float:
        """Detect potential hallucinations in summary"""
        # Simplified hallucination detection
        original_words = set(original_text.lower().split())
        summary_words = set(summary.lower().split())
        
        # Words in summary but not in original (potential hallucinations)
        hallucinated_words = summary_words - original_words
        
        # Filter out common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        hallucinated_words = hallucinated_words - common_words
        
        if not summary_words:
            return 0.0
        
        return len(hallucinated_words) / len(summary_words)
    
    def _calculate_topic_relevance(self, text: str, expected_topics: List[str]) -> float:
        """Calculate topic relevance score"""
        text_lower = text.lower()
        matches = sum(1 for topic in expected_topics if topic.lower() in text_lower)
        return matches / len(expected_topics) if expected_topics else 0.0

class TestChaosEngineering:
    """Chaos engineering tests for resilience validation"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tester = SuperhumanNewsIngestionTester(TEST_CONFIG)
        self.api_base = TEST_CONFIG["api_base_url"]
    
    def test_api_resilience_under_load(self):
        """Test API resilience under simulated load"""
        concurrent_requests = 50
        success_count = 0
        total_response_time = 0
        
        async def make_request():
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_base}/news", timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return True, response_time
                return False, response_time
            except:
                return False, 10000  # Timeout penalty
        
        # Simulate concurrent load
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            
            for future in concurrent.futures.as_completed(futures):
                success, response_time = future.result()
                if success:
                    success_count += 1
                total_response_time += response_time
        
        success_rate = success_count / concurrent_requests
        avg_response_time = total_response_time / concurrent_requests
        
        # Validate resilience metrics
        assert success_rate >= 0.95, f"Success rate too low under load: {success_rate}"
        assert avg_response_time <= 5000, f"Average response time too high: {avg_response_time}ms"
    
    def test_cache_eviction_recovery(self):
        """Test system recovery after cache eviction"""
        if not self.tester.redis_client:
            pytest.skip("Redis not available for cache testing")
        
        try:
            # Warm up cache
            response = requests.get(f"{self.api_base}/news")
            assert response.status_code == 200
            
            # Measure baseline performance
            start_time = time.time()
            response = requests.get(f"{self.api_base}/news")
            baseline_time = (time.time() - start_time) * 1000
            
            # Simulate cache eviction
            self.tester.redis_client.flushall()
            
            # Measure performance after cache eviction
            start_time = time.time()
            response = requests.get(f"{self.api_base}/news")
            recovery_time = (time.time() - start_time) * 1000
            
            # Validate recovery
            assert response.status_code == 200, "API failed after cache eviction"
            
            # Allow for reasonable performance degradation
            max_degradation = baseline_time * 3  # 3x slower is acceptable
            assert recovery_time <= max_degradation, f"Recovery too slow: {recovery_time}ms vs baseline {baseline_time}ms"
            
        except Exception as e:
            pytest.fail(f"Cache eviction recovery test failed: {str(e)}")
    
    def test_network_latency_simulation(self):
        """Test system behavior under network latency"""
        latency_scenarios = [100, 500, 1000]  # milliseconds
        
        for latency_ms in latency_scenarios:
            try:
                # Simulate network latency (simplified)
                time.sleep(latency_ms / 1000)
                
                start_time = time.time()
                response = requests.get(f"{self.api_base}/news", timeout=30)
                response_time = (time.time() - start_time) * 1000
                
                # Validate that system handles latency gracefully
                assert response.status_code == 200, f"API failed under {latency_ms}ms latency"
                
                # Response time should be reasonable considering added latency
                expected_max_time = latency_ms + 5000  # Base time + latency + buffer
                assert response_time <= expected_max_time, f"Response time too high under latency: {response_time}ms"
                
            except Exception as e:
                pytest.fail(f"Network latency test failed for {latency_ms}ms: {str(e)}")

class TestPerformanceBenchmarks:
    """Performance benchmarking and optimization validation"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tester = SuperhumanNewsIngestionTester(TEST_CONFIG)
        self.api_base = TEST_CONFIG["api_base_url"]
    
    def test_api_response_times(self):
        """Test API response time benchmarks"""
        endpoints = [
            "/news",
            "/summarize",
            "/news-explorer"
        ]
        
        for endpoint in endpoints:
            response_times = []
            
            # Measure multiple requests for statistical significance
            for _ in range(10):
                start_time = time.time()
                
                try:
                    if endpoint == "/news-explorer":
                        response = requests.post(
                            f"{self.api_base}{endpoint}",
                            json={"query": "technology news"},
                            timeout=30
                        )
                    elif endpoint == "/summarize":
                        response = requests.post(
                            f"{self.api_base}{endpoint}",
                            json={"text": "Sample news article for testing summarization performance."},
                            timeout=30
                        )
                    else:
                        response = requests.get(f"{self.api_base}{endpoint}", timeout=30)
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        response_times.append(response_time)
                    
                except Exception as e:
                    print(f"Request failed for {endpoint}: {e}")
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                # Performance thresholds
                threshold = TEST_CONFIG["performance_thresholds"]["api_response_ms"]
                
                assert avg_response_time <= threshold, f"{endpoint} average response time too high: {avg_response_time}ms"
                assert max_response_time <= threshold * 2, f"{endpoint} max response time too high: {max_response_time}ms"
                
                print(f"{endpoint} performance: avg={avg_response_time:.2f}ms, min={min_response_time:.2f}ms, max={max_response_time:.2f}ms")
    
    def test_cache_performance(self):
        """Test caching layer performance"""
        if not self.tester.redis_client:
            pytest.skip("Redis not available for cache testing")
        
        cache_hits = 0
        total_requests = 20
        
        # Make repeated requests to test cache hit rate
        for i in range(total_requests):
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_base}/news")
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    # Assume faster responses indicate cache hits
                    if response_time < 500:  # 500ms threshold for cache hit
                        cache_hits += 1
                
            except Exception as e:
                print(f"Cache performance test request failed: {e}")
        
        cache_hit_rate = cache_hits / total_requests
        expected_hit_rate = TEST_CONFIG["performance_thresholds"]["cache_hit_rate"]
        
        assert cache_hit_rate >= expected_hit_rate, f"Cache hit rate too low: {cache_hit_rate} < {expected_hit_rate}"
        
        print(f"Cache hit rate: {cache_hit_rate:.2%} ({cache_hits}/{total_requests})")

class TestSecurityValidation:
    """Security testing and vulnerability assessment"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_base = TEST_CONFIG["api_base_url"]
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks"""
        sql_injection_payloads = [
            "'; DROP TABLE news; --",
            "' OR '1'='1",
            "'; SELECT * FROM users; --",
            "' UNION SELECT password FROM users --"
        ]
        
        for payload in sql_injection_payloads:
            try:
                # Test search endpoint with malicious payload
                response = requests.post(
                    f"{self.api_base}/news-explorer",
                    json={"query": payload},
                    timeout=10
                )
                
                # Should not return 500 error or expose database errors
                assert response.status_code != 500, f"SQL injection may have caused server error with payload: {payload}"
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    # Check for database error messages
                    error_indicators = ["sql", "database", "mysql", "postgres", "sqlite", "error"]
                    for indicator in error_indicators:
                        assert indicator not in response_text, f"Potential SQL injection vulnerability detected with payload: {payload}"
                
            except requests.exceptions.Timeout:
                # Timeout might indicate successful injection causing database issues
                pytest.fail(f"Request timeout with SQL injection payload: {payload}")
            except Exception as e:
                print(f"SQL injection test error with payload '{payload}': {e}")
    
    def test_xss_protection(self):
        """Test protection against Cross-Site Scripting (XSS) attacks"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>"
        ]
        
        for payload in xss_payloads:
            try:
                response = requests.post(
                    f"{self.api_base}/news-explorer",
                    json={"query": payload},
                    timeout=10
                )
                
                if response.status_code == 200:
                    response_text = response.text
                    
                    # Check if payload is reflected without proper escaping
                    assert payload not in response_text, f"Potential XSS vulnerability: payload reflected unescaped: {payload}"
                    
                    # Check for script tags in response
                    assert "<script" not in response_text.lower(), f"Script tags found in response with payload: {payload}"
                
            except Exception as e:
                print(f"XSS test error with payload '{payload}': {e}")
    
    def test_rate_limiting(self):
        """Test API rate limiting protection"""
        rapid_requests = 100
        success_count = 0
        rate_limited_count = 0
        
        # Make rapid requests to test rate limiting
        for i in range(rapid_requests):
            try:
                response = requests.get(f"{self.api_base}/news", timeout=5)
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:  # Too Many Requests
                    rate_limited_count += 1
                
            except Exception as e:
                print(f"Rate limiting test request {i} failed: {e}")
        
        # Should have some rate limiting in place
        total_processed = success_count + rate_limited_count
        if total_processed > 0:
            rate_limit_percentage = rate_limited_count / total_processed
            
            # Expect some rate limiting for rapid requests
            assert rate_limit_percentage > 0.1, f"Rate limiting may be insufficient: {rate_limit_percentage:.2%} requests limited"
        
        print(f"Rate limiting test: {success_count} successful, {rate_limited_count} rate limited out of {rapid_requests} requests")

if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no",
        "--html=reports/test_report.html",
        "--self-contained-html"
    ])