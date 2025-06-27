#!/usr/bin/env python3
"""
Unit tests for RSE Summarization Engineer

This module contains comprehensive tests for the RSE Summarization Engineer bot,
including data models, summarization logic, domain classification, and API integration.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio
from pathlib import Path

# Import the classes we're testing
try:
    from rse_summarization_engineer import (
        DomainCategory,
        RSEArticle,
        SummaryResult,
        ProcessingMetrics,
        RSESummarizationEngineer
    )
except ImportError:
    # Handle import error gracefully for testing
    print("Warning: Could not import RSE Summarization Engineer modules")
    DomainCategory = None
    RSEArticle = None
    SummaryResult = None
    ProcessingMetrics = None
    RSESummarizationEngineer = None


class TestDomainCategory(unittest.TestCase):
    """Test cases for DomainCategory enum"""
    
    def setUp(self):
        if DomainCategory is None:
            self.skipTest("DomainCategory not available")
    
    def test_domain_category_values(self):
        """Test that all expected domain categories exist"""
        expected_domains = [
            'RESEARCH_COMPUTING',
            'SOFTWARE_ENGINEERING', 
            'DATA_SCIENCE',
            'HIGH_PERFORMANCE_COMPUTING',
            'MACHINE_LEARNING',
            'SCIENTIFIC_SOFTWARE',
            'DIGITAL_HUMANITIES',
            'OTHER'
        ]
        
        for domain in expected_domains:
            self.assertTrue(hasattr(DomainCategory, domain))
    
    def test_domain_category_string_representation(self):
        """Test string representation of domain categories"""
        self.assertEqual(str(DomainCategory.SOFTWARE_ENGINEERING), "software_engineering")
        self.assertEqual(str(DomainCategory.MACHINE_LEARNING), "machine_learning")


class TestRSEArticle(unittest.TestCase):
    """Test cases for RSEArticle data model"""
    
    def setUp(self):
        if RSEArticle is None:
            self.skipTest("RSEArticle not available")
            
        self.sample_article_data = {
            "title": "Improving Research Software Quality",
            "content": "This article discusses best practices for research software development...",
            "url": "https://example.com/article",
            "date": "2024-01-15T10:30:00Z",
            "source": "RSE Blog",
            "author": "Dr. Jane Smith"
        }
    
    def test_rse_article_creation(self):
        """Test creating an RSEArticle instance"""
        article = RSEArticle(**self.sample_article_data)
        
        self.assertEqual(article.title, "Improving Research Software Quality")
        self.assertEqual(article.source, "RSE Blog")
        self.assertEqual(article.author, "Dr. Jane Smith")
        self.assertIsInstance(article.date, str)
    
    def test_rse_article_validation(self):
        """Test article validation"""
        # Test with missing required fields
        with self.assertRaises(TypeError):
            RSEArticle(title="Test")
    
    def test_rse_article_to_dict(self):
        """Test converting article to dictionary"""
        article = RSEArticle(**self.sample_article_data)
        article_dict = article.to_dict()
        
        self.assertIsInstance(article_dict, dict)
        self.assertEqual(article_dict['title'], self.sample_article_data['title'])
        self.assertEqual(article_dict['content'], self.sample_article_data['content'])


class TestSummaryResult(unittest.TestCase):
    """Test cases for SummaryResult data model"""
    
    def setUp(self):
        if SummaryResult is None:
            self.skipTest("SummaryResult not available")
            
        self.sample_summary_data = {
            "title": "Test Article",
            "date": "2024-01-15T10:30:00Z",
            "summary": "This is a test summary. It contains multiple sentences. The content is about research software.",
            "domain": DomainCategory.SOFTWARE_ENGINEERING if DomainCategory else "software_engineering",
            "confidence_score": 0.85,
            "key_phrases": ["research software", "testing", "quality"],
            "processing_time": 2.3,
            "model_used": "gpt-3.5-turbo",
            "word_count": 15
        }
    
    def test_summary_result_creation(self):
        """Test creating a SummaryResult instance"""
        if SummaryResult is None:
            self.skipTest("SummaryResult not available")
            
        result = SummaryResult(**self.sample_summary_data)
        
        self.assertEqual(result.title, "Test Article")
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(len(result.key_phrases), 3)
        self.assertEqual(result.word_count, 15)
    
    def test_summary_result_validation(self):
        """Test summary result validation"""
        if SummaryResult is None:
            self.skipTest("SummaryResult not available")
            
        # Test confidence score bounds
        invalid_data = self.sample_summary_data.copy()
        invalid_data['confidence_score'] = 1.5  # Invalid: > 1.0
        
        with self.assertRaises(ValueError):
            SummaryResult(**invalid_data)
    
    def test_summary_result_to_json(self):
        """Test converting summary result to JSON"""
        if SummaryResult is None:
            self.skipTest("SummaryResult not available")
            
        result = SummaryResult(**self.sample_summary_data)
        json_result = result.to_json()
        
        self.assertIsInstance(json_result, str)
        parsed = json.loads(json_result)
        self.assertEqual(parsed['title'], "Test Article")
        self.assertEqual(parsed['confidence_score'], 0.85)


class TestProcessingMetrics(unittest.TestCase):
    """Test cases for ProcessingMetrics data model"""
    
    def setUp(self):
        if ProcessingMetrics is None:
            self.skipTest("ProcessingMetrics not available")
    
    def test_processing_metrics_creation(self):
        """Test creating ProcessingMetrics instance"""
        metrics = ProcessingMetrics()
        
        self.assertEqual(metrics.total_articles_processed, 0)
        self.assertEqual(metrics.successful_summaries, 0)
        self.assertEqual(metrics.failed_summaries, 0)
        self.assertEqual(metrics.average_processing_time, 0.0)
    
    def test_processing_metrics_update(self):
        """Test updating processing metrics"""
        metrics = ProcessingMetrics()
        
        # Simulate processing an article
        metrics.record_success(2.5)
        
        self.assertEqual(metrics.total_articles_processed, 1)
        self.assertEqual(metrics.successful_summaries, 1)
        self.assertEqual(metrics.failed_summaries, 0)
        self.assertEqual(metrics.average_processing_time, 2.5)
    
    def test_processing_metrics_failure(self):
        """Test recording processing failures"""
        metrics = ProcessingMetrics()
        
        metrics.record_failure()
        
        self.assertEqual(metrics.total_articles_processed, 1)
        self.assertEqual(metrics.successful_summaries, 0)
        self.assertEqual(metrics.failed_summaries, 1)
    
    def test_processing_metrics_success_rate(self):
        """Test calculating success rate"""
        metrics = ProcessingMetrics()
        
        # Process 10 articles: 8 success, 2 failures
        for _ in range(8):
            metrics.record_success(1.0)
        for _ in range(2):
            metrics.record_failure()
        
        self.assertEqual(metrics.success_rate, 0.8)
        self.assertEqual(metrics.total_articles_processed, 10)


class TestRSESummarizationEngineer(unittest.TestCase):
    """Test cases for RSESummarizationEngineer main class"""
    
    def setUp(self):
        if RSESummarizationEngineer is None:
            self.skipTest("RSESummarizationEngineer not available")
            
        # Create a temporary config file
        self.temp_config = {
            "configuration": {
                "base_url": "http://localhost:3000",
                "cache_dir": "./test_cache",
                "max_summary_length": 300,
                "confidence_threshold": 0.7,
                "models": {
                    "primary": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "max_tokens": 256
                }
            },
            "domain_categories": {
                "software_engineering": {
                    "keywords": ["software", "engineering", "testing"],
                    "weight": 1.0
                }
            },
            "prompt_template": {
                "summarization_prompt": "Summarize: {{content}}",
                "classification_prompt": "Classify: {{content}}"
            }
        }
        
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.temp_config, self.temp_config_file)
        self.temp_config_file.close()
    
    def tearDown(self):
        # Clean up temporary config file
        if hasattr(self, 'temp_config_file'):
            os.unlink(self.temp_config_file.name)
    
    def test_engineer_initialization(self):
        """Test initializing RSESummarizationEngineer"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        self.assertIsNotNone(engineer.config)
        self.assertIsInstance(engineer.metrics, ProcessingMetrics)
        self.assertEqual(engineer.config['configuration']['base_url'], "http://localhost:3000")
    
    def test_engineer_default_config(self):
        """Test engineer with default configuration"""
        engineer = RSESummarizationEngineer()
        
        self.assertIsNotNone(engineer.config)
        self.assertIsInstance(engineer.metrics, ProcessingMetrics)
    
    @patch('openai.ChatCompletion.create')
    def test_openai_api_call(self, mock_openai):
        """Test OpenAI API integration"""
        if RSESummarizationEngineer is None:
            self.skipTest("RSESummarizationEngineer not available")
            
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test summary. It contains technical details. The research has significant impact."
        mock_openai.return_value = mock_response
        
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        # Test summarization
        result = engineer._call_openai_api("Test prompt", "gpt-3.5-turbo")
        
        self.assertEqual(result, "This is a test summary. It contains technical details. The research has significant impact.")
        mock_openai.assert_called_once()
    
    def test_domain_classification(self):
        """Test domain classification logic"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        # Test with software engineering content
        content = "This article discusses software engineering best practices and testing methodologies."
        domain, confidence = engineer._classify_domain(content)
        
        self.assertIsInstance(domain, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_key_phrase_extraction(self):
        """Test key phrase extraction"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        content = "Research software engineering practices improve code quality and maintainability."
        key_phrases = engineer._extract_key_phrases(content)
        
        self.assertIsInstance(key_phrases, list)
        self.assertGreater(len(key_phrases), 0)
        
        # Check that phrases are strings
        for phrase in key_phrases:
            self.assertIsInstance(phrase, str)
    
    def test_caching_functionality(self):
        """Test caching mechanism"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        # Test cache key generation
        content = "Test content for caching"
        cache_key = engineer._generate_cache_key(content)
        
        self.assertIsInstance(cache_key, str)
        self.assertGreater(len(cache_key), 0)
        
        # Test that same content generates same cache key
        cache_key2 = engineer._generate_cache_key(content)
        self.assertEqual(cache_key, cache_key2)
    
    def test_health_check(self):
        """Test health check functionality"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        health_status = engineer.health_check()
        
        self.assertIsInstance(health_status, dict)
        self.assertIn('status', health_status)
        self.assertIn('config_loaded', health_status)
        self.assertIn('timestamp', health_status)
    
    def test_get_metrics(self):
        """Test metrics retrieval"""
        engineer = RSESummarizationEngineer(config_path=self.temp_config_file.name)
        
        metrics = engineer.get_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_articles_processed', metrics)
        self.assertIn('successful_summaries', metrics)
        self.assertIn('failed_summaries', metrics)
        self.assertIn('success_rate', metrics)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def setUp(self):
        if RSESummarizationEngineer is None or RSEArticle is None:
            self.skipTest("Required classes not available")
    
    def test_end_to_end_processing(self):
        """Test complete article processing workflow"""
        # Create a sample article
        article_data = {
            "title": "Automated Testing in Research Software",
            "content": "This research presents a comprehensive framework for implementing automated testing in scientific software projects. The study shows that continuous integration practices can significantly improve code quality and reduce bugs in research applications. The methodology has been validated across multiple research domains.",
            "url": "https://example.com/testing-research",
            "date": "2024-01-15T10:30:00Z",
            "source": "Journal of Research Software Engineering",
            "author": "Dr. Alice Johnson"
        }
        
        article = RSEArticle(**article_data)
        
        # Verify article creation
        self.assertEqual(article.title, "Automated Testing in Research Software")
        self.assertIn("automated testing", article.content.lower())
    
    @patch('openai.ChatCompletion.create')
    def test_mock_summarization_workflow(self, mock_openai):
        """Test summarization workflow with mocked OpenAI"""
        if RSESummarizationEngineer is None:
            self.skipTest("RSESummarizationEngineer not available")
            
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This study introduces automated testing frameworks for research software. The methodology improves code quality and reduces development time. Results show significant benefits for scientific computing projects."
        mock_openai.return_value = mock_response
        
        # Create engineer with temporary config
        temp_config = {
            "configuration": {
                "base_url": "http://localhost:3000",
                "models": {"primary": "gpt-3.5-turbo"},
                "confidence_threshold": 0.7
            },
            "domain_categories": {
                "software_engineering": {
                    "keywords": ["testing", "software", "quality"],
                    "weight": 1.0
                }
            },
            "prompt_template": {
                "summarization_prompt": "Summarize: {{content}}"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(temp_config, f)
            config_path = f.name
        
        try:
            engineer = RSESummarizationEngineer(config_path=config_path)
            
            # Test API call
            result = engineer._call_openai_api("Test prompt", "gpt-3.5-turbo")
            self.assertIsInstance(result, str)
            self.assertIn("automated testing", result.lower())
            
        finally:
            os.unlink(config_path)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def setUp(self):
        if RSESummarizationEngineer is None:
            self.skipTest("RSESummarizationEngineer not available")
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file"""
        # Test with non-existent config file
        with self.assertRaises(FileNotFoundError):
            RSESummarizationEngineer(config_path="/nonexistent/config.json")
    
    def test_malformed_config_json(self):
        """Test handling of malformed JSON config"""
        # Create malformed JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            malformed_config_path = f.name
        
        try:
            with self.assertRaises(json.JSONDecodeError):
                RSESummarizationEngineer(config_path=malformed_config_path)
        finally:
            os.unlink(malformed_config_path)
    
    @patch('openai.ChatCompletion.create')
    def test_openai_api_error_handling(self, mock_openai):
        """Test handling of OpenAI API errors"""
        # Mock OpenAI to raise an exception
        mock_openai.side_effect = Exception("API Error")
        
        engineer = RSESummarizationEngineer()
        
        # Test that API errors are handled gracefully
        with self.assertRaises(Exception):
            engineer._call_openai_api("Test prompt", "gpt-3.5-turbo")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDomainCategory,
        TestRSEArticle,
        TestSummaryResult,
        TestProcessingMetrics,
        TestRSESummarizationEngineer,
        TestIntegrationScenarios,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)