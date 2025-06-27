#!/usr/bin/env python3
"""
Validation Script for RSE Summarization Engineer

This script validates the correct implementation and functionality of the
RSE Summarization Engineer bot, including data models, configuration loading,
and core processing capabilities.
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from rse_summarization_engineer import (
            DomainCategory,
            RSEArticle,
            SummaryResult,
            ProcessingMetrics,
            RSESummarizationEngineer
        )
        print("✅ All imports successful")
        return True, {
            'DomainCategory': DomainCategory,
            'RSEArticle': RSEArticle,
            'SummaryResult': SummaryResult,
            'ProcessingMetrics': ProcessingMetrics,
            'RSESummarizationEngineer': RSESummarizationEngineer
        }
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, None

def test_domain_category(classes):
    """Test DomainCategory enum functionality"""
    print("\nTesting DomainCategory...")
    
    try:
        DomainCategory = classes['DomainCategory']
        
        # Test enum values
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
            assert hasattr(DomainCategory, domain), f"Missing domain: {domain}"
        
        # Test string representation
        assert str(DomainCategory.SOFTWARE_ENGINEERING) == "software_engineering"
        assert str(DomainCategory.MACHINE_LEARNING) == "machine_learning"
        
        print("✅ DomainCategory tests passed")
        return True
    except Exception as e:
        print(f"❌ DomainCategory test failed: {e}")
        import traceback
        import sys
        traceback.print_exc()
        sys.stdout.flush()
        print(f"\nDETAILED ERROR: {str(e)}")
        print(f"ERROR TYPE: {type(e).__name__}")
        sys.stdout.flush()
        return False

def test_rse_article(classes):
    """Test RSEArticle data model"""
    print("\nTesting RSEArticle...")
    
    try:
        RSEArticle = classes['RSEArticle']
        
        # Test article creation
        article_data = {
            "title": "Test RSE Article",
            "content": "This is a test article about research software engineering practices and methodologies.",
            "url": "https://example.com/test-article",
            "date": "2024-01-15T10:30:00Z",
            "source": "Test Journal",
            "author": "Dr. Test Author"
        }
        
        article = RSEArticle(**article_data)
        
        # Validate properties
        assert article.title == "Test RSE Article"
        assert article.source == "Test Journal"
        assert article.author == "Dr. Test Author"
        assert "research software engineering" in article.content
        
        # Test to_dict method if available
        if hasattr(article, 'to_dict'):
            article_dict = article.to_dict()
            assert isinstance(article_dict, dict)
            assert article_dict['title'] == article_data['title']
        
        print("✅ RSEArticle tests passed")
        return True
    except Exception as e:
        print(f"❌ RSEArticle test failed: {e}")
        return False

def test_summary_result(classes):
    """Test SummaryResult data model"""
    print("\nTesting SummaryResult...")
    
    try:
        SummaryResult = classes['SummaryResult']
        DomainCategory = classes['DomainCategory']
        
        # Test summary result creation
        summary_data = {
            "title": "Test Summary",
            "date": "2024-01-15T10:30:00Z",
            "summary": "This is a test summary of an RSE article. It demonstrates the summarization capabilities. The content focuses on software engineering practices.",
            "domain": DomainCategory.SOFTWARE_ENGINEERING,
            "confidence_score": 0.85,
            "key_phrases": ["software engineering", "testing", "research software"],
            "processing_time": 2.5,
            "model_used": "gpt-3.5-turbo",
            "word_count": 25
        }
        
        result = SummaryResult(**summary_data)
        
        # Validate properties
        assert result.title == "Test Summary"
        assert result.confidence_score == 0.85
        assert len(result.key_phrases) == 3
        assert result.word_count == 25
        assert result.processing_time == 2.5
        
        # Test confidence score validation
        try:
            invalid_data = summary_data.copy()
            invalid_data['confidence_score'] = 1.5  # Invalid: > 1.0
            SummaryResult(**invalid_data)
            print("⚠️  Warning: Confidence score validation not implemented")
        except ValueError:
            pass  # Expected behavior
        
        # Test JSON serialization if available
        if hasattr(result, 'to_json'):
            json_str = result.to_json()
            assert isinstance(json_str, str)
            parsed = json.loads(json_str)
            assert parsed['title'] == "Test Summary"
        
        print("✅ SummaryResult tests passed")
        return True
    except Exception as e:
        print(f"❌ SummaryResult test failed: {e}")
        return False

def test_processing_metrics(classes):
    """Test ProcessingMetrics data model"""
    print("\nTesting ProcessingMetrics...")
    
    try:
        ProcessingMetrics = classes['ProcessingMetrics']
        
        # Test metrics creation
        metrics = ProcessingMetrics()
        
        # Validate initial state
        assert metrics.total_articles_processed == 0
        assert metrics.successful_summaries == 0
        assert metrics.failed_summaries == 0
        assert metrics.average_processing_time == 0.0
        
        # Test recording success
        if hasattr(metrics, 'record_success'):
            metrics.record_success(2.5)
            assert metrics.total_articles_processed == 1
            assert metrics.successful_summaries == 1
            assert metrics.average_processing_time == 2.5
        
        # Test recording failure
        if hasattr(metrics, 'record_failure'):
            metrics.record_failure()
            assert metrics.total_articles_processed == 2
            assert metrics.failed_summaries == 1
        
        # Test success rate calculation
        if hasattr(metrics, 'success_rate'):
            expected_rate = 1 / 2  # 1 success out of 2 total
            assert abs(metrics.success_rate - expected_rate) < 0.01
        
        print("✅ ProcessingMetrics tests passed")
        return True
    except Exception as e:
        print(f"❌ ProcessingMetrics test failed: {e}")
        return False

def test_rse_summarization_engineer(classes):
    """Test RSESummarizationEngineer main class"""
    print("\nTesting RSESummarizationEngineer...")
    
    try:
        RSESummarizationEngineer = classes['RSESummarizationEngineer']
        ProcessingMetrics = classes['ProcessingMetrics']
        
        # Create a temporary config file
        temp_config = {
            "id": "rse-summary-bot-v1",
            "name": "RSE Summarization Engineer",
            "configuration": {
                "base_url": "http://localhost:3000",
                "cache_dir": "./test_cache",
                "max_summary_length": 300,
                "confidence_threshold": 0.7,
                "models": {
                    "primary": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "max_tokens": 256
                },
                "rate_limits": {
                    "requests_per_minute": 60,
                    "concurrent_requests": 5
                }
            },
            "domain_categories": {
                "software_engineering": {
                    "description": "Software engineering practices",
                    "keywords": ["software", "engineering", "testing", "development"],
                    "weight": 1.0
                },
                "machine_learning": {
                    "description": "Machine learning and AI",
                    "keywords": ["machine learning", "ai", "neural networks", "deep learning"],
                    "weight": 1.0
                }
            },
            "prompt_template": {
                "system": "You are an RSE summarization expert.",
                "summarization_prompt": "Summarize the following RSE article: {{content}}",
                "classification_prompt": "Classify this article: {{content}}"
            },
            "authentication": {
                "openai_api_key_env": "OPENAI_API_KEY"
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(temp_config, f)
            config_path = f.name
        
        try:
            # Test engineer initialization
            engineer = RSESummarizationEngineer(config_path=config_path)
            
            # Validate initialization
            assert engineer.config is not None
            assert isinstance(engineer.metrics, ProcessingMetrics)
            assert engineer.config['configuration']['base_url'] == "http://localhost:3000"
            
            # Test default initialization
            engineer_default = RSESummarizationEngineer()
            assert engineer_default.config is not None
            
            # Test health check
            if hasattr(engineer, 'health_check'):
                health = engineer.health_check()
                assert isinstance(health, dict)
                assert 'status' in health
                assert 'config_loaded' in health
            
            # Test metrics retrieval
            if hasattr(engineer, 'get_metrics'):
                metrics = engineer.get_metrics()
                assert isinstance(metrics, dict)
                assert 'total_articles_processed' in metrics
            
            # Test domain classification
            if hasattr(engineer, '_classify_domain'):
                content = "This article discusses software engineering best practices and testing methodologies."
                domain, confidence = engineer._classify_domain(content)
                assert isinstance(domain, str)
                assert isinstance(confidence, float)
                assert 0.0 <= confidence <= 1.0
            
            # Test key phrase extraction
            if hasattr(engineer, '_extract_key_phrases'):
                content = "Research software engineering practices improve code quality."
                phrases = engineer._extract_key_phrases(content)
                assert isinstance(phrases, list)
                assert len(phrases) > 0
            
            # Test cache key generation
            if hasattr(engineer, '_generate_cache_key'):
                content = "Test content for caching"
                cache_key = engineer._generate_cache_key(content)
                assert isinstance(cache_key, str)
                assert len(cache_key) > 0
                
                # Same content should generate same key
                cache_key2 = engineer._generate_cache_key(content)
                assert cache_key == cache_key2
            
            print("✅ RSESummarizationEngineer tests passed")
            return True
            
        finally:
            # Clean up temporary config file
            os.unlink(config_path)
            
    except Exception as e:
        print(f"❌ RSESummarizationEngineer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_loading():
    """Test configuration file loading"""
    print("\nTesting configuration loading...")
    
    try:
        # Check if default config exists
        config_path = Path("agents/config/rse_summarization_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = [
                'id',
                'name',
                'configuration',
                'domain_categories',
                'prompt_template'
            ]
            
            for section in required_sections:
                assert section in config, f"Missing config section: {section}"
            
            # Validate configuration structure
            assert 'models' in config['configuration']
            assert 'primary' in config['configuration']['models']
            
            # Validate domain categories
            assert len(config['domain_categories']) > 0
            for domain, details in config['domain_categories'].items():
                assert 'keywords' in details
                assert 'weight' in details
                assert isinstance(details['keywords'], list)
            
            print("✅ Configuration loading tests passed")
            return True
        else:
            print("⚠️  Default configuration file not found, skipping config tests")
            return True
            
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("RSE Summarization Engineer Validation")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Loading", test_configuration_loading)
    ]
    
    # Run import test first
    success, classes = test_imports()
    if not success:
        print("\n❌ Critical failure: Cannot import required modules")
        print("Please ensure rse_summarization_engineer.py is in the current directory")
        return False
    
    # Run configuration test
    config_success = test_configuration_loading()
    
    # Run component tests if imports succeeded
    component_tests = [
        ("DomainCategory", lambda: test_domain_category(classes)),
        ("RSEArticle", lambda: test_rse_article(classes)),
        ("SummaryResult", lambda: test_summary_result(classes)),
        ("ProcessingMetrics", lambda: test_processing_metrics(classes)),
        ("RSESummarizationEngineer", lambda: test_rse_summarization_engineer(classes))
    ]
    
    results = []
    for test_name, test_func in component_tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Configuration: {'✅ Loaded' if config_success else '❌ Failed'}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if passed == total and config_success:
        print("\n🎉 All tests passed! RSE Summarization Engineer is ready for use.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python agents/rse_summarization_engineer.py --health-check")
        print("3. Test with sample article: python agents/rse_summarization_engineer.py --input sample.json")
        print("4. Integrate with AI News Dashboard")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)