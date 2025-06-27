#!/usr/bin/env python3
"""
Validation script for RSE GitHub Integrator
Tests all components and functionality of the RSE GitHub Integrator bot.

Usage:
    python validate_rse_github_integrator.py [--verbose]
"""

import sys
import os
import json
import yaml
import asyncio
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

try:
    from rse_github_integrator import (
        RSEGitHubIntegrator,
        ContentType,
        PRStatus,
        RSEContent,
        GitHubConfig
    )
except ImportError as e:
    print(f"❌ Failed to import RSE GitHub Integrator: {e}")
    sys.exit(1)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class RSEGitHubIntegratorValidator:
    """Validator for RSE GitHub Integrator functionality"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = []
        self.config_path = Path(__file__).parent / "agents" / "config" / "rse_github_config.yaml"
        self.schemas_dir = Path(__file__).parent / "schemas"
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def test_imports(self) -> bool:
        """Test that all required imports work correctly"""
        try:
            # Test enum imports
            assert hasattr(ContentType, 'JSON')
            assert hasattr(ContentType, 'MARKDOWN')
            assert hasattr(ContentType, 'YAML')
            
            assert hasattr(PRStatus, 'OPEN')
            assert hasattr(PRStatus, 'CLOSED')
            assert hasattr(PRStatus, 'MERGED')
            
            # Test class availability
            assert RSEContent is not None
            assert GitHubConfig is not None
            assert RSEGitHubIntegrator is not None
            
            self.log("All imports successful")
            return True
            
        except (ImportError, AssertionError, AttributeError) as e:
            self.log(f"Import test failed: {e}", "ERROR")
            return False
    
    def test_config_loading(self) -> bool:
        """Test configuration file loading"""
        try:
            if not self.config_path.exists():
                raise ValidationError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate required configuration sections
            required_sections = [
                'integrator', 'github', 'news_directory', 'validation',
                'ci_integration', 'webhook', 'security', 'logging'
            ]
            
            for section in required_sections:
                if section not in config:
                    raise ValidationError(f"Missing required config section: {section}")
            
            # Validate specific config values
            assert config['integrator']['id'] == 'rse-github-bot-v1'
            assert config['github']['base_branch'] == 'main'
            assert config['news_directory'] == './news'
            assert isinstance(config['validation']['enabled'], bool)
            
            self.log("Configuration loading successful")
            return True
            
        except (yaml.YAMLError, ValidationError, AssertionError, FileNotFoundError) as e:
            self.log(f"Configuration test failed: {e}", "ERROR")
            return False
    
    def test_schema_files(self) -> bool:
        """Test that all schema files exist and are valid JSON"""
        try:
            schema_files = [
                'rse_content.json',
                'rse_markdown.json',
                'rse_yaml.json'
            ]
            
            for schema_file in schema_files:
                schema_path = self.schemas_dir / schema_file
                
                if not schema_path.exists():
                    raise ValidationError(f"Schema file not found: {schema_path}")
                
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                # Validate basic schema structure
                assert '$schema' in schema
                assert 'title' in schema
                assert 'type' in schema
                assert 'properties' in schema
                
                self.log(f"Schema file {schema_file} is valid")
            
            self.log("All schema files validated successfully")
            return True
            
        except (json.JSONDecodeError, ValidationError, AssertionError, FileNotFoundError) as e:
            self.log(f"Schema validation failed: {e}", "ERROR")
            return False
    
    def test_rse_content_creation(self) -> bool:
        """Test RSEContent object creation and validation"""
        try:
            # Create test content
            test_content = RSEContent(
                title="Test RSE Content",
                summary="This is a comprehensive test summary for validation purposes that meets the minimum character requirement for proper content validation.",
                domain="Machine Learning",
                keywords=["test", "validation", "rse"],
                content="# Test Content\n\nThis is test content for validation.",
                author="Test Author",
                content_type=ContentType.MARKDOWN
            )
            
            # Validate content attributes
            assert test_content.title == "Test RSE Content"
            assert test_content.domain == "Machine Learning"
            assert len(test_content.keywords) == 3
            assert test_content.content_type == ContentType.MARKDOWN
            
            # Test content validation
            validation_result = test_content.validate()
            assert validation_result['valid'] is True
            
            # Test content serialization
            content_dict = test_content.to_dict()
            assert isinstance(content_dict, dict)
            assert 'title' in content_dict
            assert 'metadata' in content_dict
            
            self.log("RSEContent creation and validation successful")
            return True
            
        except (AssertionError, Exception) as e:
            self.log(f"RSEContent test failed: {e}", "ERROR")
            return False
    
    def test_github_config_creation(self) -> bool:
        """Test GitHubConfig object creation"""
        try:
            # Create test GitHub config
            config = GitHubConfig(
                owner="test-owner",
                repo="test-repo",
                token="test-token",
                base_branch="main"
            )
            
            # Validate config attributes
            assert config.owner == "test-owner"
            assert config.repo == "test-repo"
            assert config.token == "test-token"
            assert config.base_branch == "main"
            
            # Test config validation
            assert config.validate() is True
            
            self.log("GitHubConfig creation successful")
            return True
            
        except (AssertionError, Exception) as e:
            self.log(f"GitHubConfig test failed: {e}", "ERROR")
            return False
    
    def test_integrator_initialization(self) -> bool:
        """Test RSEGitHubIntegrator initialization"""
        try:
            # Create test configuration
            github_config = GitHubConfig(
                owner="test-owner",
                repo="test-repo",
                token="test-token",
                base_branch="main"
            )
            
            # Initialize integrator
            integrator = RSEGitHubIntegrator(
                github_config=github_config,
                config_path=str(self.config_path)
            )
            
            # Validate integrator attributes
            assert integrator.github_config == github_config
            assert integrator.config_path == str(self.config_path)
            assert hasattr(integrator, 'config')
            assert hasattr(integrator, 'logger')
            
            self.log("RSEGitHubIntegrator initialization successful")
            return True
            
        except (AssertionError, Exception) as e:
            self.log(f"Integrator initialization test failed: {e}", "ERROR")
            return False
    
    def test_filename_generation(self) -> bool:
        """Test filename generation functionality"""
        try:
            # Create test integrator
            github_config = GitHubConfig(
                owner="test-owner",
                repo="test-repo",
                token="test-token",
                base_branch="main"
            )
            
            integrator = RSEGitHubIntegrator(
                github_config=github_config,
                config_path=str(self.config_path)
            )
            
            # Create test content
            test_content = RSEContent(
                title="Test Content for Filename Generation",
                summary="Test summary",
                domain="Software Engineering",
                keywords=["test"],
                content="Test content",
                author="Test Author",
                content_type=ContentType.JSON
            )
            
            # Test filename generation
            filename = integrator.generate_filename(test_content)
            
            # Validate filename format
            assert filename.startswith('rse_')
            assert filename.endswith('.json')
            assert 'software_engineering' in filename.lower() or 'software-engineering' in filename.lower()
            
            # Test different content types
            test_content.content_type = ContentType.MARKDOWN
            md_filename = integrator.generate_filename(test_content)
            assert md_filename.endswith('.md')
            
            test_content.content_type = ContentType.YAML
            yaml_filename = integrator.generate_filename(test_content)
            assert yaml_filename.endswith('.yaml')
            
            self.log("Filename generation successful")
            return True
            
        except (AssertionError, Exception) as e:
            self.log(f"Filename generation test failed: {e}", "ERROR")
            return False
    
    def test_content_formatting(self) -> bool:
        """Test content formatting for different types"""
        try:
            # Create test integrator
            github_config = GitHubConfig(
                owner="test-owner",
                repo="test-repo",
                token="test-token",
                base_branch="main"
            )
            
            integrator = RSEGitHubIntegrator(
                github_config=github_config,
                config_path=str(self.config_path)
            )
            
            # Create test content
            test_content = RSEContent(
                title="Test Content Formatting",
                summary="Test summary for content formatting that meets the minimum character requirement for proper validation",
                domain="Data Science",
                keywords=["test", "formatting"],
                content="# Test Content\n\nThis is test content.",
                author="Test Author",
                content_type=ContentType.JSON
            )
            
            # Test JSON formatting
            json_content = integrator.format_content(test_content)
            json_data = json.loads(json_content)
            assert 'metadata' in json_data
            assert 'title' in json_data
            
            # Test Markdown formatting
            test_content.content_type = ContentType.MARKDOWN
            md_content = integrator.format_content(test_content)
            assert md_content.startswith('---')
            assert 'title:' in md_content
            assert '---' in md_content
            
            # Test YAML formatting
            test_content.content_type = ContentType.YAML
            yaml_content = integrator.format_content(test_content)
            yaml_data = yaml.safe_load(yaml_content)
            assert 'title' in yaml_data
            assert 'summary' in yaml_data
            assert 'domain' in yaml_data
            
            self.log("Content formatting successful")
            return True
            
        except (AssertionError, json.JSONDecodeError, yaml.YAMLError, Exception) as e:
            self.log(f"Content formatting test failed: {e}", "ERROR")
            return False
    
    def test_health_check(self) -> bool:
        """Test health check functionality"""
        try:
            # Create test integrator
            github_config = GitHubConfig(
                owner="test-owner",
                repo="test-repo",
                token="test-token",
                base_branch="main"
            )
            
            integrator = RSEGitHubIntegrator(
                github_config=github_config,
                config_path=str(self.config_path)
            )
            
            # Test health check
            health_status = integrator.health_check()
            
            # Validate health check response
            assert isinstance(health_status, dict)
            assert 'status' in health_status
            assert 'timestamp' in health_status
            assert 'checks' in health_status
            
            # Validate individual checks
            checks = health_status['checks']
            assert 'config_loaded' in checks
            assert 'github_config' in checks
            assert 'schemas_available' in checks
            
            self.log("Health check successful")
            return True
            
        except (AssertionError, Exception) as e:
            self.log(f"Health check test failed: {e}", "ERROR")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all validation tests"""
        tests = [
            ("Imports", self.test_imports),
            ("Configuration Loading", self.test_config_loading),
            ("Schema Files", self.test_schema_files),
            ("RSEContent Creation", self.test_rse_content_creation),
            ("GitHubConfig Creation", self.test_github_config_creation),
            ("Integrator Initialization", self.test_integrator_initialization),
            ("Filename Generation", self.test_filename_generation),
            ("Content Formatting", self.test_content_formatting),
            ("Health Check", self.test_health_check)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        print("Starting RSE GitHub Integrator validation...\n")
        
        for test_name, test_func in tests:
            print(f"Testing {test_name}...", end=" ")
            sys.stdout.flush()
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    print("PASSED")
                    passed += 1
                else:
                    print("FAILED")
                    
            except Exception as e:
                print(f"FAILED - {e}")
                results[test_name] = False
                self.log(f"Unexpected error in {test_name}: {e}", "ERROR")
        
        print(f"\nTest Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed! RSE GitHub Integrator is ready for use.")
        else:
            print("Some tests failed. Please review the issues above.")
            
        return results

def main():
    """Main function to run validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate RSE GitHub Integrator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    validator = RSEGitHubIntegratorValidator(verbose=args.verbose)
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()