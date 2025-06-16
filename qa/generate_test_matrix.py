#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - Test Matrix Generator

Generates dynamic test matrices for CI/CD pipeline based on:
- Test suite selection
- Environment configuration
- Available resources
- Historical test data
- Risk assessment

Author: Dr. Orion "TestMaster" Vanguard
Version: 1.0.0
License: MIT
"""

import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os

@dataclass
class TestConfiguration:
    """Test configuration for matrix generation"""
    test_type: str
    priority: int
    estimated_duration: int  # minutes
    resource_requirements: Dict[str, Any]
    dependencies: List[str]
    environments: List[str]
    browsers: List[str] = None
    personas: List[str] = None
    models: List[str] = None
    experiments: List[str] = None
    agents: List[str] = None

class SuperhumanTestMatrixGenerator:
    """Intelligent test matrix generator for superhuman QA"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.test_configurations = self._initialize_test_configurations()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available"""
        return {
            "persona_testing": {
                "personas": ["casual_reader", "financial_analyst", "mobile_user", "power_user", "accessibility_user"],
                "browsers": ["chrome", "firefox", "edge"]
            },
            "ai_inference_validation": {
                "models": ["gpt-4o", "claude-3.5-sonnet", "local-llm"],
                "test_types": ["summarization", "rag_search", "news_filtering", "sentiment_analysis"]
            },
            "chaos_engineering": {
                "experiments": ["network_latency", "cache_eviction", "api_failure", "database_stress", "memory_pressure"],
                "intensities": [3, 5, 7]
            },
            "multi_agent_pipeline": {
                "agents": ["functional", "performance", "security", "accessibility", "cross_browser", "api", "integration"]
            }
        }
    
    def _initialize_test_configurations(self) -> Dict[str, TestConfiguration]:
        """Initialize test configurations with priorities and requirements"""
        return {
            "persona": TestConfiguration(
                test_type="persona",
                priority=1,
                estimated_duration=15,
                resource_requirements={"cpu": "medium", "memory": "medium", "browser": True},
                dependencies=["application_running"],
                environments=["development", "staging", "production"],
                browsers=self.config.get("persona_testing", {}).get("browsers", ["chrome"]),
                personas=self.config.get("persona_testing", {}).get("personas", ["casual_reader"])
            ),
            "ai_inference": TestConfiguration(
                test_type="ai_inference",
                priority=2,
                estimated_duration=20,
                resource_requirements={"cpu": "high", "memory": "high", "api_keys": True},
                dependencies=["application_running", "api_keys"],
                environments=["development", "staging", "production"],
                models=self.config.get("ai_inference_validation", {}).get("models", ["gpt-4o"])
            ),
            "chaos": TestConfiguration(
                test_type="chaos",
                priority=3,
                estimated_duration=30,
                resource_requirements={"cpu": "high", "memory": "high", "docker": True},
                dependencies=["application_running", "monitoring"],
                environments=["staging", "production"],
                experiments=self.config.get("chaos_engineering", {}).get("experiments", ["network_latency"])
            ),
            "neural_cache": TestConfiguration(
                test_type="neural_cache",
                priority=2,
                estimated_duration=25,
                resource_requirements={"cpu": "medium", "memory": "high", "redis": True},
                dependencies=["application_running", "redis"],
                environments=["development", "staging", "production"]
            ),
            "multi_agent": TestConfiguration(
                test_type="multi_agent",
                priority=2,
                estimated_duration=35,
                resource_requirements={"cpu": "high", "memory": "high", "parallel": True},
                dependencies=["application_running"],
                environments=["development", "staging", "production"],
                agents=self.config.get("multi_agent_pipeline", {}).get("agents", ["functional"])
            ),
            "performance": TestConfiguration(
                test_type="performance",
                priority=1,
                estimated_duration=20,
                resource_requirements={"cpu": "high", "memory": "medium", "network": True},
                dependencies=["application_running"],
                environments=["staging", "production"]
            )
        }
    
    def generate_matrix(self, 
                       suite: str = "full", 
                       environment: str = "development",
                       max_duration: int = 120,
                       max_parallel: int = 10,
                       risk_level: str = "medium") -> Dict[str, Any]:
        """Generate test matrix based on parameters"""
        
        # Determine which tests to include
        if suite == "full":
            selected_tests = list(self.test_configurations.keys())
        elif suite == "smoke":
            selected_tests = ["persona", "performance"]
        elif suite == "regression":
            selected_tests = ["persona", "ai_inference", "multi_agent"]
        elif suite == "nightly":
            selected_tests = ["chaos", "neural_cache", "performance"]
        else:
            # Single test type
            selected_tests = [suite] if suite in self.test_configurations else []
        
        # Filter tests by environment compatibility
        compatible_tests = []
        for test_type in selected_tests:
            config = self.test_configurations[test_type]
            if environment in config.environments:
                compatible_tests.append(test_type)
        
        # Apply risk-based filtering
        if risk_level == "low":
            # Exclude chaos engineering in low-risk scenarios
            compatible_tests = [t for t in compatible_tests if t != "chaos"]
        elif risk_level == "high":
            # Include all tests and increase chaos intensity
            pass
        
        # Generate matrix entries
        matrix_entries = []
        total_duration = 0
        
        for test_type in compatible_tests:
            config = self.test_configurations[test_type]
            
            # Check duration constraints
            if total_duration + config.estimated_duration > max_duration:
                continue
            
            # Generate entries based on test type
            if test_type == "persona":
                entries = self._generate_persona_matrix(config, environment)
            elif test_type == "ai_inference":
                entries = self._generate_ai_inference_matrix(config, environment)
            elif test_type == "chaos":
                entries = self._generate_chaos_matrix(config, environment, risk_level)
            elif test_type == "neural_cache":
                entries = self._generate_neural_cache_matrix(config, environment)
            elif test_type == "multi_agent":
                entries = self._generate_multi_agent_matrix(config, environment)
            elif test_type == "performance":
                entries = self._generate_performance_matrix(config, environment)
            else:
                entries = [test_type]
            
            matrix_entries.extend(entries)
            total_duration += config.estimated_duration
            
            # Respect parallel execution limits
            if len(matrix_entries) >= max_parallel:
                break
        
        # Apply intelligent prioritization
        matrix_entries = self._prioritize_matrix_entries(matrix_entries, environment)
        
        # Generate final matrix
        matrix = {
            "include": matrix_entries[:max_parallel],
            "metadata": {
                "suite": suite,
                "environment": environment,
                "estimated_duration": total_duration,
                "risk_level": risk_level,
                "generated_at": datetime.now().isoformat(),
                "total_entries": len(matrix_entries[:max_parallel])
            }
        }
        
        return matrix
    
    def _generate_persona_matrix(self, config: TestConfiguration, environment: str) -> List[str]:
        """Generate persona testing matrix"""
        entries = []
        
        # Prioritize personas based on environment
        if environment == "production":
            # Focus on critical user journeys
            priority_personas = ["casual_reader", "financial_analyst"]
            priority_browsers = ["chrome", "firefox"]
        else:
            priority_personas = config.personas[:3]  # Limit for faster feedback
            priority_browsers = config.browsers[:2]
        
        for persona in priority_personas:
            for browser in priority_browsers:
                entries.append("persona")
        
        return entries
    
    def _generate_ai_inference_matrix(self, config: TestConfiguration, environment: str) -> List[str]:
        """Generate AI inference testing matrix"""
        entries = []
        
        # Prioritize models based on environment
        if environment == "production":
            # Focus on production models
            priority_models = ["gpt-4o", "claude-3.5-sonnet"]
        else:
            priority_models = config.models
        
        test_types = self.config.get("ai_inference_validation", {}).get("test_types", ["summarization"])
        
        for model in priority_models:
            for test_type in test_types[:2]:  # Limit test types
                entries.append("ai_inference")
        
        return entries
    
    def _generate_chaos_matrix(self, config: TestConfiguration, environment: str, risk_level: str) -> List[str]:
        """Generate chaos engineering matrix"""
        entries = []
        
        if environment == "production" and risk_level == "low":
            # Skip chaos in production with low risk tolerance
            return entries
        
        # Select experiments based on risk level
        if risk_level == "high":
            experiments = config.experiments
            intensities = [5, 7]
        elif risk_level == "medium":
            experiments = config.experiments[:3]
            intensities = [3, 5]
        else:
            experiments = ["network_latency", "cache_eviction"]
            intensities = [3]
        
        for experiment in experiments:
            for intensity in intensities:
                entries.append("chaos")
        
        return entries
    
    def _generate_neural_cache_matrix(self, config: TestConfiguration, environment: str) -> List[str]:
        """Generate neural cache diagnostics matrix"""
        # Single comprehensive cache analysis
        return ["neural_cache"]
    
    def _generate_multi_agent_matrix(self, config: TestConfiguration, environment: str) -> List[str]:
        """Generate multi-agent QA matrix"""
        entries = []
        
        # Prioritize agents based on environment
        if environment == "production":
            priority_agents = ["functional", "performance", "security"]
        else:
            priority_agents = config.agents
        
        for agent in priority_agents:
            entries.append("multi_agent")
        
        return entries
    
    def _generate_performance_matrix(self, config: TestConfiguration, environment: str) -> List[str]:
        """Generate performance testing matrix"""
        # Single comprehensive performance test
        return ["performance"]
    
    def _prioritize_matrix_entries(self, entries: List[str], environment: str) -> List[str]:
        """Apply intelligent prioritization to matrix entries"""
        # Get historical failure data (mock implementation)
        failure_rates = self._get_historical_failure_rates()
        
        # Sort by priority and failure rate
        def priority_key(entry):
            config = self.test_configurations.get(entry, TestConfiguration("", 5, 0, {}, [], []))
            failure_rate = failure_rates.get(entry, 0.1)
            
            # Higher priority (lower number) and higher failure rate = higher priority
            return (config.priority, -failure_rate)
        
        return sorted(entries, key=priority_key)
    
    def _get_historical_failure_rates(self) -> Dict[str, float]:
        """Get historical failure rates for test types (mock implementation)"""
        # In a real implementation, this would query historical test data
        return {
            "persona": 0.05,
            "ai_inference": 0.15,
            "chaos": 0.25,
            "neural_cache": 0.08,
            "multi_agent": 0.12,
            "performance": 0.10
        }
    
    def generate_adaptive_matrix(self, 
                                git_diff: Optional[str] = None,
                                changed_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate adaptive test matrix based on code changes"""
        
        # Analyze code changes to determine relevant tests
        relevant_tests = self._analyze_code_changes(git_diff, changed_files)
        
        # Generate targeted matrix
        matrix_entries = []
        for test_type in relevant_tests:
            if test_type in self.test_configurations:
                matrix_entries.append(test_type)
        
        # Always include smoke tests
        if "persona" not in matrix_entries:
            matrix_entries.append("persona")
        
        return {
            "include": matrix_entries,
            "metadata": {
                "type": "adaptive",
                "based_on_changes": True,
                "changed_files": changed_files or [],
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def _analyze_code_changes(self, git_diff: Optional[str], changed_files: Optional[List[str]]) -> List[str]:
        """Analyze code changes to determine relevant test types"""
        relevant_tests = set()
        
        if not changed_files:
            return ["persona", "performance"]  # Default smoke tests
        
        for file_path in changed_files:
            file_path = file_path.lower()
            
            # API changes
            if "api/" in file_path or "route" in file_path:
                relevant_tests.update(["ai_inference", "multi_agent"])
            
            # UI/Frontend changes
            if any(ext in file_path for ext in [".tsx", ".jsx", ".css", ".scss"]):
                relevant_tests.update(["persona", "performance"])
            
            # Cache/Redis changes
            if "cache" in file_path or "redis" in file_path:
                relevant_tests.add("neural_cache")
            
            # Configuration changes
            if any(config in file_path for config in ["config", "env", "docker"]):
                relevant_tests.update(["chaos", "multi_agent"])
            
            # AI/ML model changes
            if any(ai_term in file_path for ai_term in ["ai", "ml", "model", "llm"]):
                relevant_tests.add("ai_inference")
        
        return list(relevant_tests)
    
    def save_matrix(self, matrix: Dict[str, Any], output_path: str):
        """Save generated matrix to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(matrix, f, indent=2)
        
        print(f"Test matrix saved to: {output_file}")
        print(f"Generated {matrix['metadata']['total_entries']} test configurations")
        print(f"Estimated duration: {matrix['metadata']['estimated_duration']} minutes")

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(
        description="Dr. Orion TestMaster - Intelligent Test Matrix Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_test_matrix.py --suite full --output matrix.json
  python generate_test_matrix.py --suite smoke --environment staging
  python generate_test_matrix.py --suite chaos --risk-level high
  python generate_test_matrix.py --adaptive --changed-files "app/api/news.js,components/Dashboard.tsx"
        """
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--suite",
        choices=["full", "smoke", "regression", "nightly", "persona", "ai_inference", "chaos", "neural_cache", "multi_agent", "performance"],
        default="full",
        help="Test suite to generate matrix for (default: full)"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="development",
        help="Target environment (default: development)"
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Maximum test duration in minutes (default: 120)"
    )
    
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum parallel test jobs (default: 10)"
    )
    
    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high"],
        default="medium",
        help="Risk tolerance level (default: medium)"
    )
    
    parser.add_argument(
        "--output",
        default="test_matrix.json",
        help="Output file path (default: test_matrix.json)"
    )
    
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Generate adaptive matrix based on code changes"
    )
    
    parser.add_argument(
        "--changed-files",
        nargs="*",
        help="List of changed files for adaptive matrix generation"
    )
    
    parser.add_argument(
        "--git-diff",
        help="Git diff content for adaptive matrix generation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SuperhumanTestMatrixGenerator(args.config)
    
    # Generate matrix
    if args.adaptive:
        matrix = generator.generate_adaptive_matrix(
            git_diff=args.git_diff,
            changed_files=args.changed_files
        )
    else:
        matrix = generator.generate_matrix(
            suite=args.suite,
            environment=args.environment,
            max_duration=args.max_duration,
            max_parallel=args.max_parallel,
            risk_level=args.risk_level
        )
    
    # Save matrix
    generator.save_matrix(matrix, args.output)
    
    if args.verbose:
        print("\nGenerated Matrix:")
        print(json.dumps(matrix, indent=2))

if __name__ == "__main__":
    main()