#!/usr/bin/env python3
"""
Integration Tests for Veteran Developer Agent MLOps Integration

Comprehensive integration test suite for testing the Veteran Developer Agent's
integration with MLOps infrastructure, CI/CD pipelines, and external systems.

Test Categories:
- MLOps orchestrator integration
- CI/CD pipeline integration
- Observability and monitoring
- GitHub integration
- Workflow automation
- End-to-end scenarios

Author: Veteran Developer Agent V1
Target: AI News Dashboard Development Workflow
"""

import pytest
import asyncio
import tempfile
import os
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
from datetime import datetime, timedelta
import aiohttp
from aioresponses import aioresponses

# Import the integration modules
sys.path.append(str(Path(__file__).parent.parent))
from integrations.mlops_integration import (
    MLOpsAgentIntegration,
    WorkflowType,
    IntegrationStatus,
    QualityGate,
    DeploymentStrategy
)
from veteran_developer_agent import VeteranDeveloperAgent, CodeReviewResult, Finding, Severity

class TestMLOpsIntegration:
    """Test suite for MLOps integration functionality"""
    
    @pytest.fixture
    def sample_mlops_config(self):
        """Sample MLOps configuration for testing"""
        return {
            'integration': {
                'orchestrator': {
                    'enabled': True,
                    'endpoint': 'http://localhost:8080/api/v1',
                    'timeout': 30
                },
                'observability': {
                    'prometheus_endpoint': 'http://localhost:9090',
                    'grafana_endpoint': 'http://localhost:3000',
                    'alert_manager_endpoint': 'http://localhost:9093'
                },
                'github': {
                    'auto_pr_creation': True,
                    'require_reviews': True,
                    'branch_protection': True,
                    'webhook_url': 'http://localhost:8080/webhooks/github'
                }
            },
            'workflows': {
                'code_review_to_deployment': {
                    'enabled': True,
                    'quality_gates': ['security_scan', 'performance_test', 'integration_test'],
                    'deployment_strategy': 'blue_green',
                    'rollback_on_failure': True
                },
                'architecture_optimization': {
                    'enabled': True,
                    'analysis_depth': 'comprehensive',
                    'auto_apply_safe_optimizations': True
                }
            },
            'quality_gates': {
                'security_scan': {
                    'enabled': True,
                    'fail_on_critical': True,
                    'fail_on_high_count': 5
                },
                'performance_test': {
                    'enabled': True,
                    'response_time_threshold': 200,
                    'throughput_threshold': 1000
                },
                'integration_test': {
                    'enabled': True,
                    'coverage_threshold': 80,
                    'success_rate_threshold': 95
                }
            }
        }
    
    @pytest.fixture
    def temp_mlops_config_file(self, sample_mlops_config):
        """Create temporary MLOps configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_mlops_config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def mock_orchestrator_responses(self):
        """Mock responses from MLOps orchestrator"""
        return {
            'health': {'status': 'healthy', 'version': '1.0.0'},
            'deploy': {
                'deployment_id': 'dep-12345',
                'status': 'success',
                'url': 'https://staging.ai-news-dashboard.com'
            },
            'test': {
                'test_id': 'test-67890',
                'status': 'passed',
                'coverage': 85.5,
                'tests_passed': 142,
                'tests_failed': 3
            },
            'security_scan': {
                'scan_id': 'scan-11111',
                'status': 'completed',
                'vulnerabilities_found': 2,
                'critical_count': 0,
                'high_count': 1,
                'medium_count': 1
            }
        }
    
    @pytest.fixture
    def sample_code_review_result(self):
        """Sample code review result for testing"""
        findings = [
            Finding(
                rule_id='SEC001',
                severity=Severity.HIGH,
                message='Potential SQL injection vulnerability',
                file_path='api/users.py',
                line_number=42,
                suggestion='Use parameterized queries'
            ),
            Finding(
                rule_id='PERF001',
                severity=Severity.MEDIUM,
                message='Inefficient database query',
                file_path='api/news.py',
                line_number=15,
                suggestion='Add database index'
            )
        ]
        
        return CodeReviewResult(
            summary='Found 2 issues requiring attention',
            findings=findings,
            recommendations=['Implement input validation', 'Optimize database queries'],
            next_steps=['Fix security issues', 'Performance optimization']
        )
    
    def test_mlops_integration_initialization(self, temp_mlops_config_file):
        """Test MLOps integration initialization"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        assert integration.config is not None
        assert integration.orchestrator_endpoint == 'http://localhost:8080/api/v1'
        assert integration.prometheus_endpoint == 'http://localhost:9090'
        assert integration.github_integration_enabled is True
    
    @pytest.mark.asyncio
    async def test_orchestrator_health_check(self, temp_mlops_config_file, mock_orchestrator_responses):
        """Test orchestrator health check"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            m.get(
                'http://localhost:8080/api/v1/health',
                payload=mock_orchestrator_responses['health']
            )
            
            health_status = await integration.check_orchestrator_health()
            
            assert health_status['status'] == 'healthy'
            assert health_status['version'] == '1.0.0'
    
    @pytest.mark.asyncio
    async def test_trigger_code_review_workflow(self, temp_mlops_config_file, sample_code_review_result, mock_orchestrator_responses):
        """Test triggering code review to deployment workflow"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock security scan
            m.post(
                'http://localhost:8080/api/v1/security/scan',
                payload=mock_orchestrator_responses['security_scan']
            )
            
            # Mock performance test
            m.post(
                'http://localhost:8080/api/v1/test/performance',
                payload={
                    'test_id': 'perf-test-123',
                    'status': 'passed',
                    'response_time_avg': 150,
                    'throughput': 1200
                }
            )
            
            # Mock integration test
            m.post(
                'http://localhost:8080/api/v1/test/integration',
                payload=mock_orchestrator_responses['test']
            )
            
            # Mock deployment
            m.post(
                'http://localhost:8080/api/v1/deploy/staging',
                payload=mock_orchestrator_responses['deploy']
            )
            
            # Convert code review result to context
            context = {
                'findings': [{
                    'rule_id': f.rule_id,
                    'severity': f.severity.value,
                    'message': f.message,
                    'file_path': f.file_path,
                    'line_number': f.line_number
                } for f in sample_code_review_result.findings],
                'summary': sample_code_review_result.summary
            }
            
            result = await integration.trigger_agent_workflow(
                'code_review_to_deployment',
                context
            )
            
            assert result['status'] == 'success'
            assert result['workflow_type'] == 'code_review_to_deployment'
            assert 'results' in result
            assert 'security_scan' in result['results']
            assert 'performance_test' in result['results']
            assert 'integration_test' in result['results']
            assert 'staging_deployment' in result['results']
    
    @pytest.mark.asyncio
    async def test_quality_gate_failure(self, temp_mlops_config_file, mock_orchestrator_responses):
        """Test workflow failure due to quality gate"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock security scan with critical vulnerabilities
            m.post(
                'http://localhost:8080/api/v1/security/scan',
                payload={
                    'scan_id': 'scan-fail-123',
                    'status': 'completed',
                    'vulnerabilities_found': 5,
                    'critical_count': 2,  # This should fail the quality gate
                    'high_count': 3,
                    'medium_count': 0
                }
            )
            
            context = {
                'findings': [{
                    'rule_id': 'SEC001',
                    'severity': 'critical',
                    'message': 'Critical security vulnerability',
                    'file_path': 'api/auth.py',
                    'line_number': 10
                }]
            }
            
            result = await integration.trigger_agent_workflow(
                'code_review_to_deployment',
                context
            )
            
            assert result['status'] == 'failed'
            assert 'quality_gate_failure' in result
            assert result['quality_gate_failure']['gate'] == 'security_scan'
            assert 'critical vulnerabilities' in result['quality_gate_failure']['reason'].lower()
    
    @pytest.mark.asyncio
    async def test_architecture_optimization_workflow(self, temp_mlops_config_file):
        """Test architecture optimization workflow"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock architecture analysis
            m.post(
                'http://localhost:8080/api/v1/analyze/architecture',
                payload={
                    'analysis_id': 'arch-123',
                    'status': 'completed',
                    'recommendations': [
                        {
                            'type': 'caching',
                            'component': 'api',
                            'description': 'Add Redis caching layer',
                            'estimated_improvement': '40% response time reduction'
                        },
                        {
                            'type': 'database_optimization',
                            'component': 'database',
                            'description': 'Add database indexes',
                            'estimated_improvement': '60% query performance improvement'
                        }
                    ]
                }
            )
            
            # Mock optimization application
            m.post(
                'http://localhost:8080/api/v1/optimize/apply',
                payload={
                    'optimization_id': 'opt-456',
                    'status': 'success',
                    'applied_optimizations': 2,
                    'estimated_performance_gain': '35%'
                }
            )
            
            context = {
                'components': ['api', 'database', 'cache'],
                'performance_targets': {
                    'api_response_time': '<200ms',
                    'database_query_time': '<50ms'
                }
            }
            
            result = await integration.trigger_agent_workflow(
                'architecture_optimization',
                context
            )
            
            assert result['status'] == 'success'
            assert result['workflow_type'] == 'architecture_optimization'
            assert 'optimization_results' in result
    
    @pytest.mark.asyncio
    async def test_observability_integration(self, temp_mlops_config_file):
        """Test observability and monitoring integration"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock Prometheus metrics
            m.get(
                'http://localhost:9090/api/v1/query',
                payload={
                    'status': 'success',
                    'data': {
                        'resultType': 'vector',
                        'result': [
                            {
                                'metric': {'__name__': 'api_response_time'},
                                'value': [1640995200, '150']
                            }
                        ]
                    }
                }
            )
            
            # Mock Grafana dashboard creation
            m.post(
                'http://localhost:3000/api/dashboards/db',
                payload={
                    'id': 123,
                    'slug': 'agent-monitoring',
                    'status': 'success',
                    'url': '/d/agent-monitoring/veteran-developer-agent'
                }
            )
            
            # Test metrics collection
            metrics = await integration.collect_performance_metrics()
            assert 'api_response_time' in metrics
            assert metrics['api_response_time'] == 150
            
            # Test dashboard creation
            dashboard_result = await integration.create_monitoring_dashboard()
            assert dashboard_result['status'] == 'success'
            assert 'url' in dashboard_result
    
    @pytest.mark.asyncio
    async def test_github_integration(self, temp_mlops_config_file):
        """Test GitHub integration functionality"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock GitHub API calls
            m.post(
                'https://api.github.com/repos/owner/ai-news-dashboard/pulls',
                payload={
                    'id': 123,
                    'number': 456,
                    'html_url': 'https://github.com/owner/ai-news-dashboard/pull/456',
                    'state': 'open'
                }
            )
            
            m.post(
                'https://api.github.com/repos/owner/ai-news-dashboard/issues/456/comments',
                payload={
                    'id': 789,
                    'body': 'Agent analysis complete',
                    'created_at': '2024-01-01T00:00:00Z'
                }
            )
            
            # Test PR creation
            pr_result = await integration.create_github_pr(
                title='Agent-suggested improvements',
                body='Automated improvements based on code review',
                head='feature/agent-improvements',
                base='main'
            )
            
            assert pr_result['number'] == 456
            assert pr_result['state'] == 'open'
            
            # Test comment creation
            comment_result = await integration.add_github_comment(
                pr_number=456,
                comment='Agent analysis complete with 2 recommendations'
            )
            
            assert comment_result['id'] == 789
    
    def test_integration_status_reporting(self, temp_mlops_config_file):
        """Test integration status reporting"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        status = integration.get_integration_status()
        
        assert status['status'] in ['active', 'inactive', 'degraded']
        assert 'components' in status
        assert 'orchestrator' in status['components']
        assert 'observability' in status['components']
        assert 'github' in status['components']
        assert 'configuration' in status
        assert 'metrics' in status
    
    @pytest.mark.asyncio
    async def test_workflow_rollback(self, temp_mlops_config_file, mock_orchestrator_responses):
        """Test workflow rollback functionality"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock successful deployment
            m.post(
                'http://localhost:8080/api/v1/deploy/staging',
                payload=mock_orchestrator_responses['deploy']
            )
            
            # Mock deployment failure detection
            m.get(
                'http://localhost:8080/api/v1/deploy/dep-12345/health',
                payload={
                    'status': 'unhealthy',
                    'error_rate': 15.5,
                    'response_time': 5000
                }
            )
            
            # Mock rollback
            m.post(
                'http://localhost:8080/api/v1/deploy/dep-12345/rollback',
                payload={
                    'rollback_id': 'rollback-789',
                    'status': 'success',
                    'previous_deployment_id': 'dep-11111'
                }
            )
            
            # Test rollback trigger
            rollback_result = await integration.trigger_rollback('dep-12345')
            
            assert rollback_result['status'] == 'success'
            assert rollback_result['rollback_id'] == 'rollback-789'
    
    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, temp_mlops_config_file):
        """Test handling of concurrent workflows"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock multiple workflow endpoints
            for i in range(3):
                m.post(
                    'http://localhost:8080/api/v1/security/scan',
                    payload={
                        'scan_id': f'scan-{i}',
                        'status': 'completed',
                        'vulnerabilities_found': 0
                    }
                )
                
                m.post(
                    'http://localhost:8080/api/v1/test/integration',
                    payload={
                        'test_id': f'test-{i}',
                        'status': 'passed',
                        'coverage': 85
                    }
                )
                
                m.post(
                    'http://localhost:8080/api/v1/deploy/staging',
                    payload={
                        'deployment_id': f'dep-{i}',
                        'status': 'success'
                    }
                )
            
            # Run concurrent workflows
            contexts = [
                {'findings': [], 'summary': f'Review {i}'}
                for i in range(3)
            ]
            
            tasks = [
                integration.trigger_agent_workflow('code_review_to_deployment', context)
                for context in contexts
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, temp_mlops_config_file):
        """Test workflow timeout handling"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock slow response that will timeout
            async def slow_response(url, **kwargs):
                await asyncio.sleep(2)  # Longer than timeout
                return aiohttp.web.Response(text=json.dumps({'status': 'success'}))
            
            m.post(
                'http://localhost:8080/api/v1/security/scan',
                callback=slow_response
            )
            
            # Set short timeout for testing
            integration.request_timeout = 1
            
            context = {'findings': [], 'summary': 'Test timeout'}
            
            result = await integration.trigger_agent_workflow(
                'code_review_to_deployment',
                context
            )
            
            assert result['status'] == 'failed'
            assert 'timeout' in result.get('error', '').lower()
    
    def test_configuration_validation(self, sample_mlops_config):
        """Test configuration validation"""
        # Test with invalid configuration
        invalid_config = sample_mlops_config.copy()
        del invalid_config['integration']['orchestrator']['endpoint']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            invalid_config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                MLOpsAgentIntegration(invalid_config_path)
        finally:
            os.unlink(invalid_config_path)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, temp_mlops_config_file):
        """Test error recovery mechanisms"""
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        with aioresponses() as m:
            # Mock initial failure
            m.post(
                'http://localhost:8080/api/v1/security/scan',
                status=500,
                payload={'error': 'Internal server error'}
            )
            
            # Mock successful retry
            m.post(
                'http://localhost:8080/api/v1/security/scan',
                payload={
                    'scan_id': 'scan-retry-123',
                    'status': 'completed',
                    'vulnerabilities_found': 0
                }
            )
            
            # Enable retry mechanism
            integration.max_retries = 2
            integration.retry_delay = 0.1
            
            context = {'findings': [], 'summary': 'Test retry'}
            
            result = await integration.trigger_agent_workflow(
                'code_review_to_deployment',
                context
            )
            
            # Should succeed after retry
            assert result['status'] == 'success'

class TestEndToEndScenarios:
    """End-to-end integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_development_workflow(self, temp_config_file, temp_mlops_config_file):
        """Test complete development workflow from code review to deployment"""
        # Initialize agent and integration
        agent = VeteranDeveloperAgent(temp_config_file)
        integration = MLOpsAgentIntegration(temp_mlops_config_file)
        
        # Sample code with issues
        sample_code = '''
def process_user_input(user_input):
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return execute_query(query)

def slow_function(data):
    # Performance issue: O(n²) complexity
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i] == data[j]:
                result.append(data[i])
    return result
'''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_code)
            code_file = f.name
        
        try:
            with aioresponses() as m:
                # Mock all MLOps endpoints
                m.post(
                    'http://localhost:8080/api/v1/security/scan',
                    payload={
                        'scan_id': 'scan-e2e-123',
                        'status': 'completed',
                        'vulnerabilities_found': 1,
                        'critical_count': 0,
                        'high_count': 1,
                        'medium_count': 0
                    }
                )
                
                m.post(
                    'http://localhost:8080/api/v1/test/performance',
                    payload={
                        'test_id': 'perf-e2e-123',
                        'status': 'passed',
                        'response_time_avg': 180,
                        'throughput': 1100
                    }
                )
                
                m.post(
                    'http://localhost:8080/api/v1/test/integration',
                    payload={
                        'test_id': 'int-e2e-123',
                        'status': 'passed',
                        'coverage': 82,
                        'tests_passed': 95,
                        'tests_failed': 2
                    }
                )
                
                m.post(
                    'http://localhost:8080/api/v1/deploy/staging',
                    payload={
                        'deployment_id': 'dep-e2e-123',
                        'status': 'success',
                        'url': 'https://staging-e2e.ai-news-dashboard.com'
                    }
                )
                
                # Step 1: Conduct code review
                review_result = await agent.conduct_code_review([code_file])
                
                assert len(review_result.findings) > 0
                security_findings = [f for f in review_result.findings if 'sql' in f.message.lower()]
                performance_findings = [f for f in review_result.findings if 'performance' in f.message.lower()]
                
                assert len(security_findings) > 0
                assert len(performance_findings) > 0
                
                # Step 2: Trigger MLOps workflow
                context = {
                    'findings': [{
                        'rule_id': f.rule_id,
                        'severity': f.severity.value,
                        'message': f.message,
                        'file_path': f.file_path,
                        'line_number': f.line_number
                    } for f in review_result.findings],
                    'summary': review_result.summary
                }
                
                workflow_result = await integration.trigger_agent_workflow(
                    'code_review_to_deployment',
                    context
                )
                
                # Step 3: Verify workflow completion
                assert workflow_result['status'] == 'success'
                assert 'results' in workflow_result
                assert 'security_scan' in workflow_result['results']
                assert 'performance_test' in workflow_result['results']
                assert 'integration_test' in workflow_result['results']
                assert 'staging_deployment' in workflow_result['results']
                
                # Step 4: Verify deployment
                deployment_result = workflow_result['results']['staging_deployment']
                assert deployment_result['status'] == 'success'
                assert 'url' in deployment_result
                
        finally:
            os.unlink(code_file)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])