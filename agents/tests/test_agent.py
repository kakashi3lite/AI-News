#!/usr/bin/env python3
"""
Unit Tests for Veteran Developer Agent

Comprehensive test suite for the Veteran Developer Agent functionality,
including code review, architecture analysis, and MLOps integration.

Test Categories:
- Agent initialization and configuration
- Code review capabilities
- Architecture analysis
- Security auditing
- Performance analysis
- Report generation
- Error handling

Author: Veteran Developer Agent V1
Target: AI News Dashboard Development Workflow
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import yaml
import json

# Import the agent modules
sys.path.append(str(Path(__file__).parent.parent))
from veteran_developer_agent import (
    VeteranDeveloperAgent,
    AgentCapability,
    CodeReviewResult,
    ArchitectureReviewResult,
    SecurityAuditResult,
    Finding,
    ArchitectureRecommendation,
    SecurityVulnerability,
    Severity,
    Priority
)

class TestVeteranDeveloperAgent:
    """Test suite for Veteran Developer Agent"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample agent configuration for testing"""
        return {
            'agent': {
                'name': 'Test Veteran Developer Agent',
                'experience_years': 30,
                'specializations': [
                    'Full-Stack Architecture',
                    'MLOps & AI Systems',
                    'Security & Compliance'
                ]
            },
            'capabilities': {
                'code_review': {
                    'enabled': True,
                    'languages': ['python', 'javascript', 'typescript'],
                    'security_focus': True,
                    'performance_analysis': True
                },
                'architecture_review': {
                    'enabled': True,
                    'focus_areas': ['scalability', 'security', 'maintainability']
                },
                'security_audit': {
                    'enabled': True,
                    'vulnerability_scanning': True,
                    'compliance_checking': True
                }
            },
            'project_specific': {
                'ai_news_dashboard': {
                    'components': ['api', 'dashboard', 'mlops'],
                    'critical_paths': ['/api/news', '/dashboard'],
                    'performance_targets': {
                        'api_response_time': '<200ms',
                        'dashboard_load_time': '<2s'
                    }
                }
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create temporary configuration file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        os.unlink(config_path)
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing"""
        return '''
def process_news_data(data):
    # Security issue: no input validation
    result = eval(data)  # Dangerous use of eval
    
    # Performance issue: inefficient loop
    processed = []
    for item in result:
        for i in range(len(result)):
            if item['id'] == result[i]['id']:
                processed.append(item)
    
    # Missing error handling
    return processed

def get_user_data(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_query(query)
'''
    
    @pytest.fixture
    def sample_javascript_code(self):
        """Sample JavaScript code for testing"""
        return '''
function processNewsData(data) {
    // Security issue: XSS vulnerability
    document.getElementById('content').innerHTML = data;
    
    // Performance issue: blocking operation
    let result = [];
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data.length; j++) {
            if (data[i].id === data[j].id) {
                result.push(data[i]);
            }
        }
    }
    
    // Missing error handling
    return result;
}

function getUserData(userId) {
    // Insecure API call
    fetch(`/api/users/${userId}`, {
        method: 'GET',
        // Missing authentication headers
    }).then(response => response.json());
}
'''
    
    def test_agent_initialization(self, temp_config_file):
        """Test agent initialization with configuration"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        assert agent.name == 'Test Veteran Developer Agent'
        assert agent.experience_years == 30
        assert len(agent.specializations) == 3
        assert AgentCapability.CODE_REVIEW in agent.capabilities
        assert AgentCapability.ARCHITECTURE_REVIEW in agent.capabilities
        assert AgentCapability.SECURITY_AUDIT in agent.capabilities
    
    def test_agent_initialization_invalid_config(self):
        """Test agent initialization with invalid configuration"""
        with pytest.raises(FileNotFoundError):
            VeteranDeveloperAgent('nonexistent_config.yaml')
    
    @pytest.mark.asyncio
    async def test_code_review_python(self, temp_config_file, sample_python_code):
        """Test code review functionality with Python code"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            python_file = f.name
        
        try:
            result = await agent.conduct_code_review([python_file])
            
            assert isinstance(result, CodeReviewResult)
            assert len(result.findings) > 0
            
            # Check for security findings
            security_findings = [f for f in result.findings if 'security' in f.message.lower() or 'eval' in f.message.lower()]
            assert len(security_findings) > 0
            
            # Check for performance findings
            performance_findings = [f for f in result.findings if 'performance' in f.message.lower() or 'inefficient' in f.message.lower()]
            assert len(performance_findings) > 0
            
            # Check severity levels
            critical_findings = [f for f in result.findings if f.severity == Severity.CRITICAL]
            assert len(critical_findings) > 0  # eval() should be critical
            
            assert result.summary is not None
            assert len(result.recommendations) > 0
            assert len(result.next_steps) > 0
            
        finally:
            os.unlink(python_file)
    
    @pytest.mark.asyncio
    async def test_code_review_javascript(self, temp_config_file, sample_javascript_code):
        """Test code review functionality with JavaScript code"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create temporary JavaScript file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(sample_javascript_code)
            js_file = f.name
        
        try:
            result = await agent.conduct_code_review([js_file])
            
            assert isinstance(result, CodeReviewResult)
            assert len(result.findings) > 0
            
            # Check for XSS vulnerability detection
            xss_findings = [f for f in result.findings if 'xss' in f.message.lower() or 'innerhtml' in f.message.lower()]
            assert len(xss_findings) > 0
            
            # Check for performance issues
            performance_findings = [f for f in result.findings if 'performance' in f.message.lower()]
            assert len(performance_findings) > 0
            
        finally:
            os.unlink(js_file)
    
    @pytest.mark.asyncio
    async def test_code_review_multiple_files(self, temp_config_file, sample_python_code, sample_javascript_code):
        """Test code review with multiple files"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            python_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(sample_javascript_code)
            js_file = f.name
        
        try:
            result = await agent.conduct_code_review([python_file, js_file])
            
            assert isinstance(result, CodeReviewResult)
            assert len(result.findings) > 0
            
            # Should have findings from both files
            python_findings = [f for f in result.findings if f.file_path == python_file]
            js_findings = [f for f in result.findings if f.file_path == js_file]
            
            assert len(python_findings) > 0
            assert len(js_findings) > 0
            
        finally:
            os.unlink(python_file)
            os.unlink(js_file)
    
    @pytest.mark.asyncio
    async def test_architecture_review(self, temp_config_file):
        """Test architecture review functionality"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Mock architecture components
        components = ['api', 'dashboard', 'mlops']
        
        result = await agent.review_architecture(components)
        
        assert isinstance(result, ArchitectureReviewResult)
        assert result.summary is not None
        assert len(result.findings) > 0
        assert len(result.recommendations) > 0
        
        # Check that findings cover the specified components
        component_findings = [f for f in result.findings if f.component in components]
        assert len(component_findings) > 0
    
    @pytest.mark.asyncio
    async def test_security_audit(self, temp_config_file, sample_python_code):
        """Test security audit functionality"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create temporary file with security issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            python_file = f.name
        
        try:
            result = await agent.conduct_security_audit([python_file])
            
            assert isinstance(result, SecurityAuditResult)
            assert len(result.vulnerabilities) > 0
            
            # Check for specific vulnerability types
            code_injection = [v for v in result.vulnerabilities if 'injection' in v.vulnerability_type.lower()]
            assert len(code_injection) > 0
            
            # Check severity levels
            high_severity = [v for v in result.vulnerabilities if v.severity in [Severity.HIGH, Severity.CRITICAL]]
            assert len(high_severity) > 0
            
        finally:
            os.unlink(python_file)
    
    def test_generate_report(self, temp_config_file):
        """Test report generation"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create mock results
        mock_findings = [
            Finding(
                rule_id='TEST001',
                severity=Severity.HIGH,
                message='Test finding',
                file_path='test.py',
                line_number=10,
                suggestion='Fix this issue'
            )
        ]
        
        mock_result = CodeReviewResult(
            summary='Test review summary',
            findings=mock_findings,
            recommendations=['Test recommendation'],
            next_steps=['Test next step']
        )
        
        report = agent.generate_report([mock_result])
        
        assert isinstance(report, str)
        assert 'Test review summary' in report
        assert 'TEST001' in report
        assert 'Test finding' in report
    
    @pytest.mark.asyncio
    async def test_code_review_empty_file_list(self, temp_config_file):
        """Test code review with empty file list"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        result = await agent.conduct_code_review([])
        
        assert isinstance(result, CodeReviewResult)
        assert len(result.findings) == 0
        assert 'No files' in result.summary
    
    @pytest.mark.asyncio
    async def test_code_review_nonexistent_file(self, temp_config_file):
        """Test code review with nonexistent file"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        with pytest.raises(FileNotFoundError):
            await agent.conduct_code_review(['nonexistent_file.py'])
    
    def test_agent_capabilities_configuration(self, temp_config_file):
        """Test agent capabilities configuration"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Test capability checking
        assert agent.has_capability(AgentCapability.CODE_REVIEW)
        assert agent.has_capability(AgentCapability.ARCHITECTURE_REVIEW)
        assert agent.has_capability(AgentCapability.SECURITY_AUDIT)
        
        # Test language support
        assert agent.supports_language('python')
        assert agent.supports_language('javascript')
        assert agent.supports_language('typescript')
        assert not agent.supports_language('cobol')
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, temp_config_file):
        """Test performance analysis capability"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Code with performance issues
        performance_code = '''
def slow_function(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            for k in range(len(data)):
                if data[i] == data[j] == data[k]:
                    result.append(data[i])
    return result

def inefficient_search(items, target):
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(performance_code)
            perf_file = f.name
        
        try:
            result = await agent.conduct_code_review([perf_file])
            
            # Should detect O(n³) complexity and inefficient search
            performance_findings = [f for f in result.findings if 'performance' in f.message.lower() or 'complexity' in f.message.lower()]
            assert len(performance_findings) > 0
            
        finally:
            os.unlink(perf_file)
    
    def test_configuration_validation(self, sample_config):
        """Test configuration validation"""
        # Test with invalid configuration
        invalid_config = sample_config.copy()
        del invalid_config['agent']['name']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            invalid_config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                VeteranDeveloperAgent(invalid_config_path)
        finally:
            os.unlink(invalid_config_path)
    
    @pytest.mark.asyncio
    async def test_concurrent_reviews(self, temp_config_file, sample_python_code):
        """Test concurrent code reviews"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create multiple temporary files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.py', delete=False) as f:
                f.write(sample_python_code)
                files.append(f.name)
        
        try:
            # Run concurrent reviews
            tasks = [agent.conduct_code_review([file]) for file in files]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            for result in results:
                assert isinstance(result, CodeReviewResult)
                assert len(result.findings) > 0
                
        finally:
            for file in files:
                os.unlink(file)
    
    def test_finding_serialization(self):
        """Test Finding object serialization"""
        finding = Finding(
            rule_id='TEST001',
            severity=Severity.HIGH,
            message='Test message',
            file_path='test.py',
            line_number=42,
            suggestion='Test suggestion'
        )
        
        # Test dictionary conversion
        finding_dict = finding.__dict__.copy()
        finding_dict['severity'] = finding_dict['severity'].value
        
        assert finding_dict['rule_id'] == 'TEST001'
        assert finding_dict['severity'] == 'high'
        assert finding_dict['line_number'] == 42
    
    @pytest.mark.asyncio
    async def test_large_codebase_handling(self, temp_config_file):
        """Test handling of large codebase"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create a large number of files
        files = []
        large_code = '\n'.join([f'def function_{i}(): pass' for i in range(100)])
        
        for i in range(10):
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_large_{i}.py', delete=False) as f:
                f.write(large_code)
                files.append(f.name)
        
        try:
            result = await agent.conduct_code_review(files)
            
            assert isinstance(result, CodeReviewResult)
            # Should handle large codebase without errors
            assert result.summary is not None
            
        finally:
            for file in files:
                os.unlink(file)

class TestAgentIntegration:
    """Integration tests for agent with external systems"""
    
    @pytest.fixture
    def mock_mlops_integration(self):
        """Mock MLOps integration for testing"""
        with patch('agents.integrations.mlops_integration.MLOpsAgentIntegration') as mock:
            mock_instance = Mock()
            mock_instance.trigger_agent_workflow = AsyncMock(return_value={
                'status': 'success',
                'workflow_type': 'code_review_to_deployment',
                'timestamp': '2024-01-01T00:00:00Z'
            })
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_mlops_workflow_integration(self, temp_config_file, mock_mlops_integration, sample_python_code):
        """Test integration with MLOps workflows"""
        agent = VeteranDeveloperAgent(temp_config_file)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_python_code)
            python_file = f.name
        
        try:
            # Conduct code review
            review_result = await agent.conduct_code_review([python_file])
            
            # Simulate MLOps integration
            workflow_result = await mock_mlops_integration.trigger_agent_workflow(
                'code_review_to_deployment',
                {'findings': review_result.findings}
            )
            
            assert workflow_result['status'] == 'success'
            assert workflow_result['workflow_type'] == 'code_review_to_deployment'
            
        finally:
            os.unlink(python_file)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])