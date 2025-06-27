#!/usr/bin/env python3
"""
Veteran Developer Agent CLI Interface

Command-line interface for interacting with the Veteran Developer Agent,
allowing users to trigger code reviews, architecture analysis, and MLOps
integration workflows directly from the terminal.

Usage:
    python agent_cli.py review --files src/
    python agent_cli.py architecture --analyze
    python agent_cli.py workflow --type code_review_to_deployment
    python agent_cli.py status --integration

Author: Veteran Developer Agent V1
Target: AI News Dashboard Development Workflow
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from veteran_developer_agent import VeteranDeveloperAgent, AgentCapability
    from integrations.mlops_integration import MLOpsAgentIntegration
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentCLI:
    """Command-line interface for Veteran Developer Agent"""
    
    def __init__(self):
        self.agent = None
        self.integration = None
        self.config_path = None
        
    def setup_agent(self, config_path: Optional[str] = None):
        """Initialize agent and integration"""
        try:
            self.config_path = config_path or str(Path(__file__).parent.parent / "config" / "agent_config.yaml")
            self.agent = VeteranDeveloperAgent(self.config_path)
            self.integration = MLOpsAgentIntegration()
            logger.info("Agent and integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            sys.exit(1)
    
    async def run_code_review(self, args) -> Dict[str, Any]:
        """Run code review"""
        print("🔍 Starting comprehensive code review...")
        
        file_paths = None
        if args.files:
            if os.path.isdir(args.files):
                # Get all code files in directory
                file_paths = self._get_code_files_in_directory(args.files)
            elif os.path.isfile(args.files):
                file_paths = [args.files]
            else:
                print(f"❌ Error: {args.files} is not a valid file or directory")
                return {"status": "error", "message": "Invalid file path"}
        
        try:
            result = await self.agent.conduct_code_review(file_paths)
            
            # Display results
            self._display_code_review_results(result)
            
            # Trigger MLOps workflow if enabled
            if args.trigger_workflow and self.integration:
                print("\n🚀 Triggering MLOps workflow...")
                workflow_result = await self.integration.trigger_agent_workflow(
                    "code_review_to_deployment",
                    self._convert_agent_response_to_dict(result)
                )
                self._display_workflow_results(workflow_result)
            
            # Save report if requested
            if args.save_report:
                report_path = self._save_report(result, "code_review")
                print(f"\n📄 Report saved to: {report_path}")
            
            return {"status": "success", "result": result}
            
        except Exception as e:
            logger.error(f"Code review failed: {e}")
            print(f"❌ Code review failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_architecture_review(self, args) -> Dict[str, Any]:
        """Run architecture review"""
        print("🏗️ Starting architecture review...")
        
        try:
            components = args.components.split(',') if args.components else None
            result = await self.agent.review_architecture(components)
            
            # Display results
            self._display_architecture_results(result)
            
            # Trigger optimization workflow if enabled
            if args.optimize and self.integration:
                print("\n⚡ Triggering architecture optimization...")
                optimization_result = await self.integration.optimize_mlops_pipeline(
                    [rec.__dict__ for rec in result.findings]
                )
                self._display_optimization_results(optimization_result)
            
            # Save report if requested
            if args.save_report:
                report_path = self._save_report(result, "architecture_review")
                print(f"\n📄 Report saved to: {report_path}")
            
            return {"status": "success", "result": result}
            
        except Exception as e:
            logger.error(f"Architecture review failed: {e}")
            print(f"❌ Architecture review failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def run_workflow(self, args) -> Dict[str, Any]:
        """Run MLOps workflow"""
        if not self.integration:
            print("❌ MLOps integration not available")
            return {"status": "error", "message": "Integration not available"}
        
        print(f"🔄 Starting workflow: {args.type}")
        
        try:
            # Prepare context based on workflow type
            context = {}
            if args.context:
                context = json.loads(args.context)
            
            result = await self.integration.trigger_agent_workflow(args.type, context)
            
            # Display results
            self._display_workflow_results(result)
            
            return {"status": "success", "result": result}
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            print(f"❌ Workflow execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def show_status(self, args) -> Dict[str, Any]:
        """Show agent and integration status"""
        print("📊 Agent Status Report")
        print("=" * 50)
        
        # Agent status
        if self.agent:
            print(f"✅ Veteran Developer Agent: Active")
            print(f"   Experience: {self.agent.experience_years} years")
            print(f"   Specializations: {len(self.agent.specializations)}")
            print(f"   Config: {self.config_path}")
        else:
            print("❌ Veteran Developer Agent: Not initialized")
        
        # Integration status
        if self.integration:
            status = self.integration.get_integration_status()
            print(f"\n✅ MLOps Integration: {status['status'].title()}")
            print(f"   Automation Level: {status['configuration']['automation_level']}")
            print(f"   Success Rate: {status['metrics']['average_success_rate']:.1%}")
            print(f"   Workflows Executed: {status['metrics']['total_workflows_executed']}")
            
            print("\n🔧 Components:")
            for component, available in status['components'].items():
                status_icon = "✅" if available else "❌"
                print(f"   {status_icon} {component.replace('_', ' ').title()}")
        else:
            print("\n❌ MLOps Integration: Not available")
        
        return {"status": "success"}
    
    def _get_code_files_in_directory(self, directory: str) -> List[str]:
        """Get all code files in directory"""
        code_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs']
        code_files = []
        
        for ext in code_extensions:
            code_files.extend(Path(directory).rglob(f'*{ext}'))
        
        # Filter out unwanted directories
        exclude_dirs = {'node_modules', '.git', '__pycache__', '.next', 'dist', 'build'}
        filtered_files = []
        
        for file_path in code_files:
            if not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                filtered_files.append(str(file_path))
        
        return filtered_files
    
    def _display_code_review_results(self, result):
        """Display code review results"""
        print(f"\n📋 Code Review Results")
        print("=" * 50)
        print(f"Summary: {result.summary}")
        
        if result.findings:
            # Group findings by severity
            severity_groups = {}
            for finding in result.findings:
                severity = finding.severity.value
                if severity not in severity_groups:
                    severity_groups[severity] = []
                severity_groups[severity].append(finding)
            
            # Display by severity
            severity_icons = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '⚡',
                'low': 'ℹ️',
                'info': '💡'
            }
            
            for severity in ['critical', 'high', 'medium', 'low', 'info']:
                if severity in severity_groups:
                    findings = severity_groups[severity]
                    icon = severity_icons.get(severity, '•')
                    print(f"\n{icon} {severity.upper()} ({len(findings)} issues):")
                    
                    for finding in findings[:5]:  # Show first 5 of each severity
                        print(f"   📁 {finding.file_path}:{finding.line_number}")
                        print(f"      {finding.message}")
                        print(f"      💡 {finding.suggestion}")
                        print()
                    
                    if len(findings) > 5:
                        print(f"   ... and {len(findings) - 5} more {severity} issues")
        
        if result.recommendations:
            print(f"\n🎯 Recommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        if result.next_steps:
            print(f"\n📝 Next Steps:")
            for step in result.next_steps[:3]:
                print(f"   • {step}")
    
    def _display_architecture_results(self, result):
        """Display architecture review results"""
        print(f"\n🏗️ Architecture Review Results")
        print("=" * 50)
        print(f"Summary: {result.summary}")
        
        if result.findings:
            print(f"\n📐 Architecture Recommendations:")
            
            priority_icons = {
                'critical': '🚨',
                'high': '⚠️',
                'medium': '⚡',
                'low': 'ℹ️'
            }
            
            for finding in result.findings[:5]:
                icon = priority_icons.get(finding.priority.value, '•')
                print(f"\n{icon} {finding.component} ({finding.priority.value.upper()})")
                print(f"   Current: {finding.current_state}")
                print(f"   Recommended: {finding.recommended_state}")
                print(f"   Rationale: {finding.rationale}")
                print(f"   Effort: {finding.estimated_effort}")
        
        if result.recommendations:
            print(f"\n🎯 Key Recommendations:")
            for i, rec in enumerate(result.recommendations[:3], 1):
                print(f"   {i}. {rec}")
    
    def _display_workflow_results(self, result):
        """Display workflow execution results"""
        print(f"\n🔄 Workflow Results: {result['workflow_type']}")
        print("=" * 50)
        print(f"Status: {result['status'].upper()}")
        print(f"Timestamp: {result['timestamp']}")
        
        if 'results' in result:
            print(f"\n📊 Phase Results:")
            for phase_name, phase_result in result['results'].items():
                status_icon = "✅" if phase_result.get('status') == 'success' else "❌"
                print(f"   {status_icon} {phase_name}: {phase_result.get('status', 'unknown')}")
                
                # Show additional details for some phases
                if phase_name == 'automated_testing' and 'tests_passed' in phase_result:
                    print(f"      Tests: {phase_result['tests_passed']} passed, {phase_result['tests_failed']} failed")
                    print(f"      Coverage: {phase_result['coverage']}%")
                
                elif phase_name == 'staging_deployment' and 'url' in phase_result:
                    print(f"      URL: {phase_result['url']}")
                    print(f"      Deployment ID: {phase_result['deployment_id']}")
    
    def _display_optimization_results(self, result):
        """Display optimization results"""
        print(f"\n⚡ Optimization Results")
        print("=" * 50)
        print(f"Optimizations Applied: {result['total_improvements']}")
        print(f"Estimated Performance Gain: {result['estimated_performance_gain']}")
        
        if result['optimizations_applied']:
            print(f"\n🔧 Applied Optimizations:")
            for opt in result['optimizations_applied']:
                print(f"   • {opt['type']}: {opt['description']}")
                if 'estimated_time_savings' in opt:
                    print(f"     Time Savings: {opt['estimated_time_savings']}")
                if 'estimated_downtime_reduction' in opt:
                    print(f"     Downtime Reduction: {opt['estimated_downtime_reduction']}")
    
    def _convert_agent_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert agent response to dictionary for workflow integration"""
        findings_dict = []
        for finding in response.findings:
            if hasattr(finding, 'severity'):
                findings_dict.append({
                    'rule_id': finding.rule_id,
                    'severity': finding.severity.value,
                    'message': finding.message,
                    'file_path': finding.file_path,
                    'line_number': finding.line_number
                })
        
        # Separate by severity for workflow processing
        critical_findings = [f for f in findings_dict if f['severity'] == 'critical']
        security_findings = [f for f in findings_dict if 'security' in f.get('message', '').lower()]
        
        return {
            'critical_findings': critical_findings,
            'security_findings': security_findings,
            'all_findings': findings_dict,
            'summary': response.summary,
            'recommendations': response.recommendations
        }
    
    def _save_report(self, result, report_type: str) -> str:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.md"
        
        # Create reports directory if it doesn't exist
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / filename
        
        # Generate report content
        if hasattr(self.agent, 'generate_report'):
            report_content = self.agent.generate_report([result])
        else:
            # Fallback simple report
            report_content = f"""
# {report_type.replace('_', ' ').title()} Report

**Generated:** {datetime.now().isoformat()}
**Summary:** {result.summary}

## Findings
{len(result.findings)} issues found

## Recommendations
{chr(10).join(f'- {rec}' for rec in result.recommendations)}

## Next Steps
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(result.next_steps))}
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Veteran Developer Agent CLI - 30 Years of Software Engineering Experience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s review --files src/ --save-report
  %(prog)s review --files app/components/NewsDashboard.js --trigger-workflow
  %(prog)s architecture --analyze --optimize
  %(prog)s workflow --type code_review_to_deployment
  %(prog)s status --integration

For more information, visit: https://github.com/your-repo/ai-news-dashboard
"""
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to agent configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Code review command
    review_parser = subparsers.add_parser('review', help='Conduct code review')
    review_parser.add_argument(
        '--files',
        type=str,
        help='File or directory to review (default: current directory)'
    )
    review_parser.add_argument(
        '--trigger-workflow',
        action='store_true',
        help='Trigger MLOps workflow after review'
    )
    review_parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed report to file'
    )
    
    # Architecture review command
    arch_parser = subparsers.add_parser('architecture', help='Review system architecture')
    arch_parser.add_argument(
        '--components',
        type=str,
        help='Comma-separated list of components to analyze'
    )
    arch_parser.add_argument(
        '--optimize',
        action='store_true',
        help='Trigger optimization workflow'
    )
    arch_parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed report to file'
    )
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Execute MLOps workflow')
    workflow_parser.add_argument(
        '--type',
        type=str,
        required=True,
        choices=['code_review_to_deployment', 'architecture_optimization'],
        help='Type of workflow to execute'
    )
    workflow_parser.add_argument(
        '--context',
        type=str,
        help='JSON context for workflow execution'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show agent status')
    status_parser.add_argument(
        '--integration',
        action='store_true',
        help='Show MLOps integration status'
    )
    
    return parser

async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = AgentCLI()
    cli.setup_agent(args.config)
    
    try:
        if args.command == 'review':
            await cli.run_code_review(args)
        elif args.command == 'architecture':
            await cli.run_architecture_review(args)
        elif args.command == 'workflow':
            await cli.run_workflow(args)
        elif args.command == 'status':
            cli.show_status(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())