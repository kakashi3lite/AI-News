#!/usr/bin/env python3
"""
Headless Agentic CI/CD Pipeline Automation
Dr. Aurora "CodeForge" Synth's Automated Lint/Test/Optimization Pipeline

Integrates with the agentic code optimizer to provide continuous,
automated code quality improvement without human intervention.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

# Import our agentic optimizer
try:
    from agentic_code_optimizer import AgenticCodeOptimizer, CodeMetrics
except ImportError:
    print("Warning: Agentic code optimizer not found. Some features will be disabled.")
    AgenticCodeOptimizer = None
    CodeMetrics = None

class PipelineStage(Enum):
    """Pipeline execution stages"""
    SETUP = "setup"
    LINT = "lint"
    FORMAT = "format"
    TYPE_CHECK = "type_check"
    SECURITY_SCAN = "security_scan"
    TEST = "test"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    AGENTIC_OPTIMIZATION = "agentic_optimization"
    CHAOS_TESTING = "chaos_testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    CLEANUP = "cleanup"

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"

@dataclass
class PipelineResult:
    """Result of a pipeline stage execution"""
    stage: PipelineStage
    status: PipelineStatus
    duration: float
    output: str = ""
    error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class PipelineConfig:
    """Configuration for the headless pipeline"""
    # Repository settings
    repo_path: str
    branch: str = "main"
    
    # Pipeline settings
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))
    parallel_execution: bool = True
    fail_fast: bool = False
    retry_count: int = 2
    timeout: int = 3600  # seconds
    
    # Tool configurations
    python_version: str = "3.8"
    requirements_file: str = "requirements.txt"
    test_directory: str = "tests"
    coverage_threshold: float = 80.0
    
    # Quality gates
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    
    # Notification settings
    notifications: Dict[str, Any] = field(default_factory=dict)
    
    # Agentic optimization settings
    agentic_enabled: bool = True
    optimization_level: str = "moderate"
    auto_apply_fixes: bool = False

class ToolExecutor:
    """Executes various development tools in headless mode"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger("tool_executor")
        
    async def execute_command(self, command: List[str], cwd: Optional[str] = None, 
                            timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Execute a command and return exit code, stdout, stderr"""
        if cwd is None:
            cwd = self.config.repo_path
            
        if timeout is None:
            timeout = 300  # 5 minutes default
            
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            return process.returncode, stdout.decode(), stderr.decode()
            
        except asyncio.TimeoutError:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            return -1, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Command failed: {' '.join(command)}: {e}")
            return -1, "", str(e)
    
    async def run_pylint(self) -> PipelineResult:
        """Run pylint for code quality analysis"""
        start_time = time.time()
        
        command = [
            "python", "-m", "pylint",
            "--output-format=json",
            "--reports=yes",
            "--score=yes",
            ".
        ]
        
        exit_code, stdout, stderr = await self.execute_command(command)
        duration = time.time() - start_time
        
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.WARNING
        
        # Parse pylint output
        metrics = {}
        recommendations = []
        
        try:
            if stdout:
                pylint_data = json.loads(stdout)
                if isinstance(pylint_data, list):
                    # Count issues by type
                    issue_counts = {}
                    for issue in pylint_data:
                        issue_type = issue.get('type', 'unknown')
                        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                        
                        # Add recommendations for critical issues
                        if issue_type in ['error', 'fatal']:
                            recommendations.append(f"Fix {issue_type}: {issue.get('message', '')}")
                    
                    metrics['issue_counts'] = issue_counts
                    metrics['total_issues'] = len(pylint_data)
        except json.JSONDecodeError:
            # Fallback for non-JSON output
            if "Your code has been rated at" in stdout:
                # Extract score
                import re
                score_match = re.search(r'rated at ([\d\.]+)/10', stdout)
                if score_match:
                    metrics['pylint_score'] = float(score_match.group(1))
        
        return PipelineResult(
            stage=PipelineStage.LINT,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def run_black(self) -> PipelineResult:
        """Run black for code formatting"""
        start_time = time.time()
        
        # First check what would be changed
        check_command = ["python", "-m", "black", "--check", "--diff", "."]
        exit_code, stdout, stderr = await self.execute_command(check_command)
        
        if exit_code != 0:
            # Apply formatting
            format_command = ["python", "-m", "black", "."]
            exit_code, format_stdout, format_stderr = await self.execute_command(format_command)
            stdout += "\n" + format_stdout
            stderr += "\n" + format_stderr
        
        duration = time.time() - start_time
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.FAILED
        
        # Count formatted files
        formatted_files = stdout.count("reformatted")
        metrics = {
            'formatted_files': formatted_files,
            'formatting_applied': formatted_files > 0
        }
        
        recommendations = []
        if formatted_files > 0:
            recommendations.append(f"Formatted {formatted_files} files for consistency")
        
        return PipelineResult(
            stage=PipelineStage.FORMAT,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def run_mypy(self) -> PipelineResult:
        """Run mypy for type checking"""
        start_time = time.time()
        
        command = [
            "python", "-m", "mypy",
            "--json-report", ".mypy_cache/reports",
            "--html-report", ".mypy_cache/html",
            "."
        ]
        
        exit_code, stdout, stderr = await self.execute_command(command)
        duration = time.time() - start_time
        
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.WARNING
        
        # Parse mypy output
        metrics = {}
        recommendations = []
        
        # Count errors and warnings
        error_count = stdout.count("error:")
        warning_count = stdout.count("warning:")
        note_count = stdout.count("note:")
        
        metrics.update({
            'type_errors': error_count,
            'type_warnings': warning_count,
            'type_notes': note_count
        })
        
        if error_count > 0:
            recommendations.append(f"Fix {error_count} type errors for better code safety")
        if warning_count > 0:
            recommendations.append(f"Address {warning_count} type warnings")
        
        return PipelineResult(
            stage=PipelineStage.TYPE_CHECK,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def run_bandit(self) -> PipelineResult:
        """Run bandit for security analysis"""
        start_time = time.time()
        
        command = [
            "python", "-m", "bandit",
            "-r", ".",
            "-f", "json",
            "-o", "bandit-report.json"
        ]
        
        exit_code, stdout, stderr = await self.execute_command(command)
        duration = time.time() - start_time
        
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.WARNING
        
        # Parse bandit output
        metrics = {}
        recommendations = []
        
        try:
            report_file = Path(self.config.repo_path) / "bandit-report.json"
            if report_file.exists():
                with open(report_file) as f:
                    bandit_data = json.load(f)
                    
                results = bandit_data.get('results', [])
                metrics.update({
                    'security_issues': len(results),
                    'high_severity': len([r for r in results if r.get('issue_severity') == 'HIGH']),
                    'medium_severity': len([r for r in results if r.get('issue_severity') == 'MEDIUM']),
                    'low_severity': len([r for r in results if r.get('issue_severity') == 'LOW'])
                })
                
                # Add recommendations for high severity issues
                high_severity_issues = [r for r in results if r.get('issue_severity') == 'HIGH']
                for issue in high_severity_issues[:3]:  # Top 3
                    recommendations.append(f"Security: {issue.get('test_name', 'Unknown issue')}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to parse bandit report: {e}")
        
        return PipelineResult(
            stage=PipelineStage.SECURITY_SCAN,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def run_pytest(self) -> PipelineResult:
        """Run pytest for testing"""
        start_time = time.time()
        
        command = [
            "python", "-m", "pytest",
            "--json-report", "--json-report-file=pytest-report.json",
            "--cov=.", "--cov-report=json:coverage.json",
            "--cov-report=html:htmlcov",
            "--tb=short",
            "-v"
        ]
        
        exit_code, stdout, stderr = await self.execute_command(command, timeout=600)
        duration = time.time() - start_time
        
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.FAILED
        
        # Parse test results
        metrics = {}
        recommendations = []
        
        try:
            # Parse pytest JSON report
            report_file = Path(self.config.repo_path) / "pytest-report.json"
            if report_file.exists():
                with open(report_file) as f:
                    pytest_data = json.load(f)
                    
                summary = pytest_data.get('summary', {})
                metrics.update({
                    'tests_total': summary.get('total', 0),
                    'tests_passed': summary.get('passed', 0),
                    'tests_failed': summary.get('failed', 0),
                    'tests_skipped': summary.get('skipped', 0),
                    'test_duration': pytest_data.get('duration', duration)
                })
                
            # Parse coverage report
            coverage_file = Path(self.config.repo_path) / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    
                totals = coverage_data.get('totals', {})
                coverage_percent = totals.get('percent_covered', 0)
                metrics.update({
                    'coverage_percent': coverage_percent,
                    'lines_covered': totals.get('covered_lines', 0),
                    'lines_missing': totals.get('missing_lines', 0)
                })
                
                if coverage_percent < self.config.coverage_threshold:
                    recommendations.append(
                        f"Increase test coverage from {coverage_percent:.1f}% to {self.config.coverage_threshold}%"
                    )
                    
        except Exception as e:
            self.logger.warning(f"Failed to parse test reports: {e}")
        
        # Add recommendations for failed tests
        if metrics.get('tests_failed', 0) > 0:
            recommendations.append(f"Fix {metrics['tests_failed']} failing tests")
        
        return PipelineResult(
            stage=PipelineStage.TEST,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def run_performance_tests(self) -> PipelineResult:
        """Run performance benchmarks"""
        start_time = time.time()
        
        command = [
            "python", "-m", "pytest",
            "--benchmark-json=benchmark-report.json",
            "--benchmark-only",
            "-v"
        ]
        
        exit_code, stdout, stderr = await self.execute_command(command)
        duration = time.time() - start_time
        
        status = PipelineStatus.SUCCESS if exit_code == 0 else PipelineStatus.WARNING
        
        # Parse benchmark results
        metrics = {}
        recommendations = []
        
        try:
            benchmark_file = Path(self.config.repo_path) / "benchmark-report.json"
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    
                benchmarks = benchmark_data.get('benchmarks', [])
                if benchmarks:
                    avg_times = [b['stats']['mean'] for b in benchmarks]
                    metrics.update({
                        'benchmark_count': len(benchmarks),
                        'avg_execution_time': sum(avg_times) / len(avg_times),
                        'slowest_test': max(avg_times) if avg_times else 0
                    })
                    
                    # Identify slow tests
                    slow_tests = [b for b in benchmarks if b['stats']['mean'] > 1.0]  # > 1 second
                    if slow_tests:
                        recommendations.append(f"Optimize {len(slow_tests)} slow performance tests")
                        
        except Exception as e:
            self.logger.warning(f"Failed to parse benchmark report: {e}")
        
        return PipelineResult(
            stage=PipelineStage.PERFORMANCE,
            status=status,
            duration=duration,
            output=stdout,
            error=stderr,
            metrics=metrics,
            recommendations=recommendations
        )

class HeadlessPipeline:
    """Main headless pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()
        self.logger = self._setup_logging()
        self.tool_executor = ToolExecutor(config)
        self.agentic_optimizer = None
        
        # Initialize agentic optimizer if available
        if AgenticCodeOptimizer and config.agentic_enabled:
            try:
                self.agentic_optimizer = AgenticCodeOptimizer()
            except Exception as e:
                self.logger.warning(f"Failed to initialize agentic optimizer: {e}")
        
        # Pipeline state
        self.results: Dict[PipelineStage, PipelineResult] = {}
        self.start_time = None
        self.end_time = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('headless_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('headless_pipeline')
    
    async def run(self) -> Dict[PipelineStage, PipelineResult]:
        """Execute the complete headless pipeline"""
        self.start_time = datetime.now()
        
        self.console.print(Panel.fit(
            "[bold blue]Headless Agentic CI/CD Pipeline[/bold blue]\n"
            "[dim]Automated Code Quality & Optimization[/dim]",
            border_style="blue"
        ))
        
        # Create progress display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Execute pipeline stages
        if self.config.parallel_execution:
            await self._run_parallel()
        else:
            await self._run_sequential()
        
        self.end_time = datetime.now()
        
        # Generate final report
        await self._generate_report()
        
        return self.results
    
    async def _run_sequential(self):
        """Run pipeline stages sequentially"""
        stage_methods = {
            PipelineStage.LINT: self.tool_executor.run_pylint,
            PipelineStage.FORMAT: self.tool_executor.run_black,
            PipelineStage.TYPE_CHECK: self.tool_executor.run_mypy,
            PipelineStage.SECURITY_SCAN: self.tool_executor.run_bandit,
            PipelineStage.TEST: self.tool_executor.run_pytest,
            PipelineStage.PERFORMANCE: self.tool_executor.run_performance_tests,
            PipelineStage.AGENTIC_OPTIMIZATION: self._run_agentic_optimization,
            PipelineStage.CHAOS_TESTING: self._run_chaos_testing
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            total_stages = len([s for s in self.config.enabled_stages if s in stage_methods])
            main_task = progress.add_task("Pipeline Progress", total=total_stages)
            
            for stage in self.config.enabled_stages:
                if stage not in stage_methods:
                    continue
                    
                stage_task = progress.add_task(f"Running {stage.value}...", total=None)
                
                try:
                    result = await stage_methods[stage]()
                    self.results[stage] = result
                    
                    # Update progress
                    status_emoji = "✅" if result.status == PipelineStatus.SUCCESS else "⚠️" if result.status == PipelineStatus.WARNING else "❌"
                    progress.update(stage_task, description=f"{status_emoji} {stage.value} ({result.duration:.1f}s)", completed=True)
                    
                    # Check for fail-fast
                    if self.config.fail_fast and result.status == PipelineStatus.FAILED:
                        self.logger.error(f"Pipeline failed at stage {stage.value}")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Stage {stage.value} failed with exception: {e}")
                    self.results[stage] = PipelineResult(
                        stage=stage,
                        status=PipelineStatus.FAILED,
                        duration=0,
                        error=str(e)
                    )
                    
                    if self.config.fail_fast:
                        break
                
                progress.advance(main_task)
    
    async def _run_parallel(self):
        """Run compatible pipeline stages in parallel"""
        # Group stages by dependencies
        independent_stages = [
            PipelineStage.LINT,
            PipelineStage.FORMAT,
            PipelineStage.TYPE_CHECK,
            PipelineStage.SECURITY_SCAN
        ]
        
        dependent_stages = [
            PipelineStage.TEST,
            PipelineStage.PERFORMANCE,
            PipelineStage.AGENTIC_OPTIMIZATION,
            PipelineStage.CHAOS_TESTING
        ]
        
        # Run independent stages in parallel
        if any(stage in self.config.enabled_stages for stage in independent_stages):
            await self._run_stage_group(independent_stages, "Independent Analysis")
        
        # Run dependent stages sequentially
        if any(stage in self.config.enabled_stages for stage in dependent_stages):
            await self._run_stage_group(dependent_stages, "Testing & Optimization")
    
    async def _run_stage_group(self, stages: List[PipelineStage], group_name: str):
        """Run a group of pipeline stages"""
        stage_methods = {
            PipelineStage.LINT: self.tool_executor.run_pylint,
            PipelineStage.FORMAT: self.tool_executor.run_black,
            PipelineStage.TYPE_CHECK: self.tool_executor.run_mypy,
            PipelineStage.SECURITY_SCAN: self.tool_executor.run_bandit,
            PipelineStage.TEST: self.tool_executor.run_pytest,
            PipelineStage.PERFORMANCE: self.tool_executor.run_performance_tests,
            PipelineStage.AGENTIC_OPTIMIZATION: self._run_agentic_optimization,
            PipelineStage.CHAOS_TESTING: self._run_chaos_testing
        }
        
        enabled_stages = [s for s in stages if s in self.config.enabled_stages and s in stage_methods]
        
        if not enabled_stages:
            return
        
        self.console.print(f"\n[bold yellow]Running {group_name} Stages[/bold yellow]")
        
        # Create tasks for parallel execution
        tasks = []
        for stage in enabled_stages:
            task = asyncio.create_task(stage_methods[stage]())
            task.stage = stage
            tasks.append(task)
        
        # Wait for completion
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            stage_tasks = {}
            for stage in enabled_stages:
                stage_tasks[stage] = progress.add_task(f"Running {stage.value}...", total=None)
            
            # Process completed tasks
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    stage = task.stage
                    self.results[stage] = result
                    
                    # Update progress
                    status_emoji = "✅" if result.status == PipelineStatus.SUCCESS else "⚠️" if result.status == PipelineStatus.WARNING else "❌"
                    progress.update(
                        stage_tasks[stage], 
                        description=f"{status_emoji} {stage.value} ({result.duration:.1f}s)", 
                        completed=True
                    )
                    
                except Exception as e:
                    stage = task.stage
                    self.logger.error(f"Stage {stage.value} failed: {e}")
                    self.results[stage] = PipelineResult(
                        stage=stage,
                        status=PipelineStatus.FAILED,
                        duration=0,
                        error=str(e)
                    )
    
    async def _run_agentic_optimization(self) -> PipelineResult:
        """Run agentic code optimization"""
        start_time = time.time()
        
        if not self.agentic_optimizer:
            return PipelineResult(
                stage=PipelineStage.AGENTIC_OPTIMIZATION,
                status=PipelineStatus.SKIPPED,
                duration=0,
                output="Agentic optimizer not available"
            )
        
        try:
            # Initialize optimizer
            await self.agentic_optimizer.initialize()
            
            # Run optimization
            optimization_results = await self.agentic_optimizer.optimize_repository(
                self.config.repo_path
            )
            
            duration = time.time() - start_time
            
            # Extract metrics and recommendations
            report = optimization_results.get("report", {})
            summary = report.get("summary", {})
            recommendations = []
            
            # Add high priority recommendations
            high_priority = report.get("recommendations", {}).get("high_priority", [])
            for rec in high_priority:
                recommendations.append(rec.get("description", "Unknown recommendation"))
            
            return PipelineResult(
                stage=PipelineStage.AGENTIC_OPTIMIZATION,
                status=PipelineStatus.SUCCESS,
                duration=duration,
                output=json.dumps(optimization_results, indent=2),
                metrics=summary,
                recommendations=recommendations
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Agentic optimization failed: {e}")
            
            return PipelineResult(
                stage=PipelineStage.AGENTIC_OPTIMIZATION,
                status=PipelineStatus.FAILED,
                duration=duration,
                error=str(e)
            )
    
    async def _run_chaos_testing(self) -> PipelineResult:
        """Run chaos engineering experiments"""
        start_time = time.time()
        
        # Mock chaos testing implementation
        # In a real implementation, this would integrate with LitmusChaos or similar
        
        try:
            # Simulate chaos experiments
            experiments = [
                {"name": "api_latency", "duration": 60, "success": True},
                {"name": "memory_pressure", "duration": 45, "success": True},
                {"name": "network_partition", "duration": 30, "success": False}
            ]
            
            # Wait for experiments to "complete"
            await asyncio.sleep(2)
            
            duration = time.time() - start_time
            
            successful_experiments = [e for e in experiments if e["success"]]
            failed_experiments = [e for e in experiments if not e["success"]]
            
            metrics = {
                "total_experiments": len(experiments),
                "successful_experiments": len(successful_experiments),
                "failed_experiments": len(failed_experiments),
                "success_rate": len(successful_experiments) / len(experiments) * 100
            }
            
            recommendations = []
            if failed_experiments:
                for exp in failed_experiments:
                    recommendations.append(f"Improve resilience for {exp['name']} scenario")
            
            status = PipelineStatus.SUCCESS if not failed_experiments else PipelineStatus.WARNING
            
            return PipelineResult(
                stage=PipelineStage.CHAOS_TESTING,
                status=status,
                duration=duration,
                output=json.dumps(experiments, indent=2),
                metrics=metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            return PipelineResult(
                stage=PipelineStage.CHAOS_TESTING,
                status=PipelineStatus.FAILED,
                duration=duration,
                error=str(e)
            )
    
    async def _generate_report(self):
        """Generate comprehensive pipeline report"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall status
        failed_stages = [r for r in self.results.values() if r.status == PipelineStatus.FAILED]
        warning_stages = [r for r in self.results.values() if r.status == PipelineStatus.WARNING]
        
        overall_status = "SUCCESS"
        if failed_stages:
            overall_status = "FAILED"
        elif warning_stages:
            overall_status = "WARNING"
        
        # Create summary table
        table = Table(title="Pipeline Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Key Metrics", style="blue")
        
        for stage, result in self.results.items():
            status_emoji = {
                PipelineStatus.SUCCESS: "✅",
                PipelineStatus.WARNING: "⚠️",
                PipelineStatus.FAILED: "❌",
                PipelineStatus.SKIPPED: "⏭️"
            }.get(result.status, "❓")
            
            # Format key metrics
            key_metrics = []
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_percent') or key.endswith('_rate'):
                        key_metrics.append(f"{key}: {value:.1f}%")
                    else:
                        key_metrics.append(f"{key}: {value}")
            
            table.add_row(
                stage.value,
                f"{status_emoji} {result.status.value}",
                f"{result.duration:.1f}s",
                ", ".join(key_metrics[:2])  # Show top 2 metrics
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Display overall summary
        summary_panel = Panel.fit(
            f"[bold]Overall Status: {overall_status}[/bold]\n"
            f"Total Duration: {total_duration:.1f}s\n"
            f"Stages Executed: {len(self.results)}\n"
            f"Failed: {len(failed_stages)}, Warnings: {len(warning_stages)}",
            title="Pipeline Summary",
            border_style="green" if overall_status == "SUCCESS" else "yellow" if overall_status == "WARNING" else "red"
        )
        
        self.console.print(summary_panel)
        
        # Display recommendations
        all_recommendations = []
        for result in self.results.values():
            all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            self.console.print("\n[bold blue]🔧 Recommendations:[/bold blue]")
            for i, rec in enumerate(all_recommendations[:10], 1):  # Top 10
                self.console.print(f"  {i}. {rec}")
        
        # Save detailed report
        await self._save_detailed_report()
    
    async def _save_detailed_report(self):
        """Save detailed pipeline report to file"""
        report_data = {
            "pipeline_execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration": (self.end_time - self.start_time).total_seconds(),
                "config": {
                    "repo_path": self.config.repo_path,
                    "enabled_stages": [s.value for s in self.config.enabled_stages],
                    "parallel_execution": self.config.parallel_execution,
                    "optimization_level": self.config.optimization_level
                }
            },
            "stage_results": {},
            "summary": {
                "total_stages": len(self.results),
                "successful_stages": len([r for r in self.results.values() if r.status == PipelineStatus.SUCCESS]),
                "warning_stages": len([r for r in self.results.values() if r.status == PipelineStatus.WARNING]),
                "failed_stages": len([r for r in self.results.values() if r.status == PipelineStatus.FAILED]),
                "skipped_stages": len([r for r in self.results.values() if r.status == PipelineStatus.SKIPPED])
            },
            "recommendations": []
        }
        
        # Add stage results
        for stage, result in self.results.items():
            report_data["stage_results"][stage.value] = {
                "status": result.status.value,
                "duration": result.duration,
                "metrics": result.metrics,
                "recommendations": result.recommendations,
                "has_error": bool(result.error),
                "error_message": result.error if result.error else None
            }
            
            # Collect all recommendations
            report_data["recommendations"].extend(result.recommendations)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.config.repo_path) / f"pipeline_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.console.print(f"\n[dim]Detailed report saved to: {report_file}[/dim]")

# CLI Interface
async def main():
    """Main CLI interface for headless pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Headless Agentic CI/CD Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python headless_pipeline.py /path/to/repo
  python headless_pipeline.py /path/to/repo --stages lint format test
  python headless_pipeline.py /path/to/repo --parallel --agentic
        """
    )
    
    parser.add_argument(
        "repo_path",
        help="Path to the repository"
    )
    
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[s.value for s in PipelineStage],
        default=[s.value for s in PipelineStage if s != PipelineStage.SETUP and s != PipelineStage.CLEANUP],
        help="Pipeline stages to execute"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution where possible"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Enable agentic optimization"
    )
    
    parser.add_argument(
        "--optimization-level",
        choices=["conservative", "moderate", "aggressive", "experimental"],
        default="moderate",
        help="Agentic optimization level"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Test coverage threshold percentage"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Pipeline timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Convert stage names to enums
    enabled_stages = [PipelineStage(stage) for stage in args.stages]
    
    # Create pipeline configuration
    config = PipelineConfig(
        repo_path=args.repo_path,
        enabled_stages=enabled_stages,
        parallel_execution=args.parallel,
        fail_fast=args.fail_fast,
        timeout=args.timeout,
        agentic_enabled=args.agentic,
        optimization_level=args.optimization_level,
        coverage_threshold=args.coverage_threshold
    )
    
    # Validate repository path
    if not Path(args.repo_path).exists():
        print(f"Error: Repository path does not exist: {args.repo_path}")
        sys.exit(1)
    
    try:
        # Create and run pipeline
        pipeline = HeadlessPipeline(config)
        results = await pipeline.run()
        
        # Determine exit code based on results
        failed_stages = [r for r in results.values() if r.status == PipelineStatus.FAILED]
        exit_code = 1 if failed_stages else 0
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())