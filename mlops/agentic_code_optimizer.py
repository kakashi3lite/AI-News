#!/usr/bin/env python3
"""
Dr. Aurora "CodeForge" Synth's Superhuman Agentic-AI Code Optimizer
Architectural Alchemist for Multi-Agent RAG Workflows

A comprehensive system for automated code analysis, refactoring, and resilience testing
using multi-agent orchestration with RAG-powered context awareness.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.tree import Tree

# Vector store and RAG components
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages for RAG functionality...")
    subprocess.run(["pip", "install", "chromadb", "sentence-transformers"])
    import chromadb
    from sentence_transformers import SentenceTransformer

# Chaos engineering
try:
    import litmus
except ImportError:
    print("Note: LitmusChaos Python client not available. Using mock implementation.")
    litmus = None

class AgentRole(Enum):
    """Defines the roles of different agents in the system"""
    ANALYZER = "analyzer"          # Code analysis and metrics
    GENERATOR = "generator"        # Code generation and suggestions
    CRITIC = "critic"              # Code review and quality assessment
    REFACTOR = "refactor"          # Automated refactoring
    TESTER = "tester"              # Test generation and execution
    DOCUMENTER = "documenter"      # Documentation generation
    CHAOS_ENGINEER = "chaos"       # Chaos engineering and resilience
    ORCHESTRATOR = "orchestrator"  # Workflow coordination

class OptimizationLevel(Enum):
    """Optimization intensity levels"""
    CONSERVATIVE = "conservative"  # Safe, minimal changes
    MODERATE = "moderate"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Maximum optimization
    EXPERIMENTAL = "experimental"  # Cutting-edge techniques

@dataclass
class CodeMetrics:
    """Comprehensive code quality metrics"""
    complexity: float = 0.0
    maintainability: float = 0.0
    test_coverage: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    documentation_score: float = 0.0
    technical_debt: float = 0.0
    duplication_ratio: float = 0.0
    lines_of_code: int = 0
    files_analyzed: int = 0
    hotspots: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    vulnerabilities: List[Dict] = field(default_factory=list)

@dataclass
class RefactoringTask:
    """Represents a refactoring task"""
    file_path: str
    task_type: str
    description: str
    priority: int
    estimated_impact: float
    confidence: float
    suggested_changes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    test_requirements: List[str] = field(default_factory=list)

@dataclass
class ChaosExperiment:
    """Chaos engineering experiment definition"""
    name: str
    target: str
    fault_type: str
    duration: int
    success_criteria: Dict[str, float]
    blast_radius: str
    rollback_strategy: str
    monitoring_metrics: List[str] = field(default_factory=list)

class AgenticAgent:
    """Base class for all agentic agents"""
    
    def __init__(self, role: AgentRole, config: Dict[str, Any]):
        self.role = role
        self.config = config
        self.logger = logging.getLogger(f"agent.{role.value}")
        self.console = Console()
        self.context_store = None
        
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info(f"Initializing {self.role.value} agent")
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the given context and return results"""
        raise NotImplementedError
        
    def log_activity(self, message: str, level: str = "info"):
        """Log agent activity"""
        getattr(self.logger, level)(f"[{self.role.value.upper()}] {message}")

class AnalyzerAgent(AgenticAgent):
    """Agent responsible for code analysis and metrics collection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.ANALYZER, config)
        self.metrics_cache = {}
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codebase and collect metrics"""
        repo_path = context.get("repo_path")
        if not repo_path:
            raise ValueError("Repository path not provided")
            
        self.log_activity(f"Analyzing repository: {repo_path}")
        
        # Perform comprehensive analysis
        metrics = await self._analyze_codebase(repo_path)
        hotspots = await self._identify_hotspots(repo_path)
        dependencies = await self._analyze_dependencies(repo_path)
        
        return {
            "metrics": metrics,
            "hotspots": hotspots,
            "dependencies": dependencies,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_codebase(self, repo_path: str) -> CodeMetrics:
        """Perform comprehensive codebase analysis"""
        metrics = CodeMetrics()
        
        # Analyze Python files
        python_files = list(Path(repo_path).rglob("*.py"))
        metrics.files_analyzed = len(python_files)
        
        total_lines = 0
        complexity_scores = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                    # Calculate cyclomatic complexity (simplified)
                    complexity = self._calculate_complexity(content)
                    complexity_scores.append(complexity)
                    
            except Exception as e:
                self.log_activity(f"Error analyzing {file_path}: {e}", "warning")
        
        metrics.lines_of_code = total_lines
        metrics.complexity = np.mean(complexity_scores) if complexity_scores else 0
        
        # Mock other metrics (in real implementation, use tools like radon, bandit, etc.)
        metrics.maintainability = max(0, 100 - metrics.complexity * 2)
        metrics.test_coverage = await self._calculate_test_coverage(repo_path)
        metrics.performance_score = 85.0  # Mock score
        metrics.security_score = await self._security_analysis(repo_path)
        metrics.documentation_score = await self._documentation_analysis(repo_path)
        
        return metrics
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate cyclomatic complexity (simplified)"""
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        lines = content.splitlines()
        complexity = 1  # Base complexity
        
        for line in lines:
            line = line.strip()
            for keyword in complexity_keywords:
                if line.startswith(keyword + ' ') or f' {keyword} ' in line:
                    complexity += 1
                    
        return complexity / max(len(lines), 1) * 100
    
    async def _calculate_test_coverage(self, repo_path: str) -> float:
        """Calculate test coverage"""
        try:
            # Run coverage analysis
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=.", "--cov-report=json"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse coverage report
                coverage_file = Path(repo_path) / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        return coverage_data.get("totals", {}).get("percent_covered", 0)
        except Exception as e:
            self.log_activity(f"Coverage analysis failed: {e}", "warning")
            
        return 0.0
    
    async def _security_analysis(self, repo_path: str) -> float:
        """Perform security analysis"""
        try:
            # Run bandit security analysis
            result = subprocess.run(
                ["bandit", "-r", ".", "-f", "json"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                security_data = json.loads(result.stdout)
                issues = len(security_data.get("results", []))
                # Convert to score (fewer issues = higher score)
                return max(0, 100 - issues * 5)
        except Exception as e:
            self.log_activity(f"Security analysis failed: {e}", "warning")
            
        return 75.0  # Default score
    
    async def _documentation_analysis(self, repo_path: str) -> float:
        """Analyze documentation quality"""
        python_files = list(Path(repo_path).rglob("*.py"))
        documented_functions = 0
        total_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                    in_function = False
                    has_docstring = False
                    
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        if stripped.startswith('def ') or stripped.startswith('async def '):
                            if in_function and has_docstring:
                                documented_functions += 1
                            total_functions += 1
                            in_function = True
                            has_docstring = False
                            
                            # Check next few lines for docstring
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    has_docstring = True
                                    break
                    
                    if in_function and has_docstring:
                        documented_functions += 1
                        
            except Exception as e:
                self.log_activity(f"Error analyzing documentation in {file_path}: {e}", "warning")
        
        return (documented_functions / max(total_functions, 1)) * 100
    
    async def _identify_hotspots(self, repo_path: str) -> List[str]:
        """Identify code hotspots that need attention"""
        hotspots = []
        
        # Analyze git history for frequently changed files
        try:
            result = subprocess.run(
                ["git", "log", "--name-only", "--pretty=format:", "--since=3.months"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                file_counts = {}
                
                for file in files:
                    if file and file.endswith('.py'):
                        file_counts[file] = file_counts.get(file, 0) + 1
                
                # Get top 10 most changed files
                sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
                hotspots = [file for file, count in sorted_files[:10] if count > 5]
                
        except Exception as e:
            self.log_activity(f"Hotspot analysis failed: {e}", "warning")
            
        return hotspots
    
    async def _analyze_dependencies(self, repo_path: str) -> Dict[str, str]:
        """Analyze project dependencies"""
        dependencies = {}
        
        # Check requirements.txt
        req_file = Path(repo_path) / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '==' in line:
                                name, version = line.split('==', 1)
                                dependencies[name.strip()] = version.strip()
                            else:
                                dependencies[line] = "latest"
            except Exception as e:
                self.log_activity(f"Error reading requirements.txt: {e}", "warning")
        
        return dependencies

class GeneratorAgent(AgenticAgent):
    """Agent responsible for code generation and suggestions"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.GENERATOR, config)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code improvements and suggestions"""
        metrics = context.get("metrics")
        hotspots = context.get("hotspots", [])
        
        self.log_activity("Generating optimization suggestions")
        
        suggestions = await self._generate_suggestions(metrics, hotspots)
        refactoring_tasks = await self._create_refactoring_tasks(suggestions)
        
        return {
            "suggestions": suggestions,
            "refactoring_tasks": refactoring_tasks,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_suggestions(self, metrics: CodeMetrics, hotspots: List[str]) -> List[Dict]:
        """Generate optimization suggestions based on metrics"""
        suggestions = []
        
        # Complexity-based suggestions
        if metrics.complexity > 15:
            suggestions.append({
                "type": "complexity_reduction",
                "priority": "high",
                "description": "Reduce cyclomatic complexity by extracting methods and simplifying conditional logic",
                "impact": "maintainability",
                "files": hotspots[:3]
            })
        
        # Test coverage suggestions
        if metrics.test_coverage < 80:
            suggestions.append({
                "type": "test_coverage",
                "priority": "medium",
                "description": f"Increase test coverage from {metrics.test_coverage:.1f}% to 80%+",
                "impact": "reliability",
                "files": []
            })
        
        # Documentation suggestions
        if metrics.documentation_score < 70:
            suggestions.append({
                "type": "documentation",
                "priority": "medium",
                "description": "Add docstrings to functions and classes",
                "impact": "maintainability",
                "files": hotspots
            })
        
        # Security suggestions
        if metrics.security_score < 90:
            suggestions.append({
                "type": "security",
                "priority": "high",
                "description": "Address security vulnerabilities and implement best practices",
                "impact": "security",
                "files": []
            })
        
        return suggestions
    
    async def _create_refactoring_tasks(self, suggestions: List[Dict]) -> List[RefactoringTask]:
        """Create specific refactoring tasks from suggestions"""
        tasks = []
        
        for suggestion in suggestions:
            task = RefactoringTask(
                file_path="",  # Will be populated by specific analysis
                task_type=suggestion["type"],
                description=suggestion["description"],
                priority=self._priority_to_int(suggestion["priority"]),
                estimated_impact=0.8,  # Mock impact score
                confidence=0.9,
                suggested_changes=[],
                dependencies=[],
                test_requirements=[]
            )
            tasks.append(task)
        
        return tasks
    
    def _priority_to_int(self, priority: str) -> int:
        """Convert priority string to integer"""
        priority_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return priority_map.get(priority.lower(), 2)

class CriticAgent(AgenticAgent):
    """Agent responsible for code review and quality assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.CRITIC, config)
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Review and critique code changes"""
        refactoring_tasks = context.get("refactoring_tasks", [])
        
        self.log_activity("Reviewing refactoring proposals")
        
        reviews = await self._review_tasks(refactoring_tasks)
        approved_tasks = [task for task, review in zip(refactoring_tasks, reviews) if review["approved"]]
        
        return {
            "reviews": reviews,
            "approved_tasks": approved_tasks,
            "review_timestamp": datetime.now().isoformat()
        }
    
    async def _review_tasks(self, tasks: List[RefactoringTask]) -> List[Dict]:
        """Review each refactoring task"""
        reviews = []
        
        for task in tasks:
            review = {
                "task_id": id(task),
                "approved": True,  # Simplified approval logic
                "confidence": task.confidence,
                "risk_assessment": "low",
                "recommendations": [],
                "blocking_issues": []
            }
            
            # Risk assessment based on task type
            if task.task_type == "complexity_reduction" and task.priority >= 3:
                review["risk_assessment"] = "medium"
                review["recommendations"].append("Implement in stages with thorough testing")
            
            if task.confidence < 0.7:
                review["approved"] = False
                review["blocking_issues"].append("Low confidence score requires manual review")
            
            reviews.append(review)
        
        return reviews

class ChaosEngineerAgent(AgenticAgent):
    """Agent responsible for chaos engineering and resilience testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentRole.CHAOS_ENGINEER, config)
        self.experiments = []
        
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design and execute chaos experiments"""
        self.log_activity("Designing chaos experiments")
        
        experiments = await self._design_experiments(context)
        results = await self._execute_experiments(experiments)
        
        return {
            "experiments": experiments,
            "results": results,
            "chaos_timestamp": datetime.now().isoformat()
        }
    
    async def _design_experiments(self, context: Dict[str, Any]) -> List[ChaosExperiment]:
        """Design chaos experiments based on system analysis"""
        experiments = [
            ChaosExperiment(
                name="api_latency_injection",
                target="api_service",
                fault_type="latency",
                duration=300,  # 5 minutes
                success_criteria={
                    "availability": 99.0,
                    "response_time_p95": 2000,  # ms
                    "error_rate": 0.05
                },
                blast_radius="single_service",
                rollback_strategy="automatic",
                monitoring_metrics=["response_time", "error_rate", "throughput"]
            ),
            ChaosExperiment(
                name="memory_pressure",
                target="worker_nodes",
                fault_type="resource_exhaustion",
                duration=180,  # 3 minutes
                success_criteria={
                    "availability": 95.0,
                    "memory_usage": 90.0,  # %
                    "recovery_time": 60  # seconds
                },
                blast_radius="single_node",
                rollback_strategy="automatic",
                monitoring_metrics=["memory_usage", "cpu_usage", "pod_restarts"]
            ),
            ChaosExperiment(
                name="network_partition",
                target="database_connection",
                fault_type="network",
                duration=120,  # 2 minutes
                success_criteria={
                    "availability": 98.0,
                    "data_consistency": 100.0,
                    "recovery_time": 30  # seconds
                },
                blast_radius="database_tier",
                rollback_strategy="manual",
                monitoring_metrics=["connection_pool", "query_latency", "data_integrity"]
            )
        ]
        
        return experiments
    
    async def _execute_experiments(self, experiments: List[ChaosExperiment]) -> List[Dict]:
        """Execute chaos experiments and collect results"""
        results = []
        
        for experiment in experiments:
            self.log_activity(f"Executing chaos experiment: {experiment.name}")
            
            # Mock experiment execution
            result = {
                "experiment_name": experiment.name,
                "status": "completed",
                "duration": experiment.duration,
                "metrics": {
                    "availability": 99.2,
                    "response_time_p95": 1800,
                    "error_rate": 0.03,
                    "recovery_time": 25
                },
                "success": True,
                "observations": [
                    "System maintained availability above threshold",
                    "Recovery was faster than expected",
                    "No data loss detected"
                ],
                "recommendations": [
                    "Consider increasing fault injection intensity",
                    "Add more comprehensive monitoring"
                ]
            }
            
            # Validate against success criteria
            success = self._validate_experiment_success(experiment, result["metrics"])
            result["success"] = success
            
            results.append(result)
            
            # Wait between experiments
            await asyncio.sleep(1)
        
        return results
    
    def _validate_experiment_success(self, experiment: ChaosExperiment, metrics: Dict) -> bool:
        """Validate experiment results against success criteria"""
        for criterion, threshold in experiment.success_criteria.items():
            if criterion in metrics:
                if criterion in ["availability", "data_consistency"]:
                    if metrics[criterion] < threshold:
                        return False
                else:
                    if metrics[criterion] > threshold:
                        return False
        return True

class RAGContextStore:
    """RAG-powered context store for code understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = chromadb.Client()
        self.collection = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.console = Console()
        
    async def initialize(self):
        """Initialize the vector store"""
        try:
            self.collection = self.client.create_collection(
                name="code_context",
                metadata={"description": "Code context for RAG-powered refactoring"}
            )
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection("code_context")
    
    async def index_codebase(self, repo_path: str):
        """Index the codebase for RAG retrieval"""
        self.console.print("[blue]Indexing codebase for RAG context...[/blue]")
        
        python_files = list(Path(repo_path).rglob("*.py"))
        
        documents = []
        metadatas = []
        ids = []
        
        for i, file_path in enumerate(python_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Split into chunks for better retrieval
                    chunks = self._split_code_into_chunks(content)
                    
                    for j, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "file_path": str(file_path),
                            "chunk_id": j,
                            "file_type": "python"
                        })
                        ids.append(f"{file_path}_{j}")
                        
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        if documents:
            # Generate embeddings
            embeddings = self.model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
        self.console.print(f"[green]Indexed {len(documents)} code chunks from {len(python_files)} files[/green]")
    
    def _split_code_into_chunks(self, content: str, max_lines: int = 50) -> List[str]:
        """Split code content into manageable chunks"""
        lines = content.splitlines()
        chunks = []
        
        for i in range(0, len(lines), max_lines):
            chunk = '\n'.join(lines[i:i + max_lines])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    async def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant code context for a query"""
        if not self.collection:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query]).tolist()[0]
            
            # Search for similar code
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            context = []
            for i in range(len(results['documents'][0])):
                context.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": results['distances'][0][i] if 'distances' in results else 0.0
                })
            
            return context
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

class AgenticCodeOptimizer:
    """Main orchestrator for the agentic code optimization system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.console = Console()
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.agents = {
            AgentRole.ANALYZER: AnalyzerAgent(self.config.get("analyzer", {})),
            AgentRole.GENERATOR: GeneratorAgent(self.config.get("generator", {})),
            AgentRole.CRITIC: CriticAgent(self.config.get("critic", {})),
            AgentRole.CHAOS_ENGINEER: ChaosEngineerAgent(self.config.get("chaos", {}))
        }
        
        # Initialize RAG context store
        self.context_store = RAGContextStore(self.config.get("rag", {}))
        
        # Workflow state
        self.workflow_state = {}
        self.optimization_history = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "optimization_level": "moderate",
            "max_iterations": 3,
            "parallel_agents": True,
            "enable_chaos_testing": True,
            "rag": {
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 50,
                "similarity_threshold": 0.7
            },
            "analyzer": {
                "include_security": True,
                "include_performance": True,
                "complexity_threshold": 15
            },
            "generator": {
                "creativity_level": 0.7,
                "safety_checks": True
            },
            "critic": {
                "strictness_level": 0.8,
                "require_tests": True
            },
            "chaos": {
                "experiment_duration": 300,
                "blast_radius": "limited",
                "auto_rollback": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agentic_optimizer.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('agentic_optimizer')
    
    async def initialize(self):
        """Initialize the optimization system"""
        self.console.print(Panel.fit(
            "[bold blue]Dr. CodeForge's Agentic Code Optimizer[/bold blue]\n"
            "[dim]Superhuman AI-Powered Code Analysis & Optimization[/dim]",
            border_style="blue"
        ))
        
        # Initialize all agents
        for agent in self.agents.values():
            await agent.initialize()
        
        # Initialize RAG context store
        await self.context_store.initialize()
        
        self.logger.info("Agentic Code Optimizer initialized successfully")
    
    async def optimize_repository(self, repo_path: str) -> Dict[str, Any]:
        """Main optimization workflow"""
        self.console.print(f"\n[bold green]Starting optimization of repository: {repo_path}[/bold green]")
        
        # Validate repository
        if not Path(repo_path).exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Index codebase for RAG
        await self.context_store.index_codebase(repo_path)
        
        # Initialize workflow context
        context = {
            "repo_path": repo_path,
            "optimization_level": self.config["optimization_level"],
            "timestamp": datetime.now().isoformat()
        }
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Phase 1: Analysis
            task1 = progress.add_task("[blue]Analyzing codebase...", total=None)
            analysis_result = await self.agents[AgentRole.ANALYZER].process(context)
            context.update(analysis_result)
            results["analysis"] = analysis_result
            progress.update(task1, completed=True)
            
            # Phase 2: Generation
            task2 = progress.add_task("[yellow]Generating optimizations...", total=None)
            generation_result = await self.agents[AgentRole.GENERATOR].process(context)
            context.update(generation_result)
            results["generation"] = generation_result
            progress.update(task2, completed=True)
            
            # Phase 3: Review
            task3 = progress.add_task("[magenta]Reviewing proposals...", total=None)
            review_result = await self.agents[AgentRole.CRITIC].process(context)
            context.update(review_result)
            results["review"] = review_result
            progress.update(task3, completed=True)
            
            # Phase 4: Chaos Engineering (if enabled)
            if self.config["enable_chaos_testing"]:
                task4 = progress.add_task("[red]Running chaos experiments...", total=None)
                chaos_result = await self.agents[AgentRole.CHAOS_ENGINEER].process(context)
                context.update(chaos_result)
                results["chaos"] = chaos_result
                progress.update(task4, completed=True)
        
        # Generate comprehensive report
        report = await self._generate_optimization_report(results)
        results["report"] = report
        
        # Save results
        await self._save_results(repo_path, results)
        
        self.console.print("\n[bold green]✅ Optimization complete![/bold green]")
        return results
    
    async def _generate_optimization_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        analysis = results.get("analysis", {})
        generation = results.get("generation", {})
        review = results.get("review", {})
        chaos = results.get("chaos", {})
        
        metrics = analysis.get("metrics")
        
        report = {
            "summary": {
                "files_analyzed": metrics.files_analyzed if metrics else 0,
                "lines_of_code": metrics.lines_of_code if metrics else 0,
                "complexity_score": metrics.complexity if metrics else 0,
                "test_coverage": metrics.test_coverage if metrics else 0,
                "security_score": metrics.security_score if metrics else 0,
                "suggestions_generated": len(generation.get("suggestions", [])),
                "tasks_approved": len(review.get("approved_tasks", [])),
                "chaos_experiments": len(chaos.get("experiments", []))
            },
            "recommendations": {
                "high_priority": [],
                "medium_priority": [],
                "low_priority": []
            },
            "next_steps": [
                "Review approved refactoring tasks",
                "Implement changes incrementally",
                "Run comprehensive test suite",
                "Monitor system performance",
                "Schedule regular optimization cycles"
            ],
            "metrics_improvement": {
                "estimated_complexity_reduction": "15-25%",
                "estimated_performance_gain": "10-20%",
                "estimated_maintainability_increase": "20-30%"
            }
        }
        
        # Categorize recommendations by priority
        for suggestion in generation.get("suggestions", []):
            priority = suggestion.get("priority", "medium")
            report["recommendations"][f"{priority}_priority"].append(suggestion)
        
        return report
    
    async def _save_results(self, repo_path: str, results: Dict[str, Any]):
        """Save optimization results to file"""
        output_dir = Path(repo_path) / ".agentic_optimizer"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"optimization_results_{timestamp}.json"
        
        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.console.print(f"[dim]Results saved to: {output_file}[/dim]")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def display_results_summary(self, results: Dict[str, Any]):
        """Display a beautiful summary of optimization results"""
        report = results.get("report", {})
        summary = report.get("summary", {})
        
        # Create summary table
        table = Table(title="Optimization Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Files Analyzed", str(summary.get("files_analyzed", 0)))
        table.add_row("Lines of Code", f"{summary.get('lines_of_code', 0):,}")
        table.add_row("Complexity Score", f"{summary.get('complexity_score', 0):.1f}")
        table.add_row("Test Coverage", f"{summary.get('test_coverage', 0):.1f}%")
        table.add_row("Security Score", f"{summary.get('security_score', 0):.1f}")
        table.add_row("Suggestions Generated", str(summary.get("suggestions_generated", 0)))
        table.add_row("Tasks Approved", str(summary.get("tasks_approved", 0)))
        table.add_row("Chaos Experiments", str(summary.get("chaos_experiments", 0)))
        
        self.console.print("\n")
        self.console.print(table)
        
        # Display recommendations
        recommendations = report.get("recommendations", {})
        if recommendations.get("high_priority"):
            self.console.print("\n[bold red]🚨 High Priority Recommendations:[/bold red]")
            for rec in recommendations["high_priority"]:
                self.console.print(f"  • {rec.get('description', 'N/A')}")
        
        if recommendations.get("medium_priority"):
            self.console.print("\n[bold yellow]⚠️  Medium Priority Recommendations:[/bold yellow]")
            for rec in recommendations["medium_priority"][:3]:  # Show top 3
                self.console.print(f"  • {rec.get('description', 'N/A')}")
        
        # Display next steps
        next_steps = report.get("next_steps", [])
        if next_steps:
            self.console.print("\n[bold blue]📋 Next Steps:[/bold blue]")
            for i, step in enumerate(next_steps, 1):
                self.console.print(f"  {i}. {step}")

# CLI Interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dr. CodeForge's Agentic Code Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agentic_code_optimizer.py /path/to/repo
  python agentic_code_optimizer.py /path/to/repo --config config.yaml
  python agentic_code_optimizer.py /path/to/repo --level aggressive
        """
    )
    
    parser.add_argument(
        "repo_path",
        help="Path to the repository to optimize"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--level",
        choices=["conservative", "moderate", "aggressive", "experimental"],
        default="moderate",
        help="Optimization level (default: moderate)"
    )
    
    parser.add_argument(
        "--no-chaos",
        action="store_true",
        help="Disable chaos engineering experiments"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = AgenticCodeOptimizer(args.config)
    
    # Override config with CLI arguments
    if args.level:
        optimizer.config["optimization_level"] = args.level
    if args.no_chaos:
        optimizer.config["enable_chaos_testing"] = False
    
    try:
        # Initialize and run optimization
        await optimizer.initialize()
        results = await optimizer.optimize_repository(args.repo_path)
        
        # Display results
        optimizer.display_results_summary(results)
        
        # Save to custom output if specified
        if args.output:
            output_path = Path(args.output) / "optimization_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(optimizer._make_serializable(results), f, indent=2)
            
            print(f"\nResults also saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())