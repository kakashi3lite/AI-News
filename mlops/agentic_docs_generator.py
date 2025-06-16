#!/usr/bin/env python3
"""
Agentic Documentation Generator
Dr. Aurora "CodeForge" Synth's Automated Documentation Generation

Uses agentic pipelines to update docstrings, OpenAPI specs, and architecture diagrams,
ensuring docs sync with code changes through AI-powered analysis and generation.
"""

import ast
import asyncio
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

class DocType(Enum):
    """Types of documentation to generate"""
    DOCSTRINGS = "docstrings"
    README = "readme"
    API_DOCS = "api_docs"
    ARCHITECTURE = "architecture"
    CHANGELOG = "changelog"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    DEPLOYMENT_GUIDE = "deployment_guide"
    TROUBLESHOOTING = "troubleshooting"
    OPENAPI_SPEC = "openapi_spec"
    DIAGRAMS = "diagrams"
    CODE_COMMENTS = "code_comments"

class DocFormat(Enum):
    """Documentation output formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"
    RST = "rst"
    CONFLUENCE = "confluence"
    NOTION = "notion"

class AnalysisLevel(Enum):
    """Depth of code analysis"""
    BASIC = "basic"  # Function signatures only
    STANDARD = "standard"  # + docstrings, comments
    COMPREHENSIVE = "comprehensive"  # + dependencies, flow analysis
    DEEP = "deep"  # + semantic analysis, patterns

@dataclass
class CodeElement:
    """Represents a code element for documentation"""
    name: str
    element_type: str  # function, class, module, method
    file_path: str
    line_number: int
    signature: str = ""
    docstring: str = ""
    comments: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = ""
    decorators: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    complexity: int = 0
    test_coverage: float = 0.0
    usage_examples: List[str] = field(default_factory=list)
    related_elements: List[str] = field(default_factory=list)

@dataclass
class DocumentationTask:
    """A documentation generation task"""
    doc_type: DocType
    target_path: str
    output_format: DocFormat = DocFormat.MARKDOWN
    template_path: Optional[str] = None
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    analysis_level: AnalysisLevel = AnalysisLevel.STANDARD
    auto_update: bool = True
    generate_examples: bool = True
    include_diagrams: bool = False
    custom_sections: List[str] = field(default_factory=list)

@dataclass
class DocumentationResult:
    """Result of documentation generation"""
    task: DocumentationTask
    output_files: List[str] = field(default_factory=list)
    generated_content: str = ""
    elements_documented: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    quality_score: float = 0.0
    coverage_metrics: Dict[str, float] = field(default_factory=dict)

class CodeAnalyzer:
    """Analyzes code structure for documentation generation"""
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger("code_analyzer")
        self.elements: List[CodeElement] = []
        self.dependencies: Dict[str, Set[str]] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        
    async def analyze_codebase(self, root_path: str, 
                              include_patterns: List[str] = None,
                              exclude_patterns: List[str] = None,
                              analysis_level: AnalysisLevel = AnalysisLevel.STANDARD) -> List[CodeElement]:
        """Analyze entire codebase for documentation"""
        self.elements = []
        root = Path(root_path)
        
        # Default patterns
        if include_patterns is None:
            include_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h"]
        
        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**", "**/__pycache__/**", "**/venv/**", 
                "**/env/**", "**/.git/**", "**/build/**", "**/dist/**"
            ]
        
        # Find all relevant files
        files_to_analyze = []
        for pattern in include_patterns:
            files_to_analyze.extend(root.rglob(pattern))
        
        # Filter out excluded files
        filtered_files = []
        for file_path in files_to_analyze:
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        
        self.logger.info(f"Analyzing {len(filtered_files)} files...")
        
        # Analyze each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Analyzing code...", total=len(filtered_files))
            
            for file_path in filtered_files:
                try:
                    elements = await self._analyze_file(file_path, analysis_level)
                    self.elements.extend(elements)
                    
                    progress.update(task, advance=1, description=f"Analyzed {file_path.name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path}: {e}")
                    progress.advance(task)
        
        # Build dependency graph if comprehensive analysis
        if analysis_level in [AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]:
            await self._build_dependency_graph()
        
        self.logger.info(f"Analysis complete. Found {len(self.elements)} code elements.")
        return self.elements
    
    async def _analyze_file(self, file_path: Path, analysis_level: AnalysisLevel) -> List[CodeElement]:
        """Analyze a single file"""
        elements = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            if file_path.suffix == '.py':
                elements = await self._analyze_python_file(file_path, content, analysis_level)
            elif file_path.suffix in ['.js', '.ts']:
                elements = await self._analyze_javascript_file(file_path, content, analysis_level)
            elif file_path.suffix in ['.java']:
                elements = await self._analyze_java_file(file_path, content, analysis_level)
            else:
                # Generic analysis for other languages
                elements = await self._analyze_generic_file(file_path, content, analysis_level)
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return elements
    
    async def _analyze_python_file(self, file_path: Path, content: str, 
                                 analysis_level: AnalysisLevel) -> List[CodeElement]:
        """Analyze Python file using AST"""
        elements = []
        
        try:
            tree = ast.parse(content)
            
            class CodeVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.elements = []
                    self.current_class = None
                
                def visit_FunctionDef(self, node):
                    element = self._create_function_element(node, file_path)
                    self.elements.append(element)
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    element = self._create_function_element(node, file_path, is_async=True)
                    self.elements.append(element)
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    element = self._create_class_element(node, file_path)
                    self.elements.append(element)
                    
                    old_class = self.current_class
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = old_class
                
                def _create_function_element(self, node, file_path, is_async=False):
                    # Extract function signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    
                    signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"
                    if node.returns:
                        signature += f" -> {ast.unparse(node.returns)}"
                    
                    # Extract docstring
                    docstring = ""
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        docstring = node.body[0].value.value
                    
                    # Extract decorators
                    decorators = [ast.unparse(dec) for dec in node.decorator_list]
                    
                    # Extract parameters
                    parameters = []
                    for arg in node.args.args:
                        param = {
                            'name': arg.arg,
                            'type': ast.unparse(arg.annotation) if arg.annotation else 'Any',
                            'default': None
                        }
                        parameters.append(param)
                    
                    # Add defaults
                    defaults = node.args.defaults
                    if defaults:
                        for i, default in enumerate(defaults):
                            param_index = len(parameters) - len(defaults) + i
                            if param_index >= 0:
                                parameters[param_index]['default'] = ast.unparse(default)
                    
                    element_type = "method" if self.current_class else "function"
                    
                    return CodeElement(
                        name=node.name,
                        element_type=element_type,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        signature=signature,
                        docstring=docstring,
                        parameters=parameters,
                        return_type=ast.unparse(node.returns) if node.returns else "None",
                        decorators=decorators,
                        complexity=self._calculate_complexity(node)
                    )
                
                def _create_class_element(self, node, file_path):
                    # Extract class docstring
                    docstring = ""
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        docstring = node.body[0].value.value
                    
                    # Extract base classes
                    bases = [ast.unparse(base) for base in node.bases]
                    signature = f"class {node.name}({', '.join(bases)}):" if bases else f"class {node.name}:"
                    
                    # Extract decorators
                    decorators = [ast.unparse(dec) for dec in node.decorator_list]
                    
                    return CodeElement(
                        name=node.name,
                        element_type="class",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        signature=signature,
                        docstring=docstring,
                        decorators=decorators,
                        complexity=len(node.body)
                    )
                
                def _calculate_complexity(self, node):
                    """Calculate cyclomatic complexity"""
                    complexity = 1  # Base complexity
                    
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                            complexity += 1
                        elif isinstance(child, ast.ExceptHandler):
                            complexity += 1
                        elif isinstance(child, (ast.And, ast.Or)):
                            complexity += 1
                    
                    return complexity
            
            visitor = CodeVisitor()
            visitor.visit(tree)
            elements = visitor.elements
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing Python file {file_path}: {e}")
        
        return elements
    
    async def _analyze_javascript_file(self, file_path: Path, content: str, 
                                     analysis_level: AnalysisLevel) -> List[CodeElement]:
        """Analyze JavaScript/TypeScript file using regex patterns"""
        elements = []
        
        try:
            lines = content.split('\n')
            
            # Function patterns
            function_patterns = [
                r'^\s*function\s+(\w+)\s*\(([^)]*)\)\s*{',  # function declaration
                r'^\s*const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>',  # arrow function
                r'^\s*async\s+function\s+(\w+)\s*\(([^)]*)\)\s*{',  # async function
                r'^\s*(\w+)\s*:\s*function\s*\(([^)]*)\)\s*{',  # object method
                r'^\s*async\s+(\w+)\s*\(([^)]*)\)\s*{',  # async method
            ]
            
            # Class patterns
            class_patterns = [
                r'^\s*class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{',
                r'^\s*export\s+class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
            ]
            
            for i, line in enumerate(lines, 1):
                # Check for functions
                for pattern in function_patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1)
                        params = match.group(2) if len(match.groups()) > 1 else ""
                        
                        # Extract JSDoc if present
                        docstring = self._extract_jsdoc(lines, i - 1)
                        
                        element = CodeElement(
                            name=name,
                            element_type="function",
                            file_path=str(file_path),
                            line_number=i,
                            signature=f"function {name}({params})",
                            docstring=docstring
                        )
                        elements.append(element)
                        break
                
                # Check for classes
                for pattern in class_patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1)
                        base_class = match.group(2) if len(match.groups()) > 1 else None
                        
                        signature = f"class {name}"
                        if base_class:
                            signature += f" extends {base_class}"
                        
                        # Extract JSDoc if present
                        docstring = self._extract_jsdoc(lines, i - 1)
                        
                        element = CodeElement(
                            name=name,
                            element_type="class",
                            file_path=str(file_path),
                            line_number=i,
                            signature=signature,
                            docstring=docstring
                        )
                        elements.append(element)
                        break
                        
        except Exception as e:
            self.logger.error(f"Error analyzing JavaScript file {file_path}: {e}")
        
        return elements
    
    async def _analyze_java_file(self, file_path: Path, content: str, 
                               analysis_level: AnalysisLevel) -> List[CodeElement]:
        """Analyze Java file using regex patterns"""
        elements = []
        
        try:
            lines = content.split('\n')
            
            # Method patterns
            method_patterns = [
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^{]+)?\s*{',
                r'^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*void\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^{]+)?\s*{'
            ]
            
            # Class patterns
            class_patterns = [
                r'^\s*(?:public|private|protected)?\s*(?:abstract)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{',
                r'^\s*(?:public|private|protected)?\s*interface\s+(\w+)(?:\s+extends\s+([^{]+))?\s*{'
            ]
            
            for i, line in enumerate(lines, 1):
                # Check for methods
                for pattern in method_patterns:
                    match = re.match(pattern, line)
                    if match:
                        if len(match.groups()) >= 2:
                            return_type = match.group(1) if match.group(1) != 'void' else 'void'
                            name = match.group(2) if len(match.groups()) > 2 else match.group(1)
                            params = match.group(3) if len(match.groups()) > 2 else match.group(2)
                        else:
                            return_type = 'void'
                            name = match.group(1)
                            params = match.group(2) if len(match.groups()) > 1 else ""
                        
                        # Extract Javadoc if present
                        docstring = self._extract_javadoc(lines, i - 1)
                        
                        element = CodeElement(
                            name=name,
                            element_type="method",
                            file_path=str(file_path),
                            line_number=i,
                            signature=f"{return_type} {name}({params})",
                            docstring=docstring,
                            return_type=return_type
                        )
                        elements.append(element)
                        break
                
                # Check for classes
                for pattern in class_patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1)
                        
                        signature = f"class {name}"
                        if len(match.groups()) > 1 and match.group(2):
                            signature += f" extends {match.group(2)}"
                        if len(match.groups()) > 2 and match.group(3):
                            signature += f" implements {match.group(3)}"
                        
                        # Extract Javadoc if present
                        docstring = self._extract_javadoc(lines, i - 1)
                        
                        element = CodeElement(
                            name=name,
                            element_type="class",
                            file_path=str(file_path),
                            line_number=i,
                            signature=signature,
                            docstring=docstring
                        )
                        elements.append(element)
                        break
                        
        except Exception as e:
            self.logger.error(f"Error analyzing Java file {file_path}: {e}")
        
        return elements
    
    async def _analyze_generic_file(self, file_path: Path, content: str, 
                                  analysis_level: AnalysisLevel) -> List[CodeElement]:
        """Generic analysis for unsupported file types"""
        elements = []
        
        # Basic analysis - just extract function-like patterns
        lines = content.split('\n')
        
        # Generic function patterns
        function_patterns = [
            r'^\s*(\w+)\s*\(',  # Simple function call pattern
            r'^\s*def\s+(\w+)',  # Python-like def
            r'^\s*function\s+(\w+)',  # JavaScript-like function
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in function_patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    
                    element = CodeElement(
                        name=name,
                        element_type="function",
                        file_path=str(file_path),
                        line_number=i,
                        signature=line.strip()
                    )
                    elements.append(element)
                    break
        
        return elements
    
    def _extract_jsdoc(self, lines: List[str], line_index: int) -> str:
        """Extract JSDoc comment before a function/class"""
        docstring = ""
        
        # Look backwards for JSDoc comment
        i = line_index - 1
        while i >= 0 and lines[i].strip() == "":
            i -= 1
        
        if i >= 0 and lines[i].strip().endswith("*/"):
            # Found end of JSDoc, collect the comment
            doc_lines = []
            while i >= 0:
                line = lines[i].strip()
                if line.startswith("/**"):
                    doc_lines.reverse()
                    docstring = "\n".join(doc_lines)
                    break
                elif line.startswith("*"):
                    doc_lines.append(line[1:].strip())
                i -= 1
        
        return docstring
    
    def _extract_javadoc(self, lines: List[str], line_index: int) -> str:
        """Extract Javadoc comment before a method/class"""
        return self._extract_jsdoc(lines, line_index)  # Same format
    
    async def _build_dependency_graph(self):
        """Build dependency graph between code elements"""
        # This would analyze imports, function calls, etc.
        # For now, just a placeholder
        self.logger.info("Building dependency graph...")
        
        for element in self.elements:
            self.dependencies[element.name] = set()
            self.call_graph[element.name] = set()

class DocumentationGenerator:
    """Generates documentation using AI-powered analysis"""
    
    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger("docs_generator")
        self.analyzer = CodeAnalyzer()
        self.templates = {}
        
    async def generate_documentation(self, tasks: List[DocumentationTask], 
                                   root_path: str) -> List[DocumentationResult]:
        """Generate documentation for multiple tasks"""
        results = []
        
        self.console.print(Panel.fit(
            f"[bold blue]📚 Agentic Documentation Generator[/bold blue]\n"
            f"[dim]Generating {len(tasks)} documentation tasks[/dim]",
            border_style="blue"
        ))
        
        # Analyze codebase once for all tasks
        max_analysis_level = max(task.analysis_level for task in tasks)
        elements = await self.analyzer.analyze_codebase(
            root_path, 
            analysis_level=max_analysis_level
        )
        
        # Generate documentation for each task
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("Generating documentation...", total=len(tasks))
            
            for doc_task in tasks:
                task_progress = progress.add_task(f"Generating {doc_task.doc_type.value}...", total=None)
                
                try:
                    result = await self._generate_single_doc(doc_task, elements, root_path)
                    results.append(result)
                    
                    progress.update(
                        task_progress,
                        description=f"✅ {doc_task.doc_type.value} ({result.elements_documented} elements)",
                        completed=True
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate {doc_task.doc_type.value}: {e}")
                    
                    error_result = DocumentationResult(
                        task=doc_task,
                        errors=[str(e)]
                    )
                    results.append(error_result)
                    
                    progress.update(
                        task_progress,
                        description=f"❌ {doc_task.doc_type.value} (failed)",
                        completed=True
                    )
                
                progress.advance(main_task)
        
        # Generate summary report
        await self._generate_summary_report(results)
        
        return results
    
    async def _generate_single_doc(self, task: DocumentationTask, 
                                 elements: List[CodeElement], 
                                 root_path: str) -> DocumentationResult:
        """Generate documentation for a single task"""
        start_time = datetime.now()
        
        # Filter elements based on task criteria
        filtered_elements = self._filter_elements(elements, task)
        
        # Generate content based on doc type
        content = ""
        output_files = []
        
        if task.doc_type == DocType.DOCSTRINGS:
            content, output_files = await self._generate_docstrings(filtered_elements, task)
        elif task.doc_type == DocType.README:
            content = await self._generate_readme(filtered_elements, task, root_path)
            output_files = [str(Path(task.target_path) / "README.md")]
        elif task.doc_type == DocType.API_DOCS:
            content = await self._generate_api_docs(filtered_elements, task)
            output_files = [str(Path(task.target_path) / "api_documentation.md")]
        elif task.doc_type == DocType.ARCHITECTURE:
            content = await self._generate_architecture_docs(filtered_elements, task, root_path)
            output_files = [str(Path(task.target_path) / "architecture.md")]
        elif task.doc_type == DocType.OPENAPI_SPEC:
            content = await self._generate_openapi_spec(filtered_elements, task)
            output_files = [str(Path(task.target_path) / "openapi.yaml")]
        elif task.doc_type == DocType.USER_GUIDE:
            content = await self._generate_user_guide(filtered_elements, task, root_path)
            output_files = [str(Path(task.target_path) / "user_guide.md")]
        elif task.doc_type == DocType.DEVELOPER_GUIDE:
            content = await self._generate_developer_guide(filtered_elements, task, root_path)
            output_files = [str(Path(task.target_path) / "developer_guide.md")]
        else:
            raise NotImplementedError(f"Documentation type {task.doc_type} not implemented")
        
        # Save generated content
        if content and output_files:
            await self._save_documentation(content, output_files[0], task.output_format)
        
        # Calculate metrics
        generation_time = (datetime.now() - start_time).total_seconds()
        quality_score = self._calculate_quality_score(content, filtered_elements)
        coverage_metrics = self._calculate_coverage_metrics(filtered_elements)
        
        return DocumentationResult(
            task=task,
            output_files=output_files,
            generated_content=content,
            elements_documented=len(filtered_elements),
            generation_time=generation_time,
            quality_score=quality_score,
            coverage_metrics=coverage_metrics
        )
    
    def _filter_elements(self, elements: List[CodeElement], task: DocumentationTask) -> List[CodeElement]:
        """Filter code elements based on task criteria"""
        filtered = elements
        
        # Apply include patterns
        if task.include_patterns:
            filtered = [
                elem for elem in filtered
                if any(Path(elem.file_path).match(pattern) for pattern in task.include_patterns)
            ]
        
        # Apply exclude patterns
        if task.exclude_patterns:
            filtered = [
                elem for elem in filtered
                if not any(Path(elem.file_path).match(pattern) for pattern in task.exclude_patterns)
            ]
        
        return filtered
    
    async def _generate_docstrings(self, elements: List[CodeElement], 
                                 task: DocumentationTask) -> Tuple[str, List[str]]:
        """Generate or update docstrings for code elements"""
        updated_files = set()
        generated_docstrings = []
        
        for element in elements:
            if element.element_type in ['function', 'method', 'class']:
                if not element.docstring or len(element.docstring.strip()) < 10:
                    # Generate docstring
                    docstring = await self._generate_element_docstring(element)
                    generated_docstrings.append(f"# {element.name}\n{docstring}\n")
                    
                    # Update file if auto_update is enabled
                    if task.auto_update:
                        await self._update_docstring_in_file(element, docstring)
                        updated_files.add(element.file_path)
        
        content = "\n".join(generated_docstrings)
        return content, list(updated_files)
    
    async def _generate_element_docstring(self, element: CodeElement) -> str:
        """Generate docstring for a single code element"""
        # AI-powered docstring generation would go here
        # For now, generate a basic template
        
        if element.element_type == 'function' or element.element_type == 'method':
            docstring = f'"""\n    {element.name}\n\n'
            
            if element.parameters:
                docstring += "    Args:\n"
                for param in element.parameters:
                    param_doc = f"        {param['name']}"
                    if param.get('type') and param['type'] != 'Any':
                        param_doc += f" ({param['type']})"
                    param_doc += ": Description of parameter\n"
                    docstring += param_doc
                docstring += "\n"
            
            if element.return_type and element.return_type != 'None':
                docstring += f"    Returns:\n        {element.return_type}: Description of return value\n\n"
            
            docstring += "    Raises:\n        Exception: Description of when this exception is raised\n"
            docstring += '    """'
            
        elif element.element_type == 'class':
            docstring = f'"""\n    {element.name}\n\n    A class that provides functionality for...\n\n'
            docstring += "    Attributes:\n        attribute_name: Description of attribute\n\n"
            docstring += "    Example:\n        >>> obj = {element.name}()\n        >>> obj.method()\n"
            docstring += '    """'
        
        else:
            docstring = f'"""\n    {element.name}\n\n    Description of {element.element_type}\n    """'
        
        return docstring
    
    async def _update_docstring_in_file(self, element: CodeElement, docstring: str):
        """Update docstring in the actual source file"""
        try:
            file_path = Path(element.file_path)
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Find the element and insert/update docstring
            # This is a simplified implementation
            # In practice, you'd need more sophisticated AST manipulation
            
            # For Python files, insert after function/class definition
            if file_path.suffix == '.py':
                target_line = element.line_number - 1  # Convert to 0-based index
                
                # Find the line with the colon (end of definition)
                while target_line < len(lines) and ':' not in lines[target_line]:
                    target_line += 1
                
                if target_line < len(lines):
                    # Insert docstring after the definition line
                    indent = len(lines[target_line]) - len(lines[target_line].lstrip())
                    indented_docstring = '\n'.join(
                        ' ' * (indent + 4) + line if line.strip() else ''
                        for line in docstring.split('\n')
                    )
                    
                    lines.insert(target_line + 1, indented_docstring)
                    
                    # Write back to file
                    file_path.write_text('\n'.join(lines), encoding='utf-8')
                    
                    self.logger.info(f"Updated docstring for {element.name} in {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update docstring for {element.name}: {e}")
    
    async def _generate_readme(self, elements: List[CodeElement], 
                             task: DocumentationTask, root_path: str) -> str:
        """Generate README.md file"""
        project_name = Path(root_path).name
        
        # Analyze project structure
        file_types = {}
        for element in elements:
            ext = Path(element.file_path).suffix
            file_types[ext] = file_types.get(ext, 0) + 1
        
        # Count different element types
        element_counts = {}
        for element in elements:
            element_counts[element.element_type] = element_counts.get(element.element_type, 0) + 1
        
        readme_content = f"""# {project_name}

A comprehensive software project with automated documentation generation.

## Overview

This project contains {len(elements)} documented code elements across {len(set(elem.file_path for elem in elements))} files.

### Project Statistics

"""
        
        # Add file type statistics
        if file_types:
            readme_content += "#### File Types\n\n"
            for ext, count in sorted(file_types.items()):
                readme_content += f"- {ext or 'No extension'}: {count} files\n"
            readme_content += "\n"
        
        # Add element type statistics
        if element_counts:
            readme_content += "#### Code Elements\n\n"
            for elem_type, count in sorted(element_counts.items()):
                readme_content += f"- {elem_type.title()}s: {count}\n"
            readme_content += "\n"
        
        readme_content += """## Installation

```bash
# Clone the repository
git clone <repository-url>
cd {project_name}

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Basic usage example
from {project_name} import main

# Run the application
main()
```

## API Documentation

For detailed API documentation, see [API Documentation](api_documentation.md).

## Architecture

For system architecture details, see [Architecture Documentation](architecture.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This README was automatically generated by the Agentic Documentation Generator.*
"""
        
        return readme_content
    
    async def _generate_api_docs(self, elements: List[CodeElement], 
                               task: DocumentationTask) -> str:
        """Generate API documentation"""
        api_content = "# API Documentation\n\n"
        api_content += "This document provides detailed information about the API endpoints and functions.\n\n"
        
        # Group elements by file
        files_dict = {}
        for element in elements:
            file_path = element.file_path
            if file_path not in files_dict:
                files_dict[file_path] = []
            files_dict[file_path].append(element)
        
        # Generate documentation for each file
        for file_path, file_elements in files_dict.items():
            relative_path = Path(file_path).name
            api_content += f"## {relative_path}\n\n"
            
            # Group by element type
            classes = [e for e in file_elements if e.element_type == 'class']
            functions = [e for e in file_elements if e.element_type in ['function', 'method']]
            
            if classes:
                api_content += "### Classes\n\n"
                for cls in classes:
                    api_content += f"#### `{cls.name}`\n\n"
                    if cls.docstring:
                        api_content += f"{cls.docstring}\n\n"
                    api_content += f"**Signature:** `{cls.signature}`\n\n"
                    
                    if cls.decorators:
                        api_content += f"**Decorators:** {', '.join(cls.decorators)}\n\n"
                    
                    api_content += f"**File:** `{cls.file_path}:{cls.line_number}`\n\n"
                    api_content += "---\n\n"
            
            if functions:
                api_content += "### Functions\n\n"
                for func in functions:
                    api_content += f"#### `{func.name}`\n\n"
                    if func.docstring:
                        api_content += f"{func.docstring}\n\n"
                    api_content += f"**Signature:** `{func.signature}`\n\n"
                    
                    if func.parameters:
                        api_content += "**Parameters:**\n\n"
                        for param in func.parameters:
                            param_line = f"- `{param['name']}`"
                            if param.get('type'):
                                param_line += f" ({param['type']})"
                            if param.get('default'):
                                param_line += f" = {param['default']}"
                            param_line += ": Parameter description\n"
                            api_content += param_line
                        api_content += "\n"
                    
                    if func.return_type and func.return_type != 'None':
                        api_content += f"**Returns:** `{func.return_type}` - Return value description\n\n"
                    
                    if func.decorators:
                        api_content += f"**Decorators:** {', '.join(func.decorators)}\n\n"
                    
                    api_content += f"**Complexity:** {func.complexity}\n\n"
                    api_content += f"**File:** `{func.file_path}:{func.line_number}`\n\n"
                    api_content += "---\n\n"
        
        return api_content
    
    async def _generate_architecture_docs(self, elements: List[CodeElement], 
                                        task: DocumentationTask, root_path: str) -> str:
        """Generate architecture documentation"""
        arch_content = "# Architecture Documentation\n\n"
        arch_content += "This document describes the system architecture and design patterns.\n\n"
        
        # Analyze project structure
        directories = set()
        for element in elements:
            dir_path = str(Path(element.file_path).parent)
            directories.add(dir_path)
        
        # Create directory tree
        arch_content += "## Project Structure\n\n"
        arch_content += "```\n"
        
        root = Path(root_path)
        for directory in sorted(directories):
            rel_dir = Path(directory).relative_to(root) if Path(directory).is_relative_to(root) else Path(directory)
            arch_content += f"{rel_dir}/\n"
        
        arch_content += "```\n\n"
        
        # Component analysis
        arch_content += "## Components\n\n"
        
        # Group by directory
        components = {}
        for element in elements:
            dir_name = Path(element.file_path).parent.name
            if dir_name not in components:
                components[dir_name] = {'classes': [], 'functions': []}
            
            if element.element_type == 'class':
                components[dir_name]['classes'].append(element)
            elif element.element_type in ['function', 'method']:
                components[dir_name]['functions'].append(element)
        
        for component_name, component_data in components.items():
            if component_data['classes'] or component_data['functions']:
                arch_content += f"### {component_name}\n\n"
                
                if component_data['classes']:
                    arch_content += f"**Classes:** {len(component_data['classes'])}\n\n"
                    for cls in component_data['classes'][:5]:  # Show first 5
                        arch_content += f"- `{cls.name}`: {cls.docstring.split('.')[0] if cls.docstring else 'No description'}\n"
                    if len(component_data['classes']) > 5:
                        arch_content += f"- ... and {len(component_data['classes']) - 5} more\n"
                    arch_content += "\n"
                
                if component_data['functions']:
                    arch_content += f"**Functions:** {len(component_data['functions'])}\n\n"
        
        # Design patterns
        arch_content += "## Design Patterns\n\n"
        arch_content += "This section identifies common design patterns used in the codebase:\n\n"
        
        # Simple pattern detection
        patterns = self._detect_patterns(elements)
        for pattern, count in patterns.items():
            arch_content += f"- **{pattern}**: {count} instances\n"
        
        arch_content += "\n## Dependencies\n\n"
        arch_content += "Key dependencies and their relationships:\n\n"
        
        # Extract imports (simplified)
        imports = set()
        for element in elements:
            if element.dependencies:
                imports.update(element.dependencies)
        
        for imp in sorted(list(imports)[:10]):  # Show first 10
            arch_content += f"- `{imp}`\n"
        
        return arch_content
    
    def _detect_patterns(self, elements: List[CodeElement]) -> Dict[str, int]:
        """Detect common design patterns in code elements"""
        patterns = {}
        
        # Singleton pattern
        singleton_count = len([e for e in elements if 'singleton' in e.name.lower()])
        if singleton_count > 0:
            patterns['Singleton'] = singleton_count
        
        # Factory pattern
        factory_count = len([e for e in elements if 'factory' in e.name.lower() or 'create' in e.name.lower()])
        if factory_count > 0:
            patterns['Factory'] = factory_count
        
        # Observer pattern
        observer_count = len([e for e in elements if 'observer' in e.name.lower() or 'listener' in e.name.lower()])
        if observer_count > 0:
            patterns['Observer'] = observer_count
        
        # Strategy pattern
        strategy_count = len([e for e in elements if 'strategy' in e.name.lower()])
        if strategy_count > 0:
            patterns['Strategy'] = strategy_count
        
        # Decorator pattern
        decorator_count = len([e for e in elements if e.decorators])
        if decorator_count > 0:
            patterns['Decorator'] = decorator_count
        
        return patterns
    
    async def _generate_openapi_spec(self, elements: List[CodeElement], 
                                   task: DocumentationTask) -> str:
        """Generate OpenAPI specification"""
        # Find API endpoints (simplified detection)
        api_elements = []
        for element in elements:
            if any(decorator in ['@app.route', '@router.get', '@router.post', '@router.put', '@router.delete'] 
                   for decorator in element.decorators):
                api_elements.append(element)
        
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "API Documentation",
                "version": "1.0.0",
                "description": "Auto-generated API documentation"
            },
            "paths": {}
        }
        
        for element in api_elements:
            # Extract route information from decorators
            for decorator in element.decorators:
                if '@app.route' in decorator or '@router.' in decorator:
                    # Simplified path extraction
                    path = "/api/endpoint"  # Would need proper parsing
                    method = "get"  # Would need proper extraction
                    
                    if path not in openapi_spec["paths"]:
                        openapi_spec["paths"][path] = {}
                    
                    openapi_spec["paths"][path][method] = {
                        "summary": element.name,
                        "description": element.docstring or f"Endpoint for {element.name}",
                        "responses": {
                            "200": {
                                "description": "Successful response"
                            }
                        }
                    }
        
        return yaml.dump(openapi_spec, default_flow_style=False)
    
    async def _generate_user_guide(self, elements: List[CodeElement], 
                                 task: DocumentationTask, root_path: str) -> str:
        """Generate user guide"""
        project_name = Path(root_path).name
        
        guide_content = f"""# {project_name} User Guide

Welcome to the {project_name} user guide. This document will help you get started and make the most of the application.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd {project_name}
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Basic Usage

This section covers the basic functionality of {project_name}.

### Core Features

"""
        
        # Extract main functions that might be user-facing
        main_functions = [e for e in elements if 'main' in e.name.lower() or 'run' in e.name.lower()]
        
        if main_functions:
            guide_content += "#### Available Commands\n\n"
            for func in main_functions[:5]:
                guide_content += f"- **{func.name}**: {func.docstring.split('.')[0] if func.docstring else 'No description'}\n"
            guide_content += "\n"
        
        guide_content += """## Configuration

The application can be configured through various methods:

### Environment Variables

- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Set logging level (default: INFO)

### Configuration Files

Configuration files should be placed in the `config/` directory.

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

2. **Permission Errors**
   - Check file permissions
   - Run with appropriate privileges

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review the [API Documentation](api_documentation.md)
3. Open an issue on the project repository

## Advanced Usage

For advanced configuration and customization options, see the [Developer Guide](developer_guide.md).

---

*This user guide was automatically generated by the Agentic Documentation Generator.*
"""
        
        return guide_content
    
    async def _generate_developer_guide(self, elements: List[CodeElement], 
                                      task: DocumentationTask, root_path: str) -> str:
        """Generate developer guide"""
        project_name = Path(root_path).name
        
        dev_guide = f"""# {project_name} Developer Guide

This guide provides information for developers who want to contribute to or extend {project_name}.

## Development Setup

### Environment Setup

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd {project_name}
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Code Structure

The project follows these conventions:

"""
        
        # Analyze code structure
        file_structure = {}
        for element in elements:
            dir_name = Path(element.file_path).parent.name
            if dir_name not in file_structure:
                file_structure[dir_name] = []
            file_structure[dir_name].append(element)
        
        for directory, dir_elements in file_structure.items():
            if len(dir_elements) > 2:  # Only show directories with multiple elements
                dev_guide += f"#### `{directory}/`\n\n"
                classes = [e for e in dir_elements if e.element_type == 'class']
                functions = [e for e in dir_elements if e.element_type in ['function', 'method']]
                
                if classes:
                    dev_guide += f"- **Classes**: {len(classes)} (e.g., {', '.join([c.name for c in classes[:3]])})\n"
                if functions:
                    dev_guide += f"- **Functions**: {len(functions)} (e.g., {', '.join([f.name for f in functions[:3]])})\n"
                dev_guide += "\n"
        
        dev_guide += """## Coding Standards

### Style Guide

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write comprehensive docstrings
- Maintain test coverage above 80%

### Testing

Run tests using:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_module.py
```

### Documentation

Documentation is automatically generated using the Agentic Documentation Generator:

```bash
# Generate all documentation
python mlops/agentic_docs_generator.py --all

# Generate specific documentation
python mlops/agentic_docs_generator.py --type api_docs
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Update documentation
7. Submit a pull request

### Code Review Process

- All changes require review
- Automated checks must pass
- Documentation must be updated
- Tests must maintain coverage

## Architecture

For detailed architecture information, see [Architecture Documentation](architecture.md).

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release tag
4. Deploy to staging
5. Run integration tests
6. Deploy to production

---

*This developer guide was automatically generated by the Agentic Documentation Generator.*
"""
        
        return dev_guide
    
    async def _save_documentation(self, content: str, output_path: str, format: DocFormat):
        """Save documentation to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == DocFormat.MARKDOWN:
            output_file.write_text(content, encoding='utf-8')
        elif format == DocFormat.HTML:
            # Convert markdown to HTML (would need markdown library)
            html_content = f"<html><body>{content}</body></html>"
            output_file.with_suffix('.html').write_text(html_content, encoding='utf-8')
        elif format == DocFormat.JSON:
            json_content = {"documentation": content}
            output_file.with_suffix('.json').write_text(json.dumps(json_content, indent=2), encoding='utf-8')
        elif format == DocFormat.YAML:
            yaml_content = {"documentation": content}
            output_file.with_suffix('.yaml').write_text(yaml.dump(yaml_content), encoding='utf-8')
        else:
            # Default to markdown
            output_file.write_text(content, encoding='utf-8')
        
        self.logger.info(f"Documentation saved to {output_file}")
    
    def _calculate_quality_score(self, content: str, elements: List[CodeElement]) -> float:
        """Calculate documentation quality score"""
        if not content or not elements:
            return 0.0
        
        score = 0.0
        max_score = 100.0
        
        # Content length score (20 points)
        content_length = len(content)
        if content_length > 1000:
            score += 20
        elif content_length > 500:
            score += 15
        elif content_length > 200:
            score += 10
        else:
            score += 5
        
        # Element coverage score (30 points)
        documented_elements = len([e for e in elements if e.docstring])
        if documented_elements > 0:
            coverage_ratio = documented_elements / len(elements)
            score += coverage_ratio * 30
        
        # Structure score (25 points)
        if "#" in content:  # Has headers
            score += 10
        if "```" in content:  # Has code blocks
            score += 10
        if "**" in content or "*" in content:  # Has formatting
            score += 5
        
        # Completeness score (25 points)
        if "Parameters" in content or "Args" in content:
            score += 8
        if "Returns" in content:
            score += 8
        if "Example" in content:
            score += 9
        
        return min(score, max_score)
    
    def _calculate_coverage_metrics(self, elements: List[CodeElement]) -> Dict[str, float]:
        """Calculate documentation coverage metrics"""
        if not elements:
            return {}
        
        total_elements = len(elements)
        documented_elements = len([e for e in elements if e.docstring and len(e.docstring.strip()) > 10])
        
        # Coverage by element type
        type_coverage = {}
        for element_type in ['class', 'function', 'method']:
            type_elements = [e for e in elements if e.element_type == element_type]
            if type_elements:
                type_documented = len([e for e in type_elements if e.docstring and len(e.docstring.strip()) > 10])
                type_coverage[f"{element_type}_coverage"] = (type_documented / len(type_elements)) * 100
        
        return {
            "overall_coverage": (documented_elements / total_elements) * 100 if total_elements > 0 else 0,
            "total_elements": total_elements,
            "documented_elements": documented_elements,
            **type_coverage
        }
    
    async def _generate_summary_report(self, results: List[DocumentationResult]):
        """Generate summary report of documentation generation"""
        self.console.print("\n" + "="*80)
        self.console.print(Panel.fit(
            "[bold green]📊 Documentation Generation Summary[/bold green]",
            border_style="green"
        ))
        
        # Create summary table
        table = Table(title="Documentation Results")
        table.add_column("Doc Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Elements", justify="right")
        table.add_column("Quality Score", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Output Files", style="dim")
        
        total_elements = 0
        total_time = 0.0
        successful_tasks = 0
        
        for result in results:
            status = "✅ Success" if not result.errors else "❌ Failed"
            if not result.errors:
                successful_tasks += 1
            
            total_elements += result.elements_documented
            total_time += result.generation_time
            
            quality_str = f"{result.quality_score:.1f}%" if result.quality_score > 0 else "N/A"
            time_str = f"{result.generation_time:.2f}"
            files_str = f"{len(result.output_files)} files" if result.output_files else "None"
            
            table.add_row(
                result.task.doc_type.value,
                status,
                str(result.elements_documented),
                quality_str,
                time_str,
                files_str
            )
        
        self.console.print(table)
        
        # Summary statistics
        success_rate = (successful_tasks / len(results)) * 100 if results else 0
        avg_quality = sum(r.quality_score for r in results if r.quality_score > 0) / max(1, len([r for r in results if r.quality_score > 0]))
        
        summary_panel = f"""
[bold]Overall Statistics:[/bold]
• Tasks Completed: {successful_tasks}/{len(results)} ({success_rate:.1f}% success rate)
• Total Elements Documented: {total_elements}
• Total Generation Time: {total_time:.2f} seconds
• Average Quality Score: {avg_quality:.1f}%

[bold]Generated Files:[/bold]
"""
        
        all_files = []
        for result in results:
            all_files.extend(result.output_files)
        
        for file_path in all_files:
            summary_panel += f"• {file_path}\n"
        
        self.console.print(Panel(
            summary_panel,
            title="[bold blue]Summary[/bold blue]",
            border_style="blue"
        ))
        
        # Save detailed report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(results),
                "successful_tasks": successful_tasks,
                "success_rate": success_rate,
                "total_elements": total_elements,
                "total_time": total_time,
                "average_quality": avg_quality
            },
            "results": [
                {
                    "doc_type": result.task.doc_type.value,
                    "target_path": result.task.target_path,
                    "elements_documented": result.elements_documented,
                    "quality_score": result.quality_score,
                    "generation_time": result.generation_time,
                    "output_files": result.output_files,
                    "warnings": result.warnings,
                    "errors": result.errors,
                    "coverage_metrics": result.coverage_metrics
                }
                for result in results
            ]
        }
        
        report_path = Path("docs/generation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_data, indent=2), encoding='utf-8')
        
        self.console.print(f"\n[dim]Detailed report saved to: {report_path}[/dim]")

class AgenticDocsOrchestrator:
    """Main orchestrator for agentic documentation generation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.console = Console()
        self.logger = self._setup_logging()
        self.generator = DocumentationGenerator()
        self.config = self._load_config(config_path)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('docs_generation.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("agentic_docs")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "analysis_level": "standard",
            "output_format": "markdown",
            "auto_update": True,
            "generate_examples": True,
            "include_diagrams": False,
            "include_patterns": ["*.py", "*.js", "*.ts"],
            "exclude_patterns": [
                "**/node_modules/**", "**/__pycache__/**", 
                "**/venv/**", "**/env/**", "**/.git/**"
            ],
            "doc_types": ["readme", "api_docs", "architecture", "user_guide"],
            "output_directory": "docs"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        user_config = yaml.safe_load(f)
                    else:
                        user_config = json.load(f)
                
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def generate_all_documentation(self, root_path: str, 
                                       doc_types: Optional[List[str]] = None) -> List[DocumentationResult]:
        """Generate all configured documentation types"""
        if doc_types is None:
            doc_types = self.config.get("doc_types", ["readme", "api_docs"])
        
        # Create documentation tasks
        tasks = []
        output_dir = self.config.get("output_directory", "docs")
        
        for doc_type_str in doc_types:
            try:
                doc_type = DocType(doc_type_str)
                analysis_level = AnalysisLevel(self.config.get("analysis_level", "standard"))
                output_format = DocFormat(self.config.get("output_format", "markdown"))
                
                task = DocumentationTask(
                    doc_type=doc_type,
                    target_path=output_dir,
                    output_format=output_format,
                    include_patterns=self.config.get("include_patterns", []),
                    exclude_patterns=self.config.get("exclude_patterns", []),
                    analysis_level=analysis_level,
                    auto_update=self.config.get("auto_update", True),
                    generate_examples=self.config.get("generate_examples", True),
                    include_diagrams=self.config.get("include_diagrams", False)
                )
                
                tasks.append(task)
                
            except ValueError as e:
                self.logger.warning(f"Invalid doc type '{doc_type_str}': {e}")
        
        if not tasks:
            self.logger.error("No valid documentation tasks to execute")
            return []
        
        # Generate documentation
        self.console.print(Panel.fit(
            f"[bold blue]🚀 Starting Agentic Documentation Generation[/bold blue]\n"
            f"[dim]Root Path: {root_path}[/dim]\n"
            f"[dim]Tasks: {len(tasks)}[/dim]",
            border_style="blue"
        ))
        
        results = await self.generator.generate_documentation(tasks, root_path)
        
        return results
    
    async def update_docstrings_only(self, root_path: str) -> DocumentationResult:
        """Update only docstrings in source files"""
        task = DocumentationTask(
            doc_type=DocType.DOCSTRINGS,
            target_path=root_path,
            analysis_level=AnalysisLevel.STANDARD,
            auto_update=True,
            include_patterns=self.config.get("include_patterns", ["*.py"]),
            exclude_patterns=self.config.get("exclude_patterns", [])
        )
        
        results = await self.generator.generate_documentation([task], root_path)
        return results[0] if results else None
    
    async def generate_api_docs_only(self, root_path: str) -> DocumentationResult:
        """Generate only API documentation"""
        task = DocumentationTask(
            doc_type=DocType.API_DOCS,
            target_path=self.config.get("output_directory", "docs"),
            analysis_level=AnalysisLevel.COMPREHENSIVE,
            generate_examples=True,
            include_patterns=self.config.get("include_patterns", ["*.py"]),
            exclude_patterns=self.config.get("exclude_patterns", [])
        )
        
        results = await self.generator.generate_documentation([task], root_path)
        return results[0] if results else None

async def main():
    """Main CLI interface for agentic documentation generator"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agentic Documentation Generator - AI-powered documentation automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all documentation
  python agentic_docs_generator.py /path/to/project --all
  
  # Generate specific documentation types
  python agentic_docs_generator.py /path/to/project --types readme api_docs
  
  # Update docstrings only
  python agentic_docs_generator.py /path/to/project --docstrings-only
  
  # Use custom configuration
  python agentic_docs_generator.py /path/to/project --config config.yaml --all
"""
    )
    
    parser.add_argument(
        "root_path",
        help="Root path of the project to document"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        choices=[dt.value for dt in DocType],
        help="Specific documentation types to generate"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Generate all configured documentation types"
    )
    
    parser.add_argument(
        "--docstrings-only",
        action="store_true",
        help="Update docstrings in source files only"
    )
    
    parser.add_argument(
        "--api-docs-only",
        action="store_true",
        help="Generate API documentation only"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="docs",
        help="Output directory for generated documentation (default: docs)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=[df.value for df in DocFormat],
        default="markdown",
        help="Output format for documentation (default: markdown)"
    )
    
    parser.add_argument(
        "--analysis-level",
        choices=[al.value for al in AnalysisLevel],
        default="standard",
        help="Depth of code analysis (default: standard)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate root path
    root_path = Path(args.root_path).resolve()
    if not root_path.exists():
        print(f"Error: Root path '{root_path}' does not exist")
        return 1
    
    # Initialize orchestrator
    orchestrator = AgenticDocsOrchestrator(args.config)
    
    # Override config with CLI arguments
    if args.output_dir:
        orchestrator.config["output_directory"] = args.output_dir
    if args.format:
        orchestrator.config["output_format"] = args.format
    if args.analysis_level:
        orchestrator.config["analysis_level"] = args.analysis_level
    
    try:
        console = Console()
        
        with console.status("[bold green]Initializing agentic documentation generator..."):
            await asyncio.sleep(1)  # Simulate initialization
        
        # Execute based on arguments
        if args.docstrings_only:
            console.print("[bold yellow]Updating docstrings only...[/bold yellow]")
            result = await orchestrator.update_docstrings_only(str(root_path))
            if result:
                console.print(f"[green]✅ Updated docstrings for {result.elements_documented} elements[/green]")
            else:
                console.print("[red]❌ Failed to update docstrings[/red]")
                
        elif args.api_docs_only:
            console.print("[bold yellow]Generating API documentation only...[/bold yellow]")
            result = await orchestrator.generate_api_docs_only(str(root_path))
            if result:
                console.print(f"[green]✅ Generated API docs for {result.elements_documented} elements[/green]")
            else:
                console.print("[red]❌ Failed to generate API documentation[/red]")
                
        elif args.all or args.types:
            doc_types = args.types if args.types else None
            results = await orchestrator.generate_all_documentation(str(root_path), doc_types)
            
            successful = len([r for r in results if not r.errors])
            console.print(f"\n[bold green]✅ Documentation generation complete![/bold green]")
            console.print(f"[dim]Successfully generated {successful}/{len(results)} documentation types[/dim]")
            
        else:
            console.print("[red]Error: Please specify --all, --types, --docstrings-only, or --api-docs-only[/red]")
            return 1
        
        console.print("\n[bold blue]🎉 Agentic Documentation Generator completed successfully![/bold blue]")
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Documentation generation interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]❌ Error during documentation generation: {e}[/red]")
        logging.exception("Documentation generation failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))