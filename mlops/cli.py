#!/usr/bin/env python3
"""
Commander DeployX CLI - The Command Interface for MLOps Orchestration

This CLI provides a unified interface for Commander Solaris "DeployX" Vivante
to manage the entire MLOps deployment lifecycle. It integrates all components
into a single, powerful command-line tool.

Features:
- Interactive deployment wizard
- Repository scanning and analysis
- Automated pipeline generation
- Real-time deployment monitoring
- Comprehensive reporting
- Emergency rollback capabilities
- Multi-environment management
- Security and compliance enforcement

Author: Commander Solaris "DeployX" Vivante
"""

import asyncio
import click
import json
import yaml
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
import warnings
warnings.filterwarnings('ignore')

# Import MLOps components
from orchestrator import MLOpsOrchestrator, OrchestrationConfig, DeploymentMode, OrchestrationStatus
from global_.multi_region_coordinator import CloudProvider

# Initialize Rich console
console = Console()

class DeployXCLI:
    """Commander DeployX CLI Interface"""
    
    def __init__(self):
        self.orchestrator = None
        self.config_path = None
        self.project_root = Path.cwd()
        
    async def initialize(self, config_path: Optional[str] = None):
        """Initialize the MLOps orchestrator"""
        console.print("[bold blue]🚀 Initializing Commander DeployX...[/bold blue]")
        
        with console.status("[bold green]Loading MLOps components..."):
            self.orchestrator = MLOpsOrchestrator(config_path)
            await self.orchestrator.initialize()
        
        console.print("[bold green]✅ Commander DeployX ready for action![/bold green]")
    
    def scan_repository(self, path: str = ".") -> Dict[str, Any]:
        """Scan repository for deployment artifacts"""
        console.print(f"[bold yellow]🔍 Scanning repository: {path}[/bold yellow]")
        
        scan_results = {
            "project_type": "unknown",
            "frameworks": [],
            "containers": [],
            "infrastructure": [],
            "ci_cd": [],
            "security": [],
            "recommendations": []
        }
        
        project_path = Path(path)
        
        # Detect project type and frameworks
        if (project_path / "package.json").exists():
            scan_results["project_type"] = "nodejs"
            scan_results["frameworks"].append("Node.js")
            
            # Check for specific frameworks
            try:
                with open(project_path / "package.json") as f:
                    package_json = json.load(f)
                    dependencies = {**package_json.get("dependencies", {}), **package_json.get("devDependencies", {})}
                    
                    if "react" in dependencies:
                        scan_results["frameworks"].append("React")
                    if "vue" in dependencies:
                        scan_results["frameworks"].append("Vue.js")
                    if "angular" in dependencies:
                        scan_results["frameworks"].append("Angular")
                    if "express" in dependencies:
                        scan_results["frameworks"].append("Express.js")
                    if "next" in dependencies:
                        scan_results["frameworks"].append("Next.js")
            except Exception:
                pass
        
        elif (project_path / "requirements.txt").exists() or (project_path / "pyproject.toml").exists():
            scan_results["project_type"] = "python"
            scan_results["frameworks"].append("Python")
            
            # Check for Python frameworks
            if (project_path / "manage.py").exists():
                scan_results["frameworks"].append("Django")
            
            try:
                if (project_path / "requirements.txt").exists():
                    with open(project_path / "requirements.txt") as f:
                        requirements = f.read().lower()
                        if "flask" in requirements:
                            scan_results["frameworks"].append("Flask")
                        if "fastapi" in requirements:
                            scan_results["frameworks"].append("FastAPI")
                        if "streamlit" in requirements:
                            scan_results["frameworks"].append("Streamlit")
            except Exception:
                pass
        
        elif (project_path / "go.mod").exists():
            scan_results["project_type"] = "go"
            scan_results["frameworks"].append("Go")
        
        elif (project_path / "Cargo.toml").exists():
            scan_results["project_type"] = "rust"
            scan_results["frameworks"].append("Rust")
        
        # Detect containers
        if (project_path / "Dockerfile").exists():
            scan_results["containers"].append("Dockerfile")
        if (project_path / "docker-compose.yml").exists() or (project_path / "docker-compose.yaml").exists():
            scan_results["containers"].append("Docker Compose")
        
        # Detect infrastructure as code
        if (project_path / "terraform").exists():
            scan_results["infrastructure"].append("Terraform")
        if (project_path / "helm").exists() or (project_path / "charts").exists():
            scan_results["infrastructure"].append("Helm")
        if (project_path / "k8s").exists() or (project_path / "kubernetes").exists():
            scan_results["infrastructure"].append("Kubernetes")
        if (project_path / "infra").exists():
            scan_results["infrastructure"].append("Infrastructure")
        
        # Detect CI/CD
        if (project_path / ".github" / "workflows").exists():
            scan_results["ci_cd"].append("GitHub Actions")
        if (project_path / ".gitlab-ci.yml").exists():
            scan_results["ci_cd"].append("GitLab CI")
        if (project_path / "Jenkinsfile").exists():
            scan_results["ci_cd"].append("Jenkins")
        if (project_path / ".circleci").exists():
            scan_results["ci_cd"].append("CircleCI")
        
        # Detect security files
        if (project_path / ".security").exists():
            scan_results["security"].append("Security Config")
        if (project_path / "SECURITY.md").exists():
            scan_results["security"].append("Security Policy")
        
        # Generate recommendations
        if not scan_results["containers"]:
            scan_results["recommendations"].append("Add Dockerfile for containerization")
        if not scan_results["infrastructure"]:
            scan_results["recommendations"].append("Add Kubernetes manifests or Helm charts")
        if not scan_results["ci_cd"]:
            scan_results["recommendations"].append("Setup CI/CD pipeline")
        if not scan_results["security"]:
            scan_results["recommendations"].append("Add security scanning and policies")
        
        return scan_results
    
    def display_scan_results(self, scan_results: Dict[str, Any]):
        """Display repository scan results"""
        # Create main panel
        scan_tree = Tree("[bold blue]📊 Repository Scan Results[/bold blue]")
        
        # Project info
        project_branch = scan_tree.add(f"[green]Project Type: {scan_results['project_type'].title()}[/green]")
        
        # Frameworks
        if scan_results["frameworks"]:
            frameworks_branch = scan_tree.add("[yellow]🔧 Frameworks[/yellow]")
            for framework in scan_results["frameworks"]:
                frameworks_branch.add(f"• {framework}")
        
        # Containers
        if scan_results["containers"]:
            containers_branch = scan_tree.add("[blue]🐳 Containerization[/blue]")
            for container in scan_results["containers"]:
                containers_branch.add(f"• {container}")
        
        # Infrastructure
        if scan_results["infrastructure"]:
            infra_branch = scan_tree.add("[purple]☁️ Infrastructure[/purple]")
            for infra in scan_results["infrastructure"]:
                infra_branch.add(f"• {infra}")
        
        # CI/CD
        if scan_results["ci_cd"]:
            cicd_branch = scan_tree.add("[cyan]🔄 CI/CD[/cyan]")
            for cicd in scan_results["ci_cd"]:
                cicd_branch.add(f"• {cicd}")
        
        # Security
        if scan_results["security"]:
            security_branch = scan_tree.add("[red]🔒 Security[/red]")
            for security in scan_results["security"]:
                security_branch.add(f"• {security}")
        
        # Recommendations
        if scan_results["recommendations"]:
            rec_branch = scan_tree.add("[magenta]💡 Recommendations[/magenta]")
            for rec in scan_results["recommendations"]:
                rec_branch.add(f"• {rec}")
        
        console.print(Panel(scan_tree, title="Repository Analysis", border_style="blue"))
    
    def create_deployment_wizard(self) -> OrchestrationConfig:
        """Interactive deployment configuration wizard"""
        console.print(Panel("[bold blue]🧙‍♂️ Deployment Configuration Wizard[/bold blue]", border_style="blue"))
        
        # Deployment mode
        console.print("\n[bold yellow]Select deployment mode:[/bold yellow]")
        modes = {
            "1": DeploymentMode.DEVELOPMENT,
            "2": DeploymentMode.STAGING,
            "3": DeploymentMode.PRODUCTION,
            "4": DeploymentMode.CANARY,
            "5": DeploymentMode.BLUE_GREEN,
            "6": DeploymentMode.ROLLING,
            "7": DeploymentMode.GLOBAL
        }
        
        for key, mode in modes.items():
            console.print(f"  {key}. {mode.value.replace('_', ' ').title()}")
        
        mode_choice = Prompt.ask("Choose deployment mode", choices=list(modes.keys()), default="5")
        deployment_mode = modes[mode_choice]
        
        # Target environments
        console.print("\n[bold yellow]Select target environments:[/bold yellow]")
        available_envs = ["development", "staging", "production", "qa", "demo"]
        
        target_environments = []
        for env in available_envs:
            if Confirm.ask(f"Deploy to {env}?", default=env in ["staging", "production"]):
                target_environments.append(env)
        
        # Cloud providers
        console.print("\n[bold yellow]Select cloud providers:[/bold yellow]")
        providers = {
            "1": CloudProvider.AWS,
            "2": CloudProvider.GCP,
            "3": CloudProvider.AZURE,
            "4": CloudProvider.ON_PREMISE
        }
        
        for key, provider in providers.items():
            console.print(f"  {key}. {provider.value.upper()}")
        
        cloud_providers = []
        provider_choices = Prompt.ask("Choose cloud providers (comma-separated)", default="1,2").split(",")
        for choice in provider_choices:
            if choice.strip() in providers:
                cloud_providers.append(providers[choice.strip()])
        
        # Security and compliance
        security_enabled = Confirm.ask("Enable security scanning?", default=True)
        compliance_frameworks = []
        
        if security_enabled:
            console.print("\n[bold yellow]Select compliance frameworks:[/bold yellow]")
            frameworks = ["soc2", "pci_dss", "gdpr", "hipaa", "iso27001"]
            for framework in frameworks:
                if Confirm.ask(f"Enable {framework.upper()}?", default=framework in ["soc2", "gdpr"]):
                    compliance_frameworks.append(framework)
        
        # Advanced options
        canary_enabled = Confirm.ask("Enable AI-enhanced canary analysis?", default=True)
        observability_enabled = Confirm.ask("Enable full-stack observability?", default=True)
        gitops_enabled = Confirm.ask("Enable GitOps integration?", default=True)
        auto_rollback = Confirm.ask("Enable automatic rollback?", default=True)
        approval_required = Confirm.ask("Require manual approval?", default=False)
        
        # Notification channels
        notification_channels = []
        if Confirm.ask("Enable Slack notifications?", default=True):
            notification_channels.append("slack")
        if Confirm.ask("Enable email notifications?", default=True):
            notification_channels.append("email")
        if Confirm.ask("Enable PagerDuty alerts?", default=False):
            notification_channels.append("pagerduty")
        
        # Timeout
        timeout_minutes = int(Prompt.ask("Deployment timeout (minutes)", default="60"))
        
        return OrchestrationConfig(
            deployment_mode=deployment_mode,
            target_environments=target_environments,
            cloud_providers=cloud_providers,
            security_enabled=security_enabled,
            compliance_frameworks=compliance_frameworks,
            canary_enabled=canary_enabled,
            observability_enabled=observability_enabled,
            gitops_enabled=gitops_enabled,
            auto_rollback=auto_rollback,
            approval_required=approval_required,
            notification_channels=notification_channels,
            timeout_minutes=timeout_minutes
        )
    
    def create_application_manifest(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create application manifest based on scan results"""
        console.print("\n[bold blue]📋 Creating Application Manifest[/bold blue]")
        
        # Basic info
        app_name = Prompt.ask("Application name", default="ai-news-dashboard")
        app_version = Prompt.ask("Application version", default="1.0.0")
        
        # Images
        images = []
        if "Dockerfile" in scan_results.get("containers", []):
            default_image = f"{app_name}:{app_version}"
            image = Prompt.ask("Container image", default=default_image)
            images.append(image)
        
        # Services
        services = []
        if scan_results["project_type"] == "nodejs":
            services = ["web", "api"]
        elif scan_results["project_type"] == "python":
            if "Django" in scan_results.get("frameworks", []):
                services = ["web", "worker"]
            elif "FastAPI" in scan_results.get("frameworks", []):
                services = ["api"]
            elif "Streamlit" in scan_results.get("frameworks", []):
                services = ["web"]
        
        # Resources
        cpu = Prompt.ask("CPU request", default="500m")
        memory = Prompt.ask("Memory request", default="1Gi")
        
        # Environment variables
        env_vars = {}
        if Confirm.ask("Add environment variables?", default=True):
            env_vars["NODE_ENV"] = "production"
            env_vars["LOG_LEVEL"] = "info"
            
            while Confirm.ask("Add custom environment variable?", default=False):
                key = Prompt.ask("Variable name")
                value = Prompt.ask("Variable value")
                env_vars[key] = value
        
        return {
            "name": app_name,
            "version": app_version,
            "images": images,
            "services": services,
            "resources": {
                "cpu": cpu,
                "memory": memory
            },
            "environment_variables": env_vars,
            "project_type": scan_results["project_type"],
            "frameworks": scan_results["frameworks"]
        }
    
    async def monitor_deployment(self, orchestration_id: str):
        """Monitor deployment progress with real-time updates"""
        console.print(f"\n[bold blue]📊 Monitoring Deployment: {orchestration_id}[/bold blue]")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Progress tracking
        phases = [
            "Initialization",
            "Security Scan",
            "Compliance Check",
            "Canary Analysis",
            "Deployment",
            "Validation",
            "Monitoring",
            "Completion"
        ]
        
        with Live(layout, refresh_per_second=2, screen=True):
            while True:
                status = self.orchestrator.get_orchestration_status(orchestration_id)
                if not status:
                    break
                
                # Header
                header_text = Text(f"🚀 Commander DeployX - Deployment {orchestration_id}", style="bold blue")
                layout["header"].update(Panel(header_text, border_style="blue"))
                
                # Main content
                main_content = self._create_deployment_dashboard(status, phases)
                layout["main"].update(main_content)
                
                # Footer
                footer_text = Text(f"Status: {status.status.value.upper()} | Phase: {status.phase.value.upper()}", style="bold")
                if status.status == OrchestrationStatus.SUCCESS:
                    footer_text.stylize("green")
                elif status.status == OrchestrationStatus.FAILED:
                    footer_text.stylize("red")
                else:
                    footer_text.stylize("yellow")
                
                layout["footer"].update(Panel(footer_text, border_style="blue"))
                
                # Check if deployment is complete
                if status.status in [OrchestrationStatus.SUCCESS, OrchestrationStatus.FAILED]:
                    await asyncio.sleep(2)  # Show final status
                    break
                
                await asyncio.sleep(2)
        
        # Show final results
        final_status = self.orchestrator.get_orchestration_status(orchestration_id)
        self._display_deployment_results(final_status)
    
    def _create_deployment_dashboard(self, status, phases) -> Panel:
        """Create deployment dashboard"""
        # Create progress table
        table = Table(title="Deployment Progress", show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="white")
        
        current_phase = status.phase.value
        
        for i, phase in enumerate(phases):
            phase_key = phase.lower().replace(" ", "_")
            
            if phase_key == current_phase:
                status_icon = "🔄 Running"
                duration = f"{(datetime.now() - status.started_at).total_seconds():.1f}s"
                details = "In progress..."
            elif i < phases.index(phase.replace("_", " ").title()):
                status_icon = "✅ Complete"
                duration = "--"
                details = "Completed"
            else:
                status_icon = "⏳ Pending"
                duration = "--"
                details = "Waiting..."
            
            table.add_row(phase, status_icon, duration, details)
        
        # Add metrics
        metrics_text = f"""
[bold]Deployment Metrics:[/bold]
• Started: {status.started_at.strftime('%H:%M:%S')}
• Duration: {(datetime.now() - status.started_at).total_seconds():.1f}s
• Errors: {len(status.errors)}
• Warnings: {len(status.warnings)}
        """
        
        # Combine table and metrics
        dashboard = Table.grid()
        dashboard.add_column()
        dashboard.add_row(table)
        dashboard.add_row("")
        dashboard.add_row(metrics_text)
        
        return Panel(dashboard, title="Deployment Dashboard", border_style="blue")
    
    def _display_deployment_results(self, status):
        """Display final deployment results"""
        console.print("\n" + "="*80)
        
        if status.status == OrchestrationStatus.SUCCESS:
            console.print("[bold green]🎉 DEPLOYMENT SUCCESSFUL! 🎉[/bold green]")
        else:
            console.print("[bold red]❌ DEPLOYMENT FAILED ❌[/bold red]")
        
        console.print("="*80)
        
        # Summary table
        summary_table = Table(title="Deployment Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        duration = (status.completed_at - status.started_at).total_seconds() if status.completed_at else 0
        
        summary_table.add_row("Deployment ID", status.id)
        summary_table.add_row("Status", status.status.value.upper())
        summary_table.add_row("Final Phase", status.phase.value.replace("_", " ").title())
        summary_table.add_row("Duration", f"{duration:.1f}s")
        summary_table.add_row("Errors", str(len(status.errors)))
        summary_table.add_row("Warnings", str(len(status.warnings)))
        
        console.print(summary_table)
        
        # Show errors if any
        if status.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in status.errors:
                console.print(f"  • {error}")
        
        # Show warnings if any
        if status.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in status.warnings:
                console.print(f"  • {warning}")
        
        # Show recommendations
        if status.recommendations:
            console.print("\n[bold blue]Recommendations:[/bold blue]")
            for rec in status.recommendations:
                console.print(f"  • {rec}")
    
    def display_metrics_dashboard(self):
        """Display orchestrator metrics dashboard"""
        metrics = self.orchestrator.get_metrics()
        
        # Create metrics table
        metrics_table = Table(title="🚀 Commander DeployX Metrics", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Description", style="white")
        
        metrics_table.add_row("Total Deployments", str(metrics["total_deployments"]), "All deployment attempts")
        metrics_table.add_row("Successful", str(metrics["successful_deployments"]), "Completed successfully")
        metrics_table.add_row("Failed", str(metrics["failed_deployments"]), "Failed deployments")
        metrics_table.add_row("Success Rate", f"{metrics['success_rate']:.1f}%", "Overall success percentage")
        metrics_table.add_row("Rollbacks", str(metrics["rollbacks"]), "Automatic rollbacks executed")
        metrics_table.add_row("Security Blocks", str(metrics["security_blocks"]), "Deployments blocked by security")
        metrics_table.add_row("Compliance Violations", str(metrics["compliance_violations"]), "Compliance issues found")
        metrics_table.add_row("Active Deployments", str(metrics["active_orchestrations"]), "Currently running")
        
        console.print(Panel(metrics_table, border_style="blue"))
    
    def list_deployments(self, status_filter: Optional[str] = None):
        """List recent deployments"""
        orchestrations = self.orchestrator.list_orchestrations()
        
        if status_filter:
            status_enum = OrchestrationStatus(status_filter.lower())
            orchestrations = [o for o in orchestrations if o.status == status_enum]
        
        # Sort by start time (most recent first)
        orchestrations.sort(key=lambda x: x.started_at, reverse=True)
        
        # Create deployments table
        deployments_table = Table(title="Recent Deployments", show_header=True, header_style="bold magenta")
        deployments_table.add_column("ID", style="cyan", no_wrap=True)
        deployments_table.add_column("Status", style="white")
        deployments_table.add_column("Phase", style="yellow")
        deployments_table.add_column("Started", style="green")
        deployments_table.add_column("Duration", style="blue")
        deployments_table.add_column("Errors", style="red")
        
        for orch in orchestrations[:10]:  # Show last 10
            status_style = "green" if orch.status == OrchestrationStatus.SUCCESS else "red" if orch.status == OrchestrationStatus.FAILED else "yellow"
            
            duration = "--"
            if orch.completed_at:
                duration = f"{(orch.completed_at - orch.started_at).total_seconds():.1f}s"
            
            deployments_table.add_row(
                orch.id,
                f"[{status_style}]{orch.status.value}[/{status_style}]",
                orch.phase.value.replace("_", " ").title(),
                orch.started_at.strftime("%H:%M:%S"),
                duration,
                str(len(orch.errors))
            )
        
        console.print(Panel(deployments_table, border_style="blue"))

# CLI Commands

@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """🚀 Commander DeployX - MLOps Deployment Orchestrator"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['cli'] = DeployXCLI()

@cli.command()
@click.option('--path', '-p', default='.', help='Repository path to scan')
@click.pass_context
def scan(ctx, path):
    """🔍 Scan repository for deployment artifacts"""
    cli_instance = ctx.obj['cli']
    
    console.print("[bold blue]🔍 Commander DeployX Repository Scanner[/bold blue]")
    
    scan_results = cli_instance.scan_repository(path)
    cli_instance.display_scan_results(scan_results)
    
    # Save scan results
    scan_file = Path(path) / ".deployX_scan.json"
    with open(scan_file, 'w') as f:
        json.dump(scan_results, f, indent=2)
    
    console.print(f"\n[green]✅ Scan results saved to {scan_file}[/green]")

@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive deployment wizard')
@click.option('--config-file', '-f', help='Deployment configuration file')
@click.option('--manifest', '-m', help='Application manifest file')
@click.pass_context
def deploy(ctx, interactive, config_file, manifest):
    """🚀 Deploy application using MLOps orchestration"""
    async def run_deployment():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        # Initialize orchestrator
        await cli_instance.initialize(config_path)
        
        # Get deployment configuration
        if interactive:
            # Scan repository first
            scan_results = cli_instance.scan_repository()
            cli_instance.display_scan_results(scan_results)
            
            # Run wizard
            deployment_config = cli_instance.create_deployment_wizard()
            application_manifest = cli_instance.create_application_manifest(scan_results)
        else:
            # Load from files
            if not config_file or not manifest:
                console.print("[red]❌ Config file and manifest required for non-interactive mode[/red]")
                return
            
            with open(config_file) as f:
                config_data = yaml.safe_load(f)
            
            with open(manifest) as f:
                application_manifest = yaml.safe_load(f)
            
            # Convert to OrchestrationConfig
            deployment_config = OrchestrationConfig(**config_data)
        
        # Confirm deployment
        console.print("\n[bold yellow]📋 Deployment Summary:[/bold yellow]")
        console.print(f"Mode: {deployment_config.deployment_mode.value}")
        console.print(f"Environments: {', '.join(deployment_config.target_environments)}")
        console.print(f"Cloud Providers: {', '.join([p.value for p in deployment_config.cloud_providers])}")
        console.print(f"Security: {'✅' if deployment_config.security_enabled else '❌'}")
        console.print(f"Canary: {'✅' if deployment_config.canary_enabled else '❌'}")
        console.print(f"Observability: {'✅' if deployment_config.observability_enabled else '❌'}")
        
        if not Confirm.ask("\nProceed with deployment?", default=True):
            console.print("[yellow]Deployment cancelled[/yellow]")
            return
        
        # Start deployment
        orchestration_id = await cli_instance.orchestrator.orchestrate_deployment(
            deployment_config, application_manifest
        )
        
        # Monitor deployment
        await cli_instance.monitor_deployment(orchestration_id)
    
    asyncio.run(run_deployment())

@cli.command()
@click.option('--deployment-id', '-d', help='Deployment ID to monitor')
@click.pass_context
def monitor(ctx, deployment_id):
    """📊 Monitor deployment progress"""
    async def run_monitor():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        await cli_instance.initialize(config_path)
        
        if deployment_id:
            await cli_instance.monitor_deployment(deployment_id)
        else:
            # Show recent deployments
            cli_instance.list_deployments()
    
    asyncio.run(run_monitor())

@cli.command()
@click.option('--deployment-id', '-d', required=True, help='Deployment ID to rollback')
@click.pass_context
def rollback(ctx, deployment_id):
    """🔄 Rollback deployment"""
    async def run_rollback():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        await cli_instance.initialize(config_path)
        
        console.print(f"[bold red]🔄 Rolling back deployment: {deployment_id}[/bold red]")
        
        if not Confirm.ask("Are you sure you want to rollback?", default=False):
            console.print("[yellow]Rollback cancelled[/yellow]")
            return
        
        # Get deployment status
        status = cli_instance.orchestrator.get_orchestration_status(deployment_id)
        if not status:
            console.print(f"[red]❌ Deployment {deployment_id} not found[/red]")
            return
        
        # Execute rollback
        await cli_instance.orchestrator._execute_rollback(deployment_id, status)
        
        console.print(f"[green]✅ Rollback completed for {deployment_id}[/green]")
    
    asyncio.run(run_rollback())

@cli.command()
@click.option('--status', '-s', help='Filter by status (running, success, failed)')
@click.pass_context
def list(ctx, status):
    """📋 List deployments"""
    async def run_list():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        await cli_instance.initialize(config_path)
        cli_instance.list_deployments(status)
    
    asyncio.run(run_list())

@cli.command()
@click.pass_context
def metrics(ctx):
    """📈 Show orchestrator metrics"""
    async def run_metrics():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        await cli_instance.initialize(config_path)
        cli_instance.display_metrics_dashboard()
    
    asyncio.run(run_metrics())

@cli.command()
@click.pass_context
def report(ctx):
    """📊 Generate comprehensive report"""
    async def run_report():
        cli_instance = ctx.obj['cli']
        config_path = ctx.obj['config']
        
        await cli_instance.initialize(config_path)
        
        report = cli_instance.orchestrator.generate_orchestrator_report()
        
        # Display report
        console.print("[bold blue]📊 Commander DeployX Comprehensive Report[/bold blue]")
        console.print(json.dumps(report, indent=2, default=str))
        
        # Save report
        report_file = f"deployX_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Report saved to {report_file}[/green]")
    
    asyncio.run(run_report())

@cli.command()
def version():
    """📋 Show version information"""
    console.print("[bold blue]🚀 Commander DeployX v1.0.0[/bold blue]")
    console.print("MLOps Deployment Orchestrator")
    console.print("By Commander Solaris 'DeployX' Vivante")
    console.print("\n[green]Ready for planetary-scale deployments! 🌍[/green]")

if __name__ == '__main__':
    cli()