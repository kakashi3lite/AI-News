#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - QA Infrastructure Deployment Script
Automated deployment and configuration of the Superhuman QA Testing Environment
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import docker
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qa_deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

console = Console()

class QAInfrastructureDeployer:
    """Automated QA infrastructure deployment and management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.console = Console()
        self.docker_client = None
        self.deployment_status = {}
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def deploy_full_infrastructure(self, 
                                       environment: str = "development",
                                       skip_build: bool = False,
                                       force_recreate: bool = False) -> Dict:
        """Deploy complete QA infrastructure"""
        
        console.print(Panel.fit(
            "[bold blue]Dr. Orion TestMaster Vanguard[/bold blue]\n"
            "[green]QA Infrastructure Deployment[/green]\n"
            f"[yellow]Environment: {environment}[/yellow]",
            title="🚀 Infrastructure Deployment"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("[cyan]Deployment Progress", total=100)
            
            try:
                # 1. Pre-deployment checks (10%)
                progress.update(main_task, description="[cyan]Running pre-deployment checks")
                await self._pre_deployment_checks(progress)
                progress.update(main_task, advance=10)
                
                # 2. Build Docker images (20%)
                if not skip_build:
                    progress.update(main_task, description="[cyan]Building Docker images")
                    await self._build_docker_images(progress, force_recreate)
                    progress.update(main_task, advance=20)
                else:
                    progress.update(main_task, advance=20)
                
                # 3. Deploy monitoring stack (25%)
                progress.update(main_task, description="[cyan]Deploying monitoring stack")
                await self._deploy_monitoring_stack(progress, environment)
                progress.update(main_task, advance=25)
                
                # 4. Deploy QA services (25%)
                progress.update(main_task, description="[cyan]Deploying QA services")
                await self._deploy_qa_services(progress, environment)
                progress.update(main_task, advance=25)
                
                # 5. Configure networking and volumes (10%)
                progress.update(main_task, description="[cyan]Configuring networking")
                await self._configure_networking(progress)
                progress.update(main_task, advance=10)
                
                # 6. Post-deployment validation (10%)
                progress.update(main_task, description="[cyan]Running post-deployment validation")
                await self._post_deployment_validation(progress)
                progress.update(main_task, advance=10)
                
                progress.update(main_task, completed=100, description="[green]✅ Deployment completed")
                
            except Exception as e:
                logger.error(f"Deployment failed: {e}")
                progress.update(main_task, description="[red]❌ Deployment failed")
                self.deployment_status['error'] = str(e)
                raise
        
        # Display deployment summary
        await self._display_deployment_summary()
        
        return self.deployment_status
    
    async def _pre_deployment_checks(self, progress):
        """Run pre-deployment checks"""
        checks = [
            ("Docker daemon", self._check_docker),
            ("Docker Compose", self._check_docker_compose),
            ("Required ports", self._check_ports),
            ("Disk space", self._check_disk_space),
            ("Configuration files", self._check_config_files)
        ]
        
        for check_name, check_func in checks:
            try:
                result = await check_func()
                self.deployment_status[f"check_{check_name.lower().replace(' ', '_')}"] = {
                    'status': 'passed' if result else 'failed',
                    'details': result
                }
                console.print(f"✅ {check_name}: {'Passed' if result else 'Failed'}")
            except Exception as e:
                logger.error(f"Pre-deployment check failed for {check_name}: {e}")
                self.deployment_status[f"check_{check_name.lower().replace(' ', '_')}"] = {
                    'status': 'error',
                    'error': str(e)
                }
                console.print(f"❌ {check_name}: Error - {e}")
    
    async def _check_docker(self) -> bool:
        """Check if Docker is running"""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False
    
    async def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available"""
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    async def _check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket
        
        required_ports = [3000, 8080, 9090, 3001, 6379, 8888, 4444, 5432]
        
        for port in required_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        logger.warning(f"Port {port} is already in use")
                        return False
            except Exception:
                pass
        
        return True
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space"""
        import shutil
        
        try:
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (1024**3)
            return free_gb >= 5  # Require at least 5GB free
        except Exception:
            return False
    
    async def _check_config_files(self) -> bool:
        """Check if required configuration files exist"""
        required_files = [
            'config.yaml',
            'docker-compose.yml',
            'prometheus/prometheus.yml',
            'grafana/superhuman_qa_dashboard.json'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        return True
    
    async def _build_docker_images(self, progress, force_recreate: bool = False):
        """Build Docker images for QA services"""
        images_to_build = [
            ('qa-orchestrator', 'Dockerfile.qa'),
            ('monitoring-dashboard', 'Dockerfile.monitoring')
        ]
        
        for image_name, dockerfile in images_to_build:
            try:
                console.print(f"🔨 Building {image_name}...")
                
                build_args = {}
                if force_recreate:
                    build_args['nocache'] = True
                
                # Build image
                image, logs = self.docker_client.images.build(
                    path=".",
                    dockerfile=dockerfile,
                    tag=f"superhuman-qa/{image_name}:latest",
                    rm=True,
                    **build_args
                )
                
                self.deployment_status[f"build_{image_name}"] = {
                    'status': 'success',
                    'image_id': image.id
                }
                
                console.print(f"✅ {image_name} built successfully")
                
            except Exception as e:
                logger.error(f"Failed to build {image_name}: {e}")
                self.deployment_status[f"build_{image_name}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
                console.print(f"❌ Failed to build {image_name}: {e}")
    
    async def _deploy_monitoring_stack(self, progress, environment: str):
        """Deploy monitoring stack (Prometheus, Grafana, etc.)"""
        try:
            console.print("📊 Deploying monitoring stack...")
            
            # Use Docker Compose to deploy monitoring services
            compose_cmd = [
                'docker-compose',
                '-f', 'docker-compose.yml',
                'up', '-d',
                'prometheus', 'grafana', 'alertmanager', 'redis', 'node-exporter'
            ]
            
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.deployment_status['monitoring_stack'] = {
                    'status': 'deployed',
                    'services': ['prometheus', 'grafana', 'alertmanager', 'redis', 'node-exporter']
                }
                console.print("✅ Monitoring stack deployed successfully")
            else:
                raise Exception(f"Docker Compose failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to deploy monitoring stack: {e}")
            self.deployment_status['monitoring_stack'] = {
                'status': 'failed',
                'error': str(e)
            }
            console.print(f"❌ Failed to deploy monitoring stack: {e}")
    
    async def _deploy_qa_services(self, progress, environment: str):
        """Deploy QA services"""
        try:
            console.print("🧪 Deploying QA services...")
            
            # Deploy QA orchestrator and related services
            compose_cmd = [
                'docker-compose',
                '-f', 'docker-compose.yml',
                'up', '-d',
                'qa-orchestrator', 'qa-monitoring-dashboard', 'selenium-hub', 
                'selenium-chrome', 'selenium-firefox', 'weaviate'
            ]
            
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.deployment_status['qa_services'] = {
                    'status': 'deployed',
                    'services': ['qa-orchestrator', 'qa-monitoring-dashboard', 'selenium-grid', 'weaviate']
                }
                console.print("✅ QA services deployed successfully")
            else:
                raise Exception(f"Docker Compose failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to deploy QA services: {e}")
            self.deployment_status['qa_services'] = {
                'status': 'failed',
                'error': str(e)
            }
            console.print(f"❌ Failed to deploy QA services: {e}")
    
    async def _configure_networking(self, progress):
        """Configure Docker networking and volumes"""
        try:
            console.print("🌐 Configuring networking...")
            
            # Create custom network if it doesn't exist
            try:
                network = self.docker_client.networks.get('superhuman-qa-network')
                console.print("✅ Network already exists")
            except docker.errors.NotFound:
                network = self.docker_client.networks.create(
                    'superhuman-qa-network',
                    driver='bridge'
                )
                console.print("✅ Network created")
            
            self.deployment_status['networking'] = {
                'status': 'configured',
                'network_id': network.id
            }
            
        except Exception as e:
            logger.error(f"Failed to configure networking: {e}")
            self.deployment_status['networking'] = {
                'status': 'failed',
                'error': str(e)
            }
            console.print(f"❌ Failed to configure networking: {e}")
    
    async def _post_deployment_validation(self, progress):
        """Validate deployment by checking service health"""
        services_to_check = {
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3001/api/health',
            'qa-orchestrator': 'http://localhost:8080/health',
            'qa-monitoring': 'http://localhost:8081/health',
            'selenium-hub': 'http://localhost:4444/wd/hub/status'
        }
        
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            for service_name, health_url in services_to_check.items():
                try:
                    console.print(f"🔍 Checking {service_name}...")
                    
                    # Wait a bit for service to start
                    await asyncio.sleep(2)
                    
                    async with session.get(health_url, timeout=10) as response:
                        if response.status == 200:
                            self.deployment_status[f"health_{service_name}"] = {
                                'status': 'healthy',
                                'url': health_url
                            }
                            console.print(f"✅ {service_name} is healthy")
                        else:
                            raise Exception(f"Health check failed with status {response.status}")
                            
                except Exception as e:
                    logger.error(f"Health check failed for {service_name}: {e}")
                    self.deployment_status[f"health_{service_name}"] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    console.print(f"❌ {service_name} health check failed: {e}")
    
    async def _display_deployment_summary(self):
        """Display deployment summary"""
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            "[bold green]🎉 Deployment Summary[/bold green]",
            title="Dr. Orion TestMaster Vanguard"
        ))
        
        # Create summary table
        table = Table(title="🚀 Service Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("URL", style="yellow")
        
        service_urls = {
            'QA Orchestrator': 'http://localhost:8080',
            'Monitoring Dashboard': 'http://localhost:8081',
            'Prometheus': 'http://localhost:9090',
            'Grafana': 'http://localhost:3001',
            'Selenium Grid': 'http://localhost:4444'
        }
        
        for service, url in service_urls.items():
            status = "🟢 Running" if f"health_{service.lower().replace(' ', '-')}" in self.deployment_status else "🔴 Unknown"
            table.add_row(service, status, url)
        
        console.print(table)
        
        # Display next steps
        console.print("\n[bold cyan]🎯 Next Steps:[/bold cyan]")
        console.print("1. Access QA Orchestrator: http://localhost:8080")
        console.print("2. View Monitoring Dashboard: http://localhost:8081")
        console.print("3. Check Grafana Dashboards: http://localhost:3001 (admin/admin)")
        console.print("4. Monitor with Prometheus: http://localhost:9090")
        console.print("5. Run QA tests: python run_qa_suite.py")
        
        console.print("\n[green]✅ QA Infrastructure is ready for superhuman testing![/green]")
    
    async def teardown_infrastructure(self):
        """Teardown QA infrastructure"""
        console.print(Panel.fit(
            "[bold red]🔥 Tearing Down QA Infrastructure[/bold red]",
            title="Infrastructure Teardown"
        ))
        
        try:
            # Stop and remove all containers
            compose_cmd = ['docker-compose', '-f', 'docker-compose.yml', 'down', '-v']
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("✅ All services stopped and removed")
            else:
                console.print(f"❌ Teardown failed: {result.stderr}")
            
            # Remove custom network
            try:
                network = self.docker_client.networks.get('superhuman-qa-network')
                network.remove()
                console.print("✅ Custom network removed")
            except docker.errors.NotFound:
                console.print("ℹ️  Custom network not found")
            
            console.print("[green]✅ Infrastructure teardown completed[/green]")
            
        except Exception as e:
            logger.error(f"Teardown failed: {e}")
            console.print(f"[red]❌ Teardown failed: {e}[/red]")
    
    async def get_service_logs(self, service_name: str, lines: int = 100):
        """Get logs for a specific service"""
        try:
            compose_cmd = ['docker-compose', 'logs', '--tail', str(lines), service_name]
            result = subprocess.run(compose_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[cyan]📋 Logs for {service_name}:[/cyan]")
                console.print(result.stdout)
            else:
                console.print(f"[red]❌ Failed to get logs for {service_name}: {result.stderr}[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Error getting logs: {e}[/red]")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Dr. Orion TestMaster Vanguard - QA Infrastructure Deployment"
    )
    
    parser.add_argument(
        'action',
        choices=['deploy', 'teardown', 'status', 'logs'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--environment', '-e',
        default='development',
        choices=['development', 'staging', 'production'],
        help='Deployment environment'
    )
    
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip Docker image building'
    )
    
    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='Force recreate Docker images'
    )
    
    parser.add_argument(
        '--service',
        help='Service name for logs action'
    )
    
    parser.add_argument(
        '--lines',
        type=int,
        default=100,
        help='Number of log lines to show'
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Initialize deployer
    deployer = QAInfrastructureDeployer(args.config)
    
    try:
        if args.action == 'deploy':
            results = asyncio.run(deployer.deploy_full_infrastructure(
                environment=args.environment,
                skip_build=args.skip_build,
                force_recreate=args.force_recreate
            ))
            
            # Check for deployment errors
            if 'error' in results:
                sys.exit(1)
                
        elif args.action == 'teardown':
            asyncio.run(deployer.teardown_infrastructure())
            
        elif args.action == 'status':
            # Show current status
            console.print("[cyan]📊 Current Infrastructure Status[/cyan]")
            # Implementation for status check
            
        elif args.action == 'logs':
            if not args.service:
                console.print("[red]❌ Service name required for logs action[/red]")
                sys.exit(1)
            
            asyncio.run(deployer.get_service_logs(args.service, args.lines))
            
    except KeyboardInterrupt:
        console.print("\n[red]❌ Operation interrupted by user[/red]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]❌ Operation failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()