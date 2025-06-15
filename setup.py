#!/usr/bin/env python3
"""
AI News Dashboard - Setup Script
Automated setup and configuration for the MLOps infrastructure
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SetupConfig:
    """Configuration for setup process"""
    project_name: str = "ai-news-dashboard"
    python_version: str = "3.10"
    node_version: str = "18"
    cuda_version: str = "11.8"
    install_gpu: bool = True
    install_dev_tools: bool = True
    setup_docker: bool = True
    setup_kubernetes: bool = False
    create_env_file: bool = True

class ProjectSetup:
    """Main setup class for AI News Dashboard"""
    
    def __init__(self, config: SetupConfig):
        self.config = config
        self.project_root = Path.cwd()
        self.mlops_dir = self.project_root / "mlops"
        
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging"""
        logger.info(f"Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )
            if result.stdout:
                logger.info(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stderr:
                logger.error(f"Error: {e.stderr.strip()}")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check if required tools are installed"""
        logger.info("Checking prerequisites...")
        
        required_tools = {
            "python": f"python --version",
            "node": f"node --version",
            "npm": f"npm --version",
            "git": f"git --version"
        }
        
        if self.config.setup_docker:
            required_tools["docker"] = "docker --version"
            required_tools["docker-compose"] = "docker-compose --version"
        
        if self.config.setup_kubernetes:
            required_tools["kubectl"] = "kubectl version --client"
        
        missing_tools = []
        for tool, command in required_tools.items():
            try:
                result = self.run_command(command, check=False)
                if result.returncode != 0:
                    missing_tools.append(tool)
                else:
                    logger.info(f"✓ {tool} is installed")
            except Exception:
                missing_tools.append(tool)
        
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        logger.info("All prerequisites are satisfied")
        return True
    
    def setup_python_environment(self) -> None:
        """Setup Python virtual environment and install dependencies"""
        logger.info("Setting up Python environment...")
        
        # Create virtual environment
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            self.run_command(f"python -m venv {venv_path}")
            logger.info("Created virtual environment")
        
        # Activate virtual environment and install dependencies
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_path = venv_path / "bin" / "pip"
        
        # Install base requirements
        requirements_file = self.mlops_dir / "requirements.txt"
        if requirements_file.exists():
            self.run_command(f"{pip_path} install --upgrade pip")
            self.run_command(f"{pip_path} install -r {requirements_file}")
            logger.info("Installed Python dependencies")
        
        # Install GPU-specific packages if requested
        if self.config.install_gpu:
            self.install_gpu_packages(pip_path)
    
    def install_gpu_packages(self, pip_path: Path) -> None:
        """Install GPU-specific packages"""
        logger.info("Installing GPU packages...")
        
        gpu_packages = [
            f"torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{self.config.cuda_version.replace('.', '')}",
            "tensorflow[and-cuda]",
            "cupy-cuda11x",
            "rapids-cudf-cu11",
            "rapids-cuml-cu11"
        ]
        
        for package in gpu_packages:
            try:
                self.run_command(f"{pip_path} install {package}", check=False)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install GPU package: {package}")
    
    def setup_node_environment(self) -> None:
        """Setup Node.js environment and install dependencies"""
        logger.info("Setting up Node.js environment...")
        
        package_json = self.project_root / "package.json"
        if package_json.exists():
            self.run_command("npm install")
            logger.info("Installed Node.js dependencies")
        else:
            # Create basic package.json
            self.create_package_json()
            self.run_command("npm install")
    
    def create_package_json(self) -> None:
        """Create basic package.json file"""
        package_config = {
            "name": self.config.project_name,
            "version": "1.0.0",
            "description": "AI News Dashboard - Advanced MLOps Infrastructure",
            "main": "index.js",
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint",
                "test": "jest",
                "docker:build": "docker-compose -f mlops/docker/docker-compose.yml build",
                "docker:up": "docker-compose -f mlops/docker/docker-compose.yml up -d",
                "docker:down": "docker-compose -f mlops/docker/docker-compose.yml down",
                "k8s:deploy": "kubectl apply -f mlops/kubernetes/",
                "k8s:delete": "kubectl delete -f mlops/kubernetes/"
            },
            "dependencies": {
                "next": "^14.0.0",
                "react": "^18.0.0",
                "react-dom": "^18.0.0",
                "@types/node": "^20.0.0",
                "@types/react": "^18.0.0",
                "@types/react-dom": "^18.0.0",
                "typescript": "^5.0.0"
            },
            "devDependencies": {
                "eslint": "^8.0.0",
                "eslint-config-next": "^14.0.0",
                "jest": "^29.0.0",
                "@testing-library/react": "^13.0.0",
                "@testing-library/jest-dom": "^6.0.0"
            },
            "keywords": [
                "ai",
                "news",
                "dashboard",
                "mlops",
                "machine-learning",
                "nlp",
                "computer-vision",
                "federated-learning",
                "quantum-computing",
                "blockchain"
            ],
            "author": "AI News Dashboard Team",
            "license": "MIT"
        }
        
        with open(self.project_root / "package.json", "w") as f:
            json.dump(package_config, f, indent=2)
        
        logger.info("Created package.json")
    
    def create_environment_file(self) -> None:
        """Create .env file with default configuration"""
        if not self.config.create_env_file:
            return
        
        logger.info("Creating environment configuration...")
        
        env_config = {
            "# Core Configuration": "",
            "NODE_ENV": "development",
            "PORT": "3000",
            "API_PORT": "8000",
            
            "# Database Configuration": "",
            "REDIS_URL": "redis://localhost:6379",
            "POSTGRES_URL": "postgresql://postgres:password@localhost:5432/newsdb",
            "MONGODB_URL": "mongodb://localhost:27017/newsdb",
            
            "# ML Configuration": "",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "OPENAI_API_KEY": "your_openai_api_key_here",
            "HUGGINGFACE_TOKEN": "your_huggingface_token_here",
            "ANTHROPIC_API_KEY": "your_anthropic_api_key_here",
            
            "# Monitoring": "",
            "PROMETHEUS_GATEWAY": "http://localhost:9090",
            "GRAFANA_URL": "http://localhost:3001",
            "GRAFANA_USERNAME": "admin",
            "GRAFANA_PASSWORD": "admin",
            
            "# Security": "",
            "JWT_SECRET": "your_jwt_secret_here",
            "ENCRYPTION_KEY": "your_encryption_key_here",
            "API_KEY": "your_api_key_here",
            
            "# News APIs": "",
            "NEWS_API_KEY": "your_news_api_key_here",
            "GUARDIAN_API_KEY": "your_guardian_api_key_here",
            "NYT_API_KEY": "your_nyt_api_key_here",
            
            "# Cloud Configuration": "",
            "AWS_ACCESS_KEY_ID": "your_aws_access_key",
            "AWS_SECRET_ACCESS_KEY": "your_aws_secret_key",
            "AWS_REGION": "us-east-1",
            "GOOGLE_CLOUD_PROJECT": "your_gcp_project_id",
            
            "# Blockchain": "",
            "ETHEREUM_RPC_URL": "https://mainnet.infura.io/v3/your_infura_key",
            "PRIVATE_KEY": "your_ethereum_private_key",
            
            "# Edge Computing": "",
            "EDGE_DEVICE_ID": "edge-001",
            "EDGE_REGION": "us-east-1",
            "EDGE_SYNC_INTERVAL": "300",
            
            "# Quantum Computing": "",
            "IBM_QUANTUM_TOKEN": "your_ibm_quantum_token",
            "QUANTUM_BACKEND": "ibmq_qasm_simulator"
        }
        
        env_file = self.project_root / ".env"
        with open(env_file, "w") as f:
            for key, value in env_config.items():
                if key.startswith("#"):
                    f.write(f"\n{key}\n")
                else:
                    f.write(f"{key}={value}\n")
        
        # Create .env.example
        env_example = self.project_root / ".env.example"
        with open(env_example, "w") as f:
            for key, value in env_config.items():
                if key.startswith("#"):
                    f.write(f"\n{key}\n")
                else:
                    if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
                        f.write(f"{key}=\n")
                    else:
                        f.write(f"{key}={value}\n")
        
        logger.info("Created environment files (.env and .env.example)")
    
    def setup_docker_environment(self) -> None:
        """Setup Docker environment"""
        if not self.config.setup_docker:
            return
        
        logger.info("Setting up Docker environment...")
        
        # Build Docker images
        docker_compose_file = self.mlops_dir / "docker" / "docker-compose.yml"
        if docker_compose_file.exists():
            try:
                self.run_command(f"docker-compose -f {docker_compose_file} build")
                logger.info("Built Docker images")
            except subprocess.CalledProcessError:
                logger.warning("Failed to build Docker images")
    
    def setup_kubernetes_environment(self) -> None:
        """Setup Kubernetes environment"""
        if not self.config.setup_kubernetes:
            return
        
        logger.info("Setting up Kubernetes environment...")
        
        k8s_dir = self.mlops_dir / "kubernetes"
        if k8s_dir.exists():
            try:
                # Apply namespace first
                self.run_command(f"kubectl apply -f {k8s_dir / 'deployment.yaml'} --dry-run=client")
                self.run_command(f"kubectl apply -f {k8s_dir / 'services.yaml'} --dry-run=client")
                logger.info("Validated Kubernetes configurations")
            except subprocess.CalledProcessError:
                logger.warning("Kubernetes configuration validation failed")
    
    def create_project_structure(self) -> None:
        """Create additional project directories"""
        logger.info("Creating project structure...")
        
        directories = [
            "data/raw",
            "data/processed",
            "data/models",
            "logs",
            "tests/unit",
            "tests/integration",
            "tests/load",
            "docs/api",
            "docs/deployment",
            "scripts",
            "notebooks",
            ".github/workflows"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep = dir_path / ".gitkeep"
            if not any(dir_path.iterdir()) and not gitkeep.exists():
                gitkeep.touch()
        
        logger.info("Created project directory structure")
    
    def create_development_scripts(self) -> None:
        """Create development helper scripts"""
        logger.info("Creating development scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Development server script
        dev_script = scripts_dir / "dev.py"
        with open(dev_script, "w") as f:
            f.write('''
#!/usr/bin/env python3
"""
Development server launcher
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start development servers"""
    project_root = Path(__file__).parent.parent
    
    # Start backend services
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "mlops.api.main:app", 
        "--reload", "--host", "0.0.0.0", "--port", "8000"
    ]
    
    # Start frontend
    frontend_cmd = ["npm", "run", "dev"]
    
    try:
        print("Starting backend server...")
        backend_process = subprocess.Popen(backend_cmd, cwd=project_root)
        
        print("Starting frontend server...")
        frontend_process = subprocess.Popen(frontend_cmd, cwd=project_root)
        
        print("Development servers started!")
        print("Backend: http://localhost:8000")
        print("Frontend: http://localhost:3000")
        print("Press Ctrl+C to stop")
        
        # Wait for processes
        backend_process.wait()
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
''')
        
        # Make script executable
        if sys.platform != "win32":
            os.chmod(dev_script, 0o755)
        
        # Test script
        test_script = scripts_dir / "test.py"
        with open(test_script, "w") as f:
            f.write('''
#!/usr/bin/env python3
"""
Test runner script
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run all tests"""
    project_root = Path(__file__).parent.parent
    
    test_commands = [
        ["python", "-m", "pytest", "tests/unit/", "-v"],
        ["python", "-m", "pytest", "tests/integration/", "-v"],
        ["npm", "test"],
        ["python", "-m", "flake8", "mlops/"],
        ["python", "-m", "black", "--check", "mlops/"],
        ["python", "-m", "mypy", "mlops/"]
    ]
    
    for cmd in test_commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=project_root, check=True)
            print(f"✓ {cmd[0]} tests passed")
        except subprocess.CalledProcessError as e:
            print(f"✗ {cmd[0]} tests failed: {e}")
            return False
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
''')
        
        if sys.platform != "win32":
            os.chmod(test_script, 0o755)
        
        logger.info("Created development scripts")
    
    def create_gitignore(self) -> None:
        """Create comprehensive .gitignore file"""
        gitignore_content = '''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# Virtual Environment
venv/
env/
ENV/
.venv/
.env/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn-integrity

# Next.js
.next/
out/

# Environment Variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/
*.pkl
*.joblib
*.h5
*.onnx
*.pt
*.pth

# Jupyter Notebooks
.ipynb_checkpoints/

# MLflow
mlruns/
mlartifacts/

# Docker
.dockerignore

# Kubernetes
*.kubeconfig

# Secrets
*.key
*.pem
*.crt
*.p12
secrets/

# Cache
.cache/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/

# Temporary
*.tmp
*.temp
.tmp/
.temp/

# Database
*.db
*.sqlite
*.sqlite3

# Blockchain
keystore/
wallet.json

# Quantum
qiskit.cfg

# Large files
*.zip
*.tar.gz
*.rar
*.7z
'''
        
        gitignore_file = self.project_root / ".gitignore"
        with open(gitignore_file, "w") as f:
            f.write(gitignore_content)
        
        logger.info("Created .gitignore file")
    
    def run_setup(self) -> None:
        """Run complete setup process"""
        logger.info(f"Starting setup for {self.config.project_name}...")
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return
            
            # Create project structure
            self.create_project_structure()
            
            # Setup environments
            self.setup_python_environment()
            self.setup_node_environment()
            
            # Create configuration files
            self.create_environment_file()
            self.create_gitignore()
            
            # Create development scripts
            self.create_development_scripts()
            
            # Setup containerization
            if self.config.setup_docker:
                self.setup_docker_environment()
            
            if self.config.setup_kubernetes:
                self.setup_kubernetes_environment()
            
            logger.info("Setup completed successfully!")
            self.print_next_steps()
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
    
    def print_next_steps(self) -> None:
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 AI News Dashboard Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("\n1. Configure environment variables:")
        print("   - Edit .env file with your API keys and configuration")
        print("   - Update database connection strings")
        print("\n2. Start development servers:")
        print("   - Run: python scripts/dev.py")
        print("   - Or use: npm run dev (frontend) + uvicorn mlops.api.main:app --reload (backend)")
        print("\n3. Access the application:")
        print("   - Frontend: http://localhost:3000")
        print("   - API: http://localhost:8000")
        print("   - API Docs: http://localhost:8000/docs")
        print("\n4. Run tests:")
        print("   - Run: python scripts/test.py")
        print("   - Or use: pytest tests/")
        print("\n5. Deploy with Docker:")
        print("   - Run: npm run docker:up")
        print("   - Access Grafana: http://localhost:3001")
        print("   - Access Prometheus: http://localhost:9090")
        print("\n6. Documentation:")
        print("   - Read: mlops/README.md")
        print("   - API Docs: http://localhost:8000/docs")
        print("\n" + "="*60)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="AI News Dashboard Setup")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU package installation")
    parser.add_argument("--no-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--kubernetes", action="store_true", help="Setup Kubernetes environment")
    parser.add_argument("--no-env", action="store_true", help="Skip .env file creation")
    parser.add_argument("--dev-only", action="store_true", help="Development setup only")
    
    args = parser.parse_args()
    
    config = SetupConfig(
        install_gpu=not args.no_gpu,
        setup_docker=not args.no_docker,
        setup_kubernetes=args.kubernetes,
        create_env_file=not args.no_env,
        install_dev_tools=True
    )
    
    setup = ProjectSetup(config)
    setup.run_setup()

if __name__ == "__main__":
    main()