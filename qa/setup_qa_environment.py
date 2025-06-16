#!/usr/bin/env python3
"""
Dr. Orion "TestMaster" Vanguard - QA Environment Setup Script

Automated setup and configuration for the Superhuman QA Testing Environment.
This script handles:
- Virtual environment creation
- Dependency installation
- Browser driver setup
- Configuration validation
- Database connections
- API key verification
- Directory structure creation

Author: Dr. Orion "TestMaster" Vanguard
Version: 1.0.0
License: MIT
"""

import os
import sys
import subprocess
import platform
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SetupResult:
    """Result of setup operation"""
    component: str
    success: bool
    message: str
    details: Optional[Dict] = None

class SuperhumanQASetup:
    """Comprehensive QA environment setup orchestrator"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.qa_dir = self.project_root / "qa"
        self.venv_dir = self.qa_dir / "qa_env"
        self.reports_dir = self.qa_dir / "reports"
        self.test_data_dir = self.qa_dir / "test_data"
        self.logs_dir = self.qa_dir / "logs"
        self.setup_results: List[SetupResult] = []
        self.python_executable = sys.executable
        
    def run_complete_setup(self, skip_venv: bool = False, skip_browsers: bool = False) -> bool:
        """Run complete QA environment setup"""
        print("🚀 Dr. Orion TestMaster - Superhuman QA Environment Setup")
        print("=" * 60)
        
        setup_steps = [
            ("System Requirements", self.check_system_requirements),
            ("Directory Structure", self.create_directory_structure),
            ("Virtual Environment", lambda: self.setup_virtual_environment() if not skip_venv else SetupResult("Virtual Environment", True, "Skipped")),
            ("Python Dependencies", self.install_dependencies),
            ("Browser Drivers", lambda: self.setup_browser_drivers() if not skip_browsers else SetupResult("Browser Drivers", True, "Skipped")),
            ("Configuration Files", self.validate_configuration),
            ("API Connections", self.test_api_connections),
            ("Database Connections", self.test_database_connections),
            ("Test Data", self.setup_test_data),
            ("Permissions", self.check_permissions)
        ]
        
        all_success = True
        
        for step_name, step_function in setup_steps:
            print(f"\n📋 Setting up {step_name}...")
            try:
                result = step_function()
                self.setup_results.append(result)
                
                if result.success:
                    print(f"✅ {step_name}: {result.message}")
                else:
                    print(f"❌ {step_name}: {result.message}")
                    all_success = False
                    
            except Exception as e:
                error_result = SetupResult(step_name, False, f"Exception: {str(e)}")
                self.setup_results.append(error_result)
                print(f"❌ {step_name}: Exception - {str(e)}")
                all_success = False
        
        # Generate setup report
        self.generate_setup_report()
        
        print("\n" + "=" * 60)
        if all_success:
            print("🎉 Superhuman QA Environment Setup Complete!")
            print("\n🚀 Ready to achieve superhuman quality assurance!")
            self.print_next_steps()
        else:
            print("⚠️  Setup completed with some issues. Check the report for details.")
            self.print_troubleshooting_guide()
        
        return all_success
    
    def check_system_requirements(self) -> SetupResult:
        """Check system requirements and compatibility"""
        requirements = {
            "python_version": (3, 9),
            "available_memory_gb": 4,
            "available_disk_gb": 2
        }
        
        issues = []
        
        # Check Python version
        python_version = sys.version_info[:2]
        if python_version < requirements["python_version"]:
            issues.append(f"Python {requirements['python_version'][0]}.{requirements['python_version'][1]}+ required, found {python_version[0]}.{python_version[1]}")
        
        # Check available memory (simplified)
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < requirements["available_memory_gb"]:
                issues.append(f"Insufficient memory: {available_memory_gb:.1f}GB available, {requirements['available_memory_gb']}GB required")
        except ImportError:
            issues.append("Cannot check memory requirements (psutil not available)")
        
        # Check disk space
        try:
            disk_usage = os.statvfs(self.project_root) if hasattr(os, 'statvfs') else None
            if disk_usage:
                available_disk_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
                if available_disk_gb < requirements["available_disk_gb"]:
                    issues.append(f"Insufficient disk space: {available_disk_gb:.1f}GB available, {requirements['available_disk_gb']}GB required")
        except:
            pass  # Skip disk check if not available
        
        if issues:
            return SetupResult("System Requirements", False, "; ".join(issues))
        
        return SetupResult(
            "System Requirements", 
            True, 
            f"All requirements met (Python {python_version[0]}.{python_version[1]}, {platform.system()})"
        )
    
    def create_directory_structure(self) -> SetupResult:
        """Create necessary directory structure"""
        directories = [
            self.qa_dir,
            self.reports_dir,
            self.test_data_dir,
            self.logs_dir,
            self.qa_dir / "personas",
            self.qa_dir / "chaos",
            self.qa_dir / "metrics",
            self.qa_dir / "screenshots",
            self.qa_dir / "artifacts"
        ]
        
        created_dirs = []
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(directory.relative_to(self.project_root)))
            except Exception as e:
                return SetupResult("Directory Structure", False, f"Failed to create {directory}: {str(e)}")
        
        return SetupResult(
            "Directory Structure", 
            True, 
            f"Created {len(created_dirs)} directories",
            {"directories": created_dirs}
        )
    
    def setup_virtual_environment(self) -> SetupResult:
        """Setup Python virtual environment"""
        try:
            # Check if virtual environment already exists
            if self.venv_dir.exists():
                return SetupResult("Virtual Environment", True, "Virtual environment already exists")
            
            # Create virtual environment
            subprocess.run([
                self.python_executable, "-m", "venv", 
                str(self.venv_dir)
            ], check=True, capture_output=True, text=True)
            
            # Determine activation script path
            if platform.system() == "Windows":
                activate_script = self.venv_dir / "Scripts" / "activate.bat"
                pip_executable = self.venv_dir / "Scripts" / "pip.exe"
            else:
                activate_script = self.venv_dir / "bin" / "activate"
                pip_executable = self.venv_dir / "bin" / "pip"
            
            # Upgrade pip in virtual environment
            subprocess.run([
                str(pip_executable), "install", "--upgrade", "pip"
            ], check=True, capture_output=True, text=True)
            
            return SetupResult(
                "Virtual Environment", 
                True, 
                f"Created at {self.venv_dir.relative_to(self.project_root)}",
                {"activate_script": str(activate_script)}
            )
            
        except subprocess.CalledProcessError as e:
            return SetupResult("Virtual Environment", False, f"Failed to create virtual environment: {e.stderr}")
        except Exception as e:
            return SetupResult("Virtual Environment", False, f"Unexpected error: {str(e)}")
    
    def install_dependencies(self) -> SetupResult:
        """Install Python dependencies"""
        requirements_file = self.qa_dir / "requirements.txt"
        
        if not requirements_file.exists():
            return SetupResult("Python Dependencies", False, "requirements.txt not found")
        
        try:
            # Determine pip executable
            if self.venv_dir.exists():
                if platform.system() == "Windows":
                    pip_executable = self.venv_dir / "Scripts" / "pip.exe"
                else:
                    pip_executable = self.venv_dir / "bin" / "pip"
            else:
                pip_executable = "pip"
            
            # Install dependencies
            result = subprocess.run([
                str(pip_executable), "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                # Count installed packages
                installed_packages = len([line for line in result.stdout.split('\n') if 'Successfully installed' in line])
                return SetupResult(
                    "Python Dependencies", 
                    True, 
                    f"Successfully installed dependencies",
                    {"pip_output": result.stdout[:500]}  # First 500 chars
                )
            else:
                return SetupResult("Python Dependencies", False, f"pip install failed: {result.stderr[:500]}")
                
        except subprocess.TimeoutExpired:
            return SetupResult("Python Dependencies", False, "Installation timed out (>10 minutes)")
        except Exception as e:
            return SetupResult("Python Dependencies", False, f"Installation error: {str(e)}")
    
    def setup_browser_drivers(self) -> SetupResult:
        """Setup browser drivers for Selenium"""
        try:
            # Determine python executable
            if self.venv_dir.exists():
                if platform.system() == "Windows":
                    python_executable = self.venv_dir / "Scripts" / "python.exe"
                else:
                    python_executable = self.venv_dir / "bin" / "python"
            else:
                python_executable = self.python_executable
            
            # Install Chrome driver
            chrome_driver_script = """
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager

try:
    chrome_path = ChromeDriverManager().install()
    print(f"Chrome driver installed: {chrome_path}")
except Exception as e:
    print(f"Chrome driver installation failed: {e}")

try:
    firefox_path = GeckoDriverManager().install()
    print(f"Firefox driver installed: {firefox_path}")
except Exception as e:
    print(f"Firefox driver installation failed: {e}")

try:
    edge_path = EdgeChromiumDriverManager().install()
    print(f"Edge driver installed: {edge_path}")
except Exception as e:
    print(f"Edge driver installation failed: {e}")
"""
            
            result = subprocess.run([
                str(python_executable), "-c", chrome_driver_script
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if "installed" in result.stdout:
                return SetupResult(
                    "Browser Drivers", 
                    True, 
                    "Browser drivers installed successfully",
                    {"installation_log": result.stdout}
                )
            else:
                return SetupResult("Browser Drivers", False, f"Driver installation issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            return SetupResult("Browser Drivers", False, "Driver installation timed out")
        except Exception as e:
            return SetupResult("Browser Drivers", False, f"Driver setup error: {str(e)}")
    
    def validate_configuration(self) -> SetupResult:
        """Validate configuration files"""
        config_file = self.qa_dir / "config.yaml"
        
        if not config_file.exists():
            return SetupResult("Configuration Files", False, "config.yaml not found")
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = [
                "dashboard", "persona_testing", "ai_inference_validation",
                "chaos_engineering", "neural_cache", "multi_agent_pipeline"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                return SetupResult(
                    "Configuration Files", 
                    False, 
                    f"Missing configuration sections: {', '.join(missing_sections)}"
                )
            
            # Validate dashboard URL
            dashboard_url = config.get("dashboard", {}).get("url")
            if not dashboard_url:
                return SetupResult("Configuration Files", False, "Dashboard URL not configured")
            
            return SetupResult(
                "Configuration Files", 
                True, 
                f"Configuration valid (dashboard: {dashboard_url})",
                {"sections": list(config.keys())}
            )
            
        except yaml.YAMLError as e:
            return SetupResult("Configuration Files", False, f"Invalid YAML: {str(e)}")
        except Exception as e:
            return SetupResult("Configuration Files", False, f"Configuration error: {str(e)}")
    
    def test_api_connections(self) -> SetupResult:
        """Test API connections and authentication"""
        # Check environment variables for API keys
        api_keys = {
            "OpenAI": os.getenv("OPENAI_API_KEY"),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "Pinecone": os.getenv("PINECONE_API_KEY")
        }
        
        missing_keys = [name for name, key in api_keys.items() if not key]
        
        if missing_keys:
            return SetupResult(
                "API Connections", 
                False, 
                f"Missing API keys: {', '.join(missing_keys)}. Set environment variables."
            )
        
        # Test OpenAI connection (if key available)
        if api_keys["OpenAI"]:
            try:
                # Simple API test (without actually making a request to avoid costs)
                if len(api_keys["OpenAI"]) > 20 and api_keys["OpenAI"].startswith("sk-"):
                    openai_status = "Key format valid"
                else:
                    openai_status = "Key format invalid"
            except Exception as e:
                openai_status = f"Error: {str(e)}"
        else:
            openai_status = "Not configured"
        
        return SetupResult(
            "API Connections", 
            True, 
            f"API keys configured (OpenAI: {openai_status})",
            {"api_status": {name: "Configured" if key else "Missing" for name, key in api_keys.items()}}
        )
    
    def test_database_connections(self) -> SetupResult:
        """Test database connections"""
        try:
            # Test Redis connection
            import redis
            
            redis_client = redis.Redis(
                host="localhost", 
                port=6379, 
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            redis_client.ping()
            redis_status = "Connected"
            
        except redis.ConnectionError:
            redis_status = "Connection failed - Redis may not be running"
        except ImportError:
            redis_status = "Redis library not installed"
        except Exception as e:
            redis_status = f"Error: {str(e)}"
        
        # For now, we'll consider Redis optional
        success = True  # Don't fail setup if Redis is not available
        
        return SetupResult(
            "Database Connections", 
            success, 
            f"Redis: {redis_status}",
            {"redis_status": redis_status}
        )
    
    def setup_test_data(self) -> SetupResult:
        """Setup test data and sample files"""
        try:
            # Create sample test data
            sample_articles = [
                {
                    "id": "test_001",
                    "title": "Sample Technology News Article",
                    "content": "This is a sample technology news article for testing purposes. It contains information about artificial intelligence, machine learning, and software development trends.",
                    "category": "technology",
                    "published_date": "2024-01-15T10:00:00Z",
                    "source": "Test News Source"
                },
                {
                    "id": "test_002",
                    "title": "Sample Business News Article",
                    "content": "This is a sample business news article discussing market trends, economic indicators, and corporate developments in the financial sector.",
                    "category": "business",
                    "published_date": "2024-01-15T11:00:00Z",
                    "source": "Test Business News"
                }
            ]
            
            # Save sample data
            test_data_file = self.test_data_dir / "sample_articles.json"
            with open(test_data_file, 'w') as f:
                json.dump(sample_articles, f, indent=2)
            
            # Create persona test scenarios
            persona_scenarios = {
                "casual_reader": [
                    "Load homepage and browse headlines",
                    "Click on interesting article",
                    "Perform basic search",
                    "Navigate between categories"
                ],
                "financial_analyst": [
                    "Search for specific company news",
                    "Filter by financial category",
                    "Analyze article sentiment",
                    "Export data for analysis"
                ]
            }
            
            persona_file = self.test_data_dir / "persona_scenarios.json"
            with open(persona_file, 'w') as f:
                json.dump(persona_scenarios, f, indent=2)
            
            return SetupResult(
                "Test Data", 
                True, 
                f"Created sample data files in {self.test_data_dir.relative_to(self.project_root)}",
                {"files_created": ["sample_articles.json", "persona_scenarios.json"]}
            )
            
        except Exception as e:
            return SetupResult("Test Data", False, f"Failed to create test data: {str(e)}")
    
    def check_permissions(self) -> SetupResult:
        """Check file and directory permissions"""
        try:
            # Test write permissions
            test_file = self.qa_dir / "permission_test.tmp"
            
            with open(test_file, 'w') as f:
                f.write("Permission test")
            
            # Test read permissions
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup
            test_file.unlink()
            
            if content == "Permission test":
                return SetupResult("Permissions", True, "Read/write permissions verified")
            else:
                return SetupResult("Permissions", False, "Permission test failed")
                
        except PermissionError:
            return SetupResult("Permissions", False, "Insufficient permissions for QA directory")
        except Exception as e:
            return SetupResult("Permissions", False, f"Permission check error: {str(e)}")
    
    def generate_setup_report(self):
        """Generate detailed setup report"""
        report_file = self.reports_dir / f"setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.system(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "project_root": str(self.project_root)
            },
            "setup_results": [
                {
                    "component": result.component,
                    "success": result.success,
                    "message": result.message,
                    "details": result.details
                }
                for result in self.setup_results
            ],
            "summary": {
                "total_components": len(self.setup_results),
                "successful": sum(1 for r in self.setup_results if r.success),
                "failed": sum(1 for r in self.setup_results if not r.success)
            }
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n📊 Setup report saved: {report_file.relative_to(self.project_root)}")
        except Exception as e:
            print(f"\n⚠️  Failed to save setup report: {str(e)}")
    
    def print_next_steps(self):
        """Print next steps for user"""
        print("\n🎯 Next Steps:")
        print("1. Activate virtual environment:")
        
        if platform.system() == "Windows":
            print(f"   {self.venv_dir}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_dir}/bin/activate")
        
        print("\n2. Set environment variables:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        print("   export PINECONE_API_KEY='your_pinecone_key'")
        
        print("\n3. Start your dashboard:")
        print("   cd .. && npm run dev")
        
        print("\n4. Run QA tests:")
        print("   python run_qa_suite.py --config config.yaml")
        
        print("\n5. View reports:")
        print(f"   Open {self.reports_dir}/qa_report_*.html")
    
    def print_troubleshooting_guide(self):
        """Print troubleshooting guide for common issues"""
        print("\n🔧 Troubleshooting Guide:")
        print("\n• Virtual Environment Issues:")
        print("  - Ensure Python 3.9+ is installed")
        print("  - Try: python -m pip install --upgrade pip")
        
        print("\n• Browser Driver Issues:")
        print("  - Install Chrome/Firefox browsers")
        print("  - Run: python -c 'from webdriver_manager.chrome import ChromeDriverManager; ChromeDriverManager().install()'")
        
        print("\n• API Connection Issues:")
        print("  - Verify API keys are valid")
        print("  - Check internet connectivity")
        print("  - Ensure API quotas are not exceeded")
        
        print("\n• Redis Connection Issues:")
        print("  - Install and start Redis server")
        print("  - Check Redis is running on localhost:6379")
        
        print("\n• Permission Issues:")
        print("  - Run with appropriate user permissions")
        print("  - Check directory write permissions")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Dr. Orion TestMaster - Superhuman QA Environment Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_qa_environment.py                    # Full setup
  python setup_qa_environment.py --skip-venv        # Skip virtual environment
  python setup_qa_environment.py --skip-browsers    # Skip browser drivers
  python setup_qa_environment.py --project-root /path/to/project

For support: https://github.com/your-repo/issues
        """
    )
    
    parser.add_argument(
        "--project-root",
        default=os.getcwd(),
        help="Project root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip virtual environment creation"
    )
    
    parser.add_argument(
        "--skip-browsers",
        action="store_true",
        help="Skip browser driver installation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = SuperhumanQASetup(args.project_root)
    
    # Run setup
    success = setup.run_complete_setup(
        skip_venv=args.skip_venv,
        skip_browsers=args.skip_browsers
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()