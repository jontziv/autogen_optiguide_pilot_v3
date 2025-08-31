#!/usr/bin/env python3
"""
Enhanced Setup Script for Inventory ROP Lab
--------------------------------------------
Handles installation, environment setup, and initial data generation
with comprehensive error handling and diagnostics.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_step(step_name: str, status: str = "info"):
    """Print step with appropriate formatting"""
    icons = {"info": "🔧", "success": "✅", "error": "❌", "warning": "⚠️"}
    icon = icons.get(status, "ℹ️")
    print(f"{icon} {step_name}")

def run_command(cmd, description, check=True, capture_output=True):
    """Run a command with proper error handling and logging"""
    print_step(f"{description}...")
    logger.info(f"Running: {cmd}")
    
    try:
        if isinstance(cmd, str):
            # For shell commands
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=check, 
                capture_output=capture_output, 
                text=True,
                timeout=300  # 5 minute timeout
            )
        else:
            # For command lists
            result = subprocess.run(
                cmd, 
                check=check, 
                capture_output=capture_output, 
                text=True,
                timeout=300
            )
        
        if result.returncode == 0:
            print_step(f"{description} completed", "success")
            if result.stdout and capture_output:
                logger.debug(f"Output: {result.stdout.strip()}")
            return True, result.stdout if capture_output else ""
        else:
            print_step(f"{description} failed", "error")
            if result.stderr and capture_output:
                logger.error(f"Error: {result.stderr.strip()}")
            return False, result.stderr if capture_output else ""
            
    except subprocess.TimeoutExpired:
        print_step(f"{description} timed out", "error")
        logger.error(f"Command timed out: {cmd}")
        return False, "Command timed out"
    except subprocess.CalledProcessError as e:
        print_step(f"{description} failed", "error")
        logger.error(f"Command failed: {e}")
        return False, str(e)
    except Exception as e:
        print_step(f"{description} failed with exception", "error")
        logger.error(f"Unexpected error: {e}")
        return False, str(e)

def check_python_version():
    """Check Python version compatibility"""
    print_step("Checking Python version")
    
    version = sys.version_info
    if version < (3, 8):
        print_step(f"Python {version.major}.{version.minor} is too old. Python 3.8+ required", "error")
        return False
    
    print_step(f"Python {version.major}.{version.minor}.{version.micro} detected", "success")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print_step("Running in virtual environment", "success")
    else:
        print_step("Not in virtual environment - consider using venv or conda", "warning")
    
    return True

def check_pip_and_update():
    """Check pip and update if needed"""
    print_step("Checking pip installation")
    
    # Check pip version
    success, output = run_command([sys.executable, "-m", "pip", "--version"], "Checking pip version")
    if not success:
        print_step("pip not available", "error")
        return False
    
    # Update pip to latest version
    success, _ = run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "Updating pip"
    )
    
    if not success:
        print_step("Failed to update pip, continuing anyway", "warning")
    
    return True

def fix_watchdog_issue():
    """Fix the specific watchdog issue that's causing Streamlit to fail"""
    print_step("Fixing watchdog compatibility issue")
    
    try:
        # Uninstall and reinstall watchdog with specific version
        commands = [
            ([sys.executable, "-m", "pip", "uninstall", "watchdog", "-y"], "Uninstalling old watchdog"),
            ([sys.executable, "-m", "pip", "install", "watchdog>=3.0.0"], "Installing compatible watchdog")
        ]
        
        for cmd, desc in commands:
            success, _ = run_command(cmd, desc)
            if not success:
                print_step(f"Warning: {desc} failed", "warning")
        
        return True
        
    except Exception as e:
        print_step(f"Watchdog fix failed: {e}", "warning")
        return False

def install_requirements():
    """Install Python requirements with proper dependency resolution"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_step("requirements.txt not found", "error")
        return False
    
    print_step("Installing requirements from requirements.txt")
    
    # First fix watchdog issue
    fix_watchdog_issue()
    
    # Install requirements
    success, output = run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--upgrade"],
        "Installing Python packages"
    )
    
    if not success:
        print_step("Requirements installation failed", "error")
        # Try installing without upgrade flag
        success, output = run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            "Retrying installation without upgrade"
        )
        if not success:
            return False
    
    # Verify critical packages
    critical_packages = ["streamlit", "pandas", "numpy", "plotly"]
    for package in critical_packages:
        success, _ = run_command(
            [sys.executable, "-c", f"import {package}; print(f'{package} OK')"],
            f"Verifying {package}",
            capture_output=False
        )
        if not success:
            print_step(f"Critical package {package} not working", "error")
            return False
    
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    print_step("Setting up environment")
    
    env_file = Path(".env")
    if not env_file.exists():
        # Create sample .env file
        sample_env = """# GROQ API Configuration
GROQ_API_KEY=your_groq_api_key_here

# NEOS Email (for optimization solver)
NEOS_EMAIL=your_email@example.com

# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Optional: Disable file watcher if having issues
# STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
"""
        try:
            env_file.write_text(sample_env)
            print_step(f"Created sample .env file at {env_file}", "success")
            print("   📝 Please edit .env with your API keys!")
        except Exception as e:
            print_step(f"Failed to create .env file: {e}", "error")
    
    # Check for required environment variables
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print_step("GROQ_API_KEY not set properly", "warning")
        print("   🔑 Get your API key from: https://console.groq.com/keys")
        return False
    else:
        print_step("GROQ_API_KEY configured", "success")
    
    # Set NEOS email if not already set
    neos_email = os.getenv("NEOS_EMAIL")
    if not neos_email or neos_email == "your_email@example.com":
        os.environ["NEOS_EMAIL"] = "demo@example.com"
        print_step("Using default NEOS_EMAIL", "warning")
    else:
        print_step("NEOS_EMAIL configured", "success")
    
    return True

def create_directories():
    """Create necessary directories with proper permissions"""
    print_step("Creating directories")
    
    dirs = ["output_data", "logs", "temp"]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        try:
            dir_path.mkdir(exist_ok=True, parents=True)
            # Set permissions (readable/writable by user)
            if hasattr(os, 'chmod'):
                os.chmod(dir_path, 0o755)
            print_step(f"Directory created: {dir_path}", "success")
        except Exception as e:
            print_step(f"Failed to create {dir_path}: {e}", "error")
            return False
    
    return True

def generate_initial_data():
    """Generate initial synthetic data"""
    print_step("Generating initial synthetic data")
    
    # Check if data already exists
    data_file = Path("output_data/synthetic_ops_data.csv")
    if data_file.exists():
        print_step("Data file already exists, skipping generation", "warning")
        return True
    
    try:
        # Import and run data generation
        data_script = Path("data_synth.py")
        if not data_script.exists():
            print_step("data_synth.py not found", "error")
            return False
        
        # Run data generation script
        success, output = run_command(
            [
                sys.executable, 
                str(data_script),
                "--days", "90",  # 3 months of data
                "--num-skus", "50", 
                "--num-locations", "5"
            ],
            "Running data generation script"
        )
        
        if success:
            print_step("Initial data generated successfully", "success")
            return True
        else:
            print_step(f"Data generation failed: {output}", "error")
            return False
            
    except Exception as e:
        print_step(f"Data generation failed: {e}", "error")
        return False

def test_streamlit():
    """Test Streamlit installation and basic functionality"""
    print_step("Testing Streamlit installation")
    
    try:
        # Test import
        success, _ = run_command(
            [sys.executable, "-c", "import streamlit; print('Streamlit import OK')"],
            "Testing Streamlit import",
            capture_output=False
        )
        
        if not success:
            print_step("Streamlit import failed", "error")
            return False
        
        # Test basic streamlit command
        success, _ = run_command(
            [sys.executable, "-m", "streamlit", "version"],
            "Testing Streamlit command"
        )
        
        if success:
            print_step("Streamlit ready", "success")
            return True
        else:
            print_step("Streamlit command test failed", "error")
            return False
            
    except Exception as e:
        print_step(f"Streamlit test failed: {e}", "error")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print_step("Testing module imports")
    
    modules = [
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("pyomo", "pyomo"),
        ("autogen", "pyautogen"),
        ("groq", "groq"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("watchdog", "watchdog")
    ]
    
    failed = []
    for display_name, module_name in modules:
        try:
            success, _ = run_command(
                [sys.executable, "-c", f"import {module_name}; print('{display_name} OK')"],
                f"Testing {display_name}",
                capture_output=False
            )
            if not success:
                failed.append(display_name)
        except Exception:
            failed.append(display_name)
    
    if failed:
        print_step(f"Failed imports: {', '.join(failed)}", "error")
        print("   💡 Try: pip install --upgrade --force-reinstall " + " ".join(failed))
        return False
    
    print_step("All imports successful", "success")
    return True

def run_health_check():
    """Run comprehensive system health check"""
    print_step("Running system health check")
    
    try:
        # Try to import and run config health check
        success, output = run_command(
            [sys.executable, "-c", """
import sys
sys.path.append('.')
from config import check_system_health
health = check_system_health()
print(f"System status: {health['status']}")
if health.get('errors'):
    print('Errors:', '; '.join(health['errors']))
if health.get('warnings'):
    print('Warnings:', '; '.join(health['warnings']))
"""],
            "Running health check",
            capture_output=False
        )
        
        return success
        
    except Exception as e:
        print_step(f"Health check failed: {e}", "warning")
        return True  # Don't fail setup for this

def main():
    """Main setup process with comprehensive error handling"""
    print("🚀 Enhanced Setup for Inventory ROP Lab")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Working directory: {Path.cwd()}")
    print("=" * 50)
    
    steps = [
        ("Python version check", check_python_version),
        ("Pip check and update", check_pip_and_update),
        ("Installing requirements", install_requirements),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Testing imports", test_imports),
        ("Testing Streamlit", test_streamlit),
        ("Generating initial data", generate_initial_data),
        ("Running health check", run_health_check),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
                if step_name in ["Python version check", "Installing requirements"]:
                    # Critical failures
                    print_step(f"Critical failure at: {step_name}", "error")
                    print("Setup cannot continue.")
                    return False
                else:
                    print_step(f"Non-critical failure at: {step_name}", "warning")
        except Exception as e:
            print_step(f"Exception in {step_name}: {e}", "error")
            failed_steps.append(step_name)
            if step_name in ["Python version check", "Installing requirements"]:
                return False
    
    print("\n" + "=" * 50)
    if not failed_steps:
        print_step("Setup completed successfully!", "success")
        print("\n🎉 Everything is ready!")
    else:
        print_step("Setup completed with warnings", "warning")
        print(f"⚠️  Issues with: {', '.join(failed_steps)}")
    
    print("\n📋 Next steps:")
    print("1. Edit .env file with your GROQ_API_KEY")
    print("2. Run the app: streamlit run app.py")
    print("3. If you encounter issues, check the logs above")
    
    print("\n🔗 Helpful links:")
    print("  • GROQ API: https://console.groq.com/")
    print("  • AutoGen docs: https://microsoft.github.io/autogen/")
    print("  • Troubleshooting: Check the logs in the 'logs' directory")
    
    return len(failed_steps) == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)