#!/usr/bin/env python3
"""
Solana Token Analysis Agent Swarm Dashboard Launcher

This script launches the Streamlit dashboard for the Solana Token Analysis Agent Swarm.
It handles checking requirements and setting up the environment.

Usage:
    python run_dashboard.py
"""

import os
import sys
import subprocess
import shutil
import time


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        print("✓ Required dependencies (streamlit, plotly) are installed.")
        return True
    except ImportError as e:
        print(f"✗ Required dependency not found: {e.name}")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    try:
        # Get the path to the requirements.txt file
        requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        print(f"Using requirements file: {requirements_path}")
        
        # Install dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
        print("✓ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {str(e)}")
        return False


def check_env():
    """Check and setup environment variables."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "mcp_servers", ".env")
    env_example = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "mcp_servers", ".env.example")
    
    if not os.path.exists(env_file) and os.path.exists(env_example):
        print("Environment file not found. Using example file...")
        shutil.copy(env_example, env_file)
        print(f"✓ Created .env file at {env_file}")
        print("⚠ Please update the .env file with your actual API keys")
    
    required_vars = ["SOLANA_PRIVATE_KEY", "SOLANA_RPC_URL"]
    missing_vars = []
    
    # Check .env file for required variables
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_contents = f.read()
            for var in required_vars:
                if f"{var}=" not in env_contents or f"{var}=\"\"" in env_contents or f"{var}=''" in env_contents:
                    missing_vars.append(var)
    else:
        missing_vars = required_vars
    
    if missing_vars:
        print("⚠ Some required environment variables are missing:")
        for var in missing_vars:
            print(f"  - {var}")
        print("The dashboard will run, but some features may not work correctly.")
    else:
        print("✓ Environment variables are set.")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "frontend", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"✗ Dashboard file not found at {dashboard_path}")
        return False
    
    print("Launching dashboard...")
    command = ["streamlit", "run", dashboard_path, "--server.headless", "true"]
    
    try:
        process = subprocess.Popen(command)
        print(f"✓ Dashboard launched. Open your browser to view it.")
        print("  If it doesn't open automatically, go to http://localhost:8501")
        print("  Press Ctrl+C to stop the dashboard.")
        
        # Keep the script running until user interrupts
        process.wait()
        return True
    except Exception as e:
        print(f"✗ Failed to launch dashboard: {str(e)}")
        return False


def main():
    """Main function to run the dashboard."""
    print("=" * 60)
    print("Solana Token Analysis Agent Swarm Dashboard Launcher")
    print("=" * 60)
    
    # Check if streamlit is installed
    if not check_dependencies():
        print("Installing missing dependencies...")
        if not install_dependencies():
            print("Please install the required dependencies manually:")
            print("  pip install -r requirements.txt")
            return
    
    # Check environment variables
    check_env()
    
    # Launch dashboard
    try:
        launch_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
