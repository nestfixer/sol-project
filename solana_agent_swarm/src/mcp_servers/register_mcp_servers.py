#!/usr/bin/env python3
"""
Register Solana Agent Swarm MCP Servers

This script registers the Solana Agent Swarm MCP servers in the MCP settings file.
It updates the configuration to make the servers available to the agent swarm.

Usage:
    python register_mcp_servers.py [--settings-path PATH]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Default settings path for different platforms
DEFAULT_SETTINGS_PATHS = {
    'win32': os.path.expanduser('~\\AppData\\Roaming\\Windsurf\\User\\globalStorage\\saoudrizwan.claude-dev\\settings\\cline_mcp_settings.json'),
    'darwin': os.path.expanduser('~/Library/Application Support/Claude/claude_desktop_config.json'),
    'linux': os.path.expanduser('~/.config/Claude/claude_desktop_config.json')
}

# Get the current directory
CURRENT_DIR = Path(__file__).resolve().parent

def get_default_settings_path():
    """Get the default settings path based on the platform."""
    return DEFAULT_SETTINGS_PATHS.get(sys.platform)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Register Solana Agent Swarm MCP servers")
    parser.add_argument('--settings-path', help='Path to the MCP settings file')
    return parser.parse_args()

def load_settings(settings_path):
    """Load the MCP settings file."""
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Settings file not found: {settings_path}")
        print("Creating new settings file...")
        return {"mcpServers": {}}
    except json.JSONDecodeError:
        print(f"Error parsing settings file: {settings_path}")
        print("Creating new settings file...")
        return {"mcpServers": {}}

def save_settings(settings_path, settings):
    """Save the MCP settings file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"Settings saved to: {settings_path}")

def register_mcp_servers(settings):
    """Register the Solana Agent Swarm MCP servers in the settings."""
    # Convert paths to absolute paths
    solana_data_path = str(CURRENT_DIR / "solana_data_mcp" / "index.js")
    pattern_learning_path = str(CURRENT_DIR / "pattern_learning_mcp" / "index.js")
    risk_assessment_path = str(CURRENT_DIR / "risk_assessment_mcp" / "index.js")
    
    # Ensure mcpServers exists
    if "mcpServers" not in settings:
        settings["mcpServers"] = {}
    
    # Register the servers
    settings["mcpServers"]["solana_data_mcp"] = {
        "command": "node",
        "args": [solana_data_path],
        "env": {
            "SOLANA_PRIVATE_KEY": "${SOLANA_PRIVATE_KEY}",
            "SOLANA_RPC_URL": "${SOLANA_RPC_URL}",
            "OPENAI_API_KEY": "${OPENAI_API_KEY}"
        },
        "disabled": False,
        "autoApprove": []
    }
    
    settings["mcpServers"]["pattern_learning_mcp"] = {
        "command": "node",
        "args": [pattern_learning_path],
        "env": {},
        "disabled": False,
        "autoApprove": []
    }
    
    settings["mcpServers"]["risk_assessment_mcp"] = {
        "command": "node",
        "args": [risk_assessment_path],
        "env": {},
        "disabled": False,
        "autoApprove": []
    }
    
    return settings

def main():
    """Main function."""
    args = parse_arguments()
    
    # Determine settings path
    settings_path = args.settings_path or get_default_settings_path()
    
    if not settings_path:
        print("Error: Could not determine settings path. Please specify with --settings-path.")
        sys.exit(1)
    
    print(f"Using settings path: {settings_path}")
    
    # Load settings
    settings = load_settings(settings_path)
    
    # Register MCP servers
    settings = register_mcp_servers(settings)
    
    # Save settings
    save_settings(settings_path, settings)
    
    print("Solana Agent Swarm MCP servers registered successfully!")
    print("\nImportant: You need to set the environment variables in your system or in the .env file:")
    print("  - SOLANA_PRIVATE_KEY: Your Solana private key in base58 format")
    print("  - SOLANA_RPC_URL: Your Solana RPC URL (default: https://api.mainnet-beta.solana.com)")
    print("  - OPENAI_API_KEY: Your OpenAI API key for AI functionalities\n")
    print("Restart Claude or your application to apply the changes.")

if __name__ == "__main__":
    main()
