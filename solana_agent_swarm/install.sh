#!/bin/bash
# Installation script for Solana Token Analysis Agent Swarm

echo "Installing Solana Token Analysis Agent Swarm..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install the package
echo "Installing the package..."
pip install -e .

# Create default configuration
echo "Creating default configuration..."
python src/cli.py --create-config

echo "Installation complete!"
echo ""
echo "To start the CLI:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the CLI: python src/cli.py"
echo ""
echo "Remember to edit config.json to add your Solana RPC endpoint and API keys!"
