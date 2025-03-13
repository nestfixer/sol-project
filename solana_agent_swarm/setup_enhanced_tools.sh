#!/bin/bash
# Setup script for enhanced tools and libraries for Solana Token Analysis Agent Swarm

echo "Setting up enhanced tools and libraries for Solana Token Analysis Agent Swarm..."

# Create necessary directories if they don't exist
mkdir -p src/mcp_servers
mkdir -p src/utils
mkdir -p src/frontend
mkdir -p src/tests

# Install Python packages
echo "Installing Python packages..."
pip install -r requirements-enhanced.txt

# Install Node.js packages for MCP servers
echo "Installing Node.js packages for MCP servers..."
cd src/mcp_servers
npm init -y
npm install @modelcontextprotocol/sdk @solana/web3.js @switchboard-xyz/solana.js
cd ../..

# Install frontend packages
echo "Installing frontend visualization packages..."
cd src/frontend
npm install
cd ../..

# Set up MCP settings
echo "Setting up MCP settings..."
if [ -d "$HOME/.windsurf/extensions/saoudrizwan.claude-dev/User/globalStorage/saoudrizwan.claude-dev/settings" ]; then
    # Copy to Claude desktop app settings
    cp mcp_settings.json "$HOME/.windsurf/extensions/saoudrizwan.claude-dev/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
    echo "MCP settings copied to Claude desktop app"
else
    echo "Claude desktop app settings directory not found. Please manually copy mcp_settings.json to your settings location."
fi

# Make sure permissions are correct for MCP servers
chmod +x src/mcp_servers/switchboard_mcp_server.js

echo "Setup complete! You can now use the enhanced tools and libraries."
echo "To use the MCP servers, make sure you have the Claude desktop app or another MCP-compatible client."
echo "To use the frontend visualization, run 'npm run dev' in the src/frontend directory."
