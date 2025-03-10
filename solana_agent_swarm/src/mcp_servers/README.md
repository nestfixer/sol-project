# Solana Agent Swarm MCP Servers

This directory contains Model Context Protocol (MCP) servers that integrate the Solana Agent Kit with the Solana Agent Swarm. These servers enable AI agents to interact with the Solana blockchain for token analysis, pattern detection, and risk assessment.

## Overview

The MCP servers consist of three specialized servers:

1. **Solana Data MCP**  
   Provides access to Solana blockchain data, token information, and market data using the Solana Agent Kit.

2. **Pattern Learning MCP**  
   Implements pattern detection and analysis with feedback loops for continuous learning.

3. **Risk Assessment MCP**  
   Performs token risk assessment, rug pull detection, and provides safety recommendations.

## Prerequisites

- Node.js 18.0.0 or higher
- Python 3.8 or higher
- Solana Agent Kit dependencies
- MCP SDK
- Claude AI or similar LLM with MCP support

## Installation

1. Install Node.js dependencies:
   ```bash
   cd solana_agent_swarm/src/mcp_servers
   npm install
   ```

2. Create an `.env` file by copying the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to add your credentials:
   - `SOLANA_PRIVATE_KEY`: Your Solana private key in base58 format
   - `SOLANA_RPC_URL`: Your Solana RPC URL (default: https://api.mainnet-beta.solana.com)
   - `OPENAI_API_KEY`: Your OpenAI API key for AI functionalities

## Register MCP Servers

Register the MCP servers with your AI assistant using the registration script:

```bash
python register_mcp_servers.py
```

This script will register the servers in the appropriate MCP settings file for your platform:
- Windows: `%APPDATA%\Windsurf\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

If your settings file is in a different location, specify it:

```bash
python register_mcp_servers.py --settings-path /path/to/your/settings.json
```

## Running the Servers

You can start all servers at once using:

```bash
node index.js
```

Or start them individually:

```bash
node solana_data_mcp/index.js
node pattern_learning_mcp/index.js
node risk_assessment_mcp/index.js
```

## Available Tools

### Solana Data MCP

- `get_token_info`: Get comprehensive information about a Solana token
- `get_token_price`: Get the current price of a Solana token
- `get_trending_tokens`: Get trending tokens on Solana
- `get_wallet_tokens`: Get token balances for a wallet
- `detect_rug_pull_risk`: Analyze a token for rug pull risk factors

### Pattern Learning MCP

- `analyze_token_pattern`: Analyze historical data to identify patterns for a specific token
- `get_price_prediction`: Get price prediction based on identified patterns
- `detect_market_patterns`: Detect general market patterns across multiple tokens
- `add_to_blacklist`: Add a wallet or token to the blacklist
- `check_blacklist`: Check if a wallet or token is blacklisted
- `provide_pattern_feedback`: Provide feedback for learning algorithm improvement

### Risk Assessment MCP

- `assess_token_risk`: Perform a comprehensive risk assessment of a Solana token
- `get_rug_pull_probability`: Calculate the probability of a token being a rug pull
- `identify_high_risk_wallets`: Identify high-risk wallets associated with a token
- `report_rug_pull`: Report a confirmed rug pull for a token
- `get_safety_recommendations`: Get safety recommendations for interacting with a token

## Data Persistence

Each MCP server maintains its own persistent data:

- Pattern Learning MCP stores patterns and blacklisted wallets/tokens
- Risk Assessment MCP stores risk assessments and confirmed rug pulls

Data is stored in JSON files in the respective server directories.

## Example Usage

Once the MCP servers are registered and running, you can use them through the AI assistant:

```
Analyze the risk for token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
```

The AI assistant will use the appropriate MCP tools to perform the analysis.

## Extending Functionality

To add new capabilities:

1. Create new tool handlers in the appropriate MCP server
2. Update the tool listing in the server's `setupToolHandlers` method
3. Implement the new functionality in the server class
4. Register the updated server

## Troubleshooting

- **MCP Connection Issues**: Ensure servers are running and registered correctly
- **API Errors**: Check environment variables and API key validity
- **Tool Execution Errors**: Examine server logs for detailed error information

For more detailed logs, set `LOG_LEVEL=debug` in your `.env` file.
