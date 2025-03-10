# Solana Token Analysis Agent Swarm

An AI agent swarm system for analyzing Solana tokens, detecting patterns, assessing risks, and identifying investment opportunities.

## Overview

This project implements a multi-agent system for real-time monitoring and analysis of Solana tokens. The agent swarm leverages SolScan's RPC and WebSocket APIs to track token events, analyze transaction patterns, assess investment risks, and identify high-potential opportunities.

### Key Features

- **Real-time monitoring** of Solana blockchain for new token events
- **Transaction pattern analysis** to identify promising investment signals
- **Risk assessment** for detecting scams, rug pulls, and other hazards
- **Investment opportunity detection** using combined pattern and risk data
- **Collaborative agent architecture** with shared knowledge base
- **Extensible framework** for adding new analysis techniques

## Architecture

The system is built as a swarm of specialized agents that work collaboratively:

1. **Data Collection Agent**: Connects to Solana RPC/WebSocket to gather token data
2. **Pattern Analysis Agent**: Identifies trading patterns that may indicate high-performing tokens
3. **Risk Assessment Agent**: Evaluates token risks, contract safety, and liquidity
4. **Investment Advisory Agent**: Combines pattern and risk data to provide investment recommendations
5. **Orchestrator Agent**: Manages agent lifecycle and coordinates communication

These agents share data through a central Knowledge Base, enabling efficient collaboration and real-time analysis.

## Installation

### Prerequisites

- Python 3.8+
- Solana RPC/WebSocket access

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/solana-token-analysis.git
   cd solana-token-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a default configuration file:
   ```
   python src/cli.py --create-config
   ```

4. Edit `config.json` to add your Solana RPC endpoint and other API keys.

## Usage

### Command-Line Interface

The project includes an interactive CLI for testing and using the system:

```
python src/cli.py
```

Once the CLI is running, type `help` to see available commands:

- `start` - Start the agent swarm
- `collect_token_data <token_address>` - Collect data for a specific token
- `analyze_patterns <token_address>` - Analyze trading patterns for a token
- `assess_risk <token_address>` - Assess risk factors for a token
- `evaluate_opportunity <token_address>` - Evaluate investment potential for a token
- `get_top_opportunities [count]` - Get the top investment opportunities

### Running as a Service

To run the agent swarm as a continuous service:

```
python src/main.py
```

## Configuration

The system uses a JSON configuration file (`config.json` by default) with the following sections:

- `solana`: Solana RPC and WebSocket connection settings
- `apis`: External API configurations
- `agents`: Agent-specific settings
- `storage`: Data storage configuration
- `logging`: Logging settings
- `token_analysis`: Analysis parameters and thresholds

Environment variables prefixed with `SOLANA_SWARM_` can also be used to override configuration values.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
