# Integration Guide: Enhanced Tools & Libraries

This guide documents the implementation of additional tools and libraries for the Solana Token Analysis Agent Swarm.

## Overview

The following enhancements have been added to the system:

1. **Analytics & Machine Learning**
   - ML utilities with scikit-learn and PyTorch integration
   - Pattern detection and anomaly detection capabilities
   - Deep learning for price prediction

2. **Solana Ecosystem Integration**
   - Helius API client for enriched transaction data
   - Jupiter Aggregator for DEX liquidity analysis
   - Switchboard oracle access via MCP server
   - SPL token utilities for compliance checking

3. **Visualization & Dashboard**
   - D3.js network graph visualization for token relationships
   - ECharts integration for comprehensive token metrics dashboards
   - TradingView Lightweight Charts for financial charting

4. **Testing & Monitoring**
   - Pytest setup for Python components
   - Jest integration for frontend components

## Installation

To install all the enhanced tools and libraries:

```bash
# Make the setup script executable
chmod +x setup_enhanced_tools.sh

# Run the setup script
./setup_enhanced_tools.sh
```

This will:
1. Install all required Python packages
2. Set up Node.js packages for MCP servers
3. Install frontend visualization packages
4. Configure MCP settings for the Claude desktop app

## Directory Structure

```
solana_agent_swarm/
├── requirements-enhanced.txt    # Enhanced Python requirements
├── mcp_settings.json           # MCP server configuration
├── setup_enhanced_tools.sh     # Setup script
├── src/
│   ├── utils/
│   │   ├── ml_utils.py         # ML utilities (scikit-learn, PyTorch)
│   │   └── solana_ecosystem.py # Solana ecosystem integrations
│   ├── frontend/
│   │   ├── package.json        # Frontend dependencies
│   │   └── visualization.js    # Visualization components
│   ├── mcp_servers/
│   │   └── switchboard_mcp_server.js # Switchboard oracle MCP server
│   └── tests/
│       └── test_ml_utils.py    # ML utilities tests
```

## Using the Machine Learning Utilities

The ML utilities provide pattern detection, anomaly detection, and price prediction capabilities:

```python
from utils.ml_utils import PatternDetector, TokenPricePredictor

# Create and train pattern detector
detector = PatternDetector()
detector.train(historical_data, pattern_labels)

# Detect anomalies
anomalies = detector.detect_anomalies(token_data)

# Classify patterns
pattern = detector.classify_pattern(token_data)

# Create price predictor
predictor = TokenPricePredictor()
await predictor.train(features, targets)
predictions = await predictor.predict(new_features)
```

## Using Solana Ecosystem Integrations

The Solana ecosystem integrations provide access to Helius API, Jupiter Aggregator, and SPL token utilities:

```python
from utils.solana_ecosystem import HeliusClient, JupiterClient, SPLTokenChecker

# Helius API client
helius = HeliusClient(api_key="your_api_key")
await helius.initialize()
token_metadata = await helius.get_token_metadata("token_address")
token_holders = await helius.get_token_holders("token_address")

# Jupiter client for liquidity analysis
jupiter = JupiterClient()
await jupiter.initialize()
price_data = await jupiter.get_token_price("token_address")
liquidity_analysis = await jupiter.analyze_liquidity_depth("token_address")

# SPL token compliance checking
spl_checker = SPLTokenChecker(rpc_url="https://api.mainnet-beta.solana.com")
await spl_checker.initialize()
token_info = await spl_checker.get_token_info("token_address")
compliance_check = await spl_checker.check_token_compliance("token_address")
```

## Using Switchboard Oracle MCP Server

The Switchboard Oracle MCP server provides access to verified price data for various tokens:

```javascript
// Example Claude prompt:
// "Use the Switchboard oracle to get the current price of SOL"

// Claude will use:
<use_mcp_tool>
<server_name>switchboard-oracle</server_name>
<tool_name>get_price_feed</tool_name>
<arguments>
{
  "feed_identifier": "SOL/USD"
}
</arguments>
</use_mcp_tool>

// To get multiple prices:
<use_mcp_tool>
<server_name>switchboard-oracle</server_name>
<tool_name>get_multiple_prices</tool_name>
<arguments>
{
  "feed_identifiers": ["SOL/USD", "BTC/USD", "ETH/USD"]
}
</arguments>
</use_mcp_tool>

// To list available feeds:
<use_mcp_tool>
<server_name>switchboard-oracle</server_name>
<tool_name>list_available_feeds</tool_name>
<arguments>
{}
</arguments>
</use_mcp_tool>
```

## Using Visualization Components

The visualization components provide powerful data visualization capabilities for the frontend:

```javascript
import { 
  TokenRelationshipGraph, 
  TokenometricsDashboard, 
  TradingViewChartManager 
} from './visualization.js';

// D3.js Network Graph
const relationshipGraph = new TokenRelationshipGraph('token-network-container');
relationshipGraph.initialize();
relationshipGraph.render(nodes, links);

// ECharts Dashboard
const dashboard = new TokenometricsDashboard();
dashboard.initialize('dashboard-container');
dashboard.updateCharts(tokenData);

// TradingView Chart
const chart = new TradingViewChartManager('tradingview-container');
chart.initialize();
chart.addCandlestickSeries(candleData);
chart.addVolumeSeries(volumeData);
```

## Running Tests

To run the machine learning utility tests:

```bash
cd solana_agent_swarm
pytest src/tests/test_ml_utils.py -v
```

To run frontend tests:

```bash
cd solana_agent_swarm/src/frontend
npm test
```

## Next Steps

1. **Integration with Existing Agents**:
   - Update the Pattern Analysis Agent to use the new ML utilities
   - Update the Data Collection Agent to use the Solana ecosystem integrations
   - Update the frontend to use the visualization components

2. **Additional Data Sources**:
   - Integrate The Graph for indexed blockchain data
   - Add more oracle data feeds from Switchboard
   - Connect to additional DEXs via Jupiter

3. **Monitoring Setup**:
   - Configure Prometheus/Grafana for system monitoring
   - Set up DataDog for application performance monitoring

4. **Testing Expansion**:
   - Create comprehensive test suites for all components
   - Implement continuous integration for automated testing
