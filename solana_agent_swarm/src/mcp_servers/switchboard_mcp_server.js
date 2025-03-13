#!/usr/bin/env node
/**
 * Switchboard Oracle MCP Server for Solana Token Analysis Agent Swarm
 * 
 * This MCP server provides access to Switchboard oracle data feeds on Solana.
 * It allows querying price feeds for accurate, verified price data.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';

// Note: These imports would need to be installed:
// npm install @solana/web3.js @switchboard-xyz/solana.js
import { Connection, PublicKey } from '@solana/web3.js';
import { AggregatorAccount, SwitchboardProgram } from '@switchboard-xyz/solana.js';

// Common price feed addresses for Switchboard oracles
const PRICE_FEEDS = {
  'BTC/USD': 'GVXRSBjFk6e6J3NbVPXohDJetcTjaeeuykUpbQF8UoMU',
  'ETH/USD': '5zxs8uhiUxwHHLrUf6Pu6nHrbB8X8KyfFUnxcw8RpxJP',
  'SOL/USD': 'GvDMxPzN1sCj7L26YDK2HnMRXEQmQ2aemov8YBtPS7vR',
  'USDT/USD': '3vxLXJqLqF3JG5TCbYycbKWRBbCJQLxQmBGCkyqEEefL',
  'USDC/USD': 'BjUgj6YCnFBZ49wF54ddBVA9qu8TeqkjuVWvCzESXes7',
  'BONK/USD': 'BCNdrXXLSgqm2NEK6hCnR2PQKpxrwUxAjBZKgQJV5S4g',
  'JTO/USD': '7yyaeuJ1GGtVBLT2z2xub5ZWYKaNhF28mj1RdV4VDFoo',
  'JUP/USD': 'PEvnEMiUXqR5vZvodgvazdhpJtgyP7YB9NMXUe5cMr1',
  'RAY/USD': '7Mf8e9PtDgdisX9L5qys7nvPBMdwTbr4QDzJBRtZL1Xn',
  'RNDR/USD': 'DSb39i3mJ2XpfQPNvBiTWf8RjPMkKMqDPKVan3NiTUQm',
  'PYTH/USD': '8uJgyn5bEMKkXdUCqEboKP3uuxyRvPXPhtThQTvXHi7e'
};

class SwitchboardOracleMcpServer {
  constructor() {
    // Initialize the MCP server
    this.server = new Server(
      {
        name: 'switchboard-oracle-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize Solana connection
    const rpcUrl = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
    this.connection = new Connection(rpcUrl, 'confirmed');
    this.program = null;
    
    // Set up tool handlers
    this.setupToolHandlers();
    
    // Set up error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'get_price_feed',
          description: 'Get current price data from Switchboard oracle feed',
          inputSchema: {
            type: 'object',
            properties: {
              feed_identifier: {
                type: 'string',
                description: 'Feed identifier (e.g., "SOL/USD") or feed address',
              },
            },
            required: ['feed_identifier'],
          },
        },
        {
          name: 'get_multiple_prices',
          description: 'Get prices for multiple assets at once',
          inputSchema: {
            type: 'object',
            properties: {
              feed_identifiers: {
                type: 'array',
                items: {
                  type: 'string',
                },
                description: 'Array of feed identifiers (e.g., ["SOL/USD", "BTC/USD"])',
              },
            },
            required: ['feed_identifiers'],
          },
        },
        {
          name: 'list_available_feeds',
          description: 'List all available price feeds',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
      ],
    }));

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      // Initialize Switchboard program if needed
      if (!this.program) {
        try {
          this.program = await SwitchboardProgram.load(this.connection);
        } catch (error) {
          throw new McpError(
            ErrorCode.InternalError,
            `Failed to initialize Switchboard program: ${error}`
          );
        }
      }
      
      // Route to appropriate handler based on tool name
      switch (request.params.name) {
        case 'get_price_feed':
          return this.handleGetPriceFeed(request.params.arguments);
        case 'get_multiple_prices':
          return this.handleGetMultiplePrices(request.params.arguments);
        case 'list_available_feeds':
          return this.handleListAvailableFeeds();
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${request.params.name}`
          );
      }
    });
  }
  
  async handleGetPriceFeed(args) {
    if (!args.feed_identifier) {
      throw new McpError(
        ErrorCode.InvalidParams,
        'Missing feed_identifier parameter'
      );
    }
    
    try {
      let feedAddress = args.feed_identifier;
      
      // Check if the input is a known pair
      if (PRICE_FEEDS[feedAddress]) {
        feedAddress = PRICE_FEEDS[feedAddress];
      }
      
      // Check if it's a valid address
      let feedPubkey;
      try {
        feedPubkey = new PublicKey(feedAddress);
      } catch (e) {
        throw new McpError(
          ErrorCode.InvalidParams,
          `Invalid feed address: ${feedAddress}`
        );
      }
      
      // Get price from the oracle
      const aggregatorAccount = new AggregatorAccount({
        program: this.program,
        publicKey: feedPubkey,
      });
      
      const result = await aggregatorAccount.fetchLatestValue();
      if (result === null) {
        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({ error: 'No value available for feed' }, null, 2),
            },
          ],
          isError: true,
        };
      }
      
      // Get metadata for the feed
      const metadata = await aggregatorAccount.loadData();
      
      // Return the price data
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              feed: args.feed_identifier,
              address: feedAddress,
              price: result.toString(),
              price_formatted: Number(result).toFixed(metadata.decimals),
              decimals: metadata.decimals,
              lastUpdatedTimestamp: new Date(metadata.latestConfirmedRound.roundOpenTimestamp.toNumber() * 1000).toISOString(),
              confidence_interval: metadata.latestConfirmedRound.stdDeviation.toString(),
              min_response: metadata.latestConfirmedRound.minResponse.toString(),
              max_response: metadata.latestConfirmedRound.maxResponse.toString()
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error fetching price feed: ${error}`,
          },
        ],
        isError: true,
      };
    }
  }
  
  async handleGetMultiplePrices(args) {
    if (!args.feed_identifiers || !Array.isArray(args.feed_identifiers)) {
      throw new McpError(
        ErrorCode.InvalidParams,
        'Missing or invalid feed_identifiers parameter'
      );
    }
    
    try {
      const results = await Promise.all(
        args.feed_identifiers.map(async (identifier) => {
          try {
            let feedAddress = identifier;
            
            // Check if the input is a known token pair
            if (PRICE_FEEDS[identifier]) {
              feedAddress = PRICE_FEEDS[identifier];
            }
            
            const feedPubkey = new PublicKey(feedAddress);
            const aggregatorAccount = new AggregatorAccount({
              program: this.program,
              publicKey: feedPubkey,
            });
            
            const result = await aggregatorAccount.fetchLatestValue();
            if (result === null) {
              return {
                feed: identifier,
                error: 'No value available',
              };
            }
            
            const metadata = await aggregatorAccount.loadData();
            
            return {
              feed: identifier,
              address: feedAddress,
              price: result.toString(),
              price_formatted: Number(result).toFixed(metadata.decimals),
              decimals: metadata.decimals,
              lastUpdatedTimestamp: new Date(metadata.latestConfirmedRound.roundOpenTimestamp.toNumber() * 1000).toISOString(),
            };
          } catch (e) {
            return {
              feed: identifier,
              error: `Error: ${e}`,
            };
          }
        })
      );
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(results, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error fetching multiple prices: ${error}`,
          },
        ],
        isError: true,
      };
    }
  }
  
  async handleListAvailableFeeds() {
    try {
      const formattedFeeds = Object.entries(PRICE_FEEDS).map(([pair, address]) => ({
        pair,
        address,
        description: `Price feed for ${pair}`
      }));
      
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(formattedFeeds, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error listing available feeds: ${error}`,
          },
        ],
        isError: true,
      };
    }
  }

  async run() {
    // Use stdio transport for MCP server
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Switchboard oracle MCP server running on stdio');
  }
}

// Create and run the server
const server = new SwitchboardOracleMcpServer();
server.run().catch(console.error);
