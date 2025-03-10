#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { SolanaAgentKit } from 'solana-agent-kit';
import dotenv from 'dotenv';
import axios from 'axios';
import { PublicKey } from '@solana/web3.js';

// Load environment variables
dotenv.config();

// Initialize Solana Agent Kit
const PRIVATE_KEY = process.env.SOLANA_PRIVATE_KEY;
const RPC_URL = process.env.SOLANA_RPC_URL || 'https://api.mainnet-beta.solana.com';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!PRIVATE_KEY) {
  console.error('SOLANA_PRIVATE_KEY environment variable is required');
  process.exit(1);
}

class SolanaDataMCP {
  constructor() {
    this.server = new Server(
      {
        name: 'solana-data-mcp',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    try {
      // Initialize Solana Agent Kit
      this.agent = new SolanaAgentKit(
        PRIVATE_KEY,
        RPC_URL,
        OPENAI_API_KEY
      );
      console.error('Solana Agent Kit initialized successfully');
    } catch (error) {
      console.error('Failed to initialize Solana Agent Kit:', error);
      process.exit(1);
    }

    // Set up handlers
    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'get_token_info',
          description: 'Get information about a Solana token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
            },
            required: ['token_address'],
          },
        },
        {
          name: 'get_token_price',
          description: 'Get the current price of a Solana token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
            },
            required: ['token_address'],
          },
        },
        {
          name: 'get_trending_tokens',
          description: 'Get trending tokens on Solana',
          inputSchema: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
        {
          name: 'get_wallet_tokens',
          description: 'Get token balances for a wallet',
          inputSchema: {
            type: 'object',
            properties: {
              wallet_address: {
                type: 'string',
                description: 'Solana wallet address',
              },
            },
            required: ['wallet_address'],
          },
        },
        {
          name: 'detect_rug_pull_risk',
          description: 'Analyze a token for rug pull risk factors',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
            },
            required: ['token_address'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        let result;

        switch (name) {
          case 'get_token_info':
            result = await this.getTokenInfo(args.token_address);
            break;
          case 'get_token_price':
            result = await this.getTokenPrice(args.token_address);
            break;
          case 'get_trending_tokens':
            result = await this.getTrendingTokens();
            break;
          case 'get_wallet_tokens':
            result = await this.getWalletTokens(args.wallet_address);
            break;
          case 'detect_rug_pull_risk':
            result = await this.detectRugPullRisk(args.token_address);
            break;
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${name}`
            );
        }

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      } catch (error) {
        console.error(`Error executing tool ${name}:`, error);
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  async getTokenInfo(tokenAddress) {
    try {
      // Use Agent Kit to get token info
      const tokenInfo = await this.agent.getTokenInfo(tokenAddress);
      return tokenInfo;
    } catch (error) {
      console.error('Error getting token info:', error);
      throw new Error(`Failed to get token info: ${error.message}`);
    }
  }

  async getTokenPrice(tokenAddress) {
    try {
      // Use Agent Kit to get token price
      const priceData = await this.agent.getTokenPriceData([tokenAddress]);
      return priceData;
    } catch (error) {
      console.error('Error getting token price:', error);
      throw new Error(`Failed to get token price: ${error.message}`);
    }
  }

  async getTrendingTokens() {
    try {
      // Use Agent Kit to get trending tokens
      const trendingTokens = await this.agent.getTrendingTokens();
      return trendingTokens;
    } catch (error) {
      console.error('Error getting trending tokens:', error);
      throw new Error(`Failed to get trending tokens: ${error.message}`);
    }
  }

  async getWalletTokens(walletAddress) {
    try {
      // Convert the address to a PublicKey
      const publicKey = new PublicKey(walletAddress);
      
      // Get token accounts - this is a custom implementation as Agent Kit doesn't have this directly
      const response = await axios.post(RPC_URL, {
        jsonrpc: '2.0',
        id: 1,
        method: 'getTokenAccountsByOwner',
        params: [
          walletAddress,
          {
            programId: 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'
          },
          {
            encoding: 'jsonParsed'
          }
        ]
      });
      
      if (response.data.error) {
        throw new Error(response.data.error.message);
      }
      
      const tokenAccounts = response.data.result.value;
      
      // Format the token data
      const tokens = tokenAccounts.map(account => {
        const parsedInfo = account.account.data.parsed.info;
        return {
          mint: parsedInfo.mint,
          amount: parsedInfo.tokenAmount.uiAmount,
          decimals: parsedInfo.tokenAmount.decimals
        };
      });
      
      return { walletAddress, tokens };
    } catch (error) {
      console.error('Error getting wallet tokens:', error);
      throw new Error(`Failed to get wallet tokens: ${error.message}`);
    }
  }

  async detectRugPullRisk(tokenAddress) {
    try {
      // This is a simplified implementation that would be expanded with real analytics
      
      // Get token information
      const tokenInfo = await this.getTokenInfo(tokenAddress);
      
      // Get token holders
      const holdersInfo = tokenInfo?.holders;
      
      // Calculate risk metrics (simplified)
      const riskMetrics = {
        concentrationRisk: 0,
        liquidityRisk: 0,
        socialRisk: 0,
        overallRisk: 0
      };
      
      // Calculate concentration risk
      if (holdersInfo?.distribution) {
        const topHolderPercentage = holdersInfo.distribution[0]?.percentage || 0;
        riskMetrics.concentrationRisk = topHolderPercentage > 50 ? 'High' : 
                                        topHolderPercentage > 30 ? 'Medium' : 'Low';
      }
      
      // Calculate liquidity risk (simplified)
      const marketInfo = tokenInfo?.market;
      if (marketInfo) {
        riskMetrics.liquidityRisk = marketInfo.volume_24h < 1000 ? 'High' : 
                                   marketInfo.volume_24h < 10000 ? 'Medium' : 'Low';
      }
      
      // Calculate social risk (simplified)
      const socialInfo = tokenInfo?.social;
      if (socialInfo) {
        riskMetrics.socialRisk = socialInfo.sentiment_score < 0.3 ? 'High' : 
                                socialInfo.sentiment_score < 0.6 ? 'Medium' : 'Low';
      }
      
      // Calculate overall risk
      const riskLevels = {
        'Low': 1,
        'Medium': 2,
        'High': 3
      };
      
      const avgRisk = (
        riskLevels[riskMetrics.concentrationRisk] + 
        riskLevels[riskMetrics.liquidityRisk] + 
        riskLevels[riskMetrics.socialRisk]
      ) / 3;
      
      riskMetrics.overallRisk = avgRisk > 2.5 ? 'High' : avgRisk > 1.5 ? 'Medium' : 'Low';
      
      return {
        tokenAddress,
        tokenName: tokenInfo?.name || 'Unknown',
        tokenSymbol: tokenInfo?.symbol || 'Unknown',
        riskMetrics,
        analysis: `This token has a ${riskMetrics.overallRisk.toLowerCase()} risk of being a rug pull based on our analysis.`,
        recommendations: riskMetrics.overallRisk === 'High' 
          ? 'Exercise extreme caution with this token. High risk of being a scam or rug pull.'
          : riskMetrics.overallRisk === 'Medium'
          ? 'Proceed with caution. Some risk factors detected.'
          : 'Lower risk profile, but always do your own research.'
      };
    } catch (error) {
      console.error('Error analyzing rug pull risk:', error);
      throw new Error(`Failed to analyze rug pull risk: ${error.message}`);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Solana Data MCP server running on stdio');
  }
}

const server = new SolanaDataMCP();
server.run().catch(console.error);
