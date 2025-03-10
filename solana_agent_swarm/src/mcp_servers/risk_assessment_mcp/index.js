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
import dotenv from 'dotenv';
import axios from 'axios';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { PublicKey } from '@solana/web3.js';

// Load environment variables
dotenv.config();

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const RISK_DB_PATH = path.join(__dirname, 'risk_db.json');
const RUG_PULL_SIGNALS_PATH = path.join(__dirname, 'rug_pull_signals.json');

class RiskAssessmentMCP {
  constructor() {
    this.server = new Server(
      {
        name: 'risk-assessment-mcp',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize risk database
    this.riskDb = {
      assessedTokens: {}, // token_address -> risk_assessment
      knownRugPulls: [], // array of confirmed rug pulls
      lastUpdated: Date.now()
    };

    // Initialize rug pull signals
    this.rugPullSignals = {
      concentration: [
        { factor: 'top_holder_percentage', threshold: 50, weight: 0.25, description: 'Percentage held by largest holder' },
        { factor: 'team_wallet_percentage', threshold: 40, weight: 0.20, description: 'Percentage held by team wallets' },
        { factor: 'holder_count', threshold: 100, weight: 0.10, description: 'Number of token holders', inverse: true }
      ],
      liquidity: [
        { factor: 'liquidity_to_mcap', threshold: 0.05, weight: 0.20, description: 'Ratio of liquidity to market cap', inverse: true },
        { factor: 'unlocked_supply', threshold: 70, weight: 0.15, description: 'Percentage of total supply that is unlocked' }
      ],
      activity: [
        { factor: 'sudden_price_increase', threshold: 100, weight: 0.15, description: 'Percentage increase in price in 24h' },
        { factor: 'unusual_selling', threshold: 0.8, weight: 0.15, description: 'Ratio of sells to total transactions' },
        { factor: 'telegram_activity', threshold: 0.3, weight: 0.10, description: 'Activity level in official channels', inverse: true },
        { factor: 'age_days', threshold: 7, weight: 0.15, description: 'Age of token in days', inverse: true }
      ]
    };

    // Load existing data
    this.loadDatabases();

    // Set up handlers
    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  async loadDatabases() {
    try {
      // Load risk database if it exists
      try {
        const riskData = await fs.readFile(RISK_DB_PATH, 'utf8');
        this.riskDb = JSON.parse(riskData);
        console.error(`Loaded ${Object.keys(this.riskDb.assessedTokens).length} token risk assessments`);
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.error('Error loading risk database:', err);
        } else {
          console.error('Risk database not found, creating new one');
          // Ensure directory exists
          await fs.mkdir(path.dirname(RISK_DB_PATH), { recursive: true });
          await this.saveRiskDb();
        }
      }

      // Load rug pull signals if it exists, otherwise save default
      try {
        const signalsData = await fs.readFile(RUG_PULL_SIGNALS_PATH, 'utf8');
        this.rugPullSignals = JSON.parse(signalsData);
        console.error('Loaded rug pull signal configurations');
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.error('Error loading rug pull signals:', err);
        } else {
          console.error('Rug pull signals file not found, creating default one');
          // Ensure directory exists
          await fs.mkdir(path.dirname(RUG_PULL_SIGNALS_PATH), { recursive: true });
          await this.saveRugPullSignals();
        }
      }
    } catch (error) {
      console.error('Error initializing databases:', error);
    }
  }

  async saveRiskDb() {
    try {
      await fs.writeFile(RISK_DB_PATH, JSON.stringify(this.riskDb, null, 2));
    } catch (error) {
      console.error('Error saving risk database:', error);
    }
  }

  async saveRugPullSignals() {
    try {
      await fs.writeFile(RUG_PULL_SIGNALS_PATH, JSON.stringify(this.rugPullSignals, null, 2));
    } catch (error) {
      console.error('Error saving rug pull signals:', error);
    }
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'assess_token_risk',
          description: 'Perform a comprehensive risk assessment of a Solana token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
              force_refresh: {
                type: 'boolean',
                description: 'Force a fresh assessment rather than using cached data',
              },
            },
            required: ['token_address'],
          },
        },
        {
          name: 'get_rug_pull_probability',
          description: 'Calculate the probability of a token being a rug pull',
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
          name: 'identify_high_risk_wallets',
          description: 'Identify high-risk wallets associated with a token',
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
          name: 'report_rug_pull',
          description: 'Report a confirmed rug pull for a token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
              evidence: {
                type: 'string',
                description: 'Description of evidence for the rug pull',
              },
              associated_wallets: {
                type: 'array',
                items: {
                  type: 'string'
                },
                description: 'Array of wallet addresses associated with the rug pull',
              },
            },
            required: ['token_address', 'evidence'],
          },
        },
        {
          name: 'get_safety_recommendations',
          description: 'Get safety recommendations for interacting with a token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
              interaction_type: {
                type: 'string',
                description: 'Type of interaction planned with the token',
                enum: ['buy', 'sell', 'stake', 'provide_liquidity', 'any'],
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
          case 'assess_token_risk':
            result = await this.assessTokenRisk(args.token_address, args.force_refresh || false);
            break;
          case 'get_rug_pull_probability':
            result = await this.getRugPullProbability(args.token_address);
            break;
          case 'identify_high_risk_wallets':
            result = await this.identifyHighRiskWallets(args.token_address);
            break;
          case 'report_rug_pull':
            result = await this.reportRugPull(args.token_address, args.evidence, args.associated_wallets || []);
            break;
          case 'get_safety_recommendations':
            result = await this.getSafetyRecommendations(args.token_address, args.interaction_type || 'any');
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

  async assessTokenRisk(tokenAddress, forceRefresh = false) {
    try {
      // Check if we have cached assessment and not forcing refresh
      if (!forceRefresh && this.riskDb.assessedTokens[tokenAddress]) {
        const assessment = this.riskDb.assessedTokens[tokenAddress];
        const cacheAge = Date.now() - assessment.assessed_at;
        
        // Use cache if less than 1 hour old
        if (cacheAge < 3600000) {
          return {
            ...assessment,
            from_cache: true
          };
        }
      }

      // In a real implementation, this would call various APIs to gather data
      // and perform a comprehensive risk assessment
      
      // For this demo, we'll simulate a risk assessment
      const riskFactors = {
        concentration: {
          top_holder_percentage: Math.random() * 80, // 0-80%
          team_wallet_percentage: Math.random() * 60, // 0-60%
          holder_count: Math.floor(Math.random() * 1000) + 10 // 10-1010
        },
        liquidity: {
          liquidity_to_mcap: Math.random() * 0.3, // 0-30%
          unlocked_supply: Math.random() * 100, // 0-100%
        },
        activity: {
          sudden_price_increase: Math.random() * 200, // 0-200%
          unusual_selling: Math.random(), // 0-1
          telegram_activity: Math.random(), // 0-1
          age_days: Math.floor(Math.random() * 365) + 1 // 1-366 days
        }
      };
      
      // Calculate risk scores
      const riskScores = {
        concentration: this.calculateRiskScore(riskFactors.concentration, this.rugPullSignals.concentration),
        liquidity: this.calculateRiskScore(riskFactors.liquidity, this.rugPullSignals.liquidity),
        activity: this.calculateRiskScore(riskFactors.activity, this.rugPullSignals.activity)
      };
      
      // Overall risk score (weighted average of category scores)
      const overallRiskScore = (
        riskScores.concentration * 0.4 + 
        riskScores.liquidity * 0.35 + 
        riskScores.activity * 0.25
      );
      
      // Risk level based on score
      let riskLevel;
      if (overallRiskScore >= 0.7) {
        riskLevel = 'High';
      } else if (overallRiskScore >= 0.4) {
        riskLevel = 'Medium';
      } else {
        riskLevel = 'Low';
      }
      
      // Generate specific concerns based on risk factors
      const concerns = [];
      
      if (riskFactors.concentration.top_holder_percentage > 40) {
        concerns.push(`Top holder controls ${riskFactors.concentration.top_holder_percentage.toFixed(1)}% of supply`);
      }
      
      if (riskFactors.liquidity.liquidity_to_mcap < 0.05) {
        concerns.push('Low liquidity relative to market cap');
      }
      
      if (riskFactors.activity.sudden_price_increase > 100) {
        concerns.push(`Unusual price increase of ${riskFactors.activity.sudden_price_increase.toFixed(1)}% recently`);
      }
      
      if (riskFactors.activity.age_days < 7) {
        concerns.push(`Very new token (${riskFactors.activity.age_days} days old)`);
      }
      
      // Create assessment result
      const assessment = {
        token_address: tokenAddress,
        risk_level: riskLevel,
        overall_risk_score: overallRiskScore,
        risk_scores: riskScores,
        risk_factors: riskFactors,
        concerns,
        rug_pull_probability: this.calculateRugPullProbability(overallRiskScore),
        assessed_at: Date.now(),
        from_cache: false
      };
      
      // Save to database
      this.riskDb.assessedTokens[tokenAddress] = assessment;
      await this.saveRiskDb();
      
      return assessment;
    } catch (error) {
      console.error('Error assessing token risk:', error);
      throw new Error(`Failed to assess token risk: ${error.message}`);
    }
  }

  calculateRiskScore(factors, signals) {
    let totalScore = 0;
    let totalWeight = 0;
    
    for (const signal of signals) {
      const { factor, threshold, weight, inverse } = signal;
      
      if (factors[factor] !== undefined) {
        let factorScore;
        
        if (inverse) {
          // For inverse factors (higher is better), score is high when value is low
          factorScore = factors[factor] < threshold ? 1 : threshold / factors[factor];
        } else {
          // For regular factors (lower is better), score is high when value is high
          factorScore = factors[factor] > threshold ? 1 : factors[factor] / threshold;
        }
        
        totalScore += factorScore * weight;
        totalWeight += weight;
      }
    }
    
    return totalWeight > 0 ? totalScore / totalWeight : 0;
  }

  calculateRugPullProbability(riskScore) {
    // Simple conversion of risk score to probability
    // In a real system this would be more sophisticated
    return Math.min(0.95, riskScore * 1.2);
  }

  async getRugPullProbability(tokenAddress) {
    try {
      // Get the full risk assessment first
      const assessment = await this.assessTokenRisk(tokenAddress);
      
      // Extract the rug pull probability
      return {
        token_address: tokenAddress,
        probability: assessment.rug_pull_probability,
        confidence: 0.8, // Confidence in our prediction
        key_factors: assessment.concerns,
        assessment_time: new Date(assessment.assessed_at).toISOString()
      };
    } catch (error) {
      console.error('Error getting rug pull probability:', error);
      throw new Error(`Failed to get rug pull probability: ${error.message}`);
    }
  }

  async identifyHighRiskWallets(tokenAddress) {
    try {
      // In a real implementation, this would analyze the token's holders
      // and identify wallets that have suspicious behavior or are known scammers
      
      // For this demo, we'll simulate high risk wallet identification
      const numWallets = Math.floor(Math.random() * 5) + 1; // 1-5 wallets
      const highRiskWallets = [];
      
      for (let i = 0; i < numWallets; i++) {
        // Generate a random Solana-like address
        const walletChars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz';
        let wallet = '';
        for (let j = 0; j < 44; j++) {
          wallet += walletChars.charAt(Math.floor(Math.random() * walletChars.length));
        }
        
        // Random risk score between 0.7 and 1.0
        const riskScore = 0.7 + (Math.random() * 0.3);
        
        // Random risk factors
        const riskFactors = [];
        const possibleFactors = [
          'Previously involved in rug pulls',
          'Suspicious transaction patterns',
          'Connected to known scam wallets',
          'High frequency of token dumps',
          'Associated with multiple failed projects'
        ];
        
        // Select 1-3 random factors
        const numFactors = Math.floor(Math.random() * 3) + 1;
        for (let j = 0; j < numFactors; j++) {
          const factor = possibleFactors[Math.floor(Math.random() * possibleFactors.length)];
          if (!riskFactors.includes(factor)) {
            riskFactors.push(factor);
          }
        }
        
        highRiskWallets.push({
          wallet_address: wallet,
          risk_score: riskScore,
          risk_factors: riskFactors,
          holdings_percentage: (Math.random() * 40).toFixed(2) + '%', // 0-40%
        });
      }
      
      // Sort by risk score (highest first)
      highRiskWallets.sort((a, b) => b.risk_score - a.risk_score);
      
      return {
        token_address: tokenAddress,
        high_risk_wallets: highRiskWallets,
        total_wallets_analyzed: Math.floor(Math.random() * 1000) + 100, // 100-1100
        analysis_time: new Date().toISOString(),
        recommendation: highRiskWallets.length > 0 
          ? 'Monitor these wallets closely for suspicious activity' 
          : 'No high-risk wallets detected'
      };
    } catch (error) {
      console.error('Error identifying high risk wallets:', error);
      throw new Error(`Failed to identify high risk wallets: ${error.message}`);
    }
  }

  async reportRugPull(tokenAddress, evidence, associatedWallets = []) {
    try {
      // Check if token is already in known rug pulls
      const existing = this.riskDb.knownRugPulls.find(rp => rp.token_address === tokenAddress);
      
      if (existing) {
        // Update existing entry
        existing.evidence.push({
          description: evidence,
          reported_at: new Date().toISOString()
        });
        
        // Add any new associated wallets
        for (const wallet of associatedWallets) {
          if (!existing.associated_wallets.includes(wallet)) {
            existing.associated_wallets.push(wallet);
          }
        }
        
        existing.updated_at = new Date().toISOString();
      } else {
        // Create new entry
        this.riskDb.knownRugPulls.push({
          token_address: tokenAddress,
          evidence: [{
            description: evidence,
            reported_at: new Date().toISOString()
          }],
          associated_wallets: associatedWallets,
          reported_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      }
      
      // Save changes
      await this.saveRiskDb();
      
      return {
        success: true,
        message: existing ? 'Updated existing rug pull report' : 'Recorded new rug pull report',
        token_address: tokenAddress,
        associated_wallets: existing ? existing.associated_wallets : associatedWallets,
        report_time: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error reporting rug pull:', error);
      throw new Error(`Failed to report rug pull: ${error.message}`);
    }
  }

  async getSafetyRecommendations(tokenAddress, interactionType = 'any') {
    try {
      // Get the risk assessment first
      const assessment = await this.assessTokenRisk(tokenAddress);
      
      // General safety recommendations
      const generalRecommendations = [
        'Always verify the token contract on a block explorer',
        'Check token liquidity before investing',
        'Research the team and project thoroughly',
        'Use a hardware wallet for enhanced security',
        'Only invest what you can afford to lose'
      ];
      
      // Specific recommendations based on risk level and interaction type
      const specificRecommendations = [];
      
      if (assessment.risk_level === 'High') {
        specificRecommendations.push('Exercise extreme caution with this high-risk token');
        
        if (interactionType === 'buy' || interactionType === 'any') {
          specificRecommendations.push('Consider avoiding this token due to high risk indicators');
          specificRecommendations.push('If proceeding, use minimal amounts for testing only');
        }
        
        if (interactionType === 'sell' || interactionType === 'any') {
          specificRecommendations.push('Consider selling in smaller batches to minimize slippage');
          specificRecommendations.push('Be aware of potential selling restrictions in the contract');
        }
        
        if (interactionType === 'provide_liquidity' || interactionType === 'any') {
          specificRecommendations.push('Providing liquidity to high-risk tokens can lead to impermanent loss');
          specificRecommendations.push('Avoid providing significant liquidity to this token');
        }
      } else if (assessment.risk_level === 'Medium') {
        specificRecommendations.push('Proceed with caution with this medium-risk token');
        
        if (interactionType === 'buy' || interactionType === 'any') {
          specificRecommendations.push('Start with small positions and monitor closely');
          specificRecommendations.push('Set strict stop-loss orders to manage risk');
        }
        
        if (interactionType === 'provide_liquidity' || interactionType === 'any') {
          specificRecommendations.push('Consider the risk-reward ratio before providing liquidity');
          specificRecommendations.push('Monitor liquidity position regularly');
        }
      } else {
        specificRecommendations.push('This token shows lower risk indicators, but always remain vigilant');
        
        if (interactionType === 'buy' || interactionType === 'any') {
          specificRecommendations.push('Follow general investment best practices');
          specificRecommendations.push('Diversify your portfolio appropriately');
        }
      }
      
      // Add recommendations specific to concerns
      for (const concern of assessment.concerns) {
        if (concern.includes('holder controls')) {
          specificRecommendations.push('High concentration risk: Monitor large holder activity closely');
        }
        if (concern.includes('Low liquidity')) {
          specificRecommendations.push('Be aware that low liquidity can lead to high slippage and price manipulation');
        }
        if (concern.includes('Unusual price increase')) {
          specificRecommendations.push('Recent price surge may be unsustainable - consider waiting for stabilization');
        }
        if (concern.includes('Very new token')) {
          specificRecommendations.push('New tokens have limited history for analysis - proceed with extra caution');
        }
      }
      
      return {
        token_address: tokenAddress,
        risk_level: assessment.risk_level,
        interaction_type: interactionType,
        general_recommendations: generalRecommendations,
        specific_recommendations: specificRecommendations,
        concerns: assessment.concerns
      };
    } catch (error) {
      console.error('Error getting safety recommendations:', error);
      throw new Error(`Failed to get safety recommendations: ${error.message}`);
    }
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Risk Assessment MCP server running on stdio');
  }
}

const server = new RiskAssessmentMCP();
server.run().catch(console.error);
