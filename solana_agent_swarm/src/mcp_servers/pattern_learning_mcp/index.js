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
import { PatternDetector, PATTERN_TYPES } from './pattern_algorithms.js';

// Load environment variables
dotenv.config();

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const PATTERNS_DB_PATH = path.join(__dirname, 'patterns_db.json');
const BLACKLIST_DB_PATH = path.join(__dirname, 'blacklist_db.json');
const HISTORICAL_DATA_PATH = path.join(__dirname, 'historical_data');

class PatternLearningMCP {
  constructor() {
    this.server = new Server(
      {
        name: 'pattern-learning-mcp',
        version: '0.1.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    // Initialize pattern database
    this.patternsDb = {
      knownPatterns: [],
      tokenPatterns: {},
      lastUpdated: Date.now()
    };

    // Initialize blacklist database
    this.blacklistDb = {
      wallets: [],
      tokens: [],
      lastUpdated: Date.now()
    };

    // Initialize advanced pattern detector
    this.patternDetector = new PatternDetector();

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
      // Load patterns database if it exists
      try {
        const patternsData = await fs.readFile(PATTERNS_DB_PATH, 'utf8');
        this.patternsDb = JSON.parse(patternsData);
        console.error(`Loaded ${Object.keys(this.patternsDb.tokenPatterns).length} token patterns`);
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.error('Error loading patterns database:', err);
        } else {
          console.error('Patterns database not found, creating new one');
          // Ensure directory exists
          await fs.mkdir(path.dirname(PATTERNS_DB_PATH), { recursive: true });
          await this.savePatternsDb();
        }
      }

      // Load blacklist database if it exists
      try {
        const blacklistData = await fs.readFile(BLACKLIST_DB_PATH, 'utf8');
        this.blacklistDb = JSON.parse(blacklistData);
        console.error(`Loaded ${this.blacklistDb.wallets.length} blacklisted wallets and ${this.blacklistDb.tokens.length} blacklisted tokens`);
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.error('Error loading blacklist database:', err);
        } else {
          console.error('Blacklist database not found, creating new one');
          // Ensure directory exists
          await fs.mkdir(path.dirname(BLACKLIST_DB_PATH), { recursive: true });
          await this.saveBlacklistDb();
        }
      }

      // Create historical data directory if it doesn't exist
      try {
        await fs.mkdir(HISTORICAL_DATA_PATH, { recursive: true });
        console.error(`Ensured historical data directory exists at ${HISTORICAL_DATA_PATH}`);
      } catch (err) {
        console.error('Error creating historical data directory:', err);
      }
    } catch (error) {
      console.error('Error initializing databases:', error);
    }
  }

  async savePatternsDb() {
    try {
      await fs.writeFile(PATTERNS_DB_PATH, JSON.stringify(this.patternsDb, null, 2));
    } catch (error) {
      console.error('Error saving patterns database:', error);
    }
  }

  async saveBlacklistDb() {
    try {
      await fs.writeFile(BLACKLIST_DB_PATH, JSON.stringify(this.blacklistDb, null, 2));
    } catch (error) {
      console.error('Error saving blacklist database:', error);
    }
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'analyze_token_pattern',
          description: 'Analyze historical data to identify patterns for a specific token',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
              time_period: {
                type: 'string',
                description: 'Time period for analysis (24h, 7d, 30d)',
                enum: ['24h', '7d', '30d'],
              },
            },
            required: ['token_address'],
          },
        },
        {
          name: 'get_price_prediction',
          description: 'Get price prediction for a token based on identified patterns',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address (mint)',
              },
              time_horizon: {
                type: 'string',
                description: 'Time horizon for prediction (1h, 24h, 7d)',
                enum: ['1h', '24h', '7d'],
              },
            },
            required: ['token_address'],
          },
        },
        {
          name: 'detect_market_patterns',
          description: 'Detect general market patterns across multiple tokens',
          inputSchema: {
            type: 'object',
            properties: {
              pattern_type: {
                type: 'string',
                description: 'Type of pattern to look for',
                enum: ['pump_dump', 'accumulation', 'distribution', 'breakout', 'all'],
              },
              min_confidence: {
                type: 'number',
                description: 'Minimum confidence score (0-1)',
              },
            },
            required: [],
          },
        },
        {
          name: 'add_to_blacklist',
          description: 'Add a wallet or token to the blacklist',
          inputSchema: {
            type: 'object',
            properties: {
              address: {
                type: 'string',
                description: 'Wallet or token address to blacklist',
              },
              type: {
                type: 'string',
                description: 'Type of address (wallet or token)',
                enum: ['wallet', 'token'],
              },
              reason: {
                type: 'string',
                description: 'Reason for blacklisting',
              },
            },
            required: ['address', 'type'],
          },
        },
        {
          name: 'check_blacklist',
          description: 'Check if a wallet or token is blacklisted',
          inputSchema: {
            type: 'object',
            properties: {
              address: {
                type: 'string',
                description: 'Wallet or token address to check',
              },
              type: {
                type: 'string',
                description: 'Type of address (wallet, token, or both)',
                enum: ['wallet', 'token', 'both'],
              },
            },
            required: ['address'],
          },
        },
        {
          name: 'provide_pattern_feedback',
          description: 'Provide feedback on a pattern detection (used for learning)',
          inputSchema: {
            type: 'object',
            properties: {
              pattern_id: {
                type: 'string',
                description: 'ID of the pattern',
              },
              token_address: {
                type: 'string',
                description: 'Token address related to the pattern',
              },
              accuracy: {
                type: 'number',
                description: 'Accuracy of the pattern detection (0-1)',
              },
              comments: {
                type: 'string',
                description: 'Additional comments or observations',
              },
            },
            required: ['pattern_id', 'token_address', 'accuracy'],
          },
        },
        {
          name: 'backtest_pattern',
          description: 'Backtest a pattern against historical token data',
          inputSchema: {
            type: 'object',
            properties: {
              token_address: {
                type: 'string',
                description: 'Solana token address to backtest against',
              },
              pattern_type: {
                type: 'string',
                description: 'Pattern type to backtest',
                enum: Object.values(PATTERN_TYPES),
              },
              confidence_threshold: {
                type: 'number',
                description: 'Minimum confidence level to trigger a trade (0-1)',
              },
              profit_target: {
                type: 'number',
                description: 'Target profit percentage (0-1)',
              },
              stop_loss: {
                type: 'number',
                description: 'Stop loss percentage (0-1)',
              },
            },
            required: ['token_address', 'pattern_type'],
          },
        }
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        let result;

        switch (name) {
          case 'analyze_token_pattern':
            result = await this.analyzeTokenPattern(args.token_address, args.time_period || '7d');
            break;
          case 'get_price_prediction':
            result = await this.getPricePrediction(args.token_address, args.time_horizon || '24h');
            break;
          case 'detect_market_patterns':
            result = await this.detectMarketPatterns(args.pattern_type || 'all', args.min_confidence || 0.7);
            break;
          case 'add_to_blacklist':
            result = await this.addToBlacklist(args.address, args.type, args.reason || 'Not specified');
            break;
          case 'check_blacklist':
            result = await this.checkBlacklist(args.address, args.type || 'both');
            break;
          case 'provide_pattern_feedback':
            result = await this.providePatternFeedback(args.pattern_id, args.token_address, args.accuracy, args.comments);
            break;
          case 'backtest_pattern':
            result = await this.backtestPattern(
              args.token_address, 
              args.pattern_type, 
              args.confidence_threshold || 0.7,
              args.profit_target || 0.1,
              args.stop_loss || 0.05
            );
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

  async fetchHistoricalData(tokenAddress, timePeriod) {
    // In a production system, this would fetch real data from APIs
    // For demo purposes, we generate simulated historical data
    
    const dataPoints = {
      '24h': 24,
      '7d': 168,
      '30d': 720
    }[timePeriod] || 168; // Default to 7d (168 hours)
    
    const timestamps = [];
    const prices = [];
    const volumes = [];
    
    const now = Date.now();
    const basePrice = 100 + (Math.random() * 900); // $100-$1000
    const hourMs = 60 * 60 * 1000;
    
    // Generate timestamps and price/volume data with some realistic patterns
    for (let i = 0; i < dataPoints; i++) {
      const timestamp = new Date(now - (dataPoints - i) * hourMs).toISOString();
      timestamps.push(timestamp);
      
      // Create some volatility and trends in the price
      const noiseLevel = basePrice * 0.03; // 3% noise
      const trendComponent = Math.sin(i / 24) * basePrice * 0.1; // Cyclical trend component
      const randomWalk = (Math.random() - 0.5) * noiseLevel; // Random walk component
      
      // Previous price or base price for first point
      const prevPrice = i > 0 ? prices[i-1] : basePrice;
      const price = prevPrice + randomWalk + (trendComponent - (prevPrice - basePrice) * 0.01);
      prices.push(Math.max(price, basePrice * 0.5)); // Ensure price doesn't go too low
      
      // Volume that correlates somewhat with price movements
      const baseVolume = 1000000 + (Math.random() * 9000000); // 1M-10M
      const priceChange = i > 0 ? Math.abs(price - prices[i-1]) / prices[i-1] : 0;
      const volumeBoost = priceChange * baseVolume * 10; // More volume on bigger price movements
      volumes.push(baseVolume + volumeBoost);
    }
    
    // Store data in file for reuse (particularly for backtesting)
    try {
      const dataObj = { tokenAddress, timePeriod, timestamps, prices, volumes, generatedAt: now };
      const historicalDataFile = path.join(HISTORICAL_DATA_PATH, `${tokenAddress}_${timePeriod}.json`);
      await fs.writeFile(historicalDataFile, JSON.stringify(dataObj, null, 2));
    } catch (error) {
      console.error(`Failed to save historical data: ${error.message}`);
      // Non-fatal, continue with in-memory data
    }
    
    return { timestamps, prices, volumes };
  }

  async analyzeTokenPattern(tokenAddress, timePeriod = '7d') {
    try {
      console.error(`Analyzing patterns for token ${tokenAddress} over ${timePeriod}`);
      
      // Fetch historical data (in production, this would be from an API)
      const historicalData = await this.fetchHistoricalData(tokenAddress, timePeriod);
      
      // Use our advanced pattern detector to find patterns
      const analysisResult = this.patternDetector.detectPatterns(historicalData, timePeriod);
      
      // Store detected patterns in our database
      if (analysisResult.patterns && analysisResult.patterns.length > 0) {
        // If we don't have patterns for this token yet, initialize the array
        if (!this.patternsDb.tokenPatterns[tokenAddress]) {
          this.patternsDb.tokenPatterns[tokenAddress] = [];
        }
        
        // Add new patterns, avoiding duplicates
        const existingIds = this.patternsDb.tokenPatterns[tokenAddress].map(p => p.id);
        const newPatterns = analysisResult.patterns.filter(p => !existingIds.includes(p.id));
        
        this.patternsDb.tokenPatterns[tokenAddress].push(...newPatterns);
        
        // Keep only the 10 most recent patterns
        if (this.patternsDb.tokenPatterns[tokenAddress].length > 10) {
          this.patternsDb.tokenPatterns[tokenAddress].sort((a, b) => 
            new Date(b.detected_at || 0).getTime() - new Date(a.detected_at || 0).getTime()
          );
          this.patternsDb.tokenPatterns[tokenAddress] = this.patternsDb.tokenPatterns[tokenAddress].slice(0, 10);
        }
        
        // Update our database
        await this.savePatternsDb();
      }
      
      // Sort patterns by confidence (highest first)
      const sortedPatterns = [...(analysisResult.patterns || [])].sort((a, b) => b.confidence - a.confidence);
      
      // Return the analysis result
      return {
        token_address: tokenAddress,
        time_period: timePeriod,
        patterns: sortedPatterns,
        analysis_summary: analysisResult.message || `Found ${sortedPatterns.length} patterns for this token in the ${timePeriod} timeframe.`,
        highest_confidence_pattern: sortedPatterns.length > 0 ? sortedPatterns[0] : null,
        analysis_performed_at: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error analyzing token pattern:', error);
      throw new Error(`Failed to analyze token pattern: ${error.message}`);
    }
  }

  async getPricePrediction(tokenAddress, timeHorizon = '24h') {
    try {
      console.error(`Generating price prediction for token ${tokenAddress} with horizon ${timeHorizon}`);
      
      // Use the longest time period to analyze patterns
      const timePeriod = '30d';
      
      // First, analyze patterns for this token
      const analysis = await this.analyzeTokenPattern(tokenAddress, timePeriod);
      
      // Fetch the historical data
      const historicalData = await this.fetchHistoricalData(tokenAddress, timePeriod);
      
      // Get the most recent price as our starting point
      const currentPrice = historicalData.prices[historicalData.prices.length - 1] || 100;
      let confidenceScore = 0.7; // Base confidence
      let predictedPriceChange = 0; // Default to no change
      
      // Calculate prediction based on detected patterns
      if (analysis.patterns && analysis.patterns.length > 0) {
        // Find the highest confidence pattern
        const bestPattern = analysis.patterns[0]; // Already sorted by confidence
        confidenceScore = bestPattern.confidence;
        
        // Different predictions based on pattern type
        switch(bestPattern.type) {
          case PATTERN_TYPES.ACCUMULATION:
            // Accumulation typically leads to upward price movement
            predictedPriceChange = 0.1 + (Math.random() * 0.1); // 10-20% increase
            break;
          case PATTERN_TYPES.DISTRIBUTION:
            // Distribution typically leads to downward price movement
            predictedPriceChange = -0.1 - (Math.random() * 0.1); // 10-20% decrease
            break;
          case PATTERN_TYPES.BREAKOUT:
            // Breakouts typically lead to strong upward movement
            predictedPriceChange = 0.15 + (Math.random() * 0.15); // 15-30% increase
            break;
          case PATTERN_TYPES.BREAKDOWN:
            // Breakdowns typically lead to strong downward movement
            predictedPriceChange = -0.15 - (Math.random() * 0.15); // 15-30% decrease
            break;
          case PATTERN_TYPES.PUMP_DUMP:
            // Pump and dump typically means a crash is coming
            predictedPriceChange = -0.2 - (Math.random() * 0.2); // 20-40% decrease
            break;
          case PATTERN_TYPES.CONSOLIDATION:
            // Consolidation typically means minimal price movement
            predictedPriceChange = (Math.random() * 0.1) - 0.05; // -5% to +5%
            break;
          default:
            // Default case - small random movement
            predictedPriceChange = (Math.random() * 0.2) - 0.1; // -10% to +10%
        }
        
        // Adjust confidence based on horizon - longer predictions are less certain
        if (timeHorizon === '7d') {
          confidenceScore *= 0.9; // 10% reduction
        }
      } else {
        // Without patterns, we make a less confident prediction
        predictedPriceChange = (Math.random() * 0.2) - 0.1; // -10% to +10%
        confidenceScore *= 0.8; // 20% reduction in confidence
      }
      
      const predictedPrice = currentPrice * (1 + predictedPriceChange);
      const direction = predictedPriceChange > 0 ? 'bullish' : (predictedPriceChange < 0 ? 'bearish' : 'neutral');
      
      return {
        token_address: tokenAddress,
        time_horizon: timeHorizon,
        current_price: `$${currentPrice.toFixed(2)}`,
        predicted_price: `$${predictedPrice.toFixed(2)}`,
        price_change: `${(predictedPriceChange * 100).toFixed(2)}%`,
        confidence: confidenceScore,
        direction,
        supporting_patterns: analysis.patterns || [],
        prediction_made_at: new Date().toISOString(),
        disclaimer: 'This prediction is for educational purposes only. Do not make investment decisions based solely on this information.'
      };
    } catch (error) {
      console.error('Error generating price prediction:', error);
      throw new Error(`Failed to generate price prediction: ${error.message}`);
    }
  }

  async detectMarketPatterns(patternType = 'all', minConfidence = 0.7) {
    try {
      console.error(`Detecting market patterns of type ${patternType} with min confidence ${minConfidence}`);
      
      // In a real implementation, this would scan across multiple tokens
      // For demo, we'll create a list of simulated tokens and analyze each
      const demoTokens = [
        { address: 'SIMULATED_TOKEN_1', name: 'Demo Token 1' },
        { address: 'SIMULATED_TOKEN_2', name: 'Demo Token 2' },
        { address: 'SIMULATED_TOKEN_3', name: 'Demo Token 3' },
        { address: 'SIMULATED_TOKEN_4', name: 'Demo Token 4' },
        { address: 'SIMULATED_TOKEN_5', name: 'Demo Token 5' }
      ];
      
      // Analyze each token for patterns
      const tokenPatterns = [];
      for (const token of demoTokens) {
        const data = await this.fetchHistoricalData(token.address, '7d');
        const analysis = this.patternDetector.detectPatterns(data, '7d');
        
        // Filter by pattern type and confidence
        let filteredPatterns = analysis.patterns || [];
        if (patternType !== 'all') {
          filteredPatterns = filteredPatterns.filter(p => p.type.toLowerCase() === patternType.toLowerCase());
        }
        filteredPatterns = filteredPatterns.filter(p => p.confidence >= minConfidence);
        
        if (filteredPatterns.length > 0) {
          tokenPatterns.push({
            token: token.address,
            tokenName: token.name,
            patterns: filteredPatterns
          });
        }
      }
      
      // Check for market-wide patterns
      // These would be similar patterns occurring across multiple tokens
      const marketPatterns = [];
      
      // Pattern counts by type
      const patternCounts = {};
      tokenPatterns.forEach(tp => {
        tp.patterns.forEach(p => {
          if (!patternCounts[p.type]) patternCounts[p.type] = 0;
          patternCounts[p.type]++;
        });
      });
      
      // If more than 2 tokens show the same pattern, it might be a market pattern
      for (const [type, count] of Object.entries(patternCounts)) {
        if (count >= 2) {
          // Calculate average confidence
          let totalConfidence = 0;
          let patternCount = 0;
          const affectedTokens = [];
          
          tokenPatterns.forEach(tp => {
            tp.patterns.forEach(p => {
              if (p.type === type) {
                totalConfidence += p.confidence;
                patternCount++;
                affectedTokens.push({
                  token: tp.token,
                  tokenName: tp.tokenName,
                  confidence: p.confidence
                });
              }
            });
          });
          
          const avgConfidence = totalConfidence / patternCount;
          
          // Map pattern type to market pattern type
          const marketPatternType = this.mapToMarketPatternType(type);
          
          marketPatterns.push({
            id: `market_pattern_${Date.now()}_${type}`,
            type: marketPatternType,
            confidence: avgConfidence,
            detected_at: new Date().toISOString(),
            description: this.generateMarketPatternDescription(marketPatternType),
            affected_tokens: affectedTokens,
            suggested_action: this.generateActionForPattern(marketPatternType)
          });
        }
      }
      
      return {
        pattern_type: patternType,
        min_confidence: minConfidence,
        detected_patterns: marketPatterns,
        tokens_analyzed: demoTokens.length,
        tokens_with_patterns: tokenPatterns.length,
        token_details: tokenPatterns,
        analysis_time: new Date().toISOString(),
        summary: `Detected ${marketPatterns.length} market patterns with confidence >= ${minConfidence}.`
      };
    } catch (error) {
      console.error('Error detecting market patterns:', error);
      throw new Error(`Failed to detect market patterns: ${error.message}`);
    }
  }

  mapToMarketPatternType(patternType) {
    // Map individual token patterns to market-wide patterns
    const mapping = {
      [PATTERN_TYPES.ACCUMULATION]: 'sector_rotation',
      [PATTERN_TYPES.DISTRIBUTION]: 'market_exhaustion',
      [PATTERN_TYPES.BREAKOUT]: 'liquidity_gap',
      [PATTERN_TYPES.CONSOLIDATION]: 'volume_divergence',
      [PATTERN_TYPES.PUMP_DUMP]: 'pump_dump_wave'
    };
    
    return mapping[patternType] || 'general_market_trend';
  }

  generateMarketPatternDescription(patternType) {
    const descriptions = {
      sector_rotation: 'Capital appears to be moving from one token sector to another, indicating changing market sentiment or catalyst events.',
      market_exhaustion: 'Multiple tokens showing signs of buyer/seller exhaustion, suggesting a potential market reversal.',
      liquidity_gap: 'Significant liquidity imbalances detected across multiple tokens, creating potential price gaps.',
      volume_divergence: 'Volume profiles across multiple tokens show divergence from price action, suggesting potential market inefficiencies.',
      pump_dump_wave: 'Multiple tokens showing coordinated pump and dump activities, indicating potential market manipulation.'
    };
    
    return descriptions[patternType] || 'Unspecified market-wide pattern detected across multiple tokens.';
  }

  generateActionForPattern(patternType) {
    const actions = {
      sector_rotation: 'Consider repositioning to capitalize on the emerging sector strength.',
      market_exhaustion: 'Exercise caution and consider reducing exposure until direction clarifies.',
      liquidity_gap: 'Watch for potential rapid price moves as markets seek to fill liquidity gaps.',
      volume_divergence: 'Monitor for potential reversals as price and volume return to alignment.',
      pump_dump_wave: 'Exercise extreme caution and avoid chasing pumps that lack fundamental backing.'
    };
    
    return actions[patternType] || 'Monitor the situation closely for further developments.';
  }

  async addToBlacklist(address, type, reason) {
    try {
      if (!address) {
        throw new Error('Address is required');
      }
      
      if (!['wallet', 'token'].includes(type)) {
        throw new Error('Type must be either "wallet" or "token"');
      }
      
      // Add to appropriate blacklist
      const entry = {
        address,
        reason,
        added_at: new Date().toISOString()
      };
      
      if (type === 'wallet') {
        // Check if already blacklisted
        if (!this.blacklistDb.wallets.some(w => w.address === address)) {
          this.blacklistDb.wallets.push(entry);
        }
      } else {
        // Check if already blacklisted
        if (!this.blacklistDb.tokens.some(t => t.address === address)) {
          this.blacklistDb.tokens.push(entry);
        }
      }
      
      // Update timestamp
      this.blacklistDb.lastUpdated = Date.now();
      
      // Save to disk
      await this.saveBlacklistDb();
      
      return {
        success: true,
        message: `Added ${address} to the ${type} blacklist`,
        entry
      };
    } catch (error) {
      console.error('Error adding to blacklist:', error);
      throw new Error(`Failed to add to blacklist: ${error.message}`);
    }
  }

  async checkBlacklist(address, type = 'both') {
    try {
      if (!address) {
        throw new Error('Address is required');
      }
      
      let isBlacklisted = false;
      let blacklistEntry = null;
      
      // Check wallet blacklist if requested
      if (type === 'wallet' || type === 'both') {
        const walletEntry = this.blacklistDb.wallets.find(w => w.address === address);
        if (walletEntry) {
          isBlacklisted = true;
          blacklistEntry = { type: 'wallet', ...walletEntry };
        }
      }
      
      // Check token blacklist if requested and not already found
      if ((type === 'token' || type === 'both') && !isBlacklisted) {
        const tokenEntry = this.blacklistDb.tokens.find(t => t.address === address);
        if (tokenEntry) {
          isBlacklisted = true;
          blacklistEntry = { type: 'token', ...tokenEntry };
        }
      }
      
      return {
        address,
        checked_type: type,
        is_blacklisted: isBlacklisted,
        details: blacklistEntry,
        checked_at: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error checking blacklist:', error);
      throw new Error(`Failed to check blacklist: ${error.message}`);
    }
  }

  async providePatternFeedback(patternId, tokenAddress, accuracy, comments = '') {
    try {
