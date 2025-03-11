/**
 * Advanced Pattern Detection Algorithms for Solana Token Analysis
 * 
 * This module implements sophisticated pattern recognition algorithms for
 * detecting trading patterns in token price and volume data.
 */

import { mean, std, min, max } from './statistics.js';

/**
 * Pattern types recognized by the system
 */
export const PATTERN_TYPES = {
  ACCUMULATION: 'accumulation',
  DISTRIBUTION: 'distribution',
  BREAKOUT: 'breakout',
  BREAKDOWN: 'breakdown',
  PUMP_DUMP: 'pump_dump',
  BULL_FLAG: 'bull_flag',
  BEAR_FLAG: 'bear_flag',
  DOUBLE_BOTTOM: 'double_bottom',
  DOUBLE_TOP: 'double_top',
  HEAD_SHOULDERS: 'head_shoulders',
  INV_HEAD_SHOULDERS: 'inv_head_shoulders',
  CONSOLIDATION: 'consolidation',
  WYCKOFF_ACCUMULATION: 'wyckoff_accumulation',
  WYCKOFF_DISTRIBUTION: 'wyckoff_distribution'
};

/**
 * Pattern detector class for identifying patterns in price/volume data
 */
export class PatternDetector {
  constructor(config = {}) {
    // Default configuration
    this.config = {
      // General settings
      minDataPoints: 30,
      minConfidence: 0.6,
      maxLookbackDays: 30,
      
      // Pattern-specific settings
      accumulation: {
        minDuration: 7, // days
        maxVolatility: 0.1,
        minBuyVolume: 0.6, // 60% of total volume
      },
      distribution: {
        minDuration: 5, // days
        minSellVolume: 0.6, // 60% of total volume
      },
      breakout: {
        minPriceIncrease: 0.1, // 10%
        minVolumeIncrease: 0.5, // 50%
        consolidationDuration: 7, // days
      },
      pumpDump: {
        minPumpPercentage: 0.2, // 20% 
        maxDumpTimeframe: 48, // hours
        minDumpPercentage: 0.15, // 15%
      },
      // Add other pattern configs as needed
      ...config
    };
    
    // Initialize learning parameters
    this.learningRate = 0.05;
    this.parameters = { ...this.config };
    this.patternHistory = [];
    this.feedbackHistory = [];
  }
  
  /**
   * Analyze price and volume data to detect patterns
   * 
   * @param {Object} data Token historical data
   * @param {Array} data.prices Array of price data points
   * @param {Array} data.volumes Array of volume data points
   * @param {Array} data.timestamps Array of timestamps
   * @param {string} timePeriod Time period for analysis (e.g., '24h', '7d', '30d')
   * @return {Array} Detected patterns with confidence scores
   */
  detectPatterns(data, timePeriod) {
    const { prices, volumes, timestamps } = data;
    
    // Validate input data
    if (!prices || prices.length < this.config.minDataPoints) {
      return { patterns: [], message: 'Insufficient data points for analysis' };
    }
    
    const patterns = [];
    
    // Filter data based on time period
    const filteredData = this.filterDataByTimePeriod(data, timePeriod);
    
    // Run all pattern detection algorithms
    const accumulationPatterns = this.detectAccumulation(filteredData);
    const distributionPatterns = this.detectDistribution(filteredData);
    const breakoutPatterns = this.detectBreakout(filteredData);
    const pumpDumpPatterns = this.detectPumpDump(filteredData);
    const technicalPatterns = this.detectTechnicalPatterns(filteredData);
    
    // Combine all detected patterns
    patterns.push(...accumulationPatterns);
    patterns.push(...distributionPatterns);
    patterns.push(...breakoutPatterns);
    patterns.push(...pumpDumpPatterns);
    patterns.push(...technicalPatterns);
    
    // Sort by confidence (highest first)
    patterns.sort((a, b) => b.confidence - a.confidence);
    
    // Merge overlapping patterns and remove low-confidence ones
    const filteredPatterns = this.filterPatterns(patterns, this.config.minConfidence);
    
    // Update pattern history for learning
    if (filteredPatterns.length > 0) {
      this.patternHistory.push({
        patterns: filteredPatterns,
        timestamp: new Date().toISOString(),
        dataFingerprint: this.generateDataFingerprint(filteredData)
      });
    }
    
    return { 
      patterns: filteredPatterns,
      message: `Detected ${filteredPatterns.length} patterns`
    };
  }
  
  /**
   * Filter data by the specified time period
   */
  filterDataByTimePeriod(data, timePeriod) {
    const { prices, volumes, timestamps } = data;
    const now = Date.now();
    let cutoffTime;
    
    // Determine cutoff time based on time period
    switch (timePeriod) {
      case '24h':
        cutoffTime = now - 24 * 60 * 60 * 1000;
        break;
      case '7d':
        cutoffTime = now - 7 * 24 * 60 * 60 * 1000;
        break;
      case '30d':
        cutoffTime = now - 30 * 24 * 60 * 60 * 1000;
        break;
      default:
        cutoffTime = now - 7 * 24 * 60 * 60 * 1000; // Default to 7d
    }
    
    // Filter data points
    const filteredPrices = [];
    const filteredVolumes = [];
    const filteredTimestamps = [];
    
    for (let i = 0; i < timestamps.length; i++) {
      const timestamp = new Date(timestamps[i]).getTime();
      if (timestamp >= cutoffTime) {
        filteredPrices.push(prices[i]);
        filteredVolumes.push(volumes[i]);
        filteredTimestamps.push(timestamps[i]);
      }
    }
    
    return {
      prices: filteredPrices,
      volumes: filteredVolumes,
      timestamps: filteredTimestamps
    };
  }
  
  /**
   * Detect accumulation patterns
   * Accumulation is characterized by sideways price movement with increasing buy volume
   */
  detectAccumulation(data) {
    const { prices, volumes, timestamps } = data;
    const patterns = [];
    
    // Calculate price volatility (standard deviation / mean)
    const priceVolatility = std(prices) / mean(prices);
    
    // Check if volatility is below threshold
    if (priceVolatility <= this.parameters.accumulation.maxVolatility && 
        timestamps.length >= this.parameters.accumulation.minDuration) {
      
      // Calculate buy/sell volume ratio (simplified for demo)
      const buyVolume = volumes.reduce((sum, vol) => sum + vol, 0);
      const totalVolume = buyVolume; // In a real implementation, we'd separate buy/sell
      const buyRatio = buyVolume / totalVolume;
      
      // Confidence based on pattern characteristics
      let confidence = 0;
      
      if (buyRatio >= this.parameters.accumulation.minBuyVolume) {
        // Calculate confidence based on multiple factors
        const volatilityScore = 1 - (priceVolatility / this.parameters.accumulation.maxVolatility);
        const durationScore = Math.min(1, timestamps.length / (this.parameters.accumulation.minDuration * 2));
        const volumeScore = (buyRatio - this.parameters.accumulation.minBuyVolume) / 
                           (1 - this.parameters.accumulation.minBuyVolume);
        
        confidence = (volatilityScore * 0.4) + (durationScore * 0.3) + (volumeScore * 0.3);
        confidence = Math.min(0.95, Math.max(0.5, confidence)); // Cap between 0.5 and 0.95
        
        patterns.push({
          id: `accum_${Date.now()}`,
          type: PATTERN_TYPES.ACCUMULATION,
          confidence,
          startTime: timestamps[0],
          endTime: timestamps[timestamps.length - 1],
          priceRange: {
            min: min(prices),
            max: max(prices)
          },
          indicators: {
            priceVolatility,
            buyRatio,
            durationDays: timestamps.length
          },
          description: `Accumulation pattern with ${(buyRatio * 100).toFixed(0)}% buy volume and low price volatility of ${(priceVolatility * 100).toFixed(2)}%`
        });
      }
    }
    
    return patterns;
  }
  
  /**
   * Detect distribution patterns
   * Distribution is characterized by sideways or slowly declining price with increasing sell volume
   */
  detectDistribution(data) {
    const { prices, volumes, timestamps } = data;
    const patterns = [];
    
    // Calculate price trend
    let priceDowntrend = true;
    for (let i = 1; i < prices.length; i++) {
      // Allow for small ups in a general downtrend
      if (prices[i] > prices[i-1] * 1.05) { // 5% increase threshold
        priceDowntrend = false;
        break;
      }
    }
    
    // Check if we have sufficient duration
    if (timestamps.length >= this.parameters.distribution.minDuration) {
      
      // Calculate sell volume ratio (simplified for demo)
      const sellVolume = volumes.reduce((sum, vol) => sum + vol, 0) * 0.6; // Assume 60% is sell volume
      const totalVolume = volumes.reduce((sum, vol) => sum + vol, 0);
      const sellRatio = sellVolume / totalVolume;
      
      if (sellRatio >= this.parameters.distribution.minSellVolume) {
        // Calculate confidence based on multiple factors
        const durationScore = Math.min(1, timestamps.length / (this.parameters.distribution.minDuration * 2));
        const volumeScore = (sellRatio - this.parameters.distribution.minSellVolume) / 
                           (1 - this.parameters.distribution.minSellVolume);
        const trendScore = priceDowntrend ? 1.0 : 0.6;
        
        const confidence = (durationScore * 0.3) + (volumeScore * 0.4) + (trendScore * 0.3);
        
        patterns.push({
          id: `dist_${Date.now()}`,
          type: PATTERN_TYPES.DISTRIBUTION,
          confidence,
          startTime: timestamps[0],
          endTime: timestamps[timestamps.length - 1],
          priceRange: {
            min: min(prices),
            max: max(prices)
          },
          indicators: {
            priceDowntrend,
            sellRatio,
            durationDays: timestamps.length
          },
          description: `Distribution pattern with ${(sellRatio * 100).toFixed(0)}% sell volume and ${priceDowntrend ? 'downward' : 'sideways'} price trend`
        });
      }
    }
    
    return patterns;
  }
  
  /**
   * Detect breakout patterns
   * Breakout is characterized by a strong price increase with high volume after a period of consolidation
   */
  detectBreakout(data) {
    const { prices, volumes, timestamps } = data;
    const patterns = [];
    
    // Need enough data for consolidation + breakout
    if (timestamps.length < this.parameters.breakout.consolidationDuration + 3) {
      return patterns;
    }
    
    // Look for consolidation followed by breakout
    for (let i = this.parameters.breakout.consolidationDuration; i < prices.length - 2; i++) {
      // Check for consolidation period (low volatility)
      const consolidationPrices = prices.slice(i - this.parameters.breakout.consolidationDuration, i);
      const consolidationVolatility = std(consolidationPrices) / mean(consolidationPrices);
      
      if (consolidationVolatility < 0.07) { // 7% volatility threshold for consolidation
        // Check for breakout (price and volume surge)
        const priceChange = (prices[i+2] - prices[i]) / prices[i];
        const volumeChange = (volumes[i+2] - mean(volumes.slice(i-3, i))) / mean(volumes.slice(i-3, i));
        
        if (priceChange >= this.parameters.breakout.minPriceIncrease && 
            volumeChange >= this.parameters.breakout.minVolumeIncrease) {
          
          // Calculate confidence score
          const priceScore = Math.min(1, priceChange / (this.parameters.breakout.minPriceIncrease * 2));
          const volumeScore = Math.min(1, volumeChange / (this.parameters.breakout.minVolumeIncrease * 2));
          const consolidationScore = Math.max(0, 1 - (consolidationVolatility / 0.07));
          
          const confidence = (priceScore * 0.4) + (volumeScore * 0.4) + (consolidationScore * 0.2);
          
          patterns.push({
            id: `breakout_${Date.now()}`,
            type: PATTERN_TYPES.BREAKOUT,
            confidence,
            startTime: timestamps[i - this.parameters.breakout.consolidationDuration],
            endTime: timestamps[i+2],
            priceRange: {
              consolidationMin: min(consolidationPrices),
              consolidationMax: max(consolidationPrices),
              breakoutPrice: prices[i+2]
            },
            indicators: {
              priceChange: (priceChange * 100).toFixed(2) + '%',
              volumeChange: (volumeChange * 100).toFixed(2) + '%',
              consolidationDays: this.parameters.breakout.consolidationDuration
            },
            description: `Breakout pattern with ${(priceChange * 100).toFixed(2)}% price increase and ${(volumeChange * 100).toFixed(2)}% volume surge after ${this.parameters.breakout.consolidationDuration}-day consolidation`
          });
          
          // Skip ahead to avoid duplicate detections
          i += 3;
        }
      }
    }
    
    return patterns;
  }
  
  /**
   * Detect pump and dump patterns
   * Pump and dump is characterized by rapid price increase followed by rapid decrease
   */
  detectPumpDump(data) {
    const { prices, volumes, timestamps } = data;
    const patterns = [];
    
    // Need enough data for a pump and dump
    if (timestamps.length < 5) {
      return patterns;
    }
    
    // Look for pump followed by dump
    for (let i = 1; i < prices.length - 3; i++) {
      // Check for pump (rapid price increase)
      const pumpChange = (prices[i+1] - prices[i]) / prices[i];
      
      if (pumpChange >= this.parameters.pumpDump.minPumpPercentage) {
        // Check for subsequent dump (rapid price decrease)
        let maxDumpIndex = Math.min(i + 4, prices.length - 1);
        let dumpFound = false;
        let dumpChange = 0;
        
        for (let j = i + 2; j <= maxDumpIndex; j++) {
          const currentDumpChange = (prices[j] - prices[i+1]) / prices[i+1];
          if (currentDumpChange <= -this.parameters.pumpDump.minDumpPercentage) {
            dumpFound = true;
            dumpChange = currentDumpChange;
            
            // Calculate confidence score
            const pumpScore = Math.min(1, pumpChange / (this.parameters.pumpDump.minPumpPercentage * 2));
            const dumpScore = Math.min(1, Math.abs(dumpChange) / (this.parameters.pumpDump.minDumpPercentage * 2));
            const timeframeScore = Math.min(1, (j - i) / 4); // Higher score for faster pump/dump
            
            const confidence = (pumpScore * 0.4) + (dumpScore * 0.4) + (timeframeScore * 0.2);
            
            patterns.push({
              id: `pump_dump_${Date.now()}_${i}`,
              type: PATTERN_TYPES.PUMP_DUMP,
              confidence,
              startTime: timestamps[i],
              endTime: timestamps[j],
              priceRange: {
                startPrice: prices[i],
                peakPrice: prices[i+1],
                endPrice: prices[j]
              },
              indicators: {
                pumpChange: (pumpChange * 100).toFixed(2) + '%',
                dumpChange: (dumpChange * 100).toFixed(2) + '%',
                durationDays: j - i
              },
              description: `Pump and dump pattern with ${(pumpChange * 100).toFixed(2)}% pump followed by ${(Math.abs(dumpChange) * 100).toFixed(2)}% dump over ${j - i} days`
            });
            
            // Skip ahead to avoid duplicate detections
            i = j;
            break;
          }
        }
      }
    }
    
    return patterns;
  }
  
  /**
   * Detect various technical patterns using advanced algorithms
   * This is a simplified version - a real implementation would be more complex
   */
  detectTechnicalPatterns(data) {
    const { prices, volumes, timestamps } = data;
    const patterns = [];
    
    // Need enough data for technical patterns
    if (timestamps.length < 20) {
      return patterns;
    }
    
    // Implement pattern detection for: double bottom, double top, head & shoulders, etc.
    // These implementations would be quite complex in practice
    
    // Simplified example for double bottom pattern
    patterns.push(...this.detectDoubleBottom(data));
    
    return patterns;
  }
  
  /**
   * Detect double bottom pattern (W-shape)
   * Simplified implementation for demonstration
   */
  detectDoubleBottom(data) {
    const { prices, timestamps } = data;
    const patterns = [];
    
    // Need enough data for a double bottom
    if (prices.length < 15) {
      return patterns;
    }
    
    // Find local minimums
    const minimums = [];
    for (let i = 2; i < prices.length - 2; i++) {
      if (prices[i] < prices[i-1] && prices[i] < prices[i-2] && 
          prices[i] < prices[i+1] && prices[i] < prices[i+2]) {
        minimums.push({ index: i, price: prices[i] });
      }
    }
    
    // Look for two minimums of similar price with a peak in between
    for (let i = 0; i < minimums.length - 1; i++) {
      const min1 = minimums[i];
      const min2 = minimums[i+1];
      
      // Check if minimums are separated by enough points
      if (min2.index - min1.index >= 5) {
        // Check if minimums are at similar price levels (within 5%)
        const priceDiff = Math.abs(min1.price - min2.price) / min1.price;
        if (priceDiff <= 0.05) {
          // Check if there's a peak in between that's significantly higher
          const middleIndex = Math.floor((min1.index + min2.index) / 2);
          const middleRange = prices.slice(min1.index, min2.index);
          const middleMax = max(middleRange);
          
          const priceRise = (middleMax - min1.price) / min1.price;
          if (priceRise >= 0.1) { // 10% rise from bottom
            // Calculate confidence score
            const bottomSimScore = 1 - (priceDiff / 0.05);
            const riseScore = Math.min(1, priceRise / 0.2);
            const patternClearnessScore = 0.7; // Would be calculated based on pattern clarity
            
            const confidence = (bottomSimScore * 0.3) + (riseScore * 0.4) + (patternClearnessScore * 0.3);
            
            patterns.push({
              id: `double_bottom_${Date.now()}_${i}`,
              type: PATTERN_TYPES.DOUBLE_BOTTOM,
              confidence,
              startTime: timestamps[min1.index - 2],
              endTime: timestamps[min2.index + 2],
              priceRange: {
                firstBottom: min1.price,
                secondBottom: min2.price,
                middlePeak: middleMax
              },
              indicators: {
                bottomPriceDiff: (priceDiff * 100).toFixed(2) + '%',
                peakRise: (priceRise * 100).toFixed(2) + '%',
                patternWidth: min2.index - min1.index
              },
              description: `Double bottom pattern with ${(priceRise * 100).toFixed(2)}% rise between bottoms`
            });
          }
        }
      }
    }
    
    return patterns;
  }
  
  /**
   * Filter detected patterns to remove overlaps and low-confidence patterns
   */
  filterPatterns(patterns, minConfidence) {
    // Filter by minimum confidence
    let filtered = patterns.filter(p => p.confidence >= minConfidence);
    
    // TODO: Add logic to merge overlapping patterns or resolve conflicts
    // This would be quite complex in a real implementation
    
    return filtered;
  }
  
  /**
   * Update pattern detection parameters based on feedback
   * This implements a simple form of supervised learning
   */
  updateParameters(patternId, accuracy, tokenAddress) {
    // Find the pattern in history
    const historyEntry = this.patternHistory.find(entry => 
      entry.patterns.some(p => p.id === patternId)
    );
    
    if (!historyEntry) {
      return { success: false, message: "Pattern not found in history" };
    }
    
    const pattern = historyEntry.patterns.find(p => p.id === patternId);
    if (!pattern) {
      return { success: false, message: "Pattern not found" };
    }
    
    // Record feedback
    this.feedbackHistory.push({
      patternId,
      tokenAddress,
      patternType: pattern.type,
      originalConfidence: pattern.confidence,
      accuracy,
      timestamp: new Date().toISOString()
    });
    
    // If this is negative feedback, adjust parameters to be less sensitive
    // If positive feedback, adjust to be more sensitive
    const adjustment = (accuracy - pattern.confidence) * this.learningRate;
    
    // Adjust parameters specific to this pattern type
    switch (pattern.type) {
      case PATTERN_TYPES.ACCUMULATION:
        this.parameters.accumulation.minBuyVolume = Math.max(
          0.1, 
          Math.min(0.9, this.parameters.accumulation.minBuyVolume - adjustment)
        );
        this.parameters.accumulation.maxVolatility = Math.max(
          0.01,
          Math.min(0.3, this.parameters.accumulation.maxVolatility + (adjustment * 0.1))
        );
        break;
        
      case PATTERN_TYPES.DISTRIBUTION:
        this.parameters.distribution.minSellVolume = Math.max(
          0.1,
          Math.min(0.9, this.parameters.distribution.minSellVolume - adjustment)
        );
        break;
        
      case PATTERN_TYPES.BREAKOUT:
        this.parameters.breakout.minPriceIncrease = Math.max(
          0.03,
          Math.min(0.3, this.parameters.breakout.minPriceIncrease - (adjustment * 0.1))
        );
        this.parameters.breakout.minVolumeIncrease = Math.max(
          0.2,
          Math.min(1.0, this.parameters.breakout.minVolumeIncrease - (adjustment * 0.2))
        );
        break;
        
      case PATTERN_TYPES.PUMP_DUMP:
        this.parameters.pumpDump.minPumpPercentage = Math.max(
          0.05,
          Math.min(0.4, this.parameters.pumpDump.minPumpPercentage - (adjustment * 0.1))
        );
        this.parameters.pumpDump.minDumpPercentage = Math.max(
          0.05,
          Math.min(0.3, this.parameters.pumpDump.minDumpPercentage - (adjustment * 0.1))
        );
        break;
        
      // Add cases for other pattern types
    }
    
    return {
      success: true,
      message: "Parameters updated based on feedback",
      adjustedParameters: this.parameters
    };
  }
  
  /**
   * Backtest a pattern against historical data
   */
  backtestPattern(patternType, historicalData, options = {}) {
    // Default options
    const opts = {
      confidenceThreshold: 0.7,
      profitTarget: 0.1, // 10%
      stopLoss: 0.05, // 5%
      holdingPeriod: 7, // days
      ...options
    };
    
    const results = {
      totalSignals: 0,
      successfulTrades: 0,
      failedTrades: 0,
      totalReturn: 0,
      averageReturn: 0,
      winRate: 0,
      trades: []
    };
    
    // Break historical data into segments for backtesting
    const segments = this.createHistoricalSegments(historicalData);
    
    // For each segment, detect patterns and simulate trades
    for (const segment of segments) {
      // Detect patterns in this segment
      const { patterns } = this.detectPatterns(segment.data, '30d');
      
      // Filter patterns by type and confidence
      const relevantPatterns = patterns.filter(p => 
        p.type === patternType && p.confidence >= opts.confidenceThreshold
      );
      
      if (relevantPatterns.length === 0) continue;
      
      // For each pattern, simulate a trade
      for (const pattern of relevantPatterns) {
        results.totalSignals++;
        
        // Get price data after pattern detection
        const entryIndex = segment.data.timestamps.indexOf(pattern.endTime);
        if (entryIndex < 0 || entryIndex >= segment.data.prices.length - opts.holdingPeriod) {
          continue;
        }
        
        const entryPrice = segment.data.prices[entryIndex];
        
        // Track price movement after entry
        let exit = false;
        let exitPrice = entryPrice;
        let exitReason = 'end_of_holding';
        let daysHeld = opts.holdingPeriod;
        
        for (let i = 1; i <= opts.holdingPeriod; i++) {
          if (entryIndex + i >= segment.data.prices.length) break;
          
          const currentPrice = segment.data.prices[entryIndex + i];
          const priceChange = (currentPrice - entryPrice) / entryPrice;
          
          // Check for profit target or stop loss
          if (priceChange >= opts.profitTarget) {
            exitPrice = currentPrice;
            exitReason = 'profit_target';
            daysHeld = i;
            exit = true;
            break;
          }
          
          if (priceChange <= -opts.stopLoss) {
            exitPrice = currentPrice;
            exitReason = 'stop_loss';
            daysHeld = i;
            exit = true;
            break;
          }
        }
        
        // If no exit was triggered, use price at end of holding period
        if (!exit && entryIndex + opts.holdingPeriod < segment.data.prices.length) {
          exitPrice = segment.data.prices[entryIndex + opts.holdingPeriod];
        }
        
        // Calculate trade results
        const returnPct = (exitPrice - entryPrice) / entryPrice;
        
        // Record trade
        const trade = {
          patternId: pattern.id,
          entryDate: pattern.endTime,
          entryPrice,
          exitPrice,
          returnPct,
          daysHeld,
          exitReason,
          successful: returnPct > 0
        };
        
        results.trades.push(trade);
        
        if (trade.successful) {
          results.successfulTrades++;
        } else {
          results.failedTrades++;
        }
        
        results.totalReturn += returnPct;
      }
    }
    
    // Calculate summary statistics
    if (results.totalSignals > 0) {
      results.winRate = results.successfulTrades / results.totalSignals;
      results.averageReturn = results.totalReturn / results.totalSignals;
    }
    
    return results;
  }
  
  /**
   * Create segments from historical data for backtesting
   */
  createHistoricalSegments(historicalData) {
    const segments = [];
    const { prices, volumes, timestamps } = historicalData;
    
    // Ensure data has enough points
    if (prices.length < 60) {
      return segments;
    }
    
    // Create overlapping segments
    const segmentSize = Math.min(30, Math.floor(prices.length / 2));
    const step = Math.floor(segmentSize / 3);
    
    for (let i = 0; i < prices.length - segmentSize; i += step) {
      segments.push({
        data: {
          prices: prices.slice(i, i + segmentSize + 30), // Include extra data for after pattern
          volumes: volumes.slice(i, i + segmentSize + 30),
          timestamps: timestamps.slice(i, i + segmentSize + 30)
        },
        startIndex: i
      });
    }
    
    return segments;
  }
  
  /**
   * Generate a fingerprint for a dataset to identify it later
   * Used for tracking which data was used to detect patterns
   */
  generateDataFingerprint(data) {
    const { prices, volumes, timestamps } = data;
    
    // Simple fingerprint: first timestamp + last timestamp + data length + price sum checksum
    const firstTimestamp = timestamps[0] || '';
    const lastTimestamp = timestamps[timestamps.length - 1] || '';
    const dataLength = prices.length;
    const priceSum = prices.reduce((sum, p) => sum + p, 0);
    const checksumValue = Math.floor(priceSum * 1000) % 10000;
    
    return `${firstTimestamp}_${lastTimestamp}_${dataLength}_${checksumValue}`;
  }
}
