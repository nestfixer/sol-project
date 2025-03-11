/**
 * Statistical functions for pattern analysis
 * 
 * This module provides common statistical functions used in
 * pattern detection algorithms.
 */

/**
 * Calculate the mean (average) of an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @return {number} Mean value
 */
export function mean(values) {
  if (!values || values.length === 0) return 0;
  
  const sum = values.reduce((total, value) => total + value, 0);
  return sum / values.length;
}

/**
 * Calculate the standard deviation of an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @return {number} Standard deviation
 */
export function std(values) {
  if (!values || values.length <= 1) return 0;
  
  const avg = mean(values);
  const squaredDiffs = values.map(value => {
    const diff = value - avg;
    return diff * diff;
  });
  
  const variance = mean(squaredDiffs);
  return Math.sqrt(variance);
}

/**
 * Find the minimum value in an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @return {number} Minimum value or Infinity if array is empty
 */
export function min(values) {
  if (!values || values.length === 0) return Infinity;
  
  return Math.min(...values);
}

/**
 * Find the maximum value in an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @return {number} Maximum value or -Infinity if array is empty
 */
export function max(values) {
  if (!values || values.length === 0) return -Infinity;
  
  return Math.max(...values);
}

/**
 * Calculate the median value of an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @return {number} Median value
 */
export function median(values) {
  if (!values || values.length === 0) return 0;
  
  // Create a sorted copy of the array
  const sorted = [...values].sort((a, b) => a - b);
  
  const half = Math.floor(sorted.length / 2);
  
  // If the array has an odd number of elements, return the middle one
  // Otherwise, return the average of the two middle elements
  if (sorted.length % 2 === 0) {
    return (sorted[half - 1] + sorted[half]) / 2;
  } else {
    return sorted[half];
  }
}

/**
 * Calculate the percentile of a sorted array of numbers
 * 
 * @param {Array<number>} sortedValues Sorted array of numbers
 * @param {number} percentile Percentile to calculate (0-100)
 * @return {number} Value at the specified percentile
 */
export function percentile(sortedValues, percentile) {
  if (!sortedValues || sortedValues.length === 0) return 0;
  
  if (percentile <= 0) return sortedValues[0];
  if (percentile >= 100) return sortedValues[sortedValues.length - 1];
  
  const index = (percentile / 100) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  
  if (lower === upper) return sortedValues[lower];
  
  const weight = index - lower;
  return (1 - weight) * sortedValues[lower] + weight * sortedValues[upper];
}

/**
 * Calculate the moving average of an array of numbers
 * 
 * @param {Array<number>} values Array of numbers
 * @param {number} window Size of the moving window
 * @return {Array<number>} Array of moving averages
 */
export function movingAverage(values, window) {
  if (!values || values.length === 0 || window <= 0) return [];
  
  const result = [];
  
  for (let i = 0; i < values.length - window + 1; i++) {
    const windowValues = values.slice(i, i + window);
    result.push(mean(windowValues));
  }
  
  return result;
}

/**
 * Calculate exponential moving average (EMA)
 * 
 * @param {Array<number>} values Array of numbers
 * @param {number} period Period for EMA calculation
 * @return {Array<number>} Array of EMA values
 */
export function ema(values, period) {
  if (!values || values.length === 0 || period <= 0) return [];
  
  const result = [];
  const multiplier = 2 / (period + 1);
  
  // Start with simple moving average for the first value
  const sma = mean(values.slice(0, period));
  result.push(sma);
  
  // Calculate EMA for the rest of the values
  let currentEma = sma;
  
  for (let i = period; i < values.length; i++) {
    currentEma = (values[i] - currentEma) * multiplier + currentEma;
    result.push(currentEma);
  }
  
  return result;
}

/**
 * Calculate the correlation coefficient between two arrays
 * 
 * @param {Array<number>} valuesX First array of numbers
 * @param {Array<number>} valuesY Second array of numbers
 * @return {number} Correlation coefficient (-1 to 1)
 */
export function correlation(valuesX, valuesY) {
  if (!valuesX || !valuesY || valuesX.length !== valuesY.length || valuesX.length === 0) {
    return 0;
  }
  
  const meanX = mean(valuesX);
  const meanY = mean(valuesY);
  
  let numerator = 0;
  let denominatorX = 0;
  let denominatorY = 0;
  
  for (let i = 0; i < valuesX.length; i++) {
    const diffX = valuesX[i] - meanX;
    const diffY = valuesY[i] - meanY;
    
    numerator += diffX * diffY;
    denominatorX += diffX * diffX;
    denominatorY += diffY * diffY;
  }
  
  if (denominatorX === 0 || denominatorY === 0) return 0;
  
  return numerator / (Math.sqrt(denominatorX) * Math.sqrt(denominatorY));
}

/**
 * Calculate the Z-score (standard score) of a value
 * 
 * @param {number} value Value to calculate Z-score for
 * @param {number} meanValue Mean of the population
 * @param {number} stdValue Standard deviation of the population
 * @return {number} Z-score
 */
export function zScore(value, meanValue, stdValue) {
  if (stdValue === 0) return 0;
  return (value - meanValue) / stdValue;
}

/**
 * Linear regression: calculate slope and intercept
 * 
 * @param {Array<number>} valuesX X values
 * @param {Array<number>} valuesY Y values
 * @return {Object} Object with slope and intercept
 */
export function linearRegression(valuesX, valuesY) {
  if (!valuesX || !valuesY || valuesX.length !== valuesY.length || valuesX.length === 0) {
    return { slope: 0, intercept: 0 };
  }
  
  const n = valuesX.length;
  const meanX = mean(valuesX);
  const meanY = mean(valuesY);
  
  let numerator = 0;
  let denominator = 0;
  
  for (let i = 0; i < n; i++) {
    numerator += (valuesX[i] - meanX) * (valuesY[i] - meanY);
    denominator += (valuesX[i] - meanX) * (valuesX[i] - meanX);
  }
  
  if (denominator === 0) {
    return { slope: 0, intercept: meanY };
  }
  
  const slope = numerator / denominator;
  const intercept = meanY - (slope * meanX);
  
  return { slope, intercept };
}
