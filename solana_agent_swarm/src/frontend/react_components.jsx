import React, { useState, useEffect } from 'react';

/**
 * Solana Token Analysis Dashboard Components
 * 
 * This file contains React components for a potential React-based
 * frontend for the Solana Token Analysis Agent Swarm.
 */

// Layout components
const DashboardLayout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-950 text-white">
      <Navbar />
      <div className="container mx-auto px-4 py-8">
        <div className="relative">
          {/* Decorative glow effects */}
          <div className="absolute -top-40 -left-40 w-80 h-80 bg-blue-500 rounded-full opacity-10 filter blur-3xl"></div>
          <div className="absolute top-60 -right-40 w-80 h-80 bg-purple-500 rounded-full opacity-10 filter blur-3xl"></div>
          
          {/* Content with glass morphism effect */}
          <div className="relative z-10 backdrop-blur-sm rounded-xl">
            {children}
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

const Navbar = () => {
  return (
    <nav className="w-full bg-gray-900 text-white px-6 py-4 shadow-md sticky top-0 z-50 border-b border-blue-900/30">
      <div className="container mx-auto flex items-center justify-between">
        {/* Logo section */}
        <div className="flex items-center space-x-2">
          <svg className="h-8 w-8 text-blue-500" viewBox="0 0 128 128" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M32 96H96L32 32V96Z" fill="currentColor" />
            <path d="M96 32H32L96 96V32Z" fill="currentColor" />
          </svg>
          <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Solana Token Analyzer
          </span>
        </div>
        
        {/* Navigation links with enhanced styling */}
        <div className="hidden md:flex space-x-8">
          <a href="#" className="hover:text-blue-400 transition duration-300 flex items-center">
            <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
            Token Analysis
          </a>
          <a href="#" className="hover:text-blue-400 transition duration-300 flex items-center">
            <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Pattern Detection
          </a>
          <a href="#" className="hover:text-blue-400 transition duration-300 flex items-center">
            <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            Risk Assessment
          </a>
          <a href="#" className="hover:text-blue-400 transition duration-300 flex items-center">
            <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            Wallet Tracking
          </a>
        </div>
        
        {/* User section with enhanced button and profile icon */}
        <div className="flex items-center space-x-4">
          <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-md transition duration-300 shadow-lg shadow-blue-500/20 flex items-center">
            <svg className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Connect Wallet
          </button>
          <div className="relative hidden md:block">
            <div className="h-9 w-9 rounded-full bg-gradient-to-r from-purple-600 to-blue-500 flex items-center justify-center cursor-pointer">
              <span className="text-xs font-bold">AI</span>
              <span className="absolute bottom-0 right-0 h-3 w-3 rounded-full bg-green-500 border-2 border-gray-900"></span>
            </div>
          </div>
        </div>
        
        {/* Mobile menu button */}
        <button className="md:hidden text-gray-300 hover:text-white focus:outline-none">
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      </div>
    </nav>
  );
};

const Footer = () => {
  return (
    <footer className="bg-gray-900 border-t border-blue-900/30 py-6 mt-10">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <span className="text-gray-400">Solana Token Analysis Agent Swarm | © 2025</span>
          </div>
          <div className="flex space-x-6">
            <a href="#" className="text-gray-400 hover:text-blue-400 transition duration-300 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Documentation
            </a>
            <a href="#" className="text-gray-400 hover:text-blue-400 transition duration-300 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              GitHub
            </a>
            <a href="#" className="text-gray-400 hover:text-blue-400 transition duration-300 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192l-3.536 3.536M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5 0a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
              Support
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

// Token Analysis Components
const TokenAnalysisDashboard = ({ tokenAddress }) => {
  const [tokenData, setTokenData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Simulated data fetch
  useEffect(() => {
    // In a real implementation, this would fetch data from an API
    setTimeout(() => {
      setTokenData({
        name: "USD Coin",
        symbol: "USDC",
        address: tokenAddress || "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        decimals: 6,
        market: {
          price_usd: 1.00,
          price_change_24h: 0.01,
          volume_24h: 42587954,
          market_cap: 32876543210
        },
        holders: {
          count: 145879,
          distribution: [
            { wallet: "Wallet 1", percentage: 12.5 },
            { wallet: "Wallet 2", percentage: 8.7 },
            { wallet: "Wallet 3", percentage: 6.3 },
            { wallet: "Wallet 4", percentage: 5.1 },
            { wallet: "Others", percentage: 67.4 }
          ]
        }
      });
      setLoading(false);
    }, 1000);
  }, [tokenAddress]);
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Token Analysis Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Token Overview */}
        <TokenOverviewCard tokenInfo={tokenData} />
        
        {/* Market Data */}
        <MarketDataCard market={tokenData.market} />
        
        {/* Holder Information */}
        <HolderInformationCard holders={tokenData.holders} />
      </div>
      
      {/* Price Chart */}
      <div className="bg-gray-800 rounded-lg p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Price History</h2>
        <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
          <p className="text-gray-400">Price chart visualization would appear here</p>
        </div>
      </div>
      
      {/* Volume Chart */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Trading Volume</h2>
        <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
          <p className="text-gray-400">Volume chart visualization would appear here</p>
        </div>
      </div>
    </div>
  );
};

const TokenOverviewCard = ({ tokenInfo }) => {
  return (
    <div className="bg-gray-800/80 backdrop-blur rounded-lg p-6 border border-blue-900/20 shadow-lg shadow-blue-900/5">
      <div className="flex items-center mb-4">
        <div className="h-10 w-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center mr-3">
          <span className="text-lg font-bold">{tokenInfo.symbol ? tokenInfo.symbol.charAt(0) : '?'}</span>
        </div>
        <h2 className="text-xl font-semibold">Token Overview</h2>
      </div>
      <div className="space-y-3">
        <div className="flex justify-between items-center py-2 border-b border-gray-700">
          <span className="text-gray-400">Name:</span>
          <span className="font-medium">{tokenInfo.name}</span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-gray-700">
          <span className="text-gray-400">Symbol:</span>
          <span className="font-medium text-blue-400">{tokenInfo.symbol}</span>
        </div>
        <div className="flex justify-between items-center py-2 border-b border-gray-700">
          <span className="text-gray-400">Address:</span>
          <span className="font-medium font-mono bg-gray-700 px-2 py-1 rounded text-sm truncate max-w-[150px]" title={tokenInfo.address}>
            {tokenInfo.address.slice(0, 6)}...{tokenInfo.address.slice(-4)}
          </span>
        </div>
        <div className="flex justify-between items-center py-2">
          <span className="text-gray-400">Decimals:</span>
          <span className="font-medium">{tokenInfo.decimals}</span>
        </div>
      </div>
    </div>
  );
};

const MarketDataCard = ({ market }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Market Data</h2>
      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="text-gray-400">Price:</span>
          <span className="font-medium">${market.price_usd.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">24h Change:</span>
          <span className={`font-medium ${market.price_change_24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {market.price_change_24h >= 0 ? '+' : ''}{market.price_change_24h.toFixed(2)}%
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">24h Volume:</span>
          <span className="font-medium">${market.volume_24h.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Market Cap:</span>
          <span className="font-medium">${market.market_cap.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
};

const HolderInformationCard = ({ holders }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Holder Information</h2>
      <div className="mb-4">
        <div className="flex justify-between">
          <span className="text-gray-400">Total Holders:</span>
          <span className="font-medium">{holders.count.toLocaleString()}</span>
        </div>
      </div>
      
      <h3 className="text-lg font-medium mb-3">Token Distribution</h3>
      <div className="space-y-2">
        {holders.distribution.map((item, index) => (
          <div key={index} className="relative pt-1">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-300">{item.wallet}</span>
              <span className="text-sm font-medium text-gray-300">{item.percentage}%</span>
            </div>
            <div className="overflow-hidden h-2 text-xs flex rounded bg-gray-700">
              <div
                style={{ width: `${item.percentage}%` }}
                className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-purple-500"
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Pattern Detection Components
const PatternDetectionDashboard = () => {
  const [patterns, setPatterns] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Simulated data fetch
  useEffect(() => {
    setTimeout(() => {
      setPatterns([
        {
          id: "pattern-1",
          type: "accumulation",
          confidence: 0.87,
          description: "Token shows signs of accumulation by large holders over the past 7 days",
          detected_at: "2025-03-10T09:43:21Z",
          indicators: {
            volume_increase: "+32%",
            large_transactions: "12 transactions > $50k",
            whale_activity: "3 known whales active"
          }
        },
        {
          id: "pattern-2",
          type: "breakout",
          confidence: 0.92,
          description: "Token shows potential for upward price breakout based on increasing buy pressure",
          detected_at: "2025-03-11T02:15:05Z",
          indicators: {
            buy_pressure: "High",
            resistance_test: "3 tests in 24h",
            volume_pattern: "Increasing with price tests"
          }
        },
        {
          id: "pattern-3",
          type: "distribution",
          confidence: 0.78,
          description: "Early signs of distribution detected with increased selling from large holders",
          detected_at: "2025-03-09T14:22:35Z",
          indicators: {
            sell_pressure: "Medium-High",
            large_sells: "8 transactions in past 48h",
            holder_change: "-2.3% among top 10 holders"
          }
        }
      ]);
      setLoading(false);
    }, 1000);
  }, []);
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Pattern Detection Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <PatternAnalysisOptions />
        <div className="lg:col-span-2">
          <PricePredictionCard />
        </div>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Detected Patterns</h2>
        
        <div className="space-y-4">
          {patterns.map(pattern => (
            <PatternCard key={pattern.id} pattern={pattern} />
          ))}
        </div>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">Pattern Visualization</h2>
        <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center">
          <p className="text-gray-400">Pattern visualization chart would appear here</p>
        </div>
      </div>
    </div>
  );
};

const PatternAnalysisOptions = () => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Pattern Analysis Options</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-gray-400 mb-2">Time Period</label>
          <select className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white">
            <option value="24h">24 Hours</option>
            <option value="7d" selected>7 Days</option>
            <option value="30d">30 Days</option>
          </select>
        </div>
        
        <div>
          <label className="block text-gray-400 mb-2">Pattern Type</label>
          <select className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white">
            <option value="all" selected>All Patterns</option>
            <option value="pump_dump">Pump & Dump</option>
            <option value="accumulation">Accumulation</option>
            <option value="distribution">Distribution</option>
            <option value="breakout">Breakout</option>
          </select>
        </div>
        
        <div>
          <label className="block text-gray-400 mb-2">
            Minimum Confidence: 0.7
          </label>
          <input 
            type="range" 
            min="0" 
            max="1" 
            step="0.05"
            defaultValue="0.7"
            className="w-full"
          />
        </div>
        
        <button className="w-full bg-purple-600 hover:bg-purple-700 text-white rounded-md py-2 px-4 mt-4">
          Analyze Token Patterns
        </button>
        
        <button className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-md py-2 px-4">
          Detect Market Patterns
        </button>
      </div>
    </div>
  );
};

const PricePredictionCard = () => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Price Prediction</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <div className="space-y-4">
            <div>
              <label className="block text-gray-400 mb-2">Time Horizon</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white">
                <option value="1h">1 Hour</option>
                <option value="24h" selected>24 Hours</option>
                <option value="7d">7 Days</option>
              </select>
            </div>
            
            <button className="w-full bg-green-600 hover:bg-green-700 text-white rounded-md py-2 px-4 mt-4">
              Get Price Prediction
            </button>
          </div>
        </div>
        
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-3">Prediction Results</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Current Price:</span>
              <span className="font-medium">$1.23</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Predicted Price:</span>
              <span className="font-medium">$1.32</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Expected Change:</span>
              <span className="font-medium text-green-500">+7.32%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Direction:</span>
              <span className="font-medium text-green-500">Upward</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Confidence:</span>
              <span className="font-medium">0.85</span>
            </div>
          </div>
          <div className="mt-4 text-xs text-gray-400">
            Prediction made at: 2025-03-11T12:15:03Z
          </div>
          <div className="mt-2 text-xs text-gray-400 italic">
            This prediction is for educational purposes only. Do not make investment decisions based solely on this information.
          </div>
        </div>
      </div>
    </div>
  );
};

const PatternCard = ({ pattern }) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <div 
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div>
          <h3 className="text-lg font-medium">
            {pattern.type.charAt(0).toUpperCase() + pattern.type.slice(1)} Pattern
          </h3>
          <div className="text-sm text-gray-400">
            Confidence: {pattern.confidence.toFixed(2)}
          </div>
        </div>
        <div>
          <svg 
            className={`h-6 w-6 transition-transform ${expanded ? 'transform rotate-180' : ''}`} 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      
      {expanded && (
        <div className="mt-4 border-t border-gray-600 pt-4">
          <div className="text-sm mb-3">{pattern.description}</div>
          
          <div className="text-sm text-gray-400 mb-2">Detected at: {new Date(pattern.detected_at).toLocaleString()}</div>
          
          <div className="mt-3">
            <h4 className="text-md font-medium mb-2">Indicators</h4>
            <div className="bg-gray-800 rounded p-3 text-sm">
              {Object.entries(pattern.indicators).map(([key, value]) => (
                <div key={key} className="flex justify-between mb-1">
                  <span className="text-gray-400">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="mt-4">
            <h4 className="text-md font-medium mb-2">Provide Feedback</h4>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Accuracy Rating</label>
                <input 
                  type="range" 
                  min="0" 
                  max="1" 
                  step="0.1"
                  defaultValue="0.5"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Comments</label>
                <textarea 
                  className="w-full bg-gray-900 border border-gray-700 rounded-md p-2 text-sm"
                  rows="2"
                ></textarea>
              </div>
              <button className="bg-blue-600 hover:bg-blue-700 text-white text-sm rounded px-3 py-1">
                Submit Feedback
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Risk Assessment Components
const RiskAssessmentDashboard = ({ tokenAddress }) => {
  const [riskData, setRiskData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Simulated data fetch
  useEffect(() => {
    setTimeout(() => {
      setRiskData({
        risk_level: "Medium",
        overall_risk_score: 0.62,
        risk_scores: {
          "Liquidity Risk": 0.45,
          "Contract Risk": 0.72,
          "Holder Concentration": 0.85,
          "Market Volatility": 0.55,
          "Age Factor": 0.38
        },
        concerns: [
          "Contract has not been externally audited",
          "High token concentration among top 10 holders (85%)",
          "Limited liquidity relative to market cap"
        ],
        rug_pull: {
          probability: 0.37,
          key_factors: [
            "Contract ownership is not renounced",
            "Minting function is enabled",
            "Liquidity tokens are not locked"
          ]
        },
        safety: {
          recommendations: [
            "Limit investment exposure to less than 2% of portfolio",
            "Set stop-loss orders at 10-15% below purchase price",
            "Monitor holder concentration changes regularly",
            "Check for contract audits before significant investment"
          ]
        }
      });
      setLoading(false);
    }, 1000);
  }, [tokenAddress]);
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Risk Assessment Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <RiskAnalysisOptions />
        <div className="lg:col-span-2">
          <RiskSummaryCard riskData={riskData} />
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <RugPullAssessmentCard rugPullData={riskData.rug_pull} />
        <SafetyRecommendationsCard safetyData={riskData.safety} />
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">High-Risk Wallets</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-700">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Wallet Address</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Risk Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Risk Factors</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Last Activity</th>
              </tr>
            </thead>
            <tbody className="bg-gray-900 divide-y divide-gray-800">
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm">0x1a2b...3c4d</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-red-500">0.89</td>
                <td className="px-6 py-4 text-sm">Wash trading, Suspicious transfer patterns</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">2h ago</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm">0x5e6f...7g8h</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-red-500">0.82</td>
                <td className="px-6 py-4 text-sm">Connected to known rug pulls, Large dumps</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">6h ago</td>
              </tr>
              <tr>
                <td className="px-6 py-4 whitespace-nowrap text-sm">0x9i0j...1k2l</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-500">0.76</td>
                <td className="px-6 py-4 text-sm">Large sell pressure, Abnormal trading patterns</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">1d ago</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

const RiskAnalysisOptions = () => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Risk Analysis Options</h2>
      
      <div className="space-y-4">
        <div className="flex items-center">
          <input type="checkbox" id="force-refresh" className="mr-2" />
          <label htmlFor="force-refresh" className="text-gray-300">Force Refresh Assessment</label>
        </div>
        
        <button className="w-full bg-red-600 hover:bg-red-700 text-white rounded-md py-2 px-4">
          Assess Token Risk
        </button>
        
        <button className="w-full bg-orange-600 hover:bg-orange-700 text-white rounded-md py-2 px-4">
          Calculate Rug Pull Probability
        </button>
        
        <div>
          <label className="block text-gray-400 mb-2">Interaction Type</label>
          <select className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white">
            <option value="any" selected>Any Interaction</option>
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
            <option value="stake">Stake</option>
            <option value="provide_liquidity">Provide Liquidity</option>
          </select>
        </div>
        
        <button className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-md py-2 px-4">
          Get Safety Recommendations
        </button>
        
        <button className="w-full bg-purple-600 hover:bg-purple-700 text-white rounded-md py-2 px-4">
          Identify High-Risk Wallets
        </button>
      </div>
    </div>
  );
};

const RiskSummaryCard = ({ riskData }) => {
  // Determine color based on risk level
  const riskColorClass = {
    "Low": "bg-green-600",
    "Medium": "bg-orange-600",
    "High": "bg-red-600"
  }[riskData.risk_level] || "bg-gray-600";
  
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Risk Assessment Summary</h2>
      
      <div className={`${riskColorClass} p-4 rounded-lg mb-6`}>
        <h3 className="text-xl font-bold text-white text-center">
          Risk Level: {riskData.risk_level}
        </h3>
      </div>
      
      <div className="text-center mb-6">
        <div className="text-2xl font-bold">Overall Risk Score</div>
        <div className="text-4xl font-bold">{riskData.overall_risk_score.toFixed(2)}</div>
      </div>
      
      {/* Risk Radar Chart Placeholder */}
      <div className="aspect-video bg-gray-700 rounded-lg flex items-center justify-center mb-6">
        <p className="text-gray-400">Risk score radar chart would appear here</p>
      </div>
      
      {/* Risk Concerns */}
      {riskData.concerns && riskData.concerns.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Risk Concerns</h3>
          <div className="space-y-2">
            {riskData.concerns.map((concern, index) => (
              <div key={index} className="bg-red-900/30 border border-red-800 rounded-md p-3 text-sm">
                {concern}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const RugPullAssessmentCard = ({ rugPullData }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Rug Pull Assessment</h2>
      
      <div className="flex flex-col items-center mb-6">
        {/* Probability Gauge */}
        <div className="w-48 h-48 relative mb-4 flex items-center justify-center">
          <svg viewBox="0 0 100 100" className="w-full h-full">
            {/* Background arc */}
            <path 
              d="M 10,50 A 40,40 0 1,1 90,50" 
              fill="none" 
              stroke="#374151" 
              strokeWidth="16" 
              strokeLinecap="round"
            />
            
            {/* Colored arc based on probability */}
            <path 
              d={`M 10,50 A 40,40 0 ${rugPullData.probability > 0.5 ? 1 : 0},1 ${10 + 80 * rugPullData.probability},${50 - 40 * Math.sin(Math.PI * rugPullData.probability)}`} 
              fill="none" 
              stroke={rugPullData.probability < 0.3 ? "#10B981" : rugPullData.probability < 0.7 ? "#F59E0B" : "#EF4444"} 
              strokeWidth="16" 
              strokeLinecap="round"
            />
            
            {/* Indicator */}
            <circle 
              cx={10 + 80 * rugPullData.probability} 
              cy={50 - 40 * Math.sin(Math.PI * rugPullData.probability)} 
              r="6" 
              fill="white"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-3xl font-bold">{(rugPullData.probability * 100).toFixed(0)}%</div>
            <div className="text-sm text-gray-400">Probability</div>
          </div>
        </div>
        
        <div className={`text-lg font-semibold ${
          rugPullData.probability < 0.3 ? 'text-green-500' : 
          rugPullData.probability < 0.7 ? 'text-yellow-500' : 
          'text-red-500'
        }`}>
          {rugPullData.probability < 0.3 ? 'Low Risk' : 
           rugPullData.probability < 0.7 ? 'Medium Risk' : 
           'High Risk'} of Rug Pull
        </div>
      </div>
      
      {/* Key Risk Factors */}
      {rugPullData.key_factors && rugPullData.key_factors.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Key Risk Factors</h3>
          <div className="space-y-2">
            {rugPullData.key_factors.map((factor, index) => (
              <div key={index} className="bg-yellow-900/30 border border-yellow-800 rounded-md p-3 text-sm">
                {factor}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const SafetyRecommendationsCard = ({ safetyData }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Safety Recommendations</h2>
      
      {safetyData.recommendations && safetyData.recommendations.length > 0 ? (
        <div className="space-y-2">
          {safetyData.recommendations.map((recommendation, index) => (
            <div key={index} className="bg-blue-900/30 border border-blue-800 rounded-md p-3 text-sm flex items-start">
              <svg className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{recommendation}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-gray-400">
          No safety recommendations available
        </div>
      )}
    </div>
  );
};

// Wallet Tracking Components
const WalletTrackingDashboard = () => {
  const [blacklist, setBlacklist] = useState({
    wallets: [
      {
        address: "0xa1b2c3d4e5f60000e7f8d9g0h1i2j3k4l5m6n7o8",
        reason: "Involved in multiple rug pulls and wash trading",
        added_at: "2025-02-15T08:42:13Z"
      },
      {
        address: "0xp9q8r7s6t5u4v3w2x1y0z0000a1b2c3d4e5f6g7",
        reason: "Suspicious token dumps and price manipulation",
        added_at: "2025-03-01T14:22:45Z"
      }
    ],
    tokens: [
      {
        address: "0xh8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z000",
        reason: "Confirmed rug pull with honey pot contract",
        added_at: "2025-02-20T18:15:27Z"
      }
    ]
  });
  const [walletTokens, setWalletTokens] = useState(null);
  const [loading, setLoading] = useState(false);
  
  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">Wallet Tracking Dashboard</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <BlacklistManagementCard 
          setBlacklist={setBlacklist} 
          setWalletTokens={setWalletTokens}
          setLoading={setLoading}
        />
        <div className="lg:col-span-2">
          <BlacklistTabsCard blacklist={blacklist} />
        </div>
      </div>
      
      {walletTokens && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">
            Tokens Held by {walletTokens.walletAddress.slice(0, 6)}...{walletTokens.walletAddress.slice(-4)}
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Token</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Amount</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Value</th>
                  </tr>
                </thead>
                <tbody className="bg-gray-900 divide-y divide-gray-800">
                  {walletTokens.tokens.map((token, index) => (
                    <tr key={index}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <div className="flex items-center">
                          <div className="h-8 w-8 rounded-full bg-gray-700 flex items-center justify-center mr-3">
                            {token.symbol ? token.symbol.charAt(0) : '?'}
                          </div>
                          <div>
                            <div>{token.symbol || 'Unknown'}</div>
                            <div className="text-xs text-gray-400">{token.mint.slice(0, 6)}...{token.mint.slice(-4)}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">{token.amount.toLocaleString()}</td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">${token.value ? token.value.toLocaleString() : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="aspect-square bg-gray-700 rounded-lg flex items-center justify-center">
              <p className="text-gray-400">Token distribution chart would appear here</p>
            </div>
          </div>
          
          <div className="mt-6">
            <button className="bg-purple-600 hover:bg-purple-700 text-white rounded-md py-2 px-4">
              Assess Wallet Token Risks
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const BlacklistManagementCard = ({ setBlacklist, setWalletTokens, setLoading }) => {
  const [addressToCheck, setAddressToCheck] = useState('');
  const [addressType, setAddressType] = useState('wallet');
  const [reason, setReason] = useState('');
  
  const handleCheckBlacklist = () => {
    // Simulated API call
    setLoading(true);
    setTimeout(() => {
      // This would be replaced with actual API logic
      alert(`Checking ${addressType} address: ${addressToCheck}`);
      setLoading(false);
    }, 1000);
  };
  
  const handleAddToBlacklist = () => {
    if (!addressToCheck || !reason) {
      alert('Please enter both an address and a reason');
      return;
    }
    
    // Simulated API call
    setLoading(true);
    setTimeout(() => {
      // This would be replaced with actual API logic
      setBlacklist(prev => {
        const newBlacklist = {...prev};
        const newEntry = {
          address: addressToCheck,
          reason: reason,
          added_at: new Date().toISOString()
        };
        
        if (addressType === 'wallet') {
          newBlacklist.wallets = [...prev.wallets, newEntry];
        } else {
          newBlacklist.tokens = [...prev.tokens, newEntry];
        }
        
        return newBlacklist;
      });
      setReason('');
      setLoading(false);
      alert(`Added ${addressType} to blacklist!`);
    }, 1000);
  };
  
  const handleGetWalletTokens = () => {
    if (!addressToCheck) {
      alert('Please enter a wallet address');
      return;
    }
    
    // Simulated API call
    setLoading(true);
    setTimeout(() => {
      // This would be replaced with actual API logic
      setWalletTokens({
        walletAddress: addressToCheck,
        tokens: [
          {
            mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            symbol: "USDC",
            amount: 1250.75,
            value: 1250.75
          },
          {
            mint: "So11111111111111111111111111111111111111112",
            symbol: "SOL",
            amount: 5.32,
            value: 745.12
          },
          {
            mint: "AFbX8oGjGpmVFywbVouvhQSRmiW2aR1mohfahi4Y2AdB",
            symbol: "GST",
            amount: 342.18,
            value: 28.56
          }
        ]
      });
      setLoading(false);
    }, 1000);
  };
  
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Blacklist Management</h2>
      
      <div className="space-y-4">
        <div>
          <label className="block text-gray-400 mb-2">Address to Check/Add</label>
          <input 
            type="text" 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white"
            placeholder="Enter wallet or token address"
            value={addressToCheck}
            onChange={(e) => setAddressToCheck(e.target.value)}
          />
        </div>
        
        <div className="flex space-x-4">
          <div className="flex items-center">
            <input 
              type="radio" 
              id="wallet-type" 
              name="address-type"
              className="mr-2" 
              checked={addressType === 'wallet'}
              onChange={() => setAddressType('wallet')}
            />
            <label htmlFor="wallet-type" className="text-gray-300">Wallet</label>
          </div>
          <div className="flex items-center">
            <input 
              type="radio" 
              id="token-type" 
              name="address-type"
              className="mr-2"
              checked={addressType === 'token'}
              onChange={() => setAddressType('token')} 
            />
            <label htmlFor="token-type" className="text-gray-300">Token</label>
          </div>
        </div>
        
        <button 
          className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-md py-2 px-4"
          onClick={handleCheckBlacklist}
        >
          Check Blacklist
        </button>
        
        <div>
          <label className="block text-gray-400 mb-2">Reason for Blacklisting</label>
          <textarea 
            className="w-full bg-gray-700 border border-gray-600 rounded-md py-2 px-3 text-white"
            rows="3"
            placeholder="Provide a reason why this address should be blacklisted"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          ></textarea>
        </div>
        
        <button 
          className="w-full bg-red-600 hover:bg-red-700 text-white rounded-md py-2 px-4"
          onClick={handleAddToBlacklist}
        >
          Add to Blacklist
        </button>
        
        <div className="border-t border-gray-700 my-4"></div>
        
        <h3 className="text-lg font-semibold mb-2">Wallet Token Analysis</h3>
        
        <button 
          className="w-full bg-purple-600 hover:bg-purple-700 text-white rounded-md py-2 px-4"
          onClick={handleGetWalletTokens}
        >
          Get Wallet Tokens
        </button>
      </div>
    </div>
  );
};

const BlacklistTabsCard = ({ blacklist }) => {
  const [activeTab, setActiveTab] = useState('wallets');
  
  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-semibold mb-4">Current Blacklist</h2>
      
      <div className="border-b border-gray-700">
        <div className="flex">
          <button
            className={`py-2 px-4 focus:outline-none ${
              activeTab === 'wallets' 
                ? 'border-b-2 border-purple-500 text-white font-medium' 
                : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setActiveTab('wallets')}
          >
            Blacklisted Wallets
          </button>
          <button
            className={`py-2 px-4 focus:outline-none ${
              activeTab === 'tokens' 
                ? 'border-b-2 border-purple-500 text-white font-medium' 
                : 'text-gray-400 hover:text-gray-300'
            }`}
            onClick={() => setActiveTab('tokens')}
          >
            Blacklisted Tokens
          </button>
        </div>
      </div>
      
      <div className="mt-4">
        {activeTab === 'wallets' ? (
          blacklist.wallets.length > 0 ? (
            <div className="space-y-3">
              {blacklist.wallets.map((wallet, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <div className="font-mono text-sm">{wallet.address.slice(0, 10)}...{wallet.address.slice(-6)}</div>
                    <div className="text-xs text-gray-400">{new Date(wallet.added_at).toLocaleString()}</div>
                  </div>
                  <div className="mt-2 text-sm text-gray-300">{wallet.reason}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-400">
              No wallets currently blacklisted
            </div>
          )
        ) : (
          blacklist.tokens.length > 0 ? (
            <div className="space-y-3">
              {blacklist.tokens.map((token, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-3">
                  <div className="flex items-center justify-between">
                    <div className="font-mono text-sm">{token.address.slice(0, 10)}...{token.address.slice(-6)}</div>
                    <div className="text-xs text-gray-400">{new Date(token.added_at).toLocaleString()}</div>
                  </div>
                  <div className="mt-2 text-sm text-gray-300">{token.reason}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-400">
              No tokens currently blacklisted
            </div>
          )
        )}
      </div>
    </div>
  );
};

// Utility components
const LoadingSpinner = () => {
  return (
    <div className="flex justify-center items-center h-64">
      <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-purple-500"></div>
    </div>
  );
};

// Export all components
export {
  DashboardLayout,
  Navbar,
  Footer,
  TokenAnalysisDashboard,
  PatternDetectionDashboard,
  RiskAssessmentDashboard,
  WalletTrackingDashboard,
  LoadingSpinner
};
