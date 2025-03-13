import React, { useState } from 'react';
import {
  DashboardLayout,
  TokenAnalysisDashboard,
  PatternDetectionDashboard,
  RiskAssessmentDashboard,
  WalletTrackingDashboard
} from './react_components';

/**
 * Main Application Component
 * 
 * This is the entry point for the React frontend for the Solana Token Analysis Agent Swarm.
 * It provides navigation between different dashboard views.
 */
const App = () => {
  const [currentView, setCurrentView] = useState('home');
  const [tokenAddress, setTokenAddress] = useState('');
  
  // Handle navigation
  const navigateTo = (view) => {
    setCurrentView(view);
  };
  
  // Handle token address input
  const handleTokenAddressSubmit = (e) => {
    e.preventDefault();
    if (tokenAddress) {
      // Navigate to token analysis with the entered address
      setCurrentView('token-analysis');
    }
  };
  
  // Render the appropriate view based on currentView state
  const renderView = () => {
    switch (currentView) {
      case 'token-analysis':
        return <TokenAnalysisDashboard tokenAddress={tokenAddress} />;
      case 'pattern-detection':
        return <PatternDetectionDashboard />;
      case 'risk-assessment':
        return <RiskAssessmentDashboard tokenAddress={tokenAddress} />;
      case 'wallet-tracking':
        return <WalletTrackingDashboard />;
      case 'home':
      default:
        return <HomePage navigateTo={navigateTo} setTokenAddress={setTokenAddress} handleTokenAddressSubmit={handleTokenAddressSubmit} />;
    }
  };
  
  return (
    <DashboardLayout>
      {/* Custom navigation for App-level routing */}
      {currentView !== 'home' && (
        <div className="mb-8">
          <div className="flex flex-wrap items-center gap-2 mb-4">
            <button 
              onClick={() => navigateTo('home')} 
              className="bg-gray-700 hover:bg-gray-600 text-white py-1 px-3 rounded text-sm"
            >
              ← Back to Home
            </button>
            
            <button 
              onClick={() => navigateTo('token-analysis')} 
              className={`py-1 px-3 rounded text-sm ${currentView === 'token-analysis' ? 'bg-purple-600 text-white' : 'bg-gray-700 hover:bg-gray-600 text-white'}`}
            >
              Token Analysis
            </button>
            
            <button 
              onClick={() => navigateTo('pattern-detection')} 
              className={`py-1 px-3 rounded text-sm ${currentView === 'pattern-detection' ? 'bg-purple-600 text-white' : 'bg-gray-700 hover:bg-gray-600 text-white'}`}
            >
              Pattern Detection
            </button>
            
            <button 
              onClick={() => navigateTo('risk-assessment')} 
              className={`py-1 px-3 rounded text-sm ${currentView === 'risk-assessment' ? 'bg-purple-600 text-white' : 'bg-gray-700 hover:bg-gray-600 text-white'}`}
            >
              Risk Assessment
            </button>
            
            <button 
              onClick={() => navigateTo('wallet-tracking')} 
              className={`py-1 px-3 rounded text-sm ${currentView === 'wallet-tracking' ? 'bg-purple-600 text-white' : 'bg-gray-700 hover:bg-gray-600 text-white'}`}
            >
              Wallet Tracking
            </button>
          </div>
          
          {/* Token address display/edit */}
          <div className="bg-gray-800 p-3 rounded-lg mb-4 flex items-center justify-between">
            <div className="flex items-center">
              <span className="text-gray-400 mr-2">Token Address:</span>
              <span className="font-mono text-sm">{tokenAddress || 'None selected'}</span>
            </div>
            <button 
              onClick={() => navigateTo('home')}
              className="text-sm bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
            >
              Change
            </button>
          </div>
        </div>
      )}
      
      {/* Render the current view */}
      {renderView()}
    </DashboardLayout>
  );
};

/**
 * Home Page Component
 * 
 * Landing page with introduction and navigation to different dashboards
 */
const HomePage = ({ navigateTo, setTokenAddress, handleTokenAddressSubmit }) => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold mb-4">Solana Token Analysis Agent Swarm</h1>
        <p className="text-xl text-gray-300">
          An AI-powered system for analyzing Solana tokens, detecting patterns,
          assessing risks, and identifying investment opportunities in real-time.
        </p>
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Analyze a Token</h2>
        <form onSubmit={handleTokenAddressSubmit}>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Enter Solana token address (mint)"
              className="flex-1 bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white"
              onChange={(e) => setTokenAddress(e.target.value)}
            />
            <button
              type="submit"
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded"
            >
              Analyze
            </button>
          </div>
        </form>
        
        <div className="mt-4 text-center">
          <p className="text-gray-400 text-sm">
            Example: EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v (USDC)
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        <DashboardCard 
          title="Token Analysis" 
          description="Analyze token information, price data, and holder distribution."
          icon={
            <svg className="h-10 w-10 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          onClick={() => navigateTo('token-analysis')}
        />
        
        <DashboardCard 
          title="Pattern Detection" 
          description="Identify trading patterns and predict price movements."
          icon={
            <svg className="h-10 w-10 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          }
          onClick={() => navigateTo('pattern-detection')}
        />
        
        <DashboardCard 
          title="Risk Assessment" 
          description="Evaluate token contract safety, liquidity, and other risk factors."
          icon={
            <svg className="h-10 w-10 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
          onClick={() => navigateTo('risk-assessment')}
        />
        
        <DashboardCard 
          title="Wallet Tracking" 
          description="Monitor wallet activities and manage blacklists."
          icon={
            <svg className="h-10 w-10 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
          }
          onClick={() => navigateTo('wallet-tracking')}
        />
      </div>
      
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">About the Project</h2>
        <p className="mb-4">
          The Solana Token Analysis Agent Swarm is an AI-powered system designed to analyze Solana tokens,
          detect patterns, assess risks, and identify investment opportunities in real-time. The system leverages
          a swarm of specialized agents that work collaboratively to provide comprehensive analysis and actionable insights.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <FeatureCard 
            title="Multi-Agent Architecture" 
            description="Collaborative agent swarm with specialized roles"
          />
          <FeatureCard 
            title="Real-Time Analysis" 
            description="Monitor and analyze Solana blockchain transactions as they occur"
          />
          <FeatureCard 
            title="Pattern Recognition" 
            description="Identify significant trading patterns and market behaviors"
          />
          <FeatureCard 
            title="Risk Assessment" 
            description="Evaluate token contract safety, liquidity, and other risk factors"
          />
        </div>
      </div>
    </div>
  );
};

/**
 * Dashboard Card Component
 * 
 * Used on the home page to navigate to different dashboards
 */
const DashboardCard = ({ title, description, icon, onClick }) => {
  return (
    <div 
      className="bg-gray-800 rounded-lg p-6 cursor-pointer hover:bg-gray-700 transition-colors"
      onClick={onClick}
    >
      <div className="flex items-start">
        <div className="flex-shrink-0 mr-4">
          {icon}
        </div>
        <div>
          <h3 className="text-xl font-semibold mb-2">{title}</h3>
          <p className="text-gray-300">{description}</p>
        </div>
      </div>
    </div>
  );
};

/**
 * Feature Card Component
 * 
 * Used on the home page to display project features
 */
const FeatureCard = ({ title, description }) => {
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <h3 className="font-semibold mb-1">{title}</h3>
      <p className="text-gray-300 text-sm">{description}</p>
    </div>
  );
};

export default App;
