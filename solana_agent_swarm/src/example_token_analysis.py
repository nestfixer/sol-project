#!/usr/bin/env python3
"""
Example Token Analysis Script

This script demonstrates how to use the Solana Agent Kit integration
through MCP servers to perform comprehensive token analysis.

It shows:
1. How to use the SolanaAgentKitBridge class
2. How to perform token risk assessment
3. How to detect and analyze trading patterns
4. How to identify high-risk wallets
5. How to integrate these capabilities into the agent swarm

Usage:
    python example_token_analysis.py [token_address]
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import the MCP integration bridge
from mcp_integration import SolanaAgentKitBridge

# Import the core agent framework components
from core.orchestrator_agent import OrchestratorAgent
from core.knowledge_base import KnowledgeBase, EntryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("token_analysis")

# Default token address (USDC on Solana)
DEFAULT_TOKEN_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

class TokenAnalysisExample:
    """
    Example class showing how to integrate Solana Agent Kit with the agent swarm
    for comprehensive token analysis.
    """
    
    def __init__(self, token_address: str):
        """
        Initialize the token analysis example.
        
        Args:
            token_address: Solana token address to analyze
        """
        self.token_address = token_address
        
        # Initialize the Solana Agent Kit bridge
        self.bridge = SolanaAgentKitBridge()
        
        # Initialize the knowledge base
        self.knowledge_base = KnowledgeBase()
        
        # Initialize the orchestrator
        self.orchestrator = OrchestratorAgent()
        
        logger.info(f"TokenAnalysisExample initialized for token: {token_address}")
    
    async def initialize(self):
        """Initialize the agent ecosystem."""
        # Start the orchestrator
        await self.orchestrator.start()
        
        # Store the knowledge base reference
        self.orchestrator.knowledge_base = self.knowledge_base
        
        logger.info("Agent ecosystem initialized")
    
    async def analyze_token(self) -> Dict[str, Any]:
        """
        Perform a comprehensive token analysis using the MCP servers.
        
        This function demonstrates the full workflow of token analysis:
        1. Collecting basic token information
        2. Assessing token risk factors
        3. Analyzing trading patterns
        4. Identifying high-risk wallets
        5. Generating price predictions
        6. Storing analysis results in knowledge base
        
        Returns:
            A dictionary containing the complete analysis results
        """
        # Step 1: Get basic token information
        logger.info(f"Getting token information for {self.token_address}")
        token_info = await self.bridge.get_token_info(self.token_address)
        
        # Step 2: Assess token risk
        logger.info(f"Assessing token risk for {self.token_address}")
        risk_assessment = await self.bridge.assess_token_risk(self.token_address)
        
        # Step 3: Analyze trading patterns
        logger.info(f"Analyzing trading patterns for {self.token_address}")
        pattern_analysis = await self.bridge.analyze_token_pattern(self.token_address, "7d")
        
        # Step 4: Identify high-risk wallets
        logger.info(f"Identifying high-risk wallets for {self.token_address}")
        high_risk_wallets = await self.bridge.identify_high_risk_wallets(self.token_address)
        
        # Step 5: Generate price predictions
        logger.info(f"Generating price predictions for {self.token_address}")
        price_prediction = await self.bridge.get_price_prediction(self.token_address, "24h")
        
        # Step 6: Get safety recommendations
        logger.info(f"Getting safety recommendations for {self.token_address}")
        safety_recommendations = await self.bridge.get_safety_recommendations(self.token_address)
        
        # Step 7: Store all results in the knowledge base
        await self._store_analysis_results(
            token_info,
            risk_assessment,
            pattern_analysis,
            high_risk_wallets,
            price_prediction,
            safety_recommendations
        )
        
        # Create and return the comprehensive analysis report
        return self._create_analysis_report(
            token_info,
            risk_assessment,
            pattern_analysis,
            high_risk_wallets,
            price_prediction,
            safety_recommendations
        )
    
    async def _store_analysis_results(self,
                                    token_info: Dict[str, Any],
                                    risk_assessment: Dict[str, Any],
                                    pattern_analysis: Dict[str, Any],
                                    high_risk_wallets: Dict[str, Any],
                                    price_prediction: Dict[str, Any],
                                    safety_recommendations: Dict[str, Any]) -> None:
        """
        Store analysis results in the knowledge base.
        
        This function demonstrates how to use the knowledge base to store
        structured analysis results that can be accessed by other agents.
        
        Args:
            token_info: Token information
            risk_assessment: Risk assessment results
            pattern_analysis: Pattern analysis results
            high_risk_wallets: High-risk wallet information
            price_prediction: Price prediction results
            safety_recommendations: Safety recommendations
        """
        # Store token information
        await self.knowledge_base.add_entry(
            entry_id=f"token_info:{self.token_address}",
            entry_type=EntryType.TOKEN_DATA,
            data=token_info,
            source_agent_id="token_analysis_example",
            tags=["token", "info"]
        )
        
        # Store risk assessment
        await self.knowledge_base.add_entry(
            entry_id=f"risk_assessment:{self.token_address}",
            entry_type=EntryType.RISK_ASSESSMENT,
            data=risk_assessment,
            source_agent_id="token_analysis_example",
            tags=["token", "risk"]
        )
        
        # Store pattern analysis
        await self.knowledge_base.add_entry(
            entry_id=f"pattern_analysis:{self.token_address}",
            entry_type=EntryType.PATTERN,
            data=pattern_analysis,
            source_agent_id="token_analysis_example",
            tags=["token", "pattern"]
        )
        
        # Store high-risk wallets
        await self.knowledge_base.add_entry(
            entry_id=f"high_risk_wallets:{self.token_address}",
            entry_type=EntryType.RISK_ASSESSMENT,
            data=high_risk_wallets,
            source_agent_id="token_analysis_example",
            tags=["token", "wallets", "risk"]
        )
        
        # Store price prediction
        await self.knowledge_base.add_entry(
            entry_id=f"price_prediction:{self.token_address}",
            entry_type=EntryType.INVESTMENT_OPPORTUNITY,
            data=price_prediction,
            source_agent_id="token_analysis_example",
            tags=["token", "prediction"]
        )
        
        # Store safety recommendations
        await self.knowledge_base.add_entry(
            entry_id=f"safety_recommendations:{self.token_address}",
            entry_type=EntryType.RISK_ASSESSMENT,
            data=safety_recommendations,
            source_agent_id="token_analysis_example",
            tags=["token", "safety"]
        )
        
        logger.info(f"Stored all analysis results for {self.token_address} in knowledge base")
    
    def _create_analysis_report(self,
                              token_info: Dict[str, Any],
                              risk_assessment: Dict[str, Any],
                              pattern_analysis: Dict[str, Any],
                              high_risk_wallets: Dict[str, Any],
                              price_prediction: Dict[str, Any],
                              safety_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive analysis report.
        
        This function demonstrates how to combine multiple data sources
        into a cohesive report that can be presented to users.
        
        Args:
            token_info: Token information
            risk_assessment: Risk assessment results
            pattern_analysis: Pattern analysis results
            high_risk_wallets: High-risk wallet information
            price_prediction: Price prediction results
            safety_recommendations: Safety recommendations
            
        Returns:
            A comprehensive analysis report
        """
        # Extract token name and symbol
        token_name = token_info.get("name", f"Token {self.token_address[:8]}")
        token_symbol = token_info.get("symbol", "UNKNOWN")
        
        # Extract current price
        current_price = token_info.get("market", {}).get("price_usd", "Unknown")
        
        # Extract risk level
        risk_level = risk_assessment.get("risk_level", "Unknown")
        rug_pull_probability = risk_assessment.get("rug_pull_probability", 0)
        
        # Extract pattern summary
        pattern_summary = pattern_analysis.get("analysis_summary", "No patterns detected")
        patterns = pattern_analysis.get("patterns", [])
        
        # Extract price prediction details
        predicted_price = price_prediction.get("predicted_price", "Unknown")
        price_direction = price_prediction.get("direction", "Unknown")
        
        # Extract top high-risk wallets
        top_risk_wallets = high_risk_wallets.get("high_risk_wallets", [])
        
        # Create the report
        report = {
            "token_address": self.token_address,
            "token_name": token_name,
            "token_symbol": token_symbol,
            "analysis_time": datetime.utcnow().isoformat(),
            "summary": {
                "current_price": current_price,
                "risk_level": risk_level,
                "rug_pull_probability": f"{rug_pull_probability * 100:.1f}%" if isinstance(rug_pull_probability, (int, float)) else "Unknown",
                "pattern_summary": pattern_summary,
                "price_prediction": {
                    "predicted_price": predicted_price,
                    "direction": price_direction
                }
            },
            "risk_factors": risk_assessment.get("concerns", []),
            "detected_patterns": [
                {
                    "type": p.get("type"),
                    "confidence": p.get("confidence"),
                    "description": p.get("description")
                }
                for p in patterns
            ],
            "high_risk_wallets": [
                {
                    "address": w.get("wallet_address"),
                    "risk_score": w.get("risk_score"),
                    "holdings": w.get("holdings_percentage")
                }
                for w in top_risk_wallets[:3]  # Top 3 riskiest wallets
            ],
            "recommendations": {
                "general": safety_recommendations.get("general_recommendations", []),
                "specific": safety_recommendations.get("specific_recommendations", [])
            }
        }
        
        return report
    
    async def cleanup(self):
        """Clean up resources."""
        # Stop the orchestrator
        await self.orchestrator.stop()
        logger.info("Resources cleaned up")


async def main():
    """Main function."""
    # Get token address from command line or use default
    token_address = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOKEN_ADDRESS
    
    # Create token analysis example
    analysis_example = TokenAnalysisExample(token_address)
    
    try:
        # Initialize the agent ecosystem
        await analysis_example.initialize()
        
        # Perform token analysis
        analysis_report = await analysis_example.analyze_token()
        
        # Print the analysis report
        print("\n" + "=" * 50)
        print(f"ANALYSIS REPORT FOR {analysis_report['token_name']} ({analysis_report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {analysis_report['token_address']}")
        print(f"Analysis Time: {analysis_report['analysis_time']}")
        print("\n--- SUMMARY ---")
        print(f"Current Price: {analysis_report['summary']['current_price']}")
        print(f"Risk Level: {analysis_report['summary']['risk_level']}")
        print(f"Rug Pull Probability: {analysis_report['summary']['rug_pull_probability']}")
        print(f"Pattern Summary: {analysis_report['summary']['pattern_summary']}")
        print(f"Price Prediction: {analysis_report['summary']['price_prediction']['predicted_price']} ({analysis_report['summary']['price_prediction']['direction']})")
        
        print("\n--- RISK FACTORS ---")
        for factor in analysis_report['risk_factors']:
            print(f"- {factor}")
        
        print("\n--- DETECTED PATTERNS ---")
        for pattern in analysis_report['detected_patterns']:
            print(f"- {pattern['type']} (Confidence: {pattern['confidence']:.2f}): {pattern['description']}")
        
        print("\n--- HIGH RISK WALLETS ---")
        for wallet in analysis_report['high_risk_wallets']:
            print(f"- {wallet['address']} (Risk Score: {wallet['risk_score']:.2f}, Holdings: {wallet['holdings']})")
        
        print("\n--- RECOMMENDATIONS ---")
        print("General:")
        for rec in analysis_report['recommendations']['general']:
            print(f"- {rec}")
        
        print("\nSpecific:")
        for rec in analysis_report['recommendations']['specific']:
            print(f"- {rec}")
        
        print("\n" + "=" * 50)
        
        # Save the report to a JSON file
        report_file = f"token_analysis_{token_address[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(analysis_report, f, indent=2)
        
        print(f"\nAnalysis report saved to: {report_file}")
        
    finally:
        # Clean up resources
        await analysis_example.cleanup()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
