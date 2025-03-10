#!/usr/bin/env python3
"""
CLI Token Analysis Tool

This script provides a complete command-line interface for analyzing
Solana tokens using the agent swarm and MCP servers. It integrates
all components into a user-friendly CLI tool.

Usage:
    python cli_token_analysis.py analyze <token_address>
    python cli_token_analysis.py risk <token_address>
    python cli_token_analysis.py patterns <token_address> [--timeframe <timeframe>]
    python cli_token_analysis.py wallets <token_address>
    python cli_token_analysis.py predict <token_address> [--horizon <horizon>]
    python cli_token_analysis.py blacklist check <address>
    python cli_token_analysis.py blacklist add <address> <type> <reason>
    python cli_token_analysis.py trending

Options:
    --timeframe <timeframe>  Timeframe for analysis (24h, 7d, 30d) [default: 7d]
    --horizon <horizon>      Time horizon for prediction (1h, 24h, 7d) [default: 24h]
    --output <format>        Output format (text, json) [default: text]
    --verbose                Enable verbose output
    --help                   Show this help message
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import the MCP integration bridge
from mcp_integration import SolanaAgentKitBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cli_token_analysis")

# Constants
DEFAULT_TOKEN_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC on Solana

class TokenAnalysisCLI:
    """Command-line interface for token analysis."""
    
    def __init__(self, bridge: Optional[SolanaAgentKitBridge] = None, verbose: bool = False):
        """
        Initialize the CLI.
        
        Args:
            bridge: Optional bridge instance (created if not provided)
            verbose: Whether to enable verbose output
        """
        self.bridge = bridge or SolanaAgentKitBridge()
        
        # Set logging level based on verbosity
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    async def analyze_token(self, token_address: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Perform a comprehensive token analysis.
        
        Args:
            token_address: Solana token address to analyze
            output_format: Output format (text, json)
            
        Returns:
            Analysis report
        """
        logger.info(f"Starting comprehensive analysis for token: {token_address}")
        
        try:
            # Step 1: Get token information
            token_info = await self.bridge.get_token_info(token_address)
            
            # Step 2: Assess token risk
            risk_assessment = await self.bridge.assess_token_risk(token_address)
            
            # Step 3: Analyze trading patterns
            pattern_analysis = await self.bridge.analyze_token_pattern(token_address, "7d")
            
            # Step 4: Identify high-risk wallets
            high_risk_wallets = await self.bridge.identify_high_risk_wallets(token_address)
            
            # Step 5: Generate price predictions
            price_prediction = await self.bridge.get_price_prediction(token_address, "24h")
            
            # Step 6: Get safety recommendations
            safety_recommendations = await self.bridge.get_safety_recommendations(token_address)
            
            # Create analysis report
            report = self._create_analysis_report(
                token_address,
                token_info,
                risk_assessment,
                pattern_analysis,
                high_risk_wallets,
                price_prediction,
                safety_recommendations
            )
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_text_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing token: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "token_address": token_address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error analyzing token {token_address}: {e}")
            
            raise
    
    async def analyze_risk(self, token_address: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Analyze token risk.
        
        Args:
            token_address: Solana token address to analyze
            output_format: Output format (text, json)
            
        Returns:
            Risk assessment
        """
        logger.info(f"Analyzing risk for token: {token_address}")
        
        try:
            # Get token information (for name/symbol)
            token_info = await self.bridge.get_token_info(token_address)
            
            # Get risk assessment
            risk_assessment = await self.bridge.assess_token_risk(token_address)
            
            # Get rug pull probability
            rug_pull_prob = await self.bridge.get_rug_pull_probability(token_address)
            
            # Create risk report
            report = {
                "token_address": token_address,
                "token_name": token_info.get("name", f"Token {token_address[:8]}"),
                "token_symbol": token_info.get("symbol", "UNKNOWN"),
                "analysis_time": datetime.utcnow().isoformat(),
                "risk_level": risk_assessment.get("risk_level", "Unknown"),
                "overall_risk_score": risk_assessment.get("overall_risk_score", 0),
                "rug_pull_probability": rug_pull_prob.get("probability", 0),
                "risk_factors": {
                    "concentration": risk_assessment.get("risk_scores", {}).get("concentration", 0),
                    "liquidity": risk_assessment.get("risk_scores", {}).get("liquidity", 0),
                    "activity": risk_assessment.get("risk_scores", {}).get("activity", 0)
                },
                "concerns": risk_assessment.get("concerns", []),
                "key_factors": rug_pull_prob.get("key_factors", [])
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_risk_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing risk: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "token_address": token_address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error analyzing risk for token {token_address}: {e}")
            
            raise
    
    async def analyze_patterns(self, token_address: str, timeframe: str = "7d", output_format: str = "text") -> Dict[str, Any]:
        """
        Analyze token trading patterns.
        
        Args:
            token_address: Solana token address to analyze
            timeframe: Timeframe for analysis (24h, 7d, 30d)
            output_format: Output format (text, json)
            
        Returns:
            Pattern analysis
        """
        logger.info(f"Analyzing patterns for token: {token_address} over {timeframe}")
        
        try:
            # Get token information (for name/symbol)
            token_info = await self.bridge.get_token_info(token_address)
            
            # Get pattern analysis
            pattern_analysis = await self.bridge.analyze_token_pattern(token_address, timeframe)
            
            # Create patterns report
            report = {
                "token_address": token_address,
                "token_name": token_info.get("name", f"Token {token_address[:8]}"),
                "token_symbol": token_info.get("symbol", "UNKNOWN"),
                "analysis_time": datetime.utcnow().isoformat(),
                "timeframe": timeframe,
                "summary": pattern_analysis.get("analysis_summary", "No patterns detected"),
                "patterns": pattern_analysis.get("patterns", []),
                "highest_confidence_pattern": pattern_analysis.get("highest_confidence_pattern", {})
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_patterns_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "token_address": token_address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error analyzing patterns for token {token_address}: {e}")
            
            raise
    
    async def identify_wallets(self, token_address: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Identify high-risk wallets for a token.
        
        Args:
            token_address: Solana token address to analyze
            output_format: Output format (text, json)
            
        Returns:
            Wallet analysis
        """
        logger.info(f"Identifying high-risk wallets for token: {token_address}")
        
        try:
            # Get token information (for name/symbol)
            token_info = await self.bridge.get_token_info(token_address)
            
            # Get high-risk wallets
            high_risk_wallets = await self.bridge.identify_high_risk_wallets(token_address)
            
            # Create wallets report
            report = {
                "token_address": token_address,
                "token_name": token_info.get("name", f"Token {token_address[:8]}"),
                "token_symbol": token_info.get("symbol", "UNKNOWN"),
                "analysis_time": datetime.utcnow().isoformat(),
                "total_wallets_analyzed": high_risk_wallets.get("total_wallets_analyzed", 0),
                "high_risk_wallets": high_risk_wallets.get("high_risk_wallets", []),
                "recommendation": high_risk_wallets.get("recommendation", "")
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_wallets_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error identifying high-risk wallets: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "token_address": token_address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error identifying high-risk wallets for token {token_address}: {e}")
            
            raise
    
    async def predict_price(self, token_address: str, horizon: str = "24h", output_format: str = "text") -> Dict[str, Any]:
        """
        Generate price prediction for a token.
        
        Args:
            token_address: Solana token address to analyze
            horizon: Time horizon for prediction (1h, 24h, 7d)
            output_format: Output format (text, json)
            
        Returns:
            Price prediction
        """
        logger.info(f"Generating price prediction for token: {token_address} over {horizon}")
        
        try:
            # Get token information (for name/symbol)
            token_info = await self.bridge.get_token_info(token_address)
            
            # Get price prediction
            price_prediction = await self.bridge.get_price_prediction(token_address, horizon)
            
            # Create prediction report
            report = {
                "token_address": token_address,
                "token_name": token_info.get("name", f"Token {token_address[:8]}"),
                "token_symbol": token_info.get("symbol", "UNKNOWN"),
                "analysis_time": datetime.utcnow().isoformat(),
                "current_price": token_info.get("market", {}).get("price_usd", "Unknown"),
                "horizon": horizon,
                "predicted_price": price_prediction.get("predicted_price", "Unknown"),
                "price_change": price_prediction.get("price_change", "Unknown"),
                "direction": price_prediction.get("direction", "Unknown"),
                "confidence": price_prediction.get("confidence", 0),
                "supporting_patterns": price_prediction.get("supporting_patterns", []),
                "disclaimer": price_prediction.get("disclaimer", "")
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_prediction_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "token_address": token_address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error predicting price for token {token_address}: {e}")
            
            raise
    
    async def check_blacklist(self, address: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Check if an address is blacklisted.
        
        Args:
            address: Address to check
            output_format: Output format (text, json)
            
        Returns:
            Blacklist check result
        """
        logger.info(f"Checking blacklist for address: {address}")
        
        try:
            # Check blacklist
            result = await self.bridge.check_blacklist(address)
            
            # Create report
            report = {
                "address": address,
                "checked_at": datetime.utcnow().isoformat(),
                "is_blacklisted": result.get("is_blacklisted", False),
                "details": result.get("details", None)
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_blacklist_check_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error checking blacklist: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "address": address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error checking blacklist for address {address}: {e}")
            
            raise
    
    async def add_to_blacklist(self, address: str, address_type: str, reason: str, output_format: str = "text") -> Dict[str, Any]:
        """
        Add an address to the blacklist.
        
        Args:
            address: Address to blacklist
            address_type: Type of address (wallet or token)
            reason: Reason for blacklisting
            output_format: Output format (text, json)
            
        Returns:
            Blacklist addition result
        """
        logger.info(f"Adding {address_type} {address} to blacklist with reason: {reason}")
        
        try:
            # Add to blacklist
            result = await self.bridge.add_to_blacklist(address, address_type, reason)
            
            # Create report
            report = {
                "address": address,
                "type": address_type,
                "reason": reason,
                "added_at": datetime.utcnow().isoformat(),
                "success": result.get("success", False),
                "message": result.get("message", "")
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_blacklist_add_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error adding to blacklist: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "address": address,
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error adding {address_type} {address} to blacklist: {e}")
            
            raise
    
    async def get_trending_tokens(self, output_format: str = "text") -> Dict[str, Any]:
        """
        Get trending tokens.
        
        Args:
            output_format: Output format (text, json)
            
        Returns:
            Trending tokens
        """
        logger.info("Getting trending tokens")
        
        try:
            # Get trending tokens
            trending_tokens = await self.bridge.get_trending_tokens()
            
            # Create report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "trending_tokens": trending_tokens
            }
            
            # Output the report
            if output_format == "json":
                print(json.dumps(report, indent=2))
            else:
                self._print_trending_tokens_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting trending tokens: {e}")
            if output_format == "json":
                error_json = {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(json.dumps(error_json, indent=2))
            else:
                print(f"Error getting trending tokens: {e}")
            
            raise
    
    def _create_analysis_report(self,
                              token_address: str,
                              token_info: Dict[str, Any],
                              risk_assessment: Dict[str, Any],
                              pattern_analysis: Dict[str, Any],
                              high_risk_wallets: Dict[str, Any],
                              price_prediction: Dict[str, Any],
                              safety_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive analysis report.
        
        Args:
            token_address: Token address
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
        token_name = token_info.get("name", f"Token {token_address[:8]}")
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
            "token_address": token_address,
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
            "detected_patterns": patterns,
            "high_risk_wallets": top_risk_wallets[:3],  # Top 3 riskiest wallets
            "recommendations": {
                "general": safety_recommendations.get("general_recommendations", []),
                "specific": safety_recommendations.get("specific_recommendations", [])
            }
        }
        
        return report
    
    def _print_text_report(self, report: Dict[str, Any]) -> None:
        """Print a text format comprehensive analysis report."""
        print("\n" + "=" * 50)
        print(f"ANALYSIS REPORT FOR {report['token_name']} ({report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {report['token_address']}")
        print(f"Analysis Time: {report['analysis_time']}")
        
        print("\n--- SUMMARY ---")
        print(f"Current Price: {report['summary']['current_price']}")
        print(f"Risk Level: {report['summary']['risk_level']}")
        print(f"Rug Pull Probability: {report['summary']['rug_pull_probability']}")
        print(f"Pattern Summary: {report['summary']['pattern_summary']}")
        print(f"Price Prediction: {report['summary']['price_prediction']['predicted_price']} ({report['summary']['price_prediction']['direction']})")
        
        print("\n--- RISK FACTORS ---")
        for factor in report['risk_factors']:
            print(f"- {factor}")
        
        print("\n--- DETECTED PATTERNS ---")
        for pattern in report['detected_patterns']:
            print(f"- {pattern.get('type', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f}): {pattern.get('description', 'No description')}")
        
        print("\n--- HIGH RISK WALLETS ---")
        for wallet in report['high_risk_wallets']:
            print(f"- {wallet.get('wallet_address', 'Unknown')} (Risk Score: {wallet.get('risk_score', 0):.2f}, Holdings: {wallet.get('holdings_percentage', 'Unknown')})")
        
        print("\n--- RECOMMENDATIONS ---")
        print("General:")
        for rec in report['recommendations']['general']:
            print(f"- {rec}")
        
        print("\nSpecific:")
        for rec in report['recommendations']['specific']:
            print(f"- {rec}")
        
        print("\n" + "=" * 50)
    
    def _print_risk_report(self, report: Dict[str, Any]) -> None:
        """Print a text format risk assessment report."""
        print("\n" + "=" * 50)
        print(f"RISK ASSESSMENT FOR {report['token_name']} ({report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {report['token_address']}")
        print(f"Analysis Time: {report['analysis_time']}")
        
        print("\n--- RISK SUMMARY ---")
        print(f"Risk Level: {report['risk_level']}")
        print(f"Overall Risk Score: {report['overall_risk_score']:.2f}")
        print(f"Rug Pull Probability: {report['rug_pull_probability'] * 100:.1f}%")
        
        print("\n--- RISK FACTORS ---")
        print(f"Concentration Risk: {report['risk_factors']['concentration']:.2f}")
        print(f"Liquidity Risk: {report['risk_factors']['liquidity']:.2f}")
        print(f"Activity Risk: {report['risk_factors']['activity']:.2f}")
        
        print("\n--- CONCERNS ---")
        for concern in report['concerns']:
            print(f"- {concern}")
        
        print("\n--- KEY FACTORS ---")
        for factor in report['key_factors']:
            print(f"- {factor}")
        
        print("\n" + "=" * 50)
    
    def _print_patterns_report(self, report: Dict[str, Any]) -> None:
        """Print a text format pattern analysis report."""
        print("\n" + "=" * 50)
        print(f"PATTERN ANALYSIS FOR {report['token_name']} ({report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {report['token_address']}")
        print(f"Analysis Time: {report['analysis_time']}")
        print(f"Timeframe: {report['timeframe']}")
        
        print("\n--- PATTERN SUMMARY ---")
        print(f"{report['summary']}")
        
        print("\n--- DETECTED PATTERNS ---")
        for pattern in report['patterns']:
            print(f"- {pattern.get('type', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f}): {pattern.get('description', 'No description')}")
        
        if report['highest_confidence_pattern']:
            print("\n--- HIGHEST CONFIDENCE PATTERN ---")
            pattern = report['highest_confidence_pattern']
            print(f"Type: {pattern.get('type', 'Unknown')}")
            print(f"Confidence: {pattern.get('confidence', 0):.2f}")
            print(f"Description: {pattern.get('description', 'No description')}")
        
        print("\n" + "=" * 50)
    
    def _print_wallets_report(self, report: Dict[str, Any]) -> None:
        """Print a text format wallet analysis report."""
        print("\n" + "=" * 50)
        print(f"HIGH-RISK WALLETS FOR {report['token_name']} ({report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {report['token_address']}")
        print(f"Analysis Time: {report['analysis_time']}")
        print(f"Total Wallets Analyzed: {report['total_wallets_analyzed']}")
        
        print("\n--- HIGH RISK WALLETS ---")
        for wallet in report['high_risk_wallets']:
            print(f"\nWallet: {wallet.get('wallet_address', 'Unknown')}")
            print(f"  Risk Score: {wallet.get('risk_score', 0):.2f}")
            print(f"  Holdings: {wallet.get('holdings_percentage', 'Unknown')}")
            print("  Risk Factors:")
            for factor in wallet.get('risk_factors', []):
                print(f"    - {factor}")
        
        if report['recommendation']:
            print(f"\nRecommendation: {report['recommendation']}")
        
        print("\n" + "=" * 50)
    
    def _print_prediction_report(self, report: Dict[str, Any]) -> None:
        """Print a text format price prediction report."""
        print("\n" + "=" * 50)
        print(f"PRICE PREDICTION FOR {report['token_name']} ({report['token_symbol']})")
        print("=" * 50)
        print(f"Token Address: {report['token_address']}")
        print(f"Analysis Time: {report['analysis_time']}")
        print(f"Time Horizon: {report['horizon']}")
        
        print("\n--- PREDICTION SUMMARY ---")
        print(f"Current Price: {report['current_price']}")
        print(f"Predicted Price: {report['predicted_price']}")
        print(f"Price Change: {report['price_change']}")
        print(f"Direction: {report['direction']}")
        print(f"Confidence: {report['confidence']:.2f}")
        
        if report['supporting_patterns']:
            print("\n--- SUPPORTING PATTERNS ---")
            for pattern in report['supporting_patterns']:
                print(f"- {pattern.get('type', 'Unknown')} (Confidence: {pattern.get('confidence', 0):.2f}): {pattern.get('description', 'No description')}")
        
        if report['disclaimer']:
            print(f"\nDisclaimer: {report['disclaimer']}")
        
        print("\n" + "=" * 50)
    
    def _print_blacklist_check_report(self, report: Dict[str, Any]) -> None:
        """Print a text format blacklist check report."""
        print("\n" + "=" * 50)
        print(f"BLACKLIST CHECK FOR {report['address']}")
        print("=" * 50)
        print(f"Checked At: {report['checked_at']}")
        
        if report['is_blacklisted']:
            print("\nRESULT: ⚠️ ADDRESS IS BLACKLISTED ⚠️")
            
            if report['details']:
                print("\nBlacklist Details:")
                print(f"  Type: {report['details'].get('type', 'Unknown')}")
                print(f"  Reason: {report['details'].get('reason', 'Unknown')}")
                print(f"  Added At: {report['details'].get('added_at', 'Unknown')}")
        else:
            print("\nRESULT: ✅ Address is not blacklisted")
        
        print("\n" + "=" * 50)
    
    def _print_blacklist_add_report(self, report: Dict[str, Any]) -> None:
        """Print a text format blacklist addition report."""
        print("\n" + "=" * 50)
        print(f"BLACKLIST ADDITION REPORT")
        print("=" * 50)
        
        if report['success']:
            print(f"\n✅ Successfully added to blacklist:")
            print(f"  Address: {report['address']}")
            print(f"  Type: {report['type']}")
            print(f"  Reason: {report['reason']}")
            print(f"  Added At: {report['added_at']}")
        else:
            print(f"\n❌
