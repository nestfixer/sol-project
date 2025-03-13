"""
Solana ecosystem integration utilities for the Solana Token Analysis Agent Swarm.
Provides integration with Helius API, Jupiter Aggregator, Switchboard, and SPL token utilities.
"""

import asyncio
import json
import base64
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import aiohttp
from loguru import logger

class HeliusClient:
    """Client for interacting with Helius API for enhanced Solana data."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.helius.xyz/v0"):
        """
        Initialize the Helius API client.
        
        Args:
            api_key: Helius API key
            base_url: Base URL for the Helius API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def cleanup(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_enriched_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Get enriched transaction data from Helius API.
        
        Args:
            signature: Transaction signature
            
        Returns:
            Enriched transaction data
        """
        await self.initialize()
        
        url = f"{self.base_url}/transactions"
        params = {"api-key": self.api_key}
        payload = {"transactions": [signature]}
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data[0] if data else {}
                else:
                    error_text = await response.text()
                    logger.error(f"Helius API error ({response.status}): {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting enriched transaction: {str(e)}")
            return {}
    
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """
        Get token metadata from Helius API.
        
        Args:
            token_address: Token mint address
            
        Returns:
            Token metadata
        """
        await self.initialize()
        
        url = f"{self.base_url}/tokens"
        params = {"api-key": self.api_key}
        payload = {"mintAccounts": [token_address]}
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data[0] if data else {}
                else:
                    error_text = await response.text()
                    logger.error(f"Helius API error ({response.status}): {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return {}
    
    async def get_token_holders(self, token_address: str) -> List[Dict[str, Any]]:
        """
        Get token holders from Helius API.
        
        Args:
            token_address: Token mint address
            
        Returns:
            List of token holders
        """
        await self.initialize()
        
        url = f"{self.base_url}/tokens/holders"
        params = {"api-key": self.api_key}
        payload = {"mintAccount": token_address}
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Helius API error ({response.status}): {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting token holders: {str(e)}")
            return []
    
    async def get_nft_events(self, collection: Optional[str] = None, 
                             start_time: Optional[int] = None,
                             end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get NFT events from Helius API.
        
        Args:
            collection: Optional collection address
            start_time: Optional start time (Unix timestamp)
            end_time: Optional end time (Unix timestamp)
            
        Returns:
            List of NFT events
        """
        await self.initialize()
        
        url = f"{self.base_url}/nft-events"
        params = {"api-key": self.api_key}
        payload = {}
        
        if collection:
            payload["query"] = {"collection": collection}
        
        if start_time:
            if "query" not in payload:
                payload["query"] = {}
            payload["query"]["startTime"] = start_time
        
        if end_time:
            if "query" not in payload:
                payload["query"] = {}
            payload["query"]["endTime"] = end_time
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Helius API error ({response.status}): {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting NFT events: {str(e)}")
            return []
    
    async def create_webhook(self, webhook_url: str, transaction_types: List[str], 
                             account_addresses: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a webhook for transaction monitoring.
        
        Args:
            webhook_url: URL to receive webhook events
            transaction_types: Types of transactions to monitor
            account_addresses: Optional list of account addresses to filter
            
        Returns:
            Webhook creation result
        """
        await self.initialize()
        
        url = f"{self.base_url}/webhooks"
        params = {"api-key": self.api_key}
        payload = {
            "webhookURL": webhook_url,
            "transactionTypes": transaction_types,
        }
        
        if account_addresses:
            payload["accountAddresses"] = account_addresses
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Helius API error ({response.status}): {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error creating webhook: {str(e)}")
            return {}


class JupiterClient:
    """Client for interacting with Jupiter Aggregator API for DEX liquidity analysis."""
    
    def __init__(self, base_url: str = "https://quote-api.jup.ag/v6"):
        """
        Initialize the Jupiter client.
        
        Args:
            base_url: Base URL for the Jupiter API
        """
        self.base_url = base_url
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def cleanup(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_quote(self, input_mint: str, output_mint: str, amount: int, 
                      slippage_bps: int = 50) -> Dict[str, Any]:
        """
        Get a swap quote from Jupiter.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Token amount in smallest units
            slippage_bps: Slippage in basis points (1 = 0.01%)
            
        Returns:
            Swap quote data
        """
        await self.initialize()
        
        url = f"{self.base_url}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": slippage_bps
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter API error ({response.status}): {error_text}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {str(e)}")
            return {}
    
    async def get_token_price(self, token_address: str, 
                          vs_token: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v") -> Optional[Dict[str, Any]]:
        """
        Get token price in USD (using USDC as reference).
        
        Args:
            token_address: Token mint address
            vs_token: Reference token (default is USDC)
            
        Returns:
            Price data or None if error
        """
        try:
            # Use small amount for price check to minimize slippage impact
            # Assuming token has 9 decimals, adjust if needed
            amount = 1_000_000_000  # 1 token with 9 decimals
            
            quote = await self.get_quote(token_address, vs_token, amount)
            
            if not quote or "outAmount" not in quote:
                return None
            
            # Calculate price based on the output amount (USDC has 6 decimals)
            price = int(quote["outAmount"]) / 1_000_000
            normalized_price = price / (amount / 1_000_000_000)  # Price per token
            
            return {
                "price": normalized_price,
                "price_impact": quote.get("priceImpactPct", 0),
                "route_count": len(quote.get("routesInfos", [])),
                "best_route": quote.get("routePlan", [])[0].get("swapInfo", {}).get("label", "") if quote.get("routePlan") else ""
            }
        except Exception as e:
            logger.error(f"Error getting token price from Jupiter: {str(e)}")
            return None
    
    async def analyze_liquidity_depth(self, token_address: str, 
                                  vs_token: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v") -> Dict[str, Any]:
        """
        Analyze liquidity depth by testing multiple trade sizes.
        
        Args:
            token_address: Token mint address
            vs_token: Reference token (default is USDC)
            
        Returns:
            Liquidity depth analysis
        """
        # Test multiple trade sizes (assuming 9 decimals)
        amounts = [
            1_000_000_000,      # 1 token
            10_000_000_000,     # 10 tokens
            100_000_000_000,    # 100 tokens
            1_000_000_000_000,  # 1,000 tokens
        ]
        
        results = []
        
        for amount in amounts:
            try:
                quote = await self.get_quote(token_address, vs_token, amount)
                if quote:
                    results.append({
                        "input_amount": amount / 1_000_000_000,  # Convert to human-readable amount
                        "output_amount": int(quote.get("outAmount", 0)) / 1_000_000,  # USDC has 6 decimals
                        "price_impact_percent": quote.get("priceImpactPct", 0) * 100,
                        "routes_available": len(quote.get("routesInfos", []))
                    })
            except Exception as e:
                logger.error(f"Error analyzing depth at amount {amount}: {str(e)}")
                break
        
        # Calculate estimated liquidity from price impact
        total_liquidity = 0
        if results:
            # Use the largest trade with acceptable price impact to estimate liquidity
            for result in results:
                impact = result.get("price_impact_percent", 0)
                if impact > 0 and impact < 10:  # Use trades with less than 10% impact
                    liquidity = result.get("input_amount", 0) / impact * 100
                    if liquidity > total_liquidity:
                        total_liquidity = liquidity
        
        # Calculate liquidity fragility (0-1, higher is more fragile)
        fragility = 0.0
        if results and len(results) > 1:
            # Calculate normalized rate of price impact increase
            max_ideal_impact = 0.0002 * results[-1]["input_amount"]  # 0.02% per token is ideal
            actual_impact = results[-1]["price_impact_percent"]
            
            if max_ideal_impact > 0:
                fragility = min(1.0, actual_impact / max_ideal_impact / 10)
        
        return {
            "slippage_data": results,
            "estimated_liquidity_usd": total_liquidity,
            "max_trade_size": max([r.get("input_amount", 0) for r in results]) if results else 0,
            "fragility_score": fragility,
            "depth_classification": self._classify_liquidity_depth(total_liquidity),
            "estimated_max_trade_without_significant_impact": self._estimate_max_trade(total_liquidity)
        }
    
    def _classify_liquidity_depth(self, total_liquidity: float) -> str:
        """
        Classify liquidity depth into categories.
        
        Args:
            total_liquidity: Total liquidity in USD
            
        Returns:
            Classification string
        """
        if total_liquidity < 10000:
            return "very_low"
        elif total_liquidity < 100000:
            return "low"
        elif total_liquidity < 1000000:
            return "medium"
        elif total_liquidity < 10000000:
            return "high"
        else:
            return "very_high"
    
    def _estimate_max_trade(self, total_liquidity: float) -> float:
        """
        Estimate maximum trade size without significant impact.
        
        Args:
            total_liquidity: Total liquidity in USD
            
        Returns:
            Maximum recommended trade size in USD
        """
        return total_liquidity * 0.05  # ~5% of total liquidity
    
    async def get_tokens_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all tokens from Jupiter.
        
        Returns:
            List of token data
        """
        await self.initialize()
        
        try:
            # Jupiter's token list endpoint
            url = "https://token.jup.ag/all"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter token list API error ({response.status}): {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting Jupiter tokens list: {str(e)}")
            return []


class SPLTokenChecker:
    """Utility for checking SPL token compliance and details."""
    
    def __init__(self, rpc_url: str):
        """
        Initialize the SPL token checker.
        
        Args:
            rpc_url: Solana RPC URL
        """
        self.rpc_url = rpc_url
        self.session = None
    
    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def cleanup(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get detailed token info from the Solana blockchain.
        
        Args:
            token_address: Token mint address
            
        Returns:
            Token information
        """
        await self.initialize()
        
        try:
            # Get token account info
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    token_address,
                    {"encoding": "jsonParsed"}
                ]
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    account_info = result.get("result", {}).get("value", {})
                    
                    # Extract token information
                    if account_info:
                        data = account_info.get("data", {})
                        if data:
                            parsed = data.get("parsed", {})
                            info = parsed.get("info", {})
                            
                            return {
                                "address": token_address,
                                "decimals": info.get("decimals", 0),
                                "supply": info.get("supply", "0"),
                                "is_initialized": info.get("isInitialized", False),
                                "mint_authority": info.get("mintAuthority", None),
                                "freeze_authority": info.get("freezeAuthority", None),
                                "owner": account_info.get("owner", ""),
                            }
                    
                    return {"address": token_address, "error": "Token data not found"}
                else:
                    error_text = await response.text()
                    logger.error(f"Solana RPC error ({response.status}): {error_text}")
                    return {"address": token_address, "error": error_text}
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return {"address": token_address, "error": str(e)}
    
    async def check_token_compliance(self, token_address: str) -> Dict[str, Any]:
        """
        Check SPL token standard compliance.
        
        Args:
            token_address: Token mint address
            
        Returns:
            Compliance check results
        """
        token_info = await self.get_token_info(token_address)
        
        # Define compliance issues to check for
        issues = []
        
        # Check if token exists
        if "error" in token_info:
            issues.append("Token mint account does not exist or cannot be accessed")
            return {
                "token_address": token_address,
                "is_compliant": False,
                "issues": issues,
                "details": token_info
            }
        
        # Check if owned by token program
        if token_info.get("owner") != "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
            issues.append("Token mint not owned by SPL Token program")
        
        # Check if initialized
        if not token_info.get("is_initialized", False):
            issues.append("Token mint is not initialized")
        
        # Check authorities - these aren't non-compliant, but worth noting
        details = {**token_info}
        
        if token_info.get("freeze_authority"):
            details["has_freeze_authority"] = True
            # Not an issue but a risk factor to note
        
        if token_info.get("mint_authority"):
            details["has_mint_authority"] = True
            # Not an issue but a risk factor to note
        
        return {
            "token_address": token_address,
            "is_compliant": len(issues) == 0,
            "issues": issues,
            "details": details
        }
    
    async def get_token_accounts(self, token_address: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all token accounts (holders) for a specific token.
        
        Args:
            token_address: Token mint address
            limit: Maximum number of accounts to return
            
        Returns:
            List of token accounts
        """
        await self.initialize()
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getProgramAccounts",
                "params": [
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                    {
                        "encoding": "jsonParsed",
                        "filters": [
                            {
                                "dataSize": 165  # Size of token accounts
                            },
                            {
                                "memcmp": {
                                    "offset": 0,
                                    "bytes": token_address
                                }
                            }
                        ],
                        "limit": limit
                    }
                ]
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    accounts = result.get("result", [])
                    
                    # Format account information
                    formatted_accounts = []
                    for account in accounts:
                        pubkey = account.get("pubkey", "")
                        parsed_info = account.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                        
                        formatted_accounts.append({
                            "address": pubkey,
                            "owner": parsed_info.get("owner", ""),
                            "amount": parsed_info.get("tokenAmount", {}).get("amount", "0"),
                            "decimals": parsed_info.get("tokenAmount", {}).get("decimals", 0),
                            "ui_amount": parsed_info.get("tokenAmount", {}).get("uiAmount", 0)
                        })
                    
                    return formatted_accounts
                else:
                    error_text = await response.text()
                    logger.error(f"Solana RPC error ({response.status}): {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error getting token accounts: {str(e)}")
            return []
