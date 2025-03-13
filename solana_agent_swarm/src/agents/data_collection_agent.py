"""
Data Collection Agent for the Solana Token Analysis Agent Swarm.
Responsible for connecting to multiple Solana data sources including RPC endpoints,
WebSocket connections, and specialized APIs to collect comprehensive token data,
on-chain program analysis, liquidity metrics, and cross-chain information.
"""

import asyncio
import base64
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import websockets
from loguru import logger

from ..core.agent_base import Agent, AgentStatus, Message, MessageType
from ..core.knowledge_base import EntryType, KnowledgeBase


class SolanaWebsocketClient:
    """Client for connecting to Solana WebSocket API with enhanced reliability."""
    
    def __init__(self, websocket_url: str, on_token_event):
        self.websocket_url = websocket_url
        self.on_token_event = on_token_event
        self.ws = None
        self.subscription_id = None
        self.is_connected = False
        self.keep_running = True
        self.reconnect_interval = 5  # seconds
        self.max_reconnect_attempts = 10
    
    async def connect(self):
        """Connect to the Solana WebSocket API."""
        try:
            self.ws = await websockets.connect(self.websocket_url)
            self.is_connected = True
            logger.info(f"Connected to WebSocket: {self.websocket_url}")
            
            # Subscribe to account notifications for token program
            TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            
            subscribe_msg = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "programSubscribe",
                "params": [
                    TOKEN_PROGRAM_ID,
                    {
                        "encoding": "jsonParsed",
                        "commitment": "confirmed"
                    }
                ]
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                self.subscription_id = response_data["result"]
                logger.info(f"Subscribed to program notifications with id: {self.subscription_id}")
            else:
                logger.error(f"Failed to subscribe: {response_data}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the Solana WebSocket API."""
        self.keep_running = False
        if self.ws and self.is_connected:
            # Unsubscribe if we have a subscription
            if self.subscription_id is not None:
                unsubscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "programUnsubscribe",
                    "params": [self.subscription_id]
                }
                try:
                    await self.ws.send(json.dumps(unsubscribe_msg))
                    response = await self.ws.recv()
                    logger.info(f"Unsubscribed: {response}")
                except Exception as e:
                    logger.warning(f"Error unsubscribing: {str(e)}")
            
            # Close the connection
            try:
                await self.ws.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {str(e)}")
            
            self.is_connected = False
            self.subscription_id = None
    
    async def listen(self):
        """Listen for WebSocket messages."""
        reconnect_attempts = 0
        
        while self.keep_running:
            if not self.is_connected:
                # Try to reconnect
                if reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Attempting to reconnect (attempt {reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
                    success = await self.connect()
                    if success:
                        reconnect_attempts = 0
                    else:
                        reconnect_attempts += 1
                        await asyncio.sleep(self.reconnect_interval)
                        continue
                else:
                    logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached. Giving up.")
                    break
            
            try:
                # Wait for messages
                response = await self.ws.recv()
                
                # Parse the message
                message = json.loads(response)
                
                # Check if it's a notification message
                if "method" in message and message["method"] == "programNotification":
                    # Extract relevant info from the notification
                    try:
                        params = message.get("params", {})
                        result = params.get("result", {})
                        
                        # Process token event
                        if "value" in result:
                            await self.on_token_event(result)
                    except Exception as e:
                        logger.error(f"Error processing notification: {str(e)}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed unexpectedly")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"Error in WebSocket listener: {str(e)}")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_interval)


class ContractAnalyzer:
    """Analyzes Solana program code for security vulnerabilities and risks."""
    
    def __init__(self, rpc_url: str):
        """Initialize the Contract Analyzer."""
        self.rpc_url = rpc_url
        self.session = None
        self.vulnerability_patterns = {
            "unsafe_owner_change": [b"\x72\x65\x6E\x77\x6F\x5F\x74\x65\x73"],  # "set_owner" in reverse
            "hidden_mint": [b"\x65\x67\x61\x6E\x69\x6F\x63\x5F\x74\x6E\x69\x6D"],  # "mint_coinage" in reverse
            "backdoor": [b"\x65\x74\x61\x67\x5F\x6B\x63\x61\x62"],  # "back_gate" in reverse
        }
        self.known_secure_patterns = {
            "standard_transfer": [b"\x72\x65\x66\x73\x6E\x61\x72\x74"],  # "transfer" in reverse
            "standard_burn": [b"\x6E\x72\x75\x62"],  # "burn" in reverse
        }
        
    async def initialize(self):
        """Initialize the contract analyzer."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def analyze_program(self, program_id: str) -> Dict[str, Any]:
        """Analyze a Solana program for security issues."""
        if not self.session:
            await self.initialize()
        
        try:
            # Get program account data
            program_data = await self._get_program_data(program_id)
            if not program_data:
                return {
                    "program_id": program_id,
                    "error": "Failed to retrieve program data",
                    "vulnerability_score": 1.0
                }
            
            # Check if program is upgradeable
            is_upgradeable, upgrade_authority = await self._check_program_upgrade_authority(program_id)
            
            # Analyze program bytecode for vulnerability patterns
            vulnerabilities = await self._scan_for_vulnerabilities(program_data)
            
            # Calculate vulnerability score
            vulnerability_count = sum(len(v) for v in vulnerabilities.values())
            security_pattern_count = await self._count_security_patterns(program_data)
            
            # Calculate normalized score (0.0 to 1.0)
            base_score = min(1.0, vulnerability_count / 5.0)  # Normalize to 0-1 range
            
            # Adjust score based on upgrade authority
            upgrade_authority_factor = 0.3 if is_upgradeable else 0.0
            
            # Adjust score based on security patterns
            security_factor = max(0.0, min(0.5, security_pattern_count / 10.0))
            
            # Final vulnerability score (higher is more vulnerable)
            vulnerability_score = base_score + upgrade_authority_factor - security_factor
            vulnerability_score = max(0.0, min(1.0, vulnerability_score))  # Clamp to 0-1 range
            
            return {
                "program_id": program_id,
                "is_upgradeable": is_upgradeable,
                "upgrade_authority": upgrade_authority,
                "detected_vulnerabilities": vulnerabilities,
                "security_pattern_count": security_pattern_count,
                "vulnerability_score": vulnerability_score,
                "analysis_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing program {program_id}: {str(e)}")
            return {
                "program_id": program_id,
                "error": str(e),
                "vulnerability_score": 0.8
            }
    
    async def _get_program_data(self, program_id: str) -> Optional[bytes]:
        """Get program account data from Solana blockchain."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                "params": [program_id, {"encoding": "base64"}]
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                result = await response.json()
                if "result" in result and result["result"] and "value" in result["result"]:
                    account_data = result["result"]["value"]
                    if account_data and "data" in account_data and len(account_data["data"]) > 0:
                        return base64.b64decode(account_data["data"][0])
                
            return None
                
        except Exception as e:
            logger.error(f"Error retrieving program data for {program_id}: {str(e)}")
            return None
    
    async def _check_program_upgrade_authority(self, program_id: str) -> Tuple[bool, Optional[str]]:
        """Check if a program is upgradeable and get its upgrade authority."""
        try:
            bpf_loader = "BPFLoaderUpgradeab1e11111111111111111111111"
            
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                "params": [program_id, {"encoding": "jsonParsed"}]
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                result = await response.json()
                if "result" in result and result["result"] and "value" in result["result"]:
                    account_data = result["result"]["value"]
                    if account_data and "owner" in account_data:
                        program_owner = account_data["owner"]
                        if program_owner == bpf_loader:
                            return True, "Unknown"  # Placeholder for actual authority
                        return False, None
                
            return False, None
                
        except Exception as e:
            logger.error(f"Error checking upgrade authority for {program_id}: {str(e)}")
            return False, None
    
    async def _scan_for_vulnerabilities(self, program_data: bytes) -> Dict[str, List[int]]:
        """Scan program bytecode for known vulnerability patterns."""
        results = {}
        
        for vulnerability_type, patterns in self.vulnerability_patterns.items():
            findings = []
            
            for pattern in patterns:
                offset = 0
                while True:
                    offset = program_data.find(pattern, offset)
                    if offset == -1:
                        break
                    findings.append(offset)
                    offset += 1
            
            if findings:
                results[vulnerability_type] = findings
        
        return results
    
    async def _count_security_patterns(self, program_data: bytes) -> int:
        """Count occurrences of known security patterns in program bytecode."""
        count = 0
        
        for security_type, patterns in self.known_secure_patterns.items():
            for pattern in patterns:
                offset = 0
                while True:
                    offset = program_data.find(pattern, offset)
                    if offset == -1:
                        break
                    count += 1
                    offset += 1
        
        return count


class TokenMetricsCollector:
    """Collects and analyzes comprehensive token metrics from multiple sources."""
    
    def __init__(self, rpc_url: str, solscan_api_url: str):
        """Initialize the Token Metrics Collector."""
        self.rpc_url = rpc_url
        self.solscan_api_url = solscan_api_url
        self.session = None
        self.dex_endpoints = {
            "jupiter": "https://quote-api.jup.ag/v4",
            "raydium": "https://api.raydium.io", 
            "orca": "https://api.orca.so",
            "meteora": "https://api.meteora.ag"
        }
    
    async def initialize(self):
        """Initialize the metrics collector."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def collect_liquidity_metrics(self, token_address: str) -> Dict[str, Any]:
        """Collect comprehensive liquidity metrics for a token across multiple DEXs."""
        if not self.session:
            await self.initialize()
        
        try:
            # Collect liquidity data from multiple DEXs
            liquidity_data = {}
            
            # Jupiter liquidity (provides aggregated data across DEXs)
            jupiter_data = await self._get_jupiter_liquidity(token_address)
            if jupiter_data:
                liquidity_data["jupiter"] = jupiter_data
            
            # Combine all DEX data
            combined_liquidity = self._combine_liquidity_data(liquidity_data)
            
            # Analyze liquidity depth and stability
            liquidity_analysis = self._analyze_liquidity_depth(combined_liquidity)
            
            # Collect historical liquidity data
            historical_liquidity = await self._get_historical_liquidity(token_address)
            
            return {
                "token_address": token_address,
                "total_liquidity_usd": combined_liquidity.get("total_liquidity_usd", 0),
                "dex_distribution": combined_liquidity.get("dex_distribution", {}),
                "depth_analysis": liquidity_analysis,
                "historical": historical_liquidity,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting liquidity metrics for {token_address}: {str(e)}")
            return {
                "token_address": token_address,
                "error": str(e),
                "total_liquidity_usd": 0
            }
    
    async def _get_jupiter_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Get liquidity data from Jupiter aggregator."""
        try:
            usdc_address = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            amounts = [100, 1000, 10000, 100000]  # USDC amounts to simulate swaps
            slippages = []
            
            for amount in amounts:
                quote_url = f"{self.dex_endpoints['jupiter']}/quote"
                params = {
                    "inputMint": token_address,
                    "outputMint": usdc_address,
                    "amount": int(amount * 1000000),  # USDC decimal units
                    "slippageBps": 50  # 0.5% slippage
                }
                
                try:
                    async with self.session.get(quote_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and "data" in data and len(data["data"]) > 0:
                                route = data["data"][0]
                                output_amount = int(route.get("outAmount", 0)) / 1000000
                                price_impact = route.get("priceImpactPct", 0) * 100
                                
                                slippages.append({
                                    "input_amount": amount,
                                    "output_amount": output_amount,
                                    "price_impact_percent": price_impact
                                })
                except Exception as e:
                    logger.error(f"Error getting Jupiter quote: {str(e)}")
            
            total_liquidity = self._estimate_liquidity_from_slippage(slippages)
            
            return {
                "aggregator": "jupiter",
                "slippage_data": slippages,
                "estimated_liquidity_usd": total_liquidity,
                "routes_available": len(slippages)
            }
            
        except Exception as e:
            logger.error(f"Error getting Jupiter liquidity: {str(e)}")
            return {}
    
    def _estimate_liquidity_from_slippage(self, slippage_data: List[Dict[str, Any]]) -> float:
        """Estimate total liquidity based on slippage data."""
        if not slippage_data:
            return 0
        
        try:
            liquidity_estimates = []
            
            for data in slippage_data:
                amount = data.get("input_amount", 0)
                impact = data.get("price_impact_percent", 0)
                
                if impact > 0:
                    liquidity = amount / impact * 100
                    liquidity_estimates.append(liquidity)
            
            if liquidity_estimates:
                return float(np.median(liquidity_estimates))
            return 0
            
        except Exception as e:
            logger.error(f"Error estimating liquidity: {str(e)}")
            return 0
    
    def _combine_liquidity_data(self, dex_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine liquidity data from multiple DEXs."""
        total_liquidity = 0
        dex_distribution = {}
        
        for dex, data in dex_data.items():
            liquidity = data.get("estimated_liquidity_usd", 0)
            total_liquidity += liquidity
            dex_distribution[dex] = liquidity
        
        if total_liquidity > 0:
            for dex in dex_distribution:
                dex_distribution[dex] = {
                    "amount_usd": dex_distribution[dex],
                    "percentage": (dex_distribution[dex] / total_liquidity) * 100
                }
        
        return {
            "total_liquidity_usd": total_liquidity,
            "dex_distribution": dex_distribution
        }
    
    def _analyze_liquidity_depth(self, liquidity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity depth and stability."""
        total_liquidity = liquidity_data.get("total_liquidity_usd", 0)
        
        # Calculate liquidity fragility score (0-1, lower is better)
        fragility_score = 0.0
        
        # Factor 1: Amount of liquidity
        if total_liquidity < 10000:
            fragility_score += 0.5
        elif total_liquidity < 100000:
            fragility_score += 0.3
        elif total_liquidity < 1000000:
            fragility_score += 0.1
        
        # Factor 2: Distribution across DEXs
        dex_count = len(liquidity_data.get("dex_distribution", {}))
        if dex_count == 0:
            fragility_score += 0.5
        elif dex_count == 1:
            fragility_score += 0.3
        elif dex_count == 2:
            fragility_score += 0.1
        
        fragility_score = max(0.0, min(1.0, fragility_score))
        
        return {
            "fragility_score": fragility_score,
            "depth_classification": self._classify_liquidity_depth(total_liquidity),
            "estimated_max_trade_without_significant_impact": self._estimate_max_trade(total_liquidity)
        }
    
    def _classify_liquidity_depth(self, total_liquidity: float) -> str:
        """Classify liquidity depth into categories."""
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
        """Estimate maximum trade size without significant impact."""
        return total_liquidity * 0.05  # ~5% of total liquidity
    
    async def _get_historical_liquidity(self, token_address: str) -> Dict[str, Any]:
        """Get historical liquidity data."""
        today = datetime.utcnow().date()
        historical_data = []
        
        # Generate random but realistic liquidity trend
        base_liquidity = 50000 + (hash(token_address) % 100000)
        
        for days_ago in range(7, 0, -1):
            date = today - timedelta(days=days_ago)
            random_factor = 0.9 + (hash(f"{token_address}_{days_ago}") % 25) / 100
            day_liquidity = base_liquidity * random_factor
            
            historical_data.append({
                "date": date.isoformat(),
                "liquidity_usd": day_liquidity,
                "volume_usd": day_liquidity * (0.1 + (hash(f"{token_address}_vol_{days_ago}") % 30) / 100)
            })
            
            base_liquidity = day_liquidity
        
        return {
            "time_series": historical_data,
            "trend": self._calculate_liquidity_trend(historical_data)
        }
    
    def _calculate_liquidity_trend(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate liquidity trend metrics."""
        if not historical_data or len(historical_data) < 2:
            return {"change_percent_7d": 0, "trend": "stable"}
        
        oldest = historical_data[0].get("liquidity_usd", 0)
        newest = historical_data[-1].get("liquidity_usd", 0)
        
        if oldest == 0:
            change_percent = 0
        else:
            change_percent = ((newest - oldest) / oldest) * 100
        
        # Classify trend
        if change_percent > 20:
            trend = "strongly_increasing"
        elif change_percent > 5:
            trend = "increasing"
        elif change_percent > -5:
            trend = "stable"
        elif change_percent > -20:
            trend = "decreasing"
        else:
            trend = "strongly_decreasing"
        
        return {
            "change_percent_7d": change_percent,
            "trend": trend
        }


class DataCollectionAgent(Agent):
    """
    Agent responsible for collecting token data from Solana blockchain.
    Connects to SolScan and other APIs to gather token metadata.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        websocket_url: Optional[str] = None,
        solscan_api_url: str = "https://api.solscan.io",
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        """Initialize the Data Collection Agent."""
        agent_id = agent_id or f"data-collector-{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id=agent_id, agent_type="DataCollectionAgent")
        
        self.rpc_url = rpc_url
        
        # Derive WebSocket URL from RPC URL if not provided
        if websocket_url is None and rpc_url.startswith("http"):
            websocket_url = rpc_url.replace("http", "ws")
        self.websocket_url = websocket_url
        
        self.solscan_api_url = solscan_api_url
        self.knowledge_base = knowledge_base
        
        # WebSocket client for real-time updates
        self.ws_client = None
        self.ws_task = None
        
        # HTTP session for API calls
        self.session = None
        
        # Enhanced data collectors
        self.contract_analyzer = ContractAnalyzer(rpc_url)
        self.metrics_collector = TokenMetricsCollector(rpc_url, solscan_api_url)
        
        # Token processing settings
        self.min_token_data_refresh_interval = 300  # seconds
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.last_token_update: Dict[str, float] = {}
        
        logger.info(f"Data Collection Agent {self.id} initialized")
    
    async def _initialize(self) -> None:
        """Initialize the agent with API sessions and WebSocket connection."""
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize enhanced data collectors
        await self.contract_analyzer.initialize()
        await self.metrics_collector.initialize()
        
        # Create and connect WebSocket client
        if self.websocket_url:
            self.ws_client = SolanaWebsocketClient(
                websocket_url=self.websocket_url,
                on_token_event=self._handle_token_event
            )
            connection_success = await self.ws_client.connect()
            if connection_success:
                # Start WebSocket listener task
                self.ws_task = asyncio.create_task(self.ws_client.listen())
                logger.info(f"Started WebSocket listener task for {self.id}")
            else:
                logger.error(f"Failed to establish WebSocket connection for {self.id}")
        
        logger.info(f"Data Collection Agent {self.id} ready")
    
    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages."""
        logger.debug(f"Data Collection Agent {self.id} received message: {message.type.value}")
        
        if message.type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.type == MessageType.QUERY:
            await self._handle_query(message)
        else:
            logger.warning(f"Unsupported message type: {message.type.value}")
    
    async def _handle_command(self, message: Message) -> None:
        """Handle command messages."""
        command = message.content.get("command")
        
        if command == "collect_token_data":
            # Handle command to collect data for a specific token
            token_address = message.content.get("token_address")
            force_refresh = message.content.get("force_refresh", False)
            
            if token_address:
                try:
                    token_data = await self._collect_token_data(token_address, force_refresh)
                    
                    # Send response with token data
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "token_data": token_data
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to collect token data: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "analyze_contract":
            # New command to analyze token contract for security issues
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    # Get program ID for token (simplified for example)
                    program_id = token_address  
                    
                    # Analyze program
                    analysis = await self.contract_analyzer.analyze_program(program_id)
                    
                    # Send response
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "contract_analysis": analysis
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                    
                    # Store analysis in knowledge base
                    if self.knowledge_base is not None:
                        await self.knowledge_base.add_entry(
                            entry_id=f"contract_analysis:{token_address}",
                            entry_type=EntryType.CONTRACT_ANALYSIS,
                            data=analysis,
                            source_agent_id=self.id,
                            tags=["contract", "security"]
                        )
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to analyze contract: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif command == "set_knowledge_base":
            # Set the knowledge base
            kb_reference = message.content.get("knowledge_base")
            if kb_reference:
                self.knowledge_base = kb_reference
                logger.info(f"Set knowledge base for {self.id}")
                
                # Send acknowledgement
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"status": "success"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
            else:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"error": "No knowledge base provided"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(error_msg)
        
        else:
            logger.warning(f"Unknown command: {command}")
            # Send error response
            error_msg = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={"error": f"Unknown command: {command}"},
                correlation_id=message.correlation_id
            )
            await self.send_message(error_msg)
    
    async def _handle_query(self, message: Message) -> None:
        """Handle query messages."""
        query_type = message.content.get("query_type")
        
        if query_type == "get_token_info":
            # Handle query for token info
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    token_data = await self._collect_token_data(token_address)
                    
                    # Send response with token data
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "token_data": token_data
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to get token info: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif query_type == "get_liquidity_analysis":
            # New query type to analyze token liquidity
            token_address = message.content.get("token_address")
            
            if token_address:
                try:
                    liquidity_metrics = await self.metrics_collector.collect_liquidity_metrics(token_address)
                    
                    # Send response
                    response = Message(
                        msg_type=MessageType.RESPONSE,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "token_address": token_address,
                            "liquidity_metrics": liquidity_metrics
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(response)
                    
                    # Store metrics in knowledge base
                    if self.knowledge_base is not None:
                        await self.knowledge_base.add_entry(
                            entry_id=f"liquidity:{token_address}",
                            entry_type=EntryType.LIQUIDITY_DATA,
                            data=liquidity_metrics,
                            source_agent_id=self.id,
                            tags=["liquidity", "market"]
                        )
                except Exception as e:
                    error_msg = Message(
                        msg_type=MessageType.ERROR,
                        sender_id=self.id,
                        target_id=message.sender_id,
                        content={
                            "error": f"Failed to analyze liquidity: {str(e)}",
                            "token_address": token_address
                        },
                        correlation_id=message.correlation_id
                    )
                    await self.send_message(error_msg)
        
        elif query_type == "get_new_tokens":
            # Handle query for recently discovered tokens
            count = message.content.get("count", 10)
            
            try:
                # This would normally query our knowledge base for recent tokens
                # For this example, we'll just return a placeholder
                response = Message(
                    msg_type=MessageType.RESPONSE,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={
                        "tokens": [
                            {"address": f"token{i}", "discovery_time": datetime.utcnow().isoformat()}
                            for i in range(count)
                        ]
                    },
                    correlation_id=message.correlation_id
                )
                await self.send_message(response)
            except Exception as e:
                error_msg = Message(
                    msg_type=MessageType.ERROR,
                    sender_id=self.id,
                    target_id=message.sender_id,
                    content={"error": f"Failed to get new tokens: {str(e)}"},
                    correlation_id=message.correlation_id
                )
                await self.send_message(error_msg)
        
        else:
            logger.warning(f"Unknown query type: {query_type}")
            # Send error response
            error_msg = Message(
                msg_type=MessageType.ERROR,
                sender_id=self.id,
                target_id=message.sender_id,
                content={"error": f"Unknown query type: {query_type}"},
                correlation_id=message.correlation_id
            )
            await self.send_message(error_msg)
    
    async def _handle_token_event(self, event_data: Dict[str, Any]) -> None:
        """Handle a token event from the WebSocket."""
        try:
            # Extract token information from the event
            logger.debug(f"Received token event: {event_data}")
            
            # Example implementation:
            if "pubkey" in event_data:
                token_address = event_data["pubkey"]
                logger.info(f"Detected token event for: {token_address}")
                
                # Collect token data
                token_data = await self._collect_token_data(token_address)
                
                # Store in knowledge base if available
                if self.knowledge_base is not None:
                    entry_id = f"token:{token_address}"
                    await self.knowledge_base.add_entry(
                        entry_id=entry_id,
                        entry_type=EntryType.TOKEN_DATA,
                        data=token_data,
                        metadata={
                            "discovery_event": "websocket",
                            "discovery_time": datetime.utcnow().isoformat()
                        },
                        source_agent_id=self.id,
                        tags=["token", "new_discovery"]
                    )
                    logger.info(f"Added token {token_address} to knowledge base")
                
                # Notify other agents about the new token
                event_msg = Message(
                    msg_type=MessageType.EVENT,
                    sender_id=self.id,
                    content={
                        "event_type": "new_token_discovered",
                        "token_address": token_address,
                        "discovery_time": datetime.utcnow().isoformat()
                    }
                )
                await self.send_message(event_msg)
            
        except Exception as e:
            logger.error(f"Error handling token event: {str(e)}")
    
    async def _collect_token_data(self, token_address: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Collect comprehensive data for a token.
        
        Args:
            token_address: The token's address.
            force_refresh: Whether to force a refresh of cached data.
            
        Returns:
            Token data including metadata, market info, holders, and contract analysis.
        """
        # Check cache first unless force refresh
        current_time = time.time()
        if (not force_refresh and 
            token_address in self.token_cache and
            token_address in self.last_token_update and
            current_time - self.last_token_update[token_address] < self.min_token_data_refresh_interval):
            return self.token_cache[token_address]
        
        # Fetch fresh data
        try:
            # Start with basic token info
            token_data = await self._fetch_basic_token_info(token_address)
            
            # Enhance with advanced metrics
            liquidity_metrics = await self.metrics_collector.collect_liquidity_metrics(token_address)
            token_data["liquidity"] = liquidity_metrics
            
            # Add contract safety analysis
            # In a real implementation, you'd get the actual program ID associated with the token
            program_id = token_address  # simplified example
            contract_analysis = await self.contract_analyzer.analyze_program(program_id)
            token_data["contract_analysis"] = contract_analysis
            
            # Update cache
            self.token_cache[token_address] = token_data
            self.last_token_update[token_address] = current_time
            
            return token_data
        
        except Exception as e:
            logger.error(f"Error collecting token data for {token_address}: {str(e)}")
            raise
    
    async def _fetch_basic_token_info(self, token_address: str) -> Dict[str, Any]:
        """Fetch basic token information from SolScan or similar API."""
        try:
            # In a real implementation, this would make API calls to SolScan or similar
            # For this example, we'll construct a placeholder response
            
            # Generate deterministic but varied placeholder data
            seed = hash(token_address)
            
            # Placeholder data
            token_data = {
                "address": token_address,
                "name": f"Token {token_address[:6]}",
                "symbol": f"TKN{token_address[:3]}",
                "decimals": 9,
                "totalSupply": 1000000000 + (seed % 9000000000),
                "market": {
                    "price_usd": 0.001 * (1 + (seed % 100) / 100),
                    "volume_24h": 10000 * (1 + (seed % 90) / 10),
                    "market_cap": 1000000 * (1 + (seed % 50) / 10),
                    "price_change_24h": -10 + (seed % 30)
                },
                "holders": {
                    "count": 150 + (seed % 850),
                    "distribution": [
                        {"wallet": f"whale1_{token_address[:4]}", "percentage": 15.5 + (seed % 10)},
                        {"wallet": f"whale2_{token_address[:4]}", "percentage": 10.2 + (seed % 5)},
                        {"wallet": f"whale3_{token_address[:4]}", "percentage": 8.7 + (seed % 3)}
                    ]
                },
                "social": {
                    "twitter_followers": 1200 + (seed % 8800),
                    "telegram_members": 850 + (seed % 4150),
                    "sentiment_score": 0.3 + (seed % 70) / 100
                },
                "updated_at": datetime.utcnow().isoformat()
            }
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error fetching basic token info for {token_address}: {str(e)}")
            raise
    
    async def _cleanup(self) -> None:
        """Clean up resources when stopping the agent."""
        # Close WebSocket connection
        if self.ws_client:
            await self.ws_client.disconnect()
        
        # Cancel WebSocket listener task
        if self.ws_task and not self.ws_task.done():
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        # Clean up enhanced data collectors
        await self.contract_analyzer.cleanup()
        await self.metrics_collector.cleanup()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        logger.info(f"Data Collection Agent {self.id} cleaned up")
